import os
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2
from sklearn.neighbors import KDTree
from einops import rearrange, repeat

from Fusion3DSeg import camera_utils as cam_utils
from Fusion3DSeg.intersections import point_inside_polyhedra




class FrameData:
    def __init__(self, tof, point_range=None, decimation=1, depth_hw=(256, 192)):
        self.point_range = point_range
        self.decimation = decimation
        self.depth_hw = depth_hw
        self.mask = np.ones(depth_hw, bool)
        dirname = Path(tof.split('PointcloudMergeResults')[0])
        with open(tof, 'rb') as fp:
            tof = pickle.load(fp)
            self.tofcamedata = [dirname/data['fileName'].strip() for data in tof]

    def __len__(self):
        return len(self.tofcamedata)

    def __getitem__(self, i):
        with open(self.tofcamedata[i], 'rb') as fp:
            data = pickle.load(fp)
        frame_name = str(data['frameNumber'])
        orgpts = np.array(data['orgPoints'])
        points = np.array(data['modPoints'])
        normals = np.array(data['modSurfaceNormals'])
        colors = np.array(data['orgColorPoints'])
        if self.point_range is not None:
            valids = self.get_valid(orgpts, self.point_range[0], self.point_range[1])
        else:
            valids = np.ones(len(points), bool)
        if self.decimation > 1:
            mask = self.mask.copy()
            mask[::self.decimation, ::self.decimation] = False
            valids[mask.reshape(-1)] = False
        return frame_name, points, normals, colors, valids

    @staticmethod
    def get_valid(points, mindist, maxdist):
        """ Function to get point valid points

        Args:
            points (np.ndarray): [N, 3] - xyz points.
            mindist (int/float): min distance.
            maxdist (int/float): max distance.

        Returns:
            np.ndarray: [N, ] - boolean mask of valid points.
        """
        # values = np.sum(points, axis=1)
        values = points[:, 2]
        mask = (values > mindist) & (values <= maxdist)
        return mask


def parse_rts(rts):
    with open(rts, 'rb') as fp: rtsdata = pickle.load(fp)
    K = rtsdata['intrinsic']
    Ks = rtsdata['intrinsicScaled']
    xyzws = rtsdata['odo_wxyz']
    wxyzs = xyzws[:, [3, 0, 1, 2]]
    translations = rtsdata['odo_xyz']
    rgb_res = rtsdata['RGB_res']
    depth_res = rtsdata['Depth_res']
    h, w, *_ = depth_res
    return Ks, w, h, wxyzs, translations


class Fusion:
    """ COLMAP inspired point cloud fusion on RTMAP-SLAM data.

    Refs:
        https://demuc.de/papers/schoenberger2016mvs.pdf
        https://github.com/colmap/colmap/blob/dev/src/mvs/fusion.cc

    Args:
        tof (str/Path): tof frame_data.txt filepath.
        rts (str/Path): rts camera data pickle filepath.
        point_range (tuple[float]): (mindist, maxdist) range of valid points.
        decimation (int): resize depth image by factor of 1/decimation. depth_hw % decimation == 0.
        save_lookups (bool): save per frame lookups at data_dir/fusion/...
    """
    def __init__(self, tof, rts, point_range=None, decimation=1, save_lookups=True):
        self.K, self.w, self.h, self.xyzws, self.translations = parse_rts(rts)
        self.frames = FrameData(tof, point_range, decimation, (self.h, self.w), )
        self.nframes = len(self.frames)
        self.npts = self.h*self.w
        self.ds_radius, self.ds_angle = None, None

        self.eyes, self.lookats, self.frustum_spoke_origins, self.frutsum_face_normals = self._get_frustum_data(
            self.K, self.w, self.h, self.xyzws, self.translations, np.arange(self.nframes)
        )

        self.pcdimg = np.arange(self.npts).reshape(self.h, self.w)
        self.pt2u, self.pt2v = np.zeros(self.npts, np.int32), np.zeros(self.npts, np.int32)
        for v_ in range(self.h):
            for u_ in range(self.w):
                pt = self.pcdimg[v_, u_]
                self.pt2u[pt] = u_
                self.pt2v[pt] = v_

        self.save_lookups = save_lookups
        if save_lookups:
            dirname = Path(tof.split('PointcloudMergeResults')[0])
            self.uv2pt_dir = dirname/'fusion'/'uv2pt'
            self.uv2pt_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _get_frustum_data(K, w, h, xyzws, translations, frame_ids=None):
        frame_ids = np.arange(len(translations)) if frame_ids is None else frame_ids
        frustum_points, frustum_edges = cam_utils.get_camera_frustum(K, w, h)
        frustum_points = cam_utils.camera2world(frustum_points, xyzws, translations, rescale=1)
        eyes, directions, lookats = cam_utils.get_frustum_unit_vectors(frustum_points)
        frusturm_normals = cam_utils.get_frustum_face_normals(eyes, frustum_points[:, 1:-1, :])

        eyes = eyes[frame_ids]
        lookats = lookats[frame_ids]
        frustum_spoke_origins = repeat(eyes[frame_ids], 'n c -> n r c', r=4)
        frutsum_face_normals = frusturm_normals[frame_ids]

        return eyes, lookats, frustum_spoke_origins, frutsum_face_normals

    @classmethod
    def patch_downsample(
            cls, points, normals, colors,
            height, width, stride,
            max_distance, min_cosine,
            pcdimg, pt2u, pt2v, non_merged=None,
        ):
        """ Function to downsample/sparsify single frame point cloud(depth reprojection).

        Args:
            points (np.ndarray[float]): [N, 3] - 3D xyz points.
            normals (np.ndarray[float]): [N, 3] - pointwise surface unit normals.
            colors (np.ndarray): [N, 3] - pointwise rgb colors.
            height (int): depth image height.
            width (int): depth image width.
            stride (int): down-sampling patch size.
            max_distance (float): distance threshold for merging.
            min_cosine (float): cosine threshold for merging.
            pcdimg (np.ndarray[int]): [height, width] - image with point index at each pixel.
                                        => (0, 1, 2, ..., N = height.width).reshape(height, width)
            pt2u (np.ndarray[int]): [N, ] - points to corresponding depth image x-coordinate lookup.
            pt2v (np.ndarray[int]): [N, ] - points to corresponding depth image y-coordinate lookup.
            non_merged (np.ndarray[bool]): [height, width] - pre-existing visited table.

        Returns:
            np.ndarray[flaot]: ds_pts - [M, 3] downsampled points. M <= N.
            np.ndarray[flaot]: ds_norms - [M, 3] downsampled normals.
            np.ndarray[flaot]: ds_clrs - [M, 3] downsampled colors.
            np.ndarray[int]: uv2pt - [heigh*width, ] - depth pixel to ds point index lookup. uv2pt[u*w + v] = ds_pt_idx.
            np.ndarray[int]: nmerges - [M, ] - No. of points merged to each ds point lookup.
        """
        def criterion(ds, points, ds_normal, normals, max_distance, min_cosine):
            dist = np.linalg.norm(points - ds[None, :], axis=-1)
            mask = dist < max_distance
            cos = np.einsum('ij, j -> i', normals, ds_normal)
            mask = mask & (cos > min_cosine)
            return mask

        indices = np.arange(len(points))
        np.random.shuffle(indices)

        non_merged = np.ones((height, width), dtype=bool) if non_merged is None else non_merged
        uv2pt = np.full(height*width, -1, np.int32)
        count = height*width
        half = stride//2
        ds_pts, ds_norms, ds_clrs, nmerges = [], [], [], []
        npts = 0
        for i_, pt in enumerate(indices):
            u_, v_ = pt2u[pt], pt2v[pt]
            if not non_merged[v_, u_]: continue
            if not count: break

            starti, endi = max(0, v_ - half), v_ + half + 1
            startj, endj = max(0, u_ - half), u_ + half + 1

            query_patch = (pcdimg[starti:endi, startj:endj]).reshape(-1)
            query_patch = query_patch[non_merged[starti:endi, startj:endj].reshape(-1)]

            pts, norms, clrs = points[query_patch], normals[query_patch], colors[query_patch]
            ds_pt, ds_norm, ds_clr = points[pt], normals[pt], colors[pt]

            mask = criterion(ds_pt, pts, ds_norm, norms, max_distance, min_cosine)

            nmergables = mask.sum()
            merged_indices = query_patch[mask]
            count -= nmergables
            ds_pts.append(np.mean(pts[mask], axis=0))
            ds_clrs.append(np.mean(clrs[mask], axis=0))
            norm = np.mean(norms[mask], axis=0)
            ds_norms.append(norm/np.linalg.norm(norm))
            nmerges.append(nmergables)
            uv2pt[merged_indices] = npts
            npts += 1

            non_merged[pt2v[merged_indices], pt2u[merged_indices]] = False

        return np.array(ds_pts), np.array(ds_norms), np.array(ds_clrs), uv2pt, np.array(nmerges)

    def fuse(self, radius=0.05, angle=10, stride=None, max_depth=10, skip=1, verbose=False):
        """ Fuction fuse + downsample framewise point clouds into combined sparse point cloud.

        Args:
            radius (float): distance threshold for merging.
            angle (int): surface normal angle threshold for merging.
            stride (int): down-sampling patch size.
            max_depth (float): maximum sensor range.
            skip (int): jump/skip frames.
            verbose (bool): print progress.
        """
        def criterion(ds, points, ds_normal, normals, max_distance, min_cosine):
            dist = np.linalg.norm(points - ds[None, :], axis=-1)
            mask = dist < max_distance
            cos = np.einsum('ij, j -> i', normals, ds_normal)
            mask = mask & (cos > min_cosine)
            return mask

        self.ds_radius, self.ds_angle = radius, angle
        stride = max(10, int(radius*200)) if stride is None else stride
        half = stride//2
        min_cosine = np.cos(np.deg2rad(angle))
        intersections = np.ones(self.npts, dtype=bool)

        for start in range(0, self.nframes):
            frame_name, ds_pts, ds_norms, ds_clrs, valid_mask = self.frames[start]
            if valid_mask.any(): break
        ds_pts, ds_norms, ds_clrs, uv2pt, nmerges = self.patch_downsample(
            ds_pts, ds_norms, ds_clrs,
            self.h, self.w, stride, radius, min_cosine,
            self.pcdimg, self.pt2u, self.pt2v,
            valid_mask.reshape(self.h, self.w),
        )
        if self.save_lookups: self._save_uv2pt(uv2pt, frame_name)
        occurences = np.ones(len(ds_pts), np.uint32)

        for j in range(start + 1, self.nframes, skip):
            if verbose: print(f'fusing frame: {j + 1}, total points = {len(ds_pts)}, previous intersections = {intersections.sum()}')
            frame_name, query_pts, query_norm, query_clr, query_valid = self.frames[j]
            if not query_valid.any(): continue

            uv2pt = np.full(self.npts, -1, np.int32)
            plane_pts, plane_norms = self.frustum_spoke_origins[j], self.frutsum_face_normals[j]
            far_plane_pt = self.eyes[j] + max_depth*self.lookats[j]
            far_plane_norm = -self.lookats[j]
            plane_pts = np.vstack([plane_pts, far_plane_pt[None, :]])
            plane_norms = np.vstack([plane_norms, far_plane_norm[None, :]])

            intersections = point_inside_polyhedra(ds_pts, plane_pts, plane_norms,)
            if intersections.any():
                x_indices = np.where(intersections)[0]
                x_pts, x_norm, x_clr = ds_pts[intersections], ds_norms[intersections], ds_clrs[intersections]
                x_merges, x_occ = nmerges[intersections], occurences[intersections]

                uv = cam_utils.points2pixel(x_pts, self.K, self.xyzws[j], self.translations[j])
                u, v = uv

                non_merged = query_valid.reshape(self.h, self.w)
                count = self.npts
                for i_, (idx, (u_, v_)) in enumerate(zip(x_indices, uv.T)):
                    if not count: break

                    starti, endi = max(0, v_ - half), v_ + half + 1
                    startj, endj = max(0, u_ - half), u_ + half + 1

                    query_patch = (self.pcdimg[starti:endi, startj:endj]).reshape(-1)
                    valid = non_merged[starti:endi, startj:endj].reshape(-1)
                    if not valid.any(): continue
                    query_patch = query_patch[valid]

                    patch_pts, patch_norms, patch_clrs = query_pts[query_patch], query_norm[query_patch], query_clr[query_patch]
                    ds_pt, ds_norm, ds_clr = x_pts[i_], x_norm[i_], x_clr[i_]

                    mask = criterion(ds_pt, patch_pts, ds_norm, patch_norms, radius, min_cosine)
                    matches = mask.sum()

                    if matches:
                        count -= matches
                        x_pts[i_] = np.mean(np.vstack([patch_pts[mask], ds_pt[None, :]]), axis=0)
                        x_clr[i_] = np.mean(np.vstack([patch_clrs[mask], ds_clr[None, :]]), axis=0)
                        norm = np.mean(np.vstack([patch_norms[mask], ds_norm[None, :]]), axis=0)
                        x_norm[i_] = norm/np.linalg.norm(norm)
                        x_merges[i_] += matches
                        x_occ[i_] += 1
                        merged_indices = query_patch[mask]
                        uv2pt[merged_indices] = idx
                        non_merged[self.pt2v[merged_indices], self.pt2u[merged_indices]] = False

                ds_pts[intersections] = x_pts
                ds_norms[intersections] = x_norm
                ds_clrs[intersections] = x_clr
                nmerges[intersections] = x_merges
                occurences[intersections] = x_occ

            if non_merged.any():
                dsq_pts, dsq_norms, dsq_clrs, dsq_uv2pt, dsq_merges = self.patch_downsample(
                    query_pts, query_norm, query_clr,
                    self.h, self.w, 2*stride, radius, min_cosine,
                    self.pcdimg, self.pt2u, self.pt2v, non_merged
                )

                uv2pt_mask = dsq_uv2pt != -1
                uv2pt[uv2pt_mask] = dsq_uv2pt[uv2pt_mask] + len(ds_pts)

                ds_pts = np.vstack([ds_pts, dsq_pts])
                ds_norms = np.vstack([ds_norms, dsq_norms])
                ds_clrs = np.vstack([ds_clrs, dsq_clrs])
                nmerges = np.hstack([nmerges, dsq_merges])
                occurences = np.hstack([occurences, np.ones(len(dsq_pts), np.uint32)])

            if self.save_lookups: self._save_uv2pt(uv2pt, frame_name)

        return ds_pts, ds_norms, ds_clrs, nmerges, occurences

    def _save_uv2pt(self, uv2pt, frame_name):
        np.save(self.uv2pt_dir/f'{frame_name}.npy', uv2pt)

    @staticmethod
    def filter(values, threshold, data=None, less_than=False):
        """ Function to filter out points and data satisfying [values > threshold] condition.

        Args:
            values (np.ndarray):[N, ] - values.
            threshold (float): threshold value.
            data (list[np.ndarray]): [M, N, ...] - list of data with legth N.
            less_than (bool): condition = values <= threshold if less_than else values >= threshold

        Returns:
            list[np.ndarray] - [M, P, ...] - list of data satisfying given condition.
        """
        mask = values <= threshold if less_than else values >= threshold
        if data is None: return mask, None
        out = []
        for d in data:
            out.append(d[mask])
        return mask, out

    def dump_data(
            self, dirname, points,
            normals=None, colors=None, nmerges=None, occurences=None,
            compute_adjacency=True,
            verbose=False,
        ):
        dirname = Path(dirname)
        dirname.mkdir(exist_ok=True, parents=True)
        if verbose: print(f'writing fusion data into "{dirname}" directory')

        data = {
            'points':points,
            'normals':normals,
            'colors':colors,
            'nmerges':nmerges,
            'occurences':occurences,
            'nframes':self.nframes,
            'depth_hw':(self.h, self.w),
        }
        with (dirname/'fusion'/'fusion_data.pkl').open('wb') as fp: pickle.dump(data, fp)
        if compute_adjacency:
            if verbose: print('computing adjacency ...')
            if self.ds_radius is None:
                adj = None
            else:
                tree = KDTree(points)
                adj = tree.query_radius(points, r=2*self.ds_radius)
            with (dirname/'fusion'/'adj.pkl').open('wb') as fp:
                pickle.dump(np.array(adj, dtype=object), fp)

        vec3 = o3d.utility.Vector3dVector
        pcd = o3d.geometry.PointCloud(vec3(points))
        if colors is not None: pcd.colors = vec3(colors)
        if normals is not None: pcd.normals = vec3(normals)
        if verbose: o3d.visualization.draw_geometries([pcd], 'fused point cloud')
        radius = str(self.ds_radius).replace('.', '_')
        o3d.io.write_point_cloud(
            str(dirname/'fusion'/f'fusion_{radius}_{self.ds_angle}.ply'), pcd,
        )

    @classmethod
    def load_data(cls, dirname):
        dirname = Path(dirname)
        with open(dirname/'fusion'/'fusion_data.pkl', 'rb') as fp:
            data = pickle.load(fp)

        out = [
            data['points'], data['normals'], data['colors'],
            data['nmerges'], data['occurences'],
            data['nframes'], data['depth_hw'],
        ]

        adjfile = dirname/'fusion'/'adj.pkl'
        if adjfile.is_file():
            with open(adjfile, 'rb') as fp: adj = pickle.load(fp)
        else: adj = None
        out.append(adj)

        return out
