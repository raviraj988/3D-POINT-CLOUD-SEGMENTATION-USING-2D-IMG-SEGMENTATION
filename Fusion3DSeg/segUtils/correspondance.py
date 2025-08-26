import os
import copy
import pickle

import tqdm
import numpy as np
from sklearn.neighbors import KDTree
# from scipy.ndimage import distance_transform_edt as dt2d
import cv2
import open3d as o3d

from Fusion3DSeg.segUtils.meshUtils import to_pcd, to_mesh





class Correspondance:
    def __init__(self, pcdimgs, invalids, imgids, pcd2xy, merge_maps, depth_hw, load=None):
        """ Correspondance

        Args:
            pcdimgs (np.ndarray): [M, H, W] - images with correponding dense point indices.
            invalids (np.ndarray): [N, ] - boolean array of invalid points.
            imgids (np.ndarray): [N, ] - dense image ids.
            pcd2xy (np.ndarray): [N, 2] - dense points to xy mappings.
            merge_maps: list[list]: [N_, ...] - list of points merged to each sparse points.
            depth_hw (tuple/list): [height, width] of depth.
            load (str): .pkl filepath.

        ToDo:
            visualization/validation
            intergration with oriented bbox
            confidence score
        """
        if load is not None:
            with open(load, 'rb') as fp:
                args = pickle.load(fp)
            self.pcdimgs, self.pcd2xy, self.imgids, self.merge_maps, self.nframes = args

        else:
            nframes = len(pcdimgs)
            # get pcdimg with sprase point indices
            for i, neighbors in enumerate(merge_maps):
                xs, ys = pcd2xy[neighbors].T
                ids = imgids[neighbors]
                invs = invalids[neighbors]
                pcdimgs[ids, ys, xs] = i
                pcdimgs[ids[invs], ys[invs], xs[invs]] = -1

            self.pcdimgs = pcdimgs
            self.pcd2xy = pcd2xy
            self.imgids = imgids
            self.merge_maps = merge_maps
            self.nframes = nframes

    def save(self, filename):
        """ Function to save object state variables.

        Args:
            filename (str): .pkl filepath.
        """
        with open(filename, 'wb') as fp:
            pickle.dump((self.pcdimgs, self.pcd2xy, self.imgids, self.merge_maps, self.nframes), fp)

    def get_point(self, images, coords):
        """ Funtion to query image coordinates to get corresponding 3D point index in point cloud.

        Args:
            images (List[int]): List of query image ids(0 - M). Eg: [1, 24, 45, ..., M - 13, ...].
            coods (np.ndarray): [len(images), 2] - (x, y) coordinates wrt img_ids.
                            Eg - len(images) = k -> [(x1, y1), (x2, y2), ..., (xk, yk)].

        Returns:
            np.ndarray: [len(coords), ] - List 3D point cloud point indices.
        """

        x, y = coords.T
        indices = self.pcdimgs[images, y, x]
        return indices

    def get_pixel(self, idx):
        """ Funtion to query index of 3D point to
            get corresponding image-ids and coordinates wrt those images.

        Args:
            idx (int/list[int]): 3D point index.

        Returns:
            [n, ] - image-ids(np.ndarray).
            [n, 2] - coordinates wrt those images(np.ndarray).
                    coordinates -> [(x1, y1), (x2, y2), ..., (xn, yn)]
                                -> (xi, yi) pixel coords of image-id = i
        """
        if isinstance(idx, int):
            indices = self.merge_maps[idx]
        else:
            indices = [self.merge_maps[i] for i in idx]
            indices = np.hstack(indices)

        imgids = self.imgids[indices]
        coords = self.pcd2xy[indices]
        return imgids, coords

    @staticmethod
    def viz_proj(
            ids, coords, images, names=None, clr=(0, 0, 255),
            size=1, outdir='./proj', save_center=False,
        ):
        if names is None:
            names = [os.path.join(outdir, str(i + 1) + '.png') for i in range(len(images))]
        else:
            names = [os.path.join(outdir, name) for name in names]
        os.makedirs(outdir, exist_ok=True)
        images = images.copy()
        images[ids, coords[:, 1], coords[:, 0]] = clr
        loader = tqdm.tqdm(enumerate(zip(images[ids], coords)), total=len(ids))
        for i, (img, (x, y)) in loader:
            # cv2.circle(img, (x, y), size, clr, thickness=-1)
            img[y, x, :] = clr
            cv2.imwrite(names[ids[i]], img)

        if save_center:
            m, h, w, *_ = images.shape
            cx, cy = w/2, h/2
            x, y = coords.T
            dist = np.linalg.norm([cx - x, cy - y], axis=0)
            idx = np.argmin(dist)
            name = os.path.basename(names[idx])
            imgpath = os.path.join(os.path.dirname(names[idx]), f'center-{ids[idx] + 1}.png')
            img = images[ids[idx]]
            cv2.circle(img, (x[idx], y[idx]), size, clr, thickness=-1)
            cv2.imwrite(imgpath, img)

    @staticmethod
    def viz_reproj(pcd, indices, clr=(1, 0, 0), save=None, show=True):
        vizpcd = copy.deepcopy(pcd)
        pclr = np.asarray(vizpcd.colors)
        pclr[indices] = clr
        vizpcd.colors = o3d.utility.Vector3dVector(np.array(pclr))
        if show:
            o3d.visualization.draw_geometries([vizpcd, ])
        if save is not None:
            o3d.io.write_point_cloud(save, vizpcd)
        return vizpcd

    # @staticmethod
    # def viz_dt(dts, pcd2xy=None, imgids=None, outdir='./dt'):
    #     os.makedirs(outdir, exist_ok=True)
    #     loader = tqdm.tqdm(enumerate(dts), total=len(dts))
    #     for i, dt in loader:
    #         img = np.zeros((dt.shape[0], dt.shape[1], 3))
    #         img[..., 0] = dt
    #         if pcd2xy is not None:
    #             x, y = pcd2xy[imgids == i].T
    #             img[y, x, 2] = img.max()
    #         img = (img - img.min())/(img.max() - img.min())
    #         img = (255*img).astype(np.uint8)
    #         cv2.imwrite(os.path.join(outdir, str(i + 1) + '.png'), img)


class PointCorrespondance:
    def __init__(self, sparse_points, dense_points, radius, nframes, depth_hw, load=None):
        """ PointCorrespondance: point image position lookup.

        Args:
            sparse_points (np.ndarray[float]): [M, 3] - sparse points.
            dense_points (np.ndarray[float]): [N, 3] - dense points. N = nframes x height x width.
            radius (float): neighbour search radius.
            nframes (int): number of frames.
            depth_hw (tuple/list): [height, width] of depth.
            load (str): .pkl filepath.
        """
        if load is not None:
            with open(load, 'rb') as fp:
                args = pickle.load(fp)
            self.pcdimgs, self.pcd2xy, self.imgids, self.merge_maps, self.nframes = args

        else:
            pcd2xy, imgids, pcdimgs = self.get_lookups(nframes, depth_hw)
            # get pcdimg with sprase point indices
            self.pcdimgs = pcdimgs
            self.pcd2xy = pcd2xy
            self.imgids = imgids
            self.merge_maps = self.get_merge_maps(sparse_points, dense_points, radius)
            self.nframes = nframes

    @classmethod
    def get_xys(cls, h, w):
        """ Function to get rowwise xy coordinates.

        Args:
            h (int): height of the image.
            w (int): width of the image.
        Returns:
            np.ndarray:  [h*w, 2] - xys.
        """
        xs, ys = np.arange(w), np.arange(h)
        xs = np.tile(xs, h).reshape(-1)
        ys = np.repeat(ys, w).reshape(-1)
        xys = np.vstack((xs, ys)).T
        return xys.copy()

    @classmethod
    def get_lookups(cls, nframes, depth_hw):
        """ Function to get image to point and vice versa lookup tables.

        Args:
            nframes (int): number of frames.
            depth_hw (tuple/list): [height, width] of depth.

        Returns:
            np.ndarray: [N, 2] - points to xy mappings.
            np.ndarray: [N, ] - point image ids.
            np.ndarray: [nframes, H, W] - images of correponding point indices.
        """
        h, w = depth_hw
        hw = h*w
        npts = nframes*hw
        pcd2xy = PointCorrespondance.get_xys(h, w)
        indices = np.arange(hw)
        pcdimg = np.empty(depth_hw, dtype=np.int32)
        x, y = pcd2xy.T
        pcdimg[y, x] = indices

        pcdimgs = [pcdimg + i*hw for i in range(nframes)]
        pcd2xys = [pcd2xy for i in range(nframes)]
        imgids = [np.full(hw, i, dtype=int) for i in range(nframes)]
        pcd2xys = np.hstack(pcd2xys)
        imgids = np.hstack(imgids)
        pcdimgs = np.dstack(pcdimgs).transpose(2, 0, 1)
        return pcd2xys, imgids, pcdimgs

    @classmethod
    def get_merge_maps(cls, sparse_points, dense_points, radius=0.1):
        tree = KDTree(dense_points, leaf_size=2)
        neighbors = tree.query_radius(sparse_points, r=radius)
        merge_maps = [[] for _ in range(len(dense_points))]
        for i, pts in enumerate(neighbors):
            for pt in pts:
                merge_maps[pt].append(i)
        return np.array(merge_maps, dtype=object)

    def save(self, filename):
        """ Function to save object state variables.

        Args:
            filename (str): .pkl filepath.
        """
        with open(filename, 'wb') as fp:
            pickle.dump((self.pcdimgs, self.pcd2xy, self.imgids, self.merge_maps, self.nframes), fp)

    def get_point(self, images, coords):
        """ Funtion to query image coordinates to get corresponding 3D point index in point cloud.

        Args:
            images (List[int]): List of query image ids(0 - M). Eg: [1, 24, 45, ..., M - 13, ...].
            coods (np.ndarray): [len(images), 2] - (x, y) coordinates wrt img_ids.
                            Eg - len(images) = k -> [(x1, y1), (x2, y2), ..., (xk, yk)].

        Returns:
            np.ndarray: indices - [p, ] - List 3D point cloud point indices.
            np.ndarray: frequency - [len(coords), ] - per pixel point counts.
        """

        x, y = coords.T
        indices = self.pcdimgs[images, y, x]
        indices = self.merge_maps[indices]
        frequency = np.array([len(idx) for idx in indices])
        indices = np.hstack(indices).astype(np.int32)
        return indices, frequency

    @staticmethod
    def viz_reproj(pcd, indices, clr=(1, 0, 0), save=None, show=True):
        vizpcd = copy.deepcopy(pcd)
        pclr = np.asarray(vizpcd.colors)
        pclr[indices] = clr
        vizpcd.colors = o3d.utility.Vector3dVector(np.array(pclr))
        if show:
            o3d.visualization.draw_geometries([vizpcd, ])
        if save is not None:
            o3d.io.write_point_cloud(save, vizpcd)
        return vizpcd