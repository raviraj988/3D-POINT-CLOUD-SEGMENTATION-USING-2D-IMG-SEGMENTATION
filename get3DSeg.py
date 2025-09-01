import os
from pathlib import Path
import json
import time

import numpy as np
import open3d as o3d
import pandas as pd

from Fusion3DSeg.segUtils.voting import VotingSegmentation
from Fusion3DSeg.segUtils.cv import split_into_instances
from Fusion3DSeg.fusion import Fusion
from Fusion3DSeg.merge_intersecting_bb import merge_bb
# import trimesh



def segment(
        dirname, mask_dir, threshold=0.5, nclasses=133,
        filter_classes=[86, 114, 115], min_pts_per_inst=100,
        verbose=True,
    ):
    """ Function to semantic + panoptic segment 3D point cloud given masks & 2D to 3D points mappings.

    Args:
        dirname (str): path to data directory.
        threshold (float): segmentation confidence threhold.
        nclasses (int): total categories in 2D segmentation.
        filter_classes (list[int]): if not None segment only categories in filter_classes.
        min_pts_per_inst (int): objects with points < min_pts_per_insts will be classfied as "unclassified".
        verbose (bool): print progress info.

    Writes:
        dirname/segmentation
            classes.npy: np.ndarray[int] - [N, ] pointwise semantic class ids.
            info.json: list[dict[str:...]] - json file with per semantic class id information.
                                            [
                                                {
                                                    "category_id": i,
                                                    "name": "category name",
                                                    "area": total_points,
                                                    "hexcolor": "css hex color"
                                                },
                                                ...
                                            ]
            votes.npy: np.ndarray[int] - [N, nclasses] pointwise category votes.
            pcd.ply: o3d.geometry.PointCloud - semantic segmentation visualization point cloud.

        dirname/panoptic_segmentation
            ids.npy: np.ndarray[int] - [N, ] pointwise panoptic ids.
            info.json: list[dict[str:...]] - json file with per panoptic id information.
                                            [
                                                {
                                                    "id": i,
                                                    "category_id": i,
                                                    "name": "category name",
                                                    "area": total_points,
                                                    "hexcolor": "css hex color"
                                                },
                                                ...
                                            ]
            pcd.ply: o3d.geometry.PointCloud - panoptic segmentation visualization point cloud.

    """
    # =============================================================
    # Data load
    # =============================================================
    coco_meta = Path(os.path.dirname(__file__)).parent/'deeplearning'/'segmentation'/'mask2former'/'coco_meta.json'
    coco_meta = coco_meta if coco_meta.is_file() else None
    dirname = Path(dirname)

    points, norms, colors, nmerges, occurences, nframes, depth_hw, adj = Fusion.load_data(dirname)
    npts = len(points)

    start_time = time.perf_counter()
    # =============================================================
    # voting based 3D point segmentation from 2D masks
    # =============================================================
    voter = VotingSegmentation(
        npts, depth_hw, mask_dir, dirname/'fusion'/'uv2pt', nclasses, votes_file=None,
    )
    votes = voter.vote(resize=True, filename=dirname/'segmentation'/'votes.npy', verbose=verbose)
    classes = voter.segment(threshold, filter_classes)

    end_time = time.perf_counter()
    if verbose: print(f'Time taken for segmentation = {end_time - start_time} seconds')

    # =============================================================
    # CV based Instance segementation
    # =============================================================
    if adj is not None:
        insts, ids, pan_info, pan_classes = split_into_instances(
            classes, adj, nclasses, filter_classes, min_pts_per_inst, verbose=verbose
        )
    else:
        print('No adjacency list available, hence skipping instance seperation.')

    # =============================================================
    # Semantic segmentation data dump + visualization
    # =============================================================
    sem_colors, sem_pcd, sem_palette, sem_info = semantic_viz(
        points, classes, nclasses, votes=None,
        coco_data=coco_meta, outdir=dirname/'segmentation'
    )
    if verbose: o3d.visualization.draw_geometries([sem_pcd], 'semantic segmentation')

    # =============================================================
    # Panoptic segmentation data dump + visualization
    # =============================================================
    if adj is None: return votes, classes
    pan_colors, pan_pcd, pan_palette, paninfo = panoptic_viz(
        points, ids, pan_info, dirname/'panoptic_segmentation', coco_meta, colors=None, alpha=1.0
    )
    if verbose: o3d.visualization.draw_geometries([pan_pcd], 'panoptic segmentation')

    master_classes(dirname)

def remove_classes(
        dirname, mask_dir, keep_classes, threshold=0.75, nclasses=133, verbose=True,
    ):
    """ Function to remove points not belonging to keep_classes.

    Args:
        dirname (str): path to data directory.
        keep_classes (list[int]): points of classes to be kept.
        threshold (float): removal class confidence threhold.
        nclasses (int): total categories in 2D segmentation.
        verbose (bool): print progress info and save visualization.

    Writes:
        dirname/segmentation
            remaining_mask.npy: np.ndarray[bool] - [N, ] -> remaining_mask[i] = True if ith point is not removed.
            remaining.ply: o3d.geometry.PointCloud - class removal visualization.
                                                    (palette: red - remaining, blue - removed).

    Returns:
        np.ndarray[bool]: [N, ] - remaining_mask -> remaining_mask[i] = True if "i-th" point is not removed.

    """
    # =============================================================
    # Data load
    # =============================================================
    classes_csv = Path(os.path.dirname(__file__)).parent/'classes.csv'
    _, _, _, _, keep_classes = load_csv(classes_csv)

    coco_meta = Path(os.path.dirname(__file__)).parent/'deeplearning'/'segmentation'/'mask2former'/'coco_meta.json'
    coco_meta = coco_meta if coco_meta.is_file() else None
    dirname = Path(dirname)

    points, norms, colors, nmerges, occurences, nframes, depth_hw, adj = Fusion.load_data(dirname)
    colors_org = colors.copy()
    npts = len(points)

    start_time = time.perf_counter()
    # =============================================================
    # voting based 3D point segmentation from 2D masks
    # =============================================================
    votes_file = dirname/'segmentation'/'votes.npy'
    votes_file = votes_file if votes_file.is_file() else None
    voter = VotingSegmentation(
        npts, depth_hw, mask_dir, dirname/'fusion'/'uv2pt', nclasses, votes_file=votes_file,
    )
    if votes_file is None:
        votes = voter.vote(resize=True, filename=dirname/'segmentation'/'votes.npy', verbose=verbose)
    classes = voter.segment(threshold, None)

    end_time = time.perf_counter()
    if verbose: print(f'Time taken for segmentation = {end_time - start_time} seconds')

    # =============================================================
    # Generate keep classes mask
    # =============================================================
    remove_classes = np.setdiff1d(np.arange(nclasses), keep_classes)
    remove_classes = np.append(remove_classes, 133)  # added unclassfied class to remove_classes
    remove_classes = np.append(remove_classes, 134)  # added unclassfied class to remove_classes
    remaining_mask = np.ones(npts, bool)
    for cls_ in remove_classes:
        mask = classes == cls_
        remaining_mask[mask] = False

    # =============================================================
    # data dump + visualization
    # =============================================================
    (dirname/'segmentation/').mkdir(exist_ok=True, parents=True)
    np.save(dirname/'segmentation/remaining_mask.npy', remaining_mask)

    if verbose:
        colors[remaining_mask] = [1, 0, 0]
        colors[np.logical_not(remaining_mask)] = [0, 0, 1]
        # colors[classes == nclasses] = [1, 1, 0]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], 'red = remaining, blue = removed')
        o3d.io.write_point_cloud(str(dirname/'segmentation'/'remaining.ply'), pcd)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining_mask]))
        pcd.colors = o3d.utility.Vector3dVector(colors_org[remaining_mask])
        pcd.normals = o3d.utility.Vector3dVector(norms[remaining_mask])
        o3d.visualization.draw_geometries([pcd], 'cleaned point cloud')
        o3d.io.write_point_cloud(str(dirname/'segmentation'/'cleaned.ply'), pcd)
    else:
        colors[remaining_mask] = [1, 0, 0]
        colors[np.logical_not(remaining_mask)] = [0, 0, 1]
        # colors[classes == nclasses] = [1, 1, 0]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(dirname/'segmentation'/'remaining.ply'), pcd)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining_mask]))
        pcd.colors = o3d.utility.Vector3dVector(colors_org[remaining_mask])
        pcd.normals = o3d.utility.Vector3dVector(norms[remaining_mask])
        o3d.io.write_point_cloud(str(dirname/'segmentation'/'cleaned.ply'), pcd)

    removed_point_classes = classes.copy()
    removed_point_classes[remaining_mask] = 133
    removed_point_classes[removed_point_classes == 134] = 133
    
    sem_colors, sem_pcd, sem_palette, sem_info = semantic_viz(
        points, removed_point_classes, nclasses, votes=None,
        coco_data=coco_meta, outdir=dirname/'segmentation'/'removed_objects_info'
    )
    
    return remaining_mask


def semantic_viz(points, classes, nclasses, votes=None, coco_data=None, outdir='./'):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    if votes is not None: np.save(outdir/'votes.npy', votes)
    np.save(outdir/'classes.npy', classes)

    def viz(points, classes, palette):
        """ Function to visualize semanted segmented 3D points.

        Args:
            points (np.ndarray): [N, 3] - points.
            classes (np.ndarray[int]): [N, ] - point classes.
            palette (np.ndarray[float]): [nclasses, 3] - classwise color palette.

        Returns:
            np.ndarray: [N, 3] - point colors.
            np.ndarray: [M, ] - no. of points per class.
        """
        colors = np.zeros_like(points)
        classwise_pts = []
        filter_classes = np.unique(classes)

        for cls_, clr in zip(filter_classes, palette[filter_classes]):
            mask = classes == cls_
            colors[mask, :] = clr
            classwise_pts.append(mask.sum())
        return colors, filter_classes, np.array(classwise_pts)

    if coco_data is not None:
        with open(coco_data, 'r') as fp: coco_data = json.load(fp)
        coco_classes = coco_data['stuff_classes']
    else:
        coco_classes = [str(i) for i in range(nclasses)]
    coco_classes.append('unclassified')

    palette = np.random.uniform(0, 1, size=(nclasses, 3))
    palette = np.vstack((palette, np.zeros((1, 3))))
    colors, class_ids, classwise_pts = viz(points, classes, palette)

    vec3 = o3d.utility.Vector3dVector
    pcd = o3d.geometry.PointCloud(vec3(points))
    pcd.colors = vec3(colors)
    o3d.io.write_point_cloud(str(outdir/'pcd.ply'), pcd)

    class_names = [coco_classes[i] for i in class_ids]

    palette = (palette*255).astype(int)
    def tocss(clr):
        clr = "".join([hex(c).replace('0x', '').zfill(2) for c in clr])
        return "#" + clr
    palette = [tocss(clr)  for clr in palette[class_ids]]

    info = []
    for catid, area, clr, name in zip(class_ids, classwise_pts, palette, class_names):
        info.append({
            'category_id':int(catid),
            'name':name,
            'area':int(area),
            'hexcolor':clr,
        })
    with open(outdir/'info.json', 'w') as fp:
        json.dump(info, fp, indent=4)
    return colors, pcd, palette, info


def panoptic_viz(points, ids, idinfo, outdir, coco_data=None, colors=None, alpha=1.0):
    """ Function to visualize panoptic segmented 3D points.

    Args:
        points (np.ndarray[float]):[N, 3] - point coordinates.
        ids (np.ndarray[int]): [N, ] - point instance ids.
        idinfo (list[dict]): [K, ] - id info - {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts}.
        coco_data (str): path to coco_meta.json
        colors (np.ndarray[float]): [N, 3] - point colors.
        alpha (float): color blending constant.
        outdir (str): path to data dir.

    Returns:
        np.ndarray[float]: [N, 3] - colors.
        o3d.geometry.PointCloud: point cloud.
        o3d.geometry.TriangleMesh: mesh.
        np.ndarray[float]: [K, 3] - color palette.
        list[dict]: [P, ] - id info - P <= K.
                            {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts, 'color':[r, g, b], 'hexcolor':#hexcolor}.
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    np.save(outdir/'ids.npy', ids)

    def tocss(clr):
        clr = "".join([hex(c).replace('0x', '').zfill(2) for c in clr])
        return "#" + clr

    classnames = None
    if coco_data is not None:
        with open(coco_data, 'r') as fp: coco_data = json.load(fp)
        classnames = coco_data['stuff_classes']
        classnames.append('unclassified')

    allids = np.unique(ids)
    idinfo = [idinfo[id_] for id_ in allids]
    nids = len(allids)

    if colors is None:
        colors = np.zeros_like(points)
    original_colors = colors.copy()

    palette = np.random.uniform(0, 1, size=(nids, 3))
    for id_, info, clr in zip(allids, idinfo, palette):
        # info['color'] = clr
        info['hexcolor'] = tocss((clr*255).astype(int))
        info['name'] = classnames[info['category_id']] if classnames is not None else str(info['category_id'])
        mask = ids == id_
        colors[mask] = (1 - alpha)*colors[mask] + alpha*clr

    with open(outdir/'info.json', 'w') as fp:
        json.dump(idinfo, fp, indent=4)

    vec3 = o3d.utility.Vector3dVector
    pcd = o3d.geometry.PointCloud(vec3(points))
    pcd.colors = vec3(colors)
    o3d.io.write_point_cloud(str(outdir/'pcd.ply'), pcd)

    return colors, pcd, palette, idinfo


def load_semantic_segmentation(semantic_dir):
    votes = np.load(os.path.join(semantic_dir, 'votes.npy'))
    classes = np.load(os.path.join(semantic_dir, 'classes.npy'))
    with open(os.path.join(semantic_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    return votes, classes, classes, np.unique(classes), info

def load_csv(data_path):
    df = pd.read_csv(data_path)
    class_id = df['Class_ID'].tolist()
    parent_name = df['Parent'].tolist()
    parent_id = df['Parent_ID'].tolist()
    flag_infojson = df['flag_infojson'].tolist()
    flag_removal = df['flag_objremoval'].tolist()
    flag_removal = np.bool_(flag_removal)
    building_classes = np.where(flag_removal == False)[0]
    building_classes = [class_id[i] for i in building_classes]
    return class_id, parent_name, parent_id, flag_infojson, building_classes

def master_classes(dirname):
    """ Function to update parent classses in info json

    Args:
        dirname (str): path to data directory.
    """
    # dirname = Path(dirname)
    classes_csv = Path(os.path.dirname(__file__)).parent/'classes.csv'
    meta_json = Path(os.path.dirname(__file__)).parent/'classes_meta.json'
    class_id, parent_name, parent_id, flag_infojson, _ = load_csv(classes_csv)
    pcd = o3d.io.read_point_cloud(str(dirname/'panoptic_segmentation'/'pcd.ply'))
    points = np.asarray(pcd.points)
    ids = np.load(dirname/'panoptic_segmentation'/'ids.npy')
    classes = np.load(dirname/'segmentation'/'classes.npy')
    parent_classes = classes.copy()
    with open(dirname/'panoptic_segmentation'/'info.json', 'r') as fp:
        info_pan = json.load(fp)
    with open(dirname/'segmentation'/'info.json', 'r') as fp:
        info_sem = json.load(fp)
    with open(meta_json, 'r') as fp:
        classes_meta = json.load(fp)

    vec3 = o3d.utility.Vector3dVector
    final_info = []
    area_unclassidfied = 0
    palette = np.array(classes_meta['colors'])
    # palette = np.vstack((palette, np.zeros((1, 3))))
    palette = np.divide(palette, 255)

    def viz(points, classes, palette):
        """ Function to visualize semanted segmented 3D points.

        Args:
            points (np.ndarray): [N, 3] - points.
            classes (np.ndarray[int]): [N, ] - point classes.
            palette (np.ndarray[float]): [nclasses, 3] - classwise color palette.

        Returns:
            np.ndarray: [N, 3] - point colors.
            np.ndarray: [M, ] - no. of points per class.
        """
        colors = np.zeros_like(points)
        classwise_pts = []
        filter_classes = np.unique(classes)

        for cls_, clr in zip(filter_classes, palette[filter_classes]):
            mask = classes == cls_
            colors[mask, :] = clr
            classwise_pts.append(mask.sum())
        return colors, filter_classes, np.array(classwise_pts)
    
    def tocss(clr):
        clr = "".join([hex(c).replace('0x', '').zfill(2) for c in clr])
        return "#" + clr

    for info in info_pan:
        if info['category_id'] in class_id:
            mask = ids == info['id']
            info['parent_id'] = parent_id[class_id.index(info['category_id'])]
            info['parent_name'] = parent_name[class_id.index(info['category_id'])]
            info['parent_hexcolor'] = tocss((palette[info['parent_id']]*255).astype(int))
            if info['category_id'] == 133:
                unclassified_instance = info['id']
                obox_points = None
            else:
                obox = o3d.geometry.OrientedBoundingBox.create_from_points(vec3(points[mask]))
                obox_points = np.asarray(obox.get_box_points()).tolist()
            info['bbox'] = obox_points
            # obox.color = [1, 0, 0]
            # print(info['name'])
            # o3d.visualization.draw_geometries([pcd, obox], 'BB point cloud')
            if flag_infojson[class_id.index(info['category_id'])]:
                final_info.append(info)
        else:
            mask = ids == info['id']
            area_unclassidfied += np.count_nonzero(mask)
            info['parent_id'] = None
            info['parent_name'] = None
            info['parent_hexcolor'] = None
            info['bbox'] = None
    # info_pan[unclassified_instance]['area'] += area_unclassidfied
    final_info[unclassified_instance]['area'] += area_unclassidfied

    for info in info_sem:
        if info['category_id'] in class_id:
            mask = classes == info['category_id']
            info['parent_id'] = parent_id[class_id.index(info['category_id'])]
            info['parent_name'] = parent_name[class_id.index(info['category_id'])]
            info['parent_hexcolor'] = tocss((palette[info['parent_id']]*255).astype(int))
            parent_classes[mask] = int(info['parent_id'])
        else:
            mask = classes == info['category_id']
            parent_classes[mask] = classes_meta['classes'].index('unclassified')

    colors, _, _ = viz(np.array(pcd.points), parent_classes, palette)
    pcd.colors = vec3(colors)
    o3d.visualization.draw_geometries([pcd], 'parent class segmented point cloud')
    o3d.io.write_point_cloud(str(dirname/'segmentation'/'final_pcd.ply'), pcd)
    
    with open(dirname/'segmentation'/'info.json', 'w') as fp:
        json.dump(info_sem, fp, indent=4)
    with open(dirname/'panoptic_segmentation'/'info.json', 'w') as fp:
        json.dump(info_pan, fp, indent=4)
    #with open(dirname/'panoptic_segmentation'/'final_info.json', 'w') as fp:
    #    json.dump(final_info, fp, indent=4)    

    merge_bb(dirname, final_info, ids, pcd)    # Intersecting Bounding Boxes


# def separate_pcd(pcd, point_ids, ids):
#     """ Function to generate multiples trimesh.base.PointCloud given point classes/ids.

#     Args:
#         pcd (trimesh.base.PointCloud): point cloud.
#         point_ids (np.ndarray): [N, ] - point ids.
#         ids (list): list of class ids.

#     Returns:
#         list[trimesh.base.Trimesh]: list of meshes
#     """
#     vertices = np.array(pcd.vertices)
#     colors = np.array(pcd.colors)
#     seqs = []
#     pcds = []
#     for id_ in ids:
#         mask = (point_ids == id_)
#         if mask.any():
#             pcd = trimesh.points.PointCloud(vertices[mask], colors[mask])
#         else:
#             pcd = trimesh.points.PointCloud()
#         pcds.append(pcd)
#     return pcds