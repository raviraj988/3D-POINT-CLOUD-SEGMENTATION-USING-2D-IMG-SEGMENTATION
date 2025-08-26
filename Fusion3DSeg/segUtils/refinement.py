import pandas as pd
import numpy as np
import open3d as o3d
import os
from Fusion3DSeg.segUtils.meshUtils import to_pcd, pick_points, to_mesh
from RTAB_utils.spatQuad import SpatQuadranion as Quat
from Fusion3DSeg.segUtils import planeUtils as Putil

def ReadVerticesConnectedFiles(file_connected_path):
    read_connected_file = pd.read_csv(file_connected_path, sep='delimiter')
    VID = read_connected_file['VIDs'].tolist()
    list_vertexs = [ list(map(int,(i.split(",")[1:]))) for i in VID[0:]]
    return list_vertexs


def GetactualIndex(SelectedPoints, Vertex, PLanewithPoints, BoundingPoints=None):
    idxlist = []
    indices = []
    for pt in SelectedPoints:
        actualvertexidx = np.where(np.all(Vertex[:, 0:3] == pt, axis=1))[0]
        if len(actualvertexidx) > 0:
            actualvertexidx = actualvertexidx[0]
            idx = [i for i, p in enumerate(PLanewithPoints[:, Putil.Col('indicies')]) if
                   p.intersection(set([actualvertexidx]))]
        else:
            idx = [i for i in range(len(PLanewithPoints)) if
                   np.any(np.all(PLanewithPoints[i, 3] == pt, axis=1) == True)]

        if len(idx) < 1:
            continue
        idx = idx[0]
        if idx in idxlist:
            continue
        idxlist.append(idx)
        for id in idxlist:
            indices.extend(list(PLanewithPoints[id, 1]))
    return np.array(idxlist), indices


def door_updation(outer_poly, inner_poly, normal_wall, max_distance = 0.2):
    '''
    Function to align door boundary points to wall boundary points based on distance between sides
    Args:
        outer_poly (np.ndarray): [4,3] - list of wall boundary points
        inner_poly (np.ndarray): [4,3] - list of door boundary points
        normal_wall (skspatial.objects.vector.Vector): [3,] - wall plane normal
        max_distance (float): maximum distance threshold to snap door with wall
    Returns:
        updated inner_poly (np.ndarray): [4,3] - list of updated door boundary points
    '''
    def ClosestPointOnLine(point_a, point_b, point):
        '''
        Fumction to calculate closest point on line to the new point
        Args:
        point_a (np.ndarray): [3,] - first point with which line is defined
        point_b (np.ndarray): [3,] - second point with which line is defined
        point (np.ndarray): [3,] - point which needs to be projected on line
        Returns:
        distance (float): distance between closest point on line to third point
        updated_point (np.ndarray): [3,] - closest point on line to third point
        '''
        ap = point-point_a
        ab = point_b-point_a
        updated_point = point_a + np.dot(ap,ab)/ np.dot(ab,ab) * ab
        distance = np.linalg.norm(updated_point - point)  
        return distance, updated_point
    inner_poly = inner_poly.copy()
    point_wall = outer_poly[0]
    t = point_wall.dot(normal_wall) - np.einsum('c, nc -> n', normal_wall, inner_poly)
    inner_poly = inner_poly + t[:, None]*normal_wall[None, :]

    for z, point in enumerate(inner_poly):
        for i in range(0, len(outer_poly)-1):
            dist, wall_inters_point = ClosestPointOnLine(outer_poly[i], outer_poly[i+1], point)
            if dist < max_distance:
                inner_poly[z] = wall_inters_point
        dist, wall_inters_point = ClosestPointOnLine(outer_poly[0], outer_poly[-1], point)
        if dist < max_distance:
            inner_poly[z] = wall_inters_point
    return inner_poly


def depth_floodfill_dl(PlaneswithPoints, Vertex, SelectedPoint, BoundingPoints, connected, outputpath, depth_threshold = 0.03, max_level = 50, viz_ply = False):
    ''' Function to refine 3d segmentation results using depth correction
    '''
    def floodfill_depth_points(points, total_points, adj, distance, threshold, max_level=10):
        """ Fucntion to find all connected points around a given point on plane with thresholded perpendicular distance.
        Algo:
            all_points, wall normal, wall_boundary_points, door_points, connected
            for neighbour in neighbours:
                if criterion(color[neighbour]) and criterion([class[neighbour]]) and criterion(level[neighbour]):
                    cluster.append(neighbour)
            criterion[class[neighbour]] = mask[neighbour]
        Args:
            point (int): seed index of point in point cloud. point < N.
            total_points (int): total points.
            adj (list[set]): [N, ...] - adjecency list.
            colors (np.ndarray[float]): [N, 3] - point colors.
            threshold (np.ndarray[float]): [r, g, b] - color threshold.
            max_level (int): max recursion/depth level.
        Returns:
            np.ndarray[int]: [N, ] - point instance ids.
        """
        inq = np.zeros(total_points, bool)
        points_q = points.tolist()
        inq[points] = True
        levels = [1 for _ in points_q]
        sma = np.average(distance[points], axis = 0)
        # sma = colors[points]
        npts = len(points_q)
        cluster = []
        given = inq.copy()
        while points_q:
            point = points_q.pop(0)
            level = levels.pop(0)
            depth = distance[point]
            if (level == max_level): continue
            if (np.abs(sma - depth) > threshold): continue
            if not given[point]:
                npts += 1
                sma = sma + (depth - sma)/npts
                cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if not inq[n]]
            if non_inq:
                points_q += non_inq
                levels += [level + 1 for _ in non_inq]
                inq[non_inq] = True
        return np.array(cluster)
    planeidx, indices = GetactualIndex(SelectedPoints=SelectedPoint, PLanewithPoints=PlaneswithPoints, Vertex=Vertex)
    all_ind = planeidx.tolist()
    cv_seg_path = outputpath + os.sep + 'cv_segmentation'
    os.makedirs(cv_seg_path, exist_ok=True)
    if os.path.exists(os.path.join(cv_seg_path, 'ids.npy')) and os.path.exists(os.path.join(cv_seg_path, 'pcd.ply')):
        seg_ply = o3d.io.read_point_cloud(os.path.join(cv_seg_path, 'pcd.ply'))
        instance_id = np.load(os.path.join(cv_seg_path, 'ids.npy'))
    else:
        instance_id_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'ids.npy'
        instance_id = np.load(instance_id_file)
        seg_ply_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'pcd.ply'
        seg_ply = o3d.io.read_point_cloud(seg_ply_file)
    colors_seg = np.asarray(seg_ply.colors)
    points_seg = np.asarray(seg_ply.points)
    normal_wall = PlaneswithPoints[all_ind[0], 0].normal
    points_all = Vertex[:, :3]
    colors = Vertex[:, 3:6]
    colors_dl = colors.copy()
    wall_BB = BoundingPoints[PlaneswithPoints[0, 2]]
    point_wall = wall_BB[0]
    point_wall = point_wall.reshape(1, 3)
    normal_wall = normal_wall.reshape(1, 3)
    point_vectors =  points_all[:, None, :] - point_wall[None, :, :]  # [N, M, 3]
    dp = np.einsum('nmc, mc -> mn', point_vectors, normal_wall)
    dp = np.abs(dp[0])
    if viz_ply:
        selected_point = pick_points(seg_ply)
    else:
        pcd = to_pcd(points_all, colors)
        selected_point = pick_points(pcd)
    door_id = instance_id[selected_point[0]]
    door_palette = colors_seg[selected_point[0]]
    door_points = np.where(instance_id == door_id)[0]
    print('deep learning segmented points', len(door_points))    
    depth_updated_points = floodfill_depth_points(door_points, total_points= len(points_all), adj = connected, distance = dp, threshold = depth_threshold, max_level=max_level)#0.03
    print('depth door grew points', len(depth_updated_points))
    if len(depth_updated_points) > 0:
        colors_dl[depth_updated_points] = [1, 0, 0] 
        colors_dl[door_points] = [0, 1, 0]
        pcd1 = to_pcd(points_all, colors_dl, viz = True)
        instance_id[depth_updated_points] = door_id
        colors_seg[depth_updated_points] = door_palette
        pcd2 = to_pcd(points_seg, colors_seg, viz = True)
        return instance_id, pcd2
    return instance_id, seg_ply


def depth_floodfill_point(PlaneswithPoints, Vertex, SelectedPoint, BoundingPoints, connected, outputpath, depth_threshold = 0.03, max_level = 50, viz_ply = False):
    ''' Function to extract door or window from wall using depth
    '''    
    def floodfill_depth_point(points, total_points, adj, distance, threshold, max_level=10):
        """ Fucntion to find all connected points around a given point on plane with thresholded perpendicular distance.
        Algo:
            all_points, wall normal, wall_boundary_points, door_points, connected
            for neighbour in neighbours:
                if criterion(color[neighbour]) and criterion([class[neighbour]]) and criterion(level[neighbour]):
                    cluster.append(neighbour)
            criterion[class[neighbour]] = mask[neighbour]
        Args:
            point (int): seed index of point in point cloud. point < N.
            total_points (int): total points.
            adj (list[set]): [N, ...] - adjecency list.
            colors (np.ndarray[float]): [N, 3] - point colors.
            threshold (np.ndarray[float]): [r, g, b] - color threshold.
            max_level (int): max recursion/depth level.
        Returns:
            np.ndarray[int]: [N, ] - point instance ids.
        """
        inq = np.zeros(total_points, bool)
        points_q = points.copy()
        inq[points] = True
        levels = [1 for _ in points_q]
        sma = np.average(distance[points], axis = 0)
        # sma = colors[points]
        npts = len(points_q)
        cluster = []
        while points_q:
            point = points_q.pop(0)
            level = levels.pop(0)
            depth = distance[point]
            if (level == max_level): continue
            if (np.abs(sma - depth) > threshold): continue
            npts += 1
            sma = sma + (depth - sma)/npts
            cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if not inq[n]]
            points_q += non_inq
            levels += [level + 1 for _ in non_inq]
            inq[non_inq] = True
        return np.array(cluster)
    cv_seg_path = outputpath + os.sep + 'cv_segmentation'
    os.makedirs(cv_seg_path, exist_ok=True)
    if os.path.exists(os.path.join(cv_seg_path, 'ids.npy')) and os.path.exists(os.path.join(cv_seg_path, 'pcd.ply')):
        seg_ply = o3d.io.read_point_cloud(os.path.join(cv_seg_path, 'pcd.ply'))
        instance_id = np.load(os.path.join(cv_seg_path, 'ids.npy'))
    else:
        instance_id_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'ids.npy'
        instance_id = np.load(instance_id_file)
        seg_ply_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'pcd.ply'
        seg_ply = o3d.io.read_point_cloud(seg_ply_file)
    colors_seg = np.asarray(seg_ply.colors)
    points_seg = np.asarray(seg_ply.points)

    planeidx, indices = GetactualIndex(SelectedPoints=SelectedPoint, PLanewithPoints=PlaneswithPoints, Vertex=Vertex)
    all_ind = planeidx.tolist()
    normal_wall = PlaneswithPoints[all_ind[0], 0].normal
    points_all = Vertex[:, :3]
    colors = Vertex[:, 3:6]
    colors_cv = colors.copy()
    wall_BB = BoundingPoints[PlaneswithPoints[0, 2]]
    # point_wall = Vertex[actualvertexidx, :3]
    point_wall = wall_BB[0]
    point_wall = point_wall.reshape(1, 3)
    normal_wall = normal_wall.reshape(1, 3)
    point_vectors =  points_all[:, None, :] - point_wall[None, :, :]  # [N, M, 3]
    dp = np.einsum('nmc, mc -> mn', point_vectors, normal_wall)
    dp = np.abs(dp[0])
    if viz_ply:
        selected_point = pick_points(seg_ply)
    else:
        pcd = to_pcd(points_all, colors)
        selected_point = pick_points(pcd)
    door_id = instance_id[selected_point[0]]
    door_palette = colors_seg[selected_point[0]]
    door_points = np.where(instance_id == door_id)[0]
    depth_updated_point = floodfill_depth_point(selected_point, total_points= len(points_all), adj = connected, distance = dp, threshold = depth_threshold, max_level=max_level)#0.03
    print('depth point grew points', len(depth_updated_point))
    if len(depth_updated_point) > 0:
        colors_cv[door_points] = [0, 1, 0]
        actual_updated = np.setdiff1d(depth_updated_point, door_points)
        print('length of acvtual depth grown points w.r.t dl', len(actual_updated))
        if len(actual_updated) > 0:
            colors_cv[actual_updated] = [1, 0, 0]
        pcd1 = to_pcd(points_all, colors_cv, viz = True)
        colors_seg[depth_updated_point] = door_palette
        instance_id[depth_updated_point] = door_id
        pcd2 = to_pcd(points_seg, colors_seg, viz = True)
        return instance_id, pcd2
    return instance_id, seg_ply


def color_floodfill_dl(Vertex, connected, outputpath, color_threshold = 0.1, max_level = 50, viz_ply = False):
    ''' Function to refine 3d segmentation results using color correction
    '''
    def floodfill_color_points(points, total_points, adj, colors, threshold, max_level=10):
        """ Fucntion to find all connected points around a plane points with similar color.
        Algo:
            for neighbour in neighbours:
                if criterion(color[neighbour]) and criterion([class[neighbour]]) and criterion(level[neighbour]):
                    cluster.append(neighbour)
            criterion[class[neighbour]] = mask[neighbour]
        Args:
            point (int): seed index of point in point cloud. point < N.
            total_points (int): total points.
            adj (list[set]): [N, ...] - adjecency list.
            colors (np.ndarray[float]): [N, 3] - point colors.
            threshold (np.ndarray[float]): [r, g, b] - color threshold.
            max_level (int): max recursion/depth level.
        Returns:
            np.ndarray[int]: [N, ] - point instance ids.
        """
        inq = np.zeros(total_points, bool)
        points_q = points.tolist()
        inq[points] = True
        levels = [1 for _ in points_q]
        sma = np.average(colors[points], axis = 0)
        npts = len(points_q)
        cluster = []
        given = inq.copy()
        while points_q:
            point = points_q.pop(0)
            level = levels.pop(0)
            clr = colors[point]
            if (level == max_level): continue
            if (np.abs(sma - clr) > threshold).any(): continue
            if not given[point]:
                npts += 1
                sma = sma + (clr - sma)/npts
                cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if not inq[n]]
            if non_inq:
                points_q += non_inq
                levels += [level + 1 for _ in non_inq]
                inq[non_inq] = True
        return np.array(cluster)
    cv_seg_path = outputpath + os.sep + 'cv_segmentation'
    os.makedirs(cv_seg_path, exist_ok=True)
    if os.path.exists(os.path.join(cv_seg_path, 'ids.npy')) and os.path.exists(os.path.join(cv_seg_path, 'pcd.ply')):
        seg_ply = o3d.io.read_point_cloud(os.path.join(cv_seg_path, 'pcd.ply'))
        instance_id = np.load(os.path.join(cv_seg_path, 'ids.npy'))
    else:
        instance_id_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'ids.npy'
        instance_id = np.load(instance_id_file)
        seg_ply_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'pcd.ply'
        seg_ply = o3d.io.read_point_cloud(seg_ply_file)
    colors_seg = np.asarray(seg_ply.colors)
    points_seg = np.asarray(seg_ply.points)
    points_all = Vertex[:, :3]
    colors = Vertex[:, 3:6]
    colors_dl = colors.copy()
    if viz_ply:
        selected_point = pick_points(seg_ply)
    else:
        pcd = to_pcd(points_all, colors)
        selected_point = pick_points(pcd)
    door_id = instance_id[selected_point[0]]
    door_palette = colors_seg[selected_point[0]]
    door_points = np.where(instance_id == door_id)[0]
    print('deep learning segmented points', len(door_points))    
    color_updated_points = floodfill_color_points(points = door_points, total_points = len(points_all), adj = connected, colors = colors, threshold = np.asarray([color_threshold, color_threshold, color_threshold]), max_level=max_level)
    print('color door grew points', len(color_updated_points))
    if len(color_updated_points) > 0:
        colors_dl[color_updated_points] = [1, 0, 0] 
        colors_dl[door_points] = [0, 1, 0]
        pcd1 = to_pcd(points_all, colors_dl, viz = True)
        colors_seg[color_updated_points] = door_palette
        instance_id[color_updated_points] = door_id
        pcd2 = to_pcd(points_seg, colors_seg, viz = True)
        return instance_id, pcd2
    return instance_id, seg_ply


def color_floodfill_point(Vertex, connected, outputpath, color_threshold = 0.1, max_level = 50, viz_ply = False):
    ''' Function to extract door or window using color thresholding
    '''
    def floodfill_color_point(points, total_points, adj, colors, threshold, max_level=10):
        """ Fucntion to find all connected points around a given point with similar color.
        Algo:
            for neighbour in neighbours:
                if criterion(color[neighbour]) and criterion([class[neighbour]]) and criterion(level[neighbour]):
                    cluster.append(neighbour)
            criterion[class[neighbour]] = mask[neighbour]
        Args:
            point (int): seed index of point in point cloud. point < N.
            total_points (int): total points.
            adj (list[set]): [N, ...] - adjecency list.
            colors (np.ndarray[float]): [N, 3] - point colors.
            threshold (np.ndarray[float]): [r, g, b] - color threshold.
            max_level (int): max recursion/depth level.
        Returns:
            np.ndarray[int]: [N, ] - point instance ids.
        """
        inq = np.zeros(total_points, bool)
        points_q = [points]
        inq[points] = True
        levels = [1 for _ in points_q]
        sma = colors[points]
        npts = 0 #len(points_q)
        cluster = []
        while points_q:
            point = points_q.pop(0)
            level = levels.pop(0)
            clr = colors[point]
            if (level == max_level): continue
            if (np.abs(sma - clr) > threshold).any(): continue
            npts += 1
            sma = sma + (clr - sma)/npts
            cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if not inq[n]]
            points_q += non_inq
            levels += [level + 1 for _ in non_inq]
            inq[non_inq] = True
        return np.array(cluster)
    cv_seg_path = outputpath + os.sep + 'cv_segmentation'
    os.makedirs(cv_seg_path, exist_ok=True)
    if os.path.exists(os.path.join(cv_seg_path, 'ids.npy')) and os.path.exists(os.path.join(cv_seg_path, 'pcd.ply')):
        seg_ply = o3d.io.read_point_cloud(os.path.join(cv_seg_path, 'pcd.ply'))
        instance_id = np.load(os.path.join(cv_seg_path, 'ids.npy'))
    else:
        instance_id_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'ids.npy'
        instance_id = np.load(instance_id_file)
        seg_ply_file = outputpath + os.sep + 'panoptic_segmentation' + os.sep + 'pcd.ply'
        seg_ply = o3d.io.read_point_cloud(seg_ply_file)
    colors_seg = np.asarray(seg_ply.colors)
    points_seg = np.asarray(seg_ply.points)
    points_all = Vertex[:, :3]
    colors = Vertex[:, 3:6]
    colors_cv = colors.copy()
    if viz_ply:
        selected_point = pick_points(seg_ply)
    else:
        pcd = to_pcd(points_all, colors)
        selected_point = pick_points(pcd)
    door_id = instance_id[selected_point[0]]
    door_palette = colors_seg[selected_point[0]]
    door_points = np.where(instance_id == door_id)[0]
    color_updated_point = floodfill_color_point(points = selected_point[0], total_points = len(points_all), adj = connected, colors = colors, threshold = np.asarray([color_threshold, color_threshold, color_threshold]), max_level=max_level)
    print('color point grew points', len(color_updated_point))
    if len(color_updated_point) > 0:
        colors_cv[door_points] = [0, 1, 0]
        actual_updated = np.setdiff1d(color_updated_point, door_points)
        print('length of acvtual color grown points w.r.t dl', len(actual_updated))
        if len(actual_updated) > 0:
            colors_cv[actual_updated] = [1, 0, 0]
        pcd1 = to_pcd(points_all, colors_cv, viz = True)
        instance_id[color_updated_point] = door_id
        colors_seg[color_updated_point] = door_palette
        pcd2 = to_pcd(points_seg, colors_seg, viz = True)
        return instance_id, pcd2
    return instance_id, seg_ply


def save_ids_ply(seg_ply, instance_ids, outputpath):
    ''' Fucntion to save instance ids npy and updated point cloud
    '''
    cv_seg_path = outputpath + os.sep + 'cv_segmentation'
    o3d.io.write_point_cloud(cv_seg_path + os.sep + 'pcd.ply', seg_ply)
    np.save(cv_seg_path + os.sep + 'ids.npy', instance_ids)


def door_floor_align(PlaneswithPoints, Vertex, SelectedPoint, BoundingPoints, connected, outputpath, flip = True):
    ''' Fucntion to get plane info and and for visualization of door floor alignment
    '''
    def axis_angle_bn_vecs(vec1, vec2):
        """ Function to get one to all angles between two list of vectors

        Args:
            vec1 (np.ndarray): [N, 3] - list of vectors.
            vec2 (np.ndarray): [M, 3] - list of vectors.
        """
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)
        cos_ = np.dot(vec1, vec2)
        axis = np.cross(vec1, vec2)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(cos_)
        return cos_, angle, axis
    def door_wall_bottom_align(door_BBp, wall_BBp, flip):
        ''' Function to align door and wall
        Args:
            door_BBp (np.ndarray): [4,3] - list of door boundary points
            wall_BBp (np.ndarray): [4,3] - list of wall boundary points
        Returns:
            updated_door_BB (np.ndarray): [4,3] - list of updated door boundary points
        '''
        door_BB = door_BBp.copy()
        wall_BB = wall_BBp.copy()
        door_BB = door_BB[door_BB[:, 2].argsort()]
        door_BB_bottom = door_BB[:2, :]
        door_vector = door_BB_bottom[1] - door_BB_bottom[0]
        
        wall_BB = wall_BB[wall_BB[:, 2].argsort()]
        wall_BB_bottom = wall_BB[:2, :]
        wall_vector = wall_BB_bottom[1] - wall_BB_bottom[0]
        _, angle, axis = axis_angle_bn_vecs(wall_vector, door_vector)
        pivot = door_BB_bottom[0]
        qt = Quat(axis = axis, angle = angle)
        if flip:
            updated_door_bb = qt.inverse.rotate(door_BBp - pivot) + pivot 
        else:
            updated_door_bb = qt.rotate(door_BBp - pivot) + pivot
        return updated_door_bb
    planeidx, indices = GetactualIndex(SelectedPoints=SelectedPoint, PLanewithPoints=PlaneswithPoints, Vertex=Vertex)
    all_ind = planeidx.tolist()
    door_BB = BoundingPoints[PlaneswithPoints[all_ind[0], 2]]
    wall_BB = BoundingPoints[PlaneswithPoints[all_ind[1], 2]]
    door_BBp = door_wall_bottom_align(door_BB, wall_BB, flip)
    BoundingPoints[PlaneswithPoints[all_ind[0], 2]] = door_BBp
    # points = np.vstack((door_BB, wall_BB))
    # color_door_BB = np.repeat([np.asarray([1, 0, 0])], 4, axis= 0)
    # color_wall_BB = np.repeat([np.asarray([0, 1, 0])], 4, axis= 0)
    # colors = np.vstack(( color_door_BB, color_wall_BB))
    # pcd = to_pcd(points, colors, viz = True)
    # mesh = to_mesh(points,
    # [[0, 1, 2],
    # [2, 3, 0],
    # [4, 5, 6],
    # [6, 7, 4],
    # [8, 9, 10],
    # [10, 11, 8]],colors, viz = True)
    # points = np.vstack((door_BBp, wall_BB))
    # colors = np.vstack(( color_door_BB, color_wall_BB))
    # pcd = to_pcd(points, colors, viz = True)
    # mesh = to_mesh(points,
    # [[0, 1, 2],
    # [2, 3, 0],
    # [4, 5, 6],
    # [6, 7, 4],
    # [8, 9, 10],
    # [10, 11, 8]],colors, viz = True)
    return PlaneswithPoints, Vertex, BoundingPoints
