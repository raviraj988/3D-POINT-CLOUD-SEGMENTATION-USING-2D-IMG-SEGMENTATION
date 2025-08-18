#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:19:26 2023

@author: Priya Sinha
"""
import json
import open3d as o3d
import numpy as np
from skspatial.objects import Line
from pathlib import Path
import time

def cal_min_max(id, id_info_per_point, pcd_points):
    point_ind = np.where(id_info_per_point == id)
    point_list = pcd_points[point_ind]
    obox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_list))
    obox_1 = np.asarray(obox.get_box_points())
    line1 = Line(point=[0, 0, 0], direction=[1, 0, 0])
    projected_points1 = list(map(line1.project_point, obox_1))
    min_x = np.min(projected_points1, axis = 0)
    min_x = min_x[np.nonzero(min_x)]
    max_x = np.max(projected_points1, axis = 0)
    max_x = max_x[np.nonzero(max_x)]

    
    line2 = Line(point=[0, 0, 0], direction=[0, 1, 0])
    projected_points2 = list(map(line2.project_point, obox_1))
    min_y = np.min(projected_points2, axis = 0)
    min_y = min_y[np.nonzero(min_y)]
    max_y = np.max(projected_points2, axis = 0)
    max_y = max_y[np.nonzero(max_y)]
    
    line3 = Line(point=[0, 0, 0], direction=[0, 0, 1])
    projected_points3 = list(map(line3.project_point, obox_1))
    min_z = np.min(projected_points3, axis = 0)
    min_z = min_z[np.nonzero(min_z)]
    max_z = np.max(projected_points3, axis = 0)
    max_z = max_z[np.nonzero(max_z)]
    
    return min_x, max_x, min_y, max_y, min_z, max_z

def check_intersection(id1, id_list, id_info_per_point, pcd_points, info_sem):
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = cal_min_max(id_list[id1], id_info_per_point, pcd_points)
    for id2 in range(1, len(id_list)):
        if id1 != id2:
            intersecting_id = []
            if info_sem[id1]["category_id"] == info_sem[id2]["category_id"]:
                min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = cal_min_max(id_list[id2], id_info_per_point, pcd_points)
                if (((min_x1 <= min_x2 and min_x2 <= max_x1) or (min_x2 <= min_x1 and min_x1 <= max_x2)) and
                   ((min_y1 <= min_y2 and min_y2 <= max_y1) or (min_y2 <= min_y1 and min_y1 <= max_y2)) and
                   ((min_z1 <= min_z2 and min_z2 <= max_z1) or (min_z2 <= min_z1 and min_z1 <= max_z2)) 
                   ):
                        intersecting_id.append(id2)
    return intersecting_id

def update_id_info(id1, int_bb, info_sem, id_info_per_point):
    points_ind = np.where(id_info_per_point == int_bb)
    info_sem[id1]["area"] += info_sem[int_bb]["area"] 
    id_info_per_point[points_ind] = id1
    return info_sem, id_info_per_point

def intersection_point_bb(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def check_intersection_open3d(id1, id_list, id_info_per_point, pcd_points, pcd, info_sem):
    intersecting_id = []
    point_ind = np.where(id_info_per_point == id1)
    point_list = pcd_points[point_ind]
    if len(point_list) < 4:
        return intersecting_id
    else:
        obox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_list))
        pcd_list_bb = obox.get_point_indices_within_bounding_box(pcd.points)
    
        for id2 in range(1, len(id_list)):
            if (id1 != id2) and (id2 < len(info_sem)-1) and (id1 < len(info_sem)-1):
                if info_sem[id1]["parent_id"] == info_sem[id2]["parent_id"]: 
                    point_ind1 = np.where(id_info_per_point == id2)
                    point_list1 = pcd_points[point_ind1]
                    if len(point_list1) < 4:
                        return intersecting_id
                    else:
                        obox1 = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_list1))
                        pcd_list_bb1 = obox1.get_point_indices_within_bounding_box(pcd.points)
                        intersection_bb = intersection_point_bb(pcd_list_bb, pcd_list_bb1)
                        if len(intersection_bb) > 0:
                            intersecting_id.append(id2)
    return intersecting_id

def visualize_pcd(pcd, info_sem, id_info_per_point, pcd_points):
    for id in info_sem:
        point_ind1 = np.where(id_info_per_point == id["id"])
        point_list1 = pcd_points[point_ind1]
        if len(point_list1) > 4:
            obox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_list1))
            obox.color = [1,0,0]
            o3d.visualization.draw_geometries([pcd]+[obox])
    return

def merge_bb(dir_name, info_sem, id_info_per_point, pcd):
    len_info_sem = len(info_sem)
    pcd_points = np.asarray(pcd.points)

    #visualize_pcd(pcd, info_sem, id_info_per_point, pcd_points)
    start_time = time.perf_counter()

    id_list = []
    for i in range(len(info_sem)): id_list.append(info_sem[i]["id"])

    for id1 in range(1, len(id_list)):
            intersecting_bb = check_intersection_open3d(id1, id_list, id_info_per_point, pcd_points, pcd, info_sem)
            if len(intersecting_bb) > 0:
                for int_bb in intersecting_bb:
                    info_sem, id_info_per_point = update_id_info(id1, int_bb, info_sem, id_info_per_point)
                for i in intersecting_bb: 
                    if i< len(info_sem):
                        del info_sem[i]
                        
    for id in range(1, len(info_sem)):
        pnt_ind1 = np.where(id_info_per_point == info_sem[id]["id"])
        pnt_list = pcd_points[pnt_ind1]
        if len(pnt_list) > 4:
            obox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pnt_list))
            obox_points = np.asarray(obox.get_box_points()).tolist()
            info_sem[id]["bbox"] = obox_points

    end_time = time.perf_counter()
    print(f'Time taken for merging {len_info_sem} to {len(info_sem)} Bounding boxes = {end_time - start_time} seconds')

    #visualize_pcd(pcd, info_sem, id_info_per_point, pcd_points)
    with open(dir_name/"panoptic_segmentation"/"final_info.json", 'w') as fp:
        json.dump(info_sem, fp, indent=4)
    with open(dir_name/"panoptic_segmentation"/"ids.npy", 'wb') as fi:
        np.save(fi, id_info_per_point)



if __name__== "__main__":
    dir_name = Path("/home/tooliqa-user/Priya/Data/AdityeSirBasement")
    with open(dir_name/"panoptic_segmentation"/"final_info.json", 'r') as fp:
        info_sem = json.load(fp)
    id_info_per_point = np.load(dir_name/"panoptic_segmentation"/"ids.npy")
    pcd_path = str(dir_name/"segmentation"/"final_pcd.ply")  
    pcd = o3d.io.read_point_cloud(pcd_path)
    merge_bb(dir_name, info_sem, id_info_per_point, pcd)
   