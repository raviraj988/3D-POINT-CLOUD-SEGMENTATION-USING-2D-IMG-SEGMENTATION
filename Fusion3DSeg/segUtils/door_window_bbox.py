from pathlib import Path
import json
import pickle

import numpy as np
import open3d as o3d


def _get_door_window_mesh(points, faces, colors=None, filename=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors =  o3d.utility.Vector3dVector(colors)
    if filename is not None:
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        o3d.io.write_triangle_mesh(filename, mesh)
    return mesh


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _point_in_triangle(points, triangle):
    """ check if points are inside triangle.

    Args:
        points (np.ndarray[float]): [N, 3] - points.
        triangle (np.ndarray[float]): [3, 3] - triangle vertices.

    Returns:
        np.ndarray[bool]: [N, ] - points inside mask.
    """
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = points - triangle[0]  # [N, 3]
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.einsum('c, nc -> n', v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.einsum('c, nc -> n', v1, v2)
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def _get_perpendicular_vectors(normal):
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    arbitrary = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(normal, arbitrary)), 1.0):
        arbitrary = np.array([0, 1, 0])

    vector1 = np.cross(normal, arbitrary)
    vector2 = np.cross(normal, vector1)

    vector1, vector2 = vector1/np.linalg.norm(vector1), vector2/np.linalg.norm(vector2)
    return vector1, vector2


def generate_mesh(input_dir, *args, **kwargs):
    dirname = Path(input_dir)
    with (dirname/'fusion/fusion_data.pkl').open('rb') as fp: data = pickle.load(fp)
    pts, clrs = data['points'], data['colors']

    ids = np.load(dirname/'panoptic_segmentation/ids.npy')
    with open(dirname/'panoptic_segmentation/info.json') as fp: info = json.load(fp)
    doorwindow = [86, 115, 116]
    doorwindow = set(doorwindow)
    cube_result = str(list((dirname/'polyfit').glob('*.off'))[0])
    mesh = o3d.io.read_triangle_mesh(cube_result)
    mesh.compute_triangle_normals()
    traingles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.triangle_normals)
    triangle_vertices = vertices[traingles]

    bbox_vertices, bbox_triangles, triangle_ids, triangle_colors = [], [], [], []
    abc = np.array([[0, 1, 2], [2, 3, 0]])
    n_bbox = 0
    angle_threshold = np.cos(np.deg2rad(10))
    for idinfo in info:
        category = idinfo['category_id']
        if category in doorwindow:
            id_ = idinfo['id']
            mask = ids == id_

            box_pts = pts[mask]
            point_vecs = box_pts[:, None, :] - triangle_vertices[None, :, 0, :]
            perp_dist = np.einsum('m n c, n c -> m n', point_vecs, normals)
            tri_dist = np.sum(np.abs(perp_dist), axis=0)
            closest = tri_dist.argmin()
            min_dist = tri_dist[closest]
            upper_dist = min_dist + 0.05*min_dist
            closest_triangles = (tri_dist < upper_dist)
            closest_vertices = triangle_vertices[closest_triangles]
            closest_normals = normals[closest_triangles]

            vecs = closest_normals[:, None, :]*perp_dist[:, closest_triangles].T[:, :, None]
            projections =  box_pts[None, :, :] - vecs

            inside_pts = []
            for i in range(len(closest_vertices)):
                inside = _point_in_triangle(projections[i], closest_vertices[i])
                total_inside = inside.sum()
                inside_pts.append(total_inside)
            idx = np.argmax(np.array(inside_pts))

            box_pts = projections[idx]
            selected_tri = closest_vertices[idx]
            norm = closest_normals[idx]

            if angle_threshold < norm.dot([0, 0, 1]):continue

            i, j = _get_perpendicular_vectors(norm)
            origin = box_pts[0]

            x = np.einsum('nc, c -> n', box_pts - origin, i)
            y = np.einsum('nc, c -> n', box_pts - origin, j)

            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
            bbox = [
                origin + xmin*i + ymax*j,
                origin + xmin*i + ymin*j,
                origin + xmax*i + ymin*j,
                origin + xmax*i + ymax*j,
            ]

            clr = np.array(_hex_to_rgb(idinfo['hexcolor']))
            bbox_vertices.append(bbox)
            bbox_triangles.append(abc + n_bbox*4)
            triangle_colors.append([clr, clr, clr, clr])
            triangle_ids.append([id_, id_])
            n_bbox += 1

    bbox_vertices = np.vstack(bbox_vertices)
    bbox_triangles = np.vstack(bbox_triangles)
    triangle_colors = np.vstack(triangle_colors)/255
    triangle_ids = np.hstack(triangle_ids).astype(np.int32)

    bbox_mesh = _get_door_window_mesh(
        bbox_vertices, bbox_triangles, triangle_colors,
        filename=str(dirname/'panoptic_segmentation/door_window_mesh.ply'),
    )
    np.save(dirname/'panoptic_segmentation/triangle_ids.npy', triangle_ids)
    return triangle_ids, bbox_mesh


if __name__ == "__main__":
    triangle_ids, bbox_mesh = generate_mesh(
        input_dir='path/to/input/data/directory',
    )
