import os
import json

import numpy as np
import cv2
import open3d as o3d



def load_o3d_camera_data(filename):
    """ Function to load intrinsics and extrinsics from .json open3d camera file.

    Args:
        filename (str/Path): cameradata .json file.

    Returns: 
        np.ndarray: [3, 3] - K - intrinsic matrix.
        np.ndarray: [4, 4] -  Extrinsic matrix.
        np.ndarray: [3, 3] - R - rotation matrix.
        np.ndarray: [3, ] - t - translation.
        np.ndarray: [3, ] - camera eye.
        tuple[int]: [2, ] - (height, width)
    """
    with open(filename) as fp:
        data = json.load(fp)
    hw = (data['intrinsic']['height'], data['intrinsic']['width'])
    K = np.array(data['intrinsic']['intrinsic_matrix']).reshape(3, 3).T
    E = np.array(data['extrinsic']).reshape(4, 4).T
    R = E.copy()[:3, :3]
    t = E.copy()[:3, 3]
    R = R.T
    eye = -R@t
    return K, E, R, t, eye, hw


def to_pcd(points, colors=None, normals=None, viz=False, filepath=None, name='Viz'):
    """ Function to convert points array into o3d.PointCloud

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        colors (np.ndarray/List): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        normals (np.ndarray): [N, 3] point normals. Defaults to None.
        viz (bool): show point cloud. Defaults to False.
        filepath (str): save point cloud as. Defaults to None.
        name (str): window name.

    Returns:
        (o3d.PointCloud): point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            pcd.colors =  o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([pcd], name)
    if filepath is not None:
        o3d.io.write_point_cloud(filepath, pcd)
    return pcd


def to_mesh(points, faces, colors=None, viz=False, filepath=None, name='Viz'):
    """ Function to convert points array into o3d.geometry.TriangleMesh

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        faces (np.ndarray): [M, 3] - list of triangle faces of points.
        colors (np.ndarray/List, optional): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.
        name (str): window name.

    Returns:
        (o3d.geometry.TriangleMesh): mesh
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            mesh.vertex_colors =  o3d.utility.Vector3dVector(colors)
        else:
            mesh.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([mesh], name, mesh_show_back_face=True)
    if filepath is not None:
        o3d.io.write_triangle_mesh(filepath, mesh)
    return mesh


def to_lines(points, edges, colors=None, viz=False, filepath=None, name='Viz'):
    """ Function to convert points and edges into o3d.geometry.LineSet

    Args:
        points (np.ndarray[float]): [N, 3] - list of xyz of points.
        edges (np.ndarray[int]): [M, 2] - list of edges.
        colors (np.ndarray/List, optional): [N, 3] edge colors or [r, g, b].
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.
        name (str): window name.

    Returns:
        (o3d.geometry.LineSet): lines
    """
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(edges)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            lines.colors =  o3d.utility.Vector3dVector(colors)
        else:
            lines.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([lines], name, mesh_show_back_face=True)
    if filepath is not None:
        o3d.io.write_line_set(filepath, lines)
    return lines


def to_uvmesh(points, faces, uvs, texture, flip=(False, False, False), viz=False, filepath=None, name='Viz'):
    """ Function to convert points array into o3d.geometry.TriangleMesh
        reference: http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        faces (np.ndarray): [M, 3] - list of triangle faces of points.
        uvs (np.ndarray): [3*M, 2] - uv coordinates of face vertices.
        texture (np.ndarray[np.uint8]): [H, W, 3] - image.
        flip (list[bool]): list booleans representing which axes of texture to flip.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.
        name (str): window name.

    Note:
        1. open3d use trianlge uvs i.e. each triangle will have 3 uv coords => len(uvs) = len(triangles)*3
           Hence sinlge vertex can have multiple uv coordinates.
        2. Most softwares support only texture maps with size upto resolution 16384x16384 = (2**16//4)x(2**16//4)

        mesh.vertices: o3d.utility.Vector3dVector
        mesh.triangles: o3d.utility.Vector3iVector
        mesh.triangle_uvs: o3d.utility.Vector2dVector
        mesh.textures: list[o3d.geometry.Image]
        mesh.triangle_material_ids: o3d.utility.IntVector - len(triangles)

    Returns:
        (o3d.geometry.TriangleMesh): mesh
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(faces), dtype=int).tolist())

    texture = texture[::-1 if flip[0] else 1, ::-1 if flip[1] else 1, ::-1 if flip[2] else 1].copy()
    mesh.textures = [o3d.geometry.Image(texture)]

    if viz:
        o3d.visualization.draw_geometries([mesh], name, mesh_show_back_face=True)
    if filepath is not None:
        o3d.io.write_triangle_mesh(filepath, mesh)
    return mesh


def to_image(
        img, norm=False, save=None,
        show=True, delay=0,
        rgb=True, bg=0,
    ):
    """ Function to show/save image

    Args:
        img (np.ndarray): [h, w, ch] image(grayscale/rgb)
        norm (bool, optional): min-max normalize image. Defaults to False.
        save (str, optional): path to save image. Defaults to None.
        show (bool, optional): show image. Defaults to True.
        delay (int, optional): cv2 window delay. Defaults to 0.

    Returns:,
        (np.ndarray): [h, w, ch] - image.
    """
    if rgb:
        img = img[..., ::-1]
    if norm:
        img = (img - img.min())/(img.max() - img.min())
    if save is not None:
        if img.max() <= 1:
            img *=255
        cv2.imwrite(save, img.astype(np.uint8))
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
    return img


def uv2rgb(mesh, texture, rgb=False, save_as=None):
    """ Function to convert uv textured mesh to rgb point colored mesh

    Args:
        meshfile (o3d.geometry.TriangleMesh): mesh with texture.
        texture (np.ndarray): [h, w, 3] - texture image.
        rgb (bool, optional): True if texture is rgb, False if bgr. Defaults to False.
        save_as (str, optional): path to save.
    Returns:
        o3d.io.TriangleMesh: colored mesh.
        np.ndarray: [N, 3] - vertices/points.
        np.ndarray: [N, 3] - vertex colors.
        np.ndarray: [M, 3] - triangle faces.
    """
    texture = texture[..., ::-1] if not rgb else texture
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    uvs = np.asarray(mesh.triangle_uvs)
    face_vertices = faces.reshape(-1)

    h, w, c = texture.shape
    u, v = uvs.T
    u = (u * w).astype(np.int32)
    v = ((1 - v) * h).astype(np.int32)

    colors = np.empty_like(vertices)
    colors[face_vertices] = texture[v, u]/255

    colored_mesh = to_mesh(vertices, faces, colors, viz=False, filepath=save_as)
    return colored_mesh, vertices, colors, faces


def vertex_triangle_mapping(triangles, nvertices):
    """ Function to get mapping - triangles of vertices and position of vertices
        author: giris d. hegde

    Args:
        triangles (np.ndarray[int]): [M, 3] - triangles.
        nvertices (int): N = totol vertices.

    Returns:
        list[list[int]]: [N, ...] - triangles of vertices.
        list[list[int]]: [N, ...] - position of vertices.
    """
    triangles_of_vertices = [[] for _  in range(nvertices)]
    position_of_vertices = [[] for _  in range(nvertices)]

    triangles = triangles.copy()
    for i, face in enumerate(triangles):
        triangles_of_vertices[face[0]].append(i)
        triangles_of_vertices[face[1]].append(i)
        triangles_of_vertices[face[2]].append(i)
        position_of_vertices[face[0]].append(0)
        position_of_vertices[face[1]].append(1)
        position_of_vertices[face[2]].append(2)

    return triangles_of_vertices, position_of_vertices


def remove_faces_by_vertices(nvertices, triangles, mask):
    """ Function to remove triangle faces given indices
        author: giris d. hegde

    Args:
        nvertices (int): N = total points.
        triangles (np.ndarray[int]): [M, 3] - triangles.
        mask (np.ndarray[bool]): [N, ] - vertices tobe removed mask.

    Returns:
        np.ndarray[bool]: [M, ] - not removed triangles mask.
        np.ndarry[int]: new not removed triangles.
        np.ndarray[int]: [N, ] - original vertex ids to new vertex id mapping.
    """
    keep_mask = np.logical_not(mask)
    remove_indices = np.where(mask)[0]

    new_vertex_ids = np.arange(keep_mask.sum())
    oldids2newids = np.zeros(nvertices, int)
    oldids2newids[keep_mask] = new_vertex_ids

    triangles_of_vertices = [[] for _  in range(nvertices)]
    position_of_vertices = [[] for _  in range(nvertices)]

    remaining_triangles = triangles.copy()
    for i, face in enumerate(triangles):
        triangles_of_vertices[face[0]].append(i)
        triangles_of_vertices[face[1]].append(i)
        triangles_of_vertices[face[2]].append(i)
        position_of_vertices[face[0]].append(0)
        position_of_vertices[face[1]].append(1)
        position_of_vertices[face[2]].append(2)
        remaining_triangles[i] = oldids2newids[face]

    not_removed = np.ones(len(triangles), bool)
    for idx in remove_indices:
        not_removed[triangles_of_vertices[idx]] = False
    remaining_triangles = remaining_triangles[not_removed]

    return not_removed, remaining_triangles, oldids2newids


def keep_faces_by_vertices(vertices, triangles, mask):
    """ Function to keep triangle touching given vertices
        author: giris d. hegde

    Args:
        vertices (np.ndarray[float]): [N, 3] - points.
        triangles (np.ndarray[int]): [M, 3] - triangles.
        mask (np.ndarray[bool]): [N, ] - keep vertices mask.

    Returns:
        np.ndarry[float]: [P, 3] - remaining vertices.
        np.ndarry[int]: [Q, 3] - remaining triangles.
    """
    nvertices = len(vertices)
    newid = np.full(nvertices, -1)

    remaining_triangles = []
    remaining_vertices = []
    pos = 0
    for i, face in enumerate(triangles):
        if mask[face].any():
            for j, vx in enumerate(face):
                if newid[vx] == -1:
                    remaining_vertices.append(vertices[vx])
                    newid[vx] = pos
                    pos += 1
                face[j] = newid[vx]
            remaining_triangles.append(face)

    return remaining_vertices, remaining_triangles


def bbox_axes(corners):
    """ Function to get x and y axes of o3d.geometry.OrientedBoundingBox

    Args:
        corners (np.ndarray): [8, 3] - corners.

    Returns:
        np.ndarray: [3, ] - origin.
        np.ndarray: [3, ] - i - x axis unit vector.
        np.ndarray: [3, ] - j - y axis unit vector.
        float: x axis length.
        float: y axis length.
    """
    edges = (corners[1:4] - corners[0][None, :])
    lengths = np.linalg.norm(edges, axis=-1)
    lengthwise = np.argsort(lengths)
    i = edges[lengthwise[2]]
    li = lengths[lengthwise[2]]
    j = edges[lengthwise[1]]
    lj = lengths[lengthwise[1]]
    origin = (corners[0] + corners[3])/2
    return origin, i/li, j/lj, li, lj


def get_triangle_clusters(mesh):
    """ Function to get connected triangle clusters.

    Args:
        mesh (o3d.geometry.TriangleMesh): 3D mesh of N vertices and M triangles.

    Retuns:
        np.ndarray[int]: [M, ] - triangle cluster indices.
        np.ndarray[int]: [P, ] - per cluster triangles.
        np.ndarray[float]: [P, ] - per cluster area.
    """
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    return triangle_clusters, cluster_n_triangles, cluster_area


def classwise_triangle_colors(triangle_classes):
    """ Function to generate triangle colors given triangle classes

    Args:
        triangle_classes (np.ndarray[int]): [M, ] - triangle class/cluster ids.

    Returns:
        np.ndarray[float]: [M, 3] - triangle colors
    """
    ids = np.unique(triangle_classes)
    id_colors = np.random.uniform(0, 1, size=(len(ids), 3))
    colors = np.zeros((len(triangle_classes), 3))
    for id_, clr in zip(ids, id_colors):
        colors[triangle_classes == id_, :] = clr
    return colors


def generate_texture(triangle_uvs, colors, hw=(100, 100), viz=False, window_name=None):
    """ Function to generate uv texture image given triangles uvs, triangle colors.

    Args:
        triangle_uvs (np.ndaray[float]): [M*3, 2] - triangle uvs.
        colors (np.ndaray[float/uint8]): [M, 3] - triangle colors.
        hw (tuple[int]): [H, W] of image.
        viz (bool): visualize.
        window_name (str): visualization window name.

    Returns:
        np.ndarray[np.uint8]: [H, W] - image.
    """
    h, w = hw
    dtype = colors.dtype
    image = np.zeros((h, w, 3), dtype)

    triangle_uvs = triangle_uvs.copy()
    triangle_uvs[:, 0] = (triangle_uvs[:, 0]*(w - 1))
    triangle_uvs[:, 1] = (triangle_uvs[:, 1]*(h - 1))
    triangle_uvs = triangle_uvs.reshape(len(colors), 3, 2)
    triangle_uvs = triangle_uvs.astype(int)

    for pts, clr in zip(triangle_uvs, colors):
        cv2.polylines(image, [pts], isClosed=True, color=clr, thickness=1)
        cv2.fillPoly(image, [pts], color=clr)

    if viz:
        cv2.imshow(str(window_name), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def read_images(names, dirname='./', prefix='', extension='png', zfill=0, gray=False, dtype=np.uint8, rgb=False):
    """ Function to read images within a directory

    Args:
        names (list[...]): list of filenames.
        dirname (str): path to directory.
        prefix (str): common prefixes.
        extension (str): file extension.
        zfill (int): fill zeros.
        gray (bool): read as grayscale.
        dtype (np.dtype): dtype of images.
        rgb (bool): rgb or bgr.

    Returns:
        list[np.ndarray]: [for name in names image(dirname/prefixname.extension)]

    """
    images = []
    for name in names:
        name = str(name)
        filename = os.path.join(dirname, prefix + name.zfill(zfill) + '.' + extension)
        img = cv2.imread(filename, 0 if gray else 1).astype(dtype)[..., ::-1 if rgb else 1]
        images.append(img)
    return images


def one_to_all_angles(vec1, vec2):
    """ Function to get one to all angles between two list of vectors

    Args:
        vec1 (np.ndarray): [N, 3] - list of vectors.
        vec2 (np.ndarray): [M, 3] - list of vectors.
    """
    vec1 /= np.linalg.norm(vec1, axis=1)[:, None]
    vec2 /= np.linalg.norm(vec2, axis=1)[:, None]
    prod = vec1[None, :, :]*vec2[:, None, :]
    cos = np.sum(prod, axis=2)
    angles = np.rad2deg(np.arccos(cos))
    return angles


def pick_points(pcd):
    print('-'*210)
    print("Press [shift + left click] - to select point")
    print("Press [shift + right click] to deselect point")
    print("Press q to close the window")
    print('-'*210)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    points = vis.get_picked_points()
    print('-'*210)
    print("Selected points:", points)
    print('-'*210)
    return points


def get_roi(img):
    img = img.copy()
    ix = -1
    iy = -1
    drawing = False
    roi = []
    h, w, *_ = img.shape

    def get_coords(rect):
        [[x1, y1], [x2, y2]] = rect
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x = [i for i in range(x1, x2 + 1)]
        y = [i for i in range(y1, y2 + 1)]
        x, y = np.meshgrid(x, y)
        return np.vstack((x.ravel(), y.ravel()))
        return list(zip(x, y))

    def draw_rectangle_with_drag(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if x < 0: x = 0
            if x >= w: x =  w - 1
            if y < 0: y = 0
            if y >= h: y =  h - 1
            ix = x
            iy = y
            roi.append([[x, y], ])

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.rectangle(img, pt1 =(ix, iy),
                            pt2 =(x, y),
                            color =(0, 255, 255),
                            thickness =-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, pt1 =(ix, iy),
                        pt2 =(x, y),
                        color =(0, 255, 255),
                        thickness =-1)
            if x < 0: x = 0
            if x >= w: x =  w - 1
            if y < 0: y = 0
            if y >= h: y =  h - 1
            roi[-1].append([x, y])

    cv2.namedWindow(winname="Drag to select ROI")
    cv2.setMouseCallback("Drag to select ROI", draw_rectangle_with_drag)
    while True:
        cv2.imshow("Drag to select ROI", img)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()

    coords = None
    for rect in roi:
        xy = get_coords(rect)
        if coords is None:
            coords = xy.copy()
        else:
            coords = np.hstack((coords, xy))
    return coords


if __name__ == '__main__':
    mesh = './data/chair/seg.obj'
    tex = './data/chair/seg_0.png'
    mesh = o3d.io.read_triangle_mesh(mesh, enable_post_processing=True)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)


    uvmesh = to_uvmesh(
        points=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        uvs=np.asarray(mesh.triangle_uvs),
        texture=cv2.imread(tex),
        flip=(True, False, True),
        viz=True,
        filepath='out2.obj',
    )
