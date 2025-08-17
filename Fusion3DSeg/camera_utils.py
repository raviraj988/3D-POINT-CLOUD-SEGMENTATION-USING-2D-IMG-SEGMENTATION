import numpy as np
from einops import rearrange, repeat

from RTAB_utils.spatQuad import SpatQuadranion




def points2pixel(points, intrinsic, quat, translation):
    """ Function project 3D-points on given camera image plane.

    Args:
        points (np.ndarray[float]): [N, 3] - xyz points.
        intrinsic (np.ndarray[float]): [3, 3] - intrinsic matrix.
        quat (np.ndarray[float]): [4, ] - (w, x, y, z) elements of camera quaternion rotation.
        translation (np.ndarray): [3, ] - (x, y, z) camera translation.

    Returns:
        np.ndrray[float]: [2, N] - projected pixel non normalized uv - coordinates.
    """
    projection = points - translation
    projection = SpatQuadranion(quat).inverse.rotate(projection)
    uv =  intrinsic@projection.T
    uv[:2] /= uv[2:3]
    uv = np.floor(uv[:2]).astype(np.int32)
    return uv


def pixel2point(u, v, K, R, eye):
    """ Function to convert pixel in camera image into 3D point.

    Refs:
        https://github.com/isl-org/Open3D/issues/2338?

    Args:
        u (int/float): pixel x-coordinate, 0 < u < width, wrt tofleft corner.
        v (int/float): pixel y-coordinate, 0 < v < height, wrt topleft corner.
        np.ndarray: [4, 4] -  Extrinsic matrix.
        K (np.ndarray): [3, 3] - intrinsic matrix.
        R (np.ndarray): [3, 3] - rotation matrix.
        eye (np.ndarray): [3, ] - camera eye position.

    Returns:
        np.ndarray: [3, ] - xyz location of corresponding 3D point.

    Note:
        Image coordinates (u, v) are measured from origin at top left corner.
        Range of u is (0, width).
        Range of v is (0, height).
    """
    Kinv = np.linalg.inv(K)
    xyz = np.array([u, v, 1])
    # pixel to camera coordinate system
    xyz = (Kinv@xyz)
    # local to global coordinate
    xyz = R@xyz + eye
    return xyz


def get_camera_frustum(K, width, height):
    """ Function to get camera pyramid.

    Refs:
        https://github.com/isl-org/Open3D/issues/2338?

    Args:
        K (np.ndarray[float]): [3, 3] calibration matrix (camera intrinsics).
        width (int): w - image width.
        height (int): w - image height.

    Returns:
        np.ndarray[float]: [5, 3] - pyramid points [eye, image_plane_corners].
        np.ndarray[int]: [8, 2] - pyramid edges [4 - eye to corners, 4 - image plane edges].
    """
    Kinv = np.linalg.inv(K)
    # points in pixel
    points_pixel = np.array([
        [0, 0, 0],  # eye
        [0, 0, 1], # bottom-left
        [width, 0, 1],  # bottom - right
        [width, height, 1], # top - right
        [0, height, 1],  # top - left
        [width/2, height/2, 1]  # look at
    ])
    # pixel to camera coordinate system
    points = (Kinv@points_pixel.T).T
    # pyramid
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
        [0, 5],
    ])
    return points, edges


def camera2world(frame_points, xyzws, translations, rescale=1000):
    """ Function to convert points in local camera coords to world coords.

    Refs:
        src/modelling/iosdataconvert.py by Dinesh Dhotrad.

    Args:
        frame_points (np.ndarray[float]): [F, N, 3] - list of points of frames.
        xyzws (np.ndarray[float]): [F, 4] - frame quaternions(x, y, z, w).
        translations (np.ndarray[float]): [F, 3] - frame translations(x, y, z)
        rescale (float): rescaling factor.

    Returns:
        np.ndarray[float]: [F, N, 3] - world frame points.
    """
    frame_points = frame_points/rescale
    F = 1
    if frame_points.ndim > 2:
        F = len(frame_points)
    if xyzws.ndim > 1:
        F = len(xyzws)
    if translations.ndim > 1:
        F = len(translations)
    if frame_points.ndim <= 2:
        frame_points = repeat(frame_points, 'n c -> f n c', f=F)
    if xyzws.ndim == 1:
        xyzws = repeat(xyzws, 'c -> f c', f=F)
    if translations.ndim == 1:
        translations = repeat(translations, 'c -> f c', f=F)

    global_points = []
    for i, (points, xyzw, t) in enumerate(zip(frame_points, xyzws, translations)):
        rot = SpatQuadranion(xyzw)
        modpts = rot.rotate(points) + t
        global_points.append(modpts)

    return np.array(global_points)


def get_frustum_unit_vectors(frustum_points):
    """ Function to get 4 directional unit vectors of camera frustum.

    Args:
        frustum_points (np.ndarray[float]): [f, 6, 3] - pyramid points of frames[eye, image_plane_corners, lookat].

    Returns:
        np.ndarray[float]: [f, 3] - camera eyes.
        np.ndarray[float]: [f, 4, 3] - frustum direction vectors.
        np.ndarray[float]: [f, 3] - frustum look at vectors.
    """
    eyes = frustum_points[:, 0:1, :]
    corners = frustum_points[:, 1:, :]
    vecs = corners - eyes
    directions = vecs/np.linalg.norm(vecs, axis=-1)[..., None]
    return eyes[:, 0, :], directions[:, :-1, :], directions[:, -1, :]


def get_frustum_face_normals(eyes, corners):
    """ Function to get 4 normal unit vectors of camera frustum.

    Args:
        eyes (np.ndarray[float]): [f, 3] - camera eyes of frames.
        corners (np.ndarray[float]): [f, 4, 3] - camera pyramid points corners of frames in ccw order.

    Returns:
        np.ndarray[float]: [f, 4, 3] - camera eyes.
    """
    a = corners
    b = np.zeros_like(a)
    b[:, :-1, :] = corners[:, 1:, :]
    b[:, -1, :] = corners[:, 0, :]
    eye2a = a - eyes[:, None, :]
    eye2b = b - eyes[:, None, :]
    normals = np.cross(eye2a, eye2b, axisa=-1, axisb=-1)
    normals = normals/np.linalg.norm(normals, axis=-1)[..., None]
    return normals
