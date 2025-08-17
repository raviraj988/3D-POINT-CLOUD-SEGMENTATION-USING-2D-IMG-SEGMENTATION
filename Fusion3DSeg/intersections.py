
import numpy as np
from einops import rearrange, repeat


def ray_x_lines(origin, direction, starts, ends):
    """ Funciton to find ray and lines intersection.

    Refs:
        https://www.codefull.net/2015/06/intersection-of-a-ray-and-a-line-segment-in-3d/

    Args:
        origin (np.ndarray[float]): [3, ] - ray origin.
        direction (np.ndarray[float]): [3, ] - ray direction.
        starts (np.ndarray[float]): [N, 3] - line origin.
        ends (np.ndarray[float]): [N, 3] - line end.

    Note:
        given 2 lines are coplanar and intersect.

    Returns:
        np.ndarray[float]: [N, 3] - intersection point.
        np.ndarray[bool]: [N, ] - valid intersections.
    """
    line_direction = ends - starts
    ray_to_line = starts - origin[None, :]  # [N, 3]
    # if (np.abs(ray_to_line.dot(direction.cross(line_direction))) >= 0.7): # Lines are not coplanar
    # 	return None, None
    perpendicular = np.cross(direction, line_direction, axisa=-1, axisb=-1)
    rlxl = np.cross(ray_to_line, line_direction, axisa=-1, axisb=-1)
    t = np.einsum('nc, nc -> n', rlxl, perpendicular)/np.einsum('nc, nc -> n', perpendicular, perpendicular)
    intersection = origin[None, :] + t[:, None]*direction[None, :]

    xs_plus_xe = np.linalg.norm(intersection - starts, axis=-1) + np.linalg.norm(intersection - ends, axis=-1)
    length = np.linalg.norm(ends - starts, axis=-1) + 1e-6
    within =  (xs_plus_xe < length)&(t > 0)

    return intersection, within


def rays_x_plane(plane_point, plane_normal, origins, directions):
    """ Function to find rays and plane intersection.

    Args:
        plane_point (np.ndarray[float]): [3, ] - plane point.
        plane_normal (np.ndarray[float]): [3, ] - plane unit normal vector.
        origins (np.ndarray[float]): [N, 3] - ray origins.
        directions (np.ndarray[float]): [N, 3] - ray direction unit vectors.

    Returns:
        np.ndarray[float]: [N, 3] - intersection points.
        np.ndarray[bool]: [N, ] - valid intersection mask.
    """
    denom = np.einsum('c, nc -> n', plane_normal,  directions)
    # valid = (denom < -1e-6)|(denom > 1e-6)
    valid = (denom < -1e-6)

    vectors = plane_point[None, :] - origins
    t = np.zeros(len(origins))
    t[valid] = (np.einsum('nc, c -> n', vectors, plane_normal)[valid])/denom[valid]
    intersections = origins + directions*t[:, None]

    return intersections, valid


def lines_x_planes(line_origins, line_ends, plane_points, plane_normals):
    """ Function to find lines and planes intersection.

    Args:
        line_origins (np.ndarray[float]): [N, 3] - line origins.
        line_ends (np.ndarray[float]): [N, 3] - line ends.
        plane_points (np.ndarray[float]): [M, 3] - plane points.
        plane_normals (np.ndarray[float]): [M, 3] - plane unit normals.

    Returns:
        np.ndarray[float]: [N, M, 3] - intersection points.
        np.ndarray[bool]: [N, M] - valid intersection mask.
    """
    directions = line_ends - line_origins
    directions = directions/np.linalg.norm(directions, axis=-1)[..., None]
    denom = np.einsum('nc, mc -> nm', directions, plane_normals)
    valid = (denom < -1e-6)|(denom > 1e-6)  # [N, M]

    vectors = plane_points[None, :, :] - line_origins[:, None, :]  # [N, M, 3]
    t = np.zeros((len(line_origins), len(plane_points)))
    t[valid] = (np.einsum('nmc, mc -> nm', vectors, plane_normals)[valid])/denom[valid]
    intersections = line_origins[:, None, :] + directions[:, None, :]*t[:, :, None]  # [N, M, 3]

    xs_plus_xe = np.linalg.norm(intersections - line_origins, axis=-1) \
        + np.linalg.norm(intersections - line_ends, axis=-1)  # [N, M]
    lengths = np.linalg.norm(line_ends - line_origins, axis=-1) + 1e-6  # [N, ]
    valid =  (xs_plus_xe < lengths[:, None])&valid

    return intersections, valid


def point_inside_polygon(points, vertices):
    """ Function to check if point lies inside polygon.

    Args:
        points (np.ndarray[flaot]): [N, 3] - points.
        vertices (np.ndarray[float]): [M, 3] - polygon vertices in ccw/cw order. M >= 3.

    Returns:
        np.ndarray[bool]: [N, ] - points inside mask.
        np.ndarray[bool]: [M, N] - points inside edges/boundary.
    """
    edges = np.zeros_like(vertices)  # [M, 3]
    edges[:-1] = vertices[1:] - vertices[:-1]
    edges[-1] = vertices[0] - vertices[-1]
    # edges[0] = vertices[0] - vertices[-1]
    # edges[1:] = vertices[1:] - vertices[:-1]
    point_vectors =  points[:, None, :] - vertices[None, :, :]  # [N, M, 3]

    dp = np.einsum('nmc, mc -> mn', point_vectors, edges)
    within_boundary = dp >= 0  # [M, N]
    signsum = np.sum(within_boundary, axis=0)  # [N, ]
    inside = (signsum == 0)|(signsum == len(edges))
    return inside, within_boundary


def plane_x_plane(n1=None, v1=None, n2=None, v2=None, lookat=None):
    """ Funciton to find intersection of two planes.

    Args:
        n1 (np.ndarray[float]): [3, ] - plane 1 normal.
        v1 (np.ndarray[float]): [N, 3] - vertices on plane 1. if n1 is None: n1 = getnormal(v1).
        n2 (np.ndarray[float]): [3, ] - plane 2 normal.
        v2 (np.ndarray[float]): [N, 3] - vertices on plane 2. if n2 is None: n2 = getnormal(v2).
        lookat (np.ndarray[float]): [3, ] - lookat vector. output direction will be aligned to look at.

    Returns:
        np.ndarray[float]: [3, ] - intersection line unit direction vectors.
    """
    n1 = np.cross(v1[1] - v1[0], v1[2] - v1[0]) if n1 is None else n1
    n2 = np.cross(v2[1] - v2[0], v2[2] - v2[0]) if n2 is None else n2

    perpendicular = np.cross(n1, n2)
    perpendicular = perpendicular/np.linalg.norm(perpendicular)
    aligned = (perpendicular.dot(lookat) > 0) if lookat is not None else True
    perpendicular = perpendicular if aligned else -perpendicular

    return perpendicular


def point_inside_polyhedra(points, plane_points, normals):
    """ Function to check if point lies inside convex polyhedra defined by surfaces.

    Args:
        points (np.ndarray[flaot]): [N, 3] - points.
        plane_points (np.ndarray[float]): [M, 3] - point on surface planes.
        normals (np.ndarray[float]): [M, 3] - surface plane normals inward direction.

    Returns:
        np.ndarray[bool]: [N, ] - points inside mask.
    """
    point_vectors =  points[:, None, :] - plane_points[None, :, :]  # [N, M, 3]

    dp = np.einsum('nmc, mc -> mn', point_vectors, normals)
    within_boundary = dp >= 0  # [M, N]
    signsum = np.sum(within_boundary, axis=0)  # [N, ]
    # inside = (signsum == 0)|(signsum == len(normals))
    inside = (signsum == len(normals))
    return inside


def points_plane_projection(points, plane_point, normal):
    """ Function to find projection of point on plane.

    Args:
        points (np.ndarray[float]): [N, 3] - points.
        plane_point (np.ndarray[float]): [3, ] - point on plane.
        normal (np.ndarray[float]): [3, ] - plane unit normal.

    Returns:
        np.ndarray[float]: [N, 3] - points projection.
    """
    t = plane_point.dot(normal) - np.einsum('c, nc -> n', normal, points)
    projections = points + t[:, None]*normal[None, :]
    return projections


def lines_plane_projection(starts, ends, plane_point, normal):
    """ Function to find projection of lines on plane.

    Args:
        starts (np.ndarray[float]): [N, 3] - start points.
        starts (np.ndarray[float]): [N, 3] - end points.
        plane_point (np.ndarray[float]): [3, ] - point on plane.
        normal (np.ndarray[float]): [3, ] - plane unit normal.

    Returns:
        np.ndarray[float]: [N, 3] - start point projections.
        np.ndarray[float]: [N, 3] - end point projections.
        np.ndarray[float]: [N, 3] - line direction unit vectors on plane.
    """
    n = len(starts)
    points = np.vstack((starts, ends))
    t = plane_point.dot(normal) - np.einsum('c, nc -> n', normal, points)
    projections = points + t[:, None]*normal[None, :]
    start_projections, end_projections = projections[:n], projections[n:]
    directions = end_projections - start_projections
    directions = directions/np.linalg.norm(directions, axis=-1)[:, None]
    return start_projections, end_projections, directions


def ray_ray_closest(a0, a1, b0, b1):
    """ Funciton to find ray and ray closest points.

    Args:
        a0 (np.ndarray[float]): [3, ] - line1 origin.
        a1 (np.ndarray[float]): [3, ] - line1 direction.
        b0 (np.ndarray[float]): [3, ] - line2 origin.
        b1 (np.ndarray[float]): [3, ] - line2 direction.

    Returns:
        np.ndarray[float]: [3, ] - pa - closest point on line1.
        np.ndarray[float]: [3, ] - pb - closest point on line2.
        float: distance between lines.
        bool: intersects.
        bool: within_a - on line segment 1.
        bool: within_b - on line segment 2.
    """
    b = b1 - b0
    length_b = np.linalg.norm(b)
    b_ = b/length_b  # [N, 12, 3] - edge unit vectors

    a = a1 - a0
    length_a = np.linalg.norm(a)
    a_ = a/length_a  # [3, ]

    perpendicular = np.cross(a_, b_)
    denom = np.linalg.norm(perpendicular)**2
    parellel = (denom == 0)

    ab = (b0 - a0)  # [3, ]
    d = rearrange([ab, b_, perpendicular], 'b c -> b c')  # [3, 3]
    deta = np.linalg.det(d)
    d = rearrange([ab, a_, perpendicular], 'b c -> b c')  # [3, 3]
    detb = np.linalg.det(d)

    ta = deta/denom
    tb = detb/denom

    pa = a0 + a_*ta # projected closest point on segment a
    pb = b0 + b_*tb # projected closest point on segment b

    distance = np.linalg.norm(pa - pb)
    intersects = distance < 1e-6

    within_a = np.linalg.norm(pa - a0) <= length_a
    within_b = np.linalg.norm(pb - b0) <= length_b

    return pa, pb, distance, intersects, within_a, within_b
