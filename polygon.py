import math
import numpy as np
import pandas as pd
import caiman as cm


def get_contours(est, comps_to_select, cthr=0.3):
    estimates_data = est.A[:, comps_to_select]
    contours = cm.utils.visualization.get_contours(estimates_data,
                                                   dims=est.imax.shape,
                                                   thr=cthr)
    return contours


def _clean(poly):
    p = [tuple(pt) for pt in poly if pt[0] is not None and pt[1] is not None]
    q = []
    for pt in p:
        if not q or q[-1] != pt:
            q.append(pt)
    if len(q) > 1 and q[0] == q[-1]:
        q.pop()
    return q


def _sub(a,b):
    return a[0]-b[0], a[1]-b[1]


def _dot(a,b):
    return a[0]*b[0]+a[1]*b[1]


def _cross(a,b):
    return a[0]*b[1]-a[1]*b[0]


def _norm2(a):
    return _dot(a,a)

def _isfinite_point(p):
    return p is not None and math.isfinite(p[0]) and math.isfinite(p[1])

def _clean(poly):
    """Drop None/NaN/Inf, collapse duplicates, remove closing duplicate."""
    q = []
    for pt in poly:
        if pt is None:
            continue
        x, y = float(pt[0]), float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if not q or (q[-1][0] != x or q[-1][1] != y):
            q.append((x, y))
    if len(q) > 1 and q[0] == q[-1]:
        q.pop()
    return q


def _dist_point_seg(p, a, b):
    ab = _sub(b,a); ap = _sub(p,a)
    denom = _norm2(ab)
    if denom == 0.0:
        return math.hypot(p[0]-a[0], p[1]-a[1])
    t = max(0.0, min(1.0, _dot(ap,ab)/denom))
    proj = (a[0]+t*ab[0], a[1]+t*ab[1])
    return math.hypot(p[0]-proj[0], p[1]-proj[1])


def _seg_seg_dist(a1,a2,b1,b2):
    # Check proper intersection
    def orient(a,b,c):
        return _cross(_sub(b,a), _sub(c,a))

    o1,o2 = orient(a1,a2,b1), orient(a1,a2,b2)
    o3,o4 = orient(b1,b2,a1), orient(b1,b2,a2)

    if (o1==0 and o2==0 and o3==0 and o4==0):
        # Colinear; distance is min point–segment among endpoints
        return min(_dist_point_seg(a1,b1,b2), _dist_point_seg(a2,b1,b2),
                   _dist_point_seg(b1,a1,a2), _dist_point_seg(b2,a1,a2))
    if (o1*o2 < 0) and (o3*o4 < 0):
        return 0.0
    return min(_dist_point_seg(a1,b1,b2), _dist_point_seg(a2,b1,b2),
               _dist_point_seg(b1,a1,a2), _dist_point_seg(b2,a1,a2))


def convex_hull(points):
    """Monotone chain hull, returns CCW hull without duplicate last=first."""
    pts = sorted(set(_clean(points)))
    if len(pts) <= 1:
        return pts

    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]   # no duplicated start/end
    # ensure CCW (positive signed area)
    area2 = sum(hull[i][0]*hull[(i+1)%len(hull)][1] - hull[(i+1)%len(hull)][0]*hull[i][1]
                for i in range(len(hull)))
    if area2 < 0:
        hull.reverse()
    return hull


def calculate_polygon_area(coordinates):
    # Filter out points with NaN coordinates
    valid_coordinates = [point for point in coordinates if not (np.isnan(point[0]) or np.isnan(point[1]))]

    # Check if we have enough valid points to form a polygon
    if len(valid_coordinates) < 3:
        return 0  # Not enough points to form a polygon

    area = 0

    # Number of vertices
    n = len(valid_coordinates)

    # Calculate area using the Shoelace formula
    for i in range(n):
        j = (i + 1) % n
        area += valid_coordinates[i][0] * valid_coordinates[j][1]
        area -= valid_coordinates[j][0] * valid_coordinates[i][1]

    # Take absolute value and divide by 2
    area = abs(area) / 2

    return area


def calculate_perimeter(contour):
    dots_dist = []
    contour_mask = np.array(pd.Series(contour[:, 0] * contour[:, 1]).notna())
    contour = contour[contour_mask]

    dot_num = len(contour) - 1
    while dot_num >= 0:
        dist = np.linalg.norm(contour[dot_num] - contour[dot_num-1])
        dots_dist.append(dist)

        dot_num -= 1

    perimeter = np.sum(dots_dist)
    return perimeter, dots_dist


def get_circularities(contours):
    circularities = []

    for i in range(len(contours)):
        contour = contours[i]["coordinates"]

        area = calculate_polygon_area(contour)
        perimeter = calculate_perimeter(contour)[0]
        circularity = perimeter**2 / (4 * np.pi * area)

        circularities.append(circularity)

    circularities = np.array(circularities)
    return circularities


def get_convexities(contours):
    convexities = []

    for i in range(len(contours)):
        contour = contours[i]["coordinates"]
        contour_mask = np.array(pd.Series(contour[:, 0] * contour[:, 1]).notna())
        contour = contour[contour_mask]

        num_dots = (len(contour))
        angles = []
        for dot in range(num_dots):
            dot_prev = contour[(dot - 1) % num_dots]
            dot_cur = contour[dot]
            dot_next = contour[(dot + 1) % num_dots]

            vec_1 = dot_prev - dot_cur
            vec_2 = dot_next - dot_cur

            cross_product = np.cross(vec_1, vec_2)
            angle = int(cross_product <= 0)
            angles.append(angle)

        if num_dots>0:
            convexity = sum(angles) / num_dots
        else:
            convexity = np.nan
        convexities.append(convexity)

    convexities = np.array(convexities)
    return convexities


def get_max_edges(contours):
    max_edges = []

    for i in range(len(contours)):
        contour = contours[i]["coordinates"]

        edges = calculate_perimeter(contour)[1]
        if len(edges) > 0:
            max_edge = max(edges)
        else:
            max_edge = 0

        max_edges.append(max_edge)

    max_edges = np.array(max_edges)
    return max_edges


def get_aspect_ratios(contours):
    """
    Compute aspect ratio (elongation) for each contour.

    Aspect ratio = max(width, height) / min(width, height) of bounding box.
    - 1.0 = square bounding box (roughly circular footprint)
    - >2.0 = elongated (potential blood vessel, merged neurons, motion artifact)
    """
    aspect_ratios = []

    for i in range(len(contours)):
        coords = contours[i]["coordinates"]
        # Filter NaN
        valid_coords = np.array([pt for pt in coords if not (np.isnan(pt[0]) or np.isnan(pt[1]))])

        if len(valid_coords) < 3:
            aspect_ratios.append(np.nan)
            continue

        # Bounding box dimensions
        x_min, x_max = valid_coords[:, 0].min(), valid_coords[:, 0].max()
        y_min, y_max = valid_coords[:, 1].min(), valid_coords[:, 1].max()

        width = x_max - x_min
        height = y_max - y_min

        if min(width, height) < 1e-6:
            aspect_ratios.append(np.nan)
        else:
            aspect_ratios.append(max(width, height) / min(width, height))

    return np.array(aspect_ratios)


def convex_polygons_min_distance(P, Q):
    P, Q = _clean(P), _clean(Q)
    P = convex_hull(P)
    Q = convex_hull(Q)
    n, m = len(P), len(Q)

    if n>=2 and m>=2:
        # find lowest (y, then x) vertices
        i = min(range(n), key=lambda k: (P[k][1], P[k][0]))
        j = min(range(m), key=lambda k: (Q[k][1], Q[k][0]))
        best = float('inf')
        cnt = 0
        # walk until we’ve advanced n+m edges
        while cnt < n + m:
            a1, a2 = P[i], P[(i+1)%n]
            b1, b2 = Q[j], Q[(j+1)%m]
            #print('ss:', _seg_seg_dist(a1,a2,b1,b2))
            best = min(best, _seg_seg_dist(a1,a2,b1,b2))
            if best == 0.0:
                return 0.0
            ea = _sub(a2,a1)
            eb = _sub(b2,b1)
            # advance by comparing edge angle via cross product sign
            if _cross(ea, eb) >= 0:
                i = (i+1) % n
            else:
                j = (j+1) % m
            cnt += 1
        return best

    else:
        return np.nan