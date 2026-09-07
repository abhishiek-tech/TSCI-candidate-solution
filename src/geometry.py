"""Zone geometry helpers.

Zones are configured either as a rectangle ``[x1, y1, x2, y2]`` (the common
case, and what every bundled scenario config uses) or as an explicit polygon
``[[x, y], ...]``. Both normalise to a polygon so the rest of the pipeline
only ever deals with one shape type.
"""

import cv2
import numpy as np

ZONE_COLOR = (60, 180, 250)


def normalize_zone(value):
    """Accept a rect [x1,y1,x2,y2] or an explicit polygon; return a polygon."""
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        x1, y1, x2, y2 = value
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return [list(point) for point in value]


def normalize_zones(zones):
    return {name: normalize_zone(polygon) for name, polygon in zones.items()}


def point_in_polygon(point, polygon):
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def locate_point(point, zones):
    """Return the name of the first zone containing the point, else None."""
    for name, polygon in zones.items():
        if point_in_polygon(point, polygon):
            return name
    return None


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def draw_zones(frame, zones):
    for name, polygon in zones.items():
        contour = np.array(polygon, dtype=np.int32)
        cv2.polylines(frame, [contour], isClosed=True, color=ZONE_COLOR, thickness=2)
        x, y = contour[0]
        cv2.putText(frame, name, (int(x) + 6, int(y) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ZONE_COLOR, 2)
