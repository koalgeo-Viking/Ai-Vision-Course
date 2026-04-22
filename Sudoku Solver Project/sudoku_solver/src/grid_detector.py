"""
grid_detector.py
────────────────
Step 2 of the Sudoku Solver pipeline.

Lessons applied:
  Lesson 4 — Hough transform / edge detection (contour = boundary of edge region)
  Lesson 6 — Harris corners / feature descriptors (corner ordering)
"""

import cv2
import numpy as np


def order_corners(pts):
    """
    Re-order 4 corner points into canonical [TL, TR, BR, BL] order.

    Lesson 6 — spatial feature ordering:
      TL has the smallest (x + y)
      BR has the largest  (x + y)
      TR has the smallest (x - y)
      BL has the largest  (x - y)
    """
    pts     = pts.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]    # TL
    ordered[2] = pts[np.argmax(s)]    # BR
    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)] # TR
    ordered[3] = pts[np.argmax(diff)] # BL
    return ordered


def find_grid_contour(morph_img, min_area=10_000):
    """
    Lesson 4: Find the largest quadrilateral contour in a binary image.
    This contour corresponds to the sudoku grid border.

    Returns the 4-point approximated contour, or raises ValueError.
    """
    contours, _ = cv2.findContours(
        morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found — check preprocessing output.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

    raise ValueError(
        "Could not find a quadrilateral grid contour. "
        "Try adjusting preprocessing parameters."
    )


def detect_grid(morph_img, original_img):
    """
    Full grid detection:
    1. Find largest quadrilateral contour   (Lesson 4)
    2. Order corners TL→TR→BR→BL           (Lesson 6)
    3. Draw debug visualisation

    Returns (corners, debug_img) or raises ValueError.
    """
    debug_img = original_img.copy()
    grid_cnt  = find_grid_contour(morph_img)

    # Draw detected border
    cv2.drawContours(debug_img, [grid_cnt], -1, (0, 255, 0), 3)

    # Order corners
    corners = order_corners(grid_cnt)

    # Annotate corners
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0)]
    for (x, y), label, color in zip(corners, labels, colors):
        cv2.circle(debug_img, (int(x), int(y)), 10, color, -1)
        cv2.putText(debug_img, label, (int(x) + 12, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return corners, debug_img
