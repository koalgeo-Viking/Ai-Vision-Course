"""
cell_extractor.py
─────────────────
Steps 3 & 4 of the Sudoku Solver pipeline.

Lessons applied:
  Lesson 7  — Homography, getPerspectiveTransform, warpPerspective
  Lesson 3  — Unsharp Masking per cell
  Lesson 9  — Region-of-interest concept (cells as ROIs)
  Lesson 10 — Cell tracking concept (consistent cell ordering)
"""

import cv2
import numpy as np


# ── Step 3: Perspective Rectification ───────────────────────────────────────

def rectify_grid(image, corners, grid_size=450):
    """
    Lesson 7: Apply perspective transform to un-distort the sudoku grid.

    Maps the 4 detected corners → perfect (grid_size × grid_size) square.

    Returns:
        warped      BGR image   (grid_size × grid_size)
        warped_gray grayscale   (grid_size × grid_size)
        H           forward homography matrix  (3×3)
        H_inv       inverse homography matrix  (3×3, used in Step 8)
    """
    src = corners.astype("float32")
    dst = np.array(
        [[0,         0        ],
         [grid_size, 0        ],
         [grid_size, grid_size],
         [0,         grid_size]],
        dtype="float32",
    )

    H     = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)

    warped      = cv2.warpPerspective(image, H, (grid_size, grid_size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    return warped, warped_gray, H, H_inv


# ── Step 4: Cell Extraction ──────────────────────────────────────────────────

def unsharp_mask(img, blur_ksize=3, amount=1.5):
    """
    Lesson 3: Unsharp Masking.
    Enhances digit edges before CNN inference.
        sharp = img + amount * (img - gaussian_blur(img))
    """
    blurred   = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


def extract_cells(warped_gray, cell_size=50):
    """
    Lesson 9/10: Slice 450×450 grid into 81 ROIs (cells).
    Each cell has Unsharp Masking applied (Lesson 3).

    Returns cells[row][col] — a 9×9 list of (cell_size × cell_size) arrays.
    """
    cells = []
    for r in range(9):
        row = []
        for c in range(9):
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            cell = warped_gray[y0:y1, x0:x1].copy()
            cell = unsharp_mask(cell)   # Lesson 3
            row.append(cell)
        cells.append(row)
    return cells
