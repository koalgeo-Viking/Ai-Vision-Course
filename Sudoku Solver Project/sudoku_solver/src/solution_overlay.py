"""
solution_overlay.py
───────────────────
Step 8 of the Sudoku Solver pipeline.

Lesson applied:
  Lesson 7 — Inverse warpPerspective to project solution back onto the
              original image coordinate system.
"""

import cv2
import numpy as np


def draw_solution_on_grid(board_recognised, board_solved,
                          grid_size=450, cell_size=50):
    """
    Draw only newly-solved digits (cells that were empty in the input)
    onto a blank BGRA canvas the same size as the rectified grid.

    Returns a (grid_size × grid_size × 4) BGRA image.
    Alpha channel is 0 where no digit is drawn.
    """
    canvas = np.zeros((grid_size, grid_size, 4), dtype=np.uint8)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 1.2
    thick  = 2

    for r in range(9):
        for c in range(9):
            orig   = (board_recognised[r][c]
                      if isinstance(board_recognised, list)
                      else int(board_recognised[r, c]))
            solved = (board_solved[r][c]
                      if isinstance(board_solved, list)
                      else int(board_solved[r, c]))

            if orig != 0:        # already present in original — skip
                continue

            text             = str(solved)
            (tw, th), _      = cv2.getTextSize(text, font, scale, thick)
            x0               = c * cell_size + (cell_size - tw) // 2
            y0               = r * cell_size + (cell_size + th) // 2

            # Blue digits with full alpha
            cv2.putText(canvas, text, (x0, y0),
                        font, scale, (255, 80, 0, 255), thick)

    return canvas


def overlay_solution(original_img, board_recognised, board_solved,
                     H_inv, grid_size=450, cell_size=50):
    """
    Lesson 7: Inverse warpPerspective — project solution back onto original.

    Steps:
      1. Render solution digits on rectified canvas
      2. Warp canvas → original perspective using H_inv
      3. Alpha-blend with original image

    Args:
        original_img       BGR image (original photo)
        board_recognised   9×9 int array (0 = empty)
        board_solved       9×9 int array (fully filled)
        H_inv              inverse homography from rectify_grid()
        grid_size          must match rectification size (default 450)
        cell_size          grid_size // 9

    Returns result BGR image same shape as original_img.
    """
    h_orig, w_orig = original_img.shape[:2]

    canvas_bgra  = draw_solution_on_grid(
        board_recognised, board_solved, grid_size, cell_size
    )
    canvas_bgr   = canvas_bgra[:, :, :3]
    canvas_alpha = canvas_bgra[:, :,  3]

    # Inverse perspective warp (Lesson 7)
    warped_back  = cv2.warpPerspective(canvas_bgr,   H_inv, (w_orig, h_orig))
    alpha_back   = cv2.warpPerspective(canvas_alpha, H_inv, (w_orig, h_orig))

    # Alpha composite
    alpha_f = (alpha_back.astype("float32") / 255.0)[..., np.newaxis]
    result  = (original_img.astype("float32") * (1 - alpha_f)
               + warped_back.astype("float32") * alpha_f)
    result  = np.clip(result, 0, 255).astype(np.uint8)

    return result
