"""
preprocessing.py
────────────────
Step 1 of the Sudoku Solver pipeline.

Lessons applied:
  Lesson 1  — OpenCV basics (imread, cvtColor)
  Lesson 2  — Colour balancing (CLAHE)
  Lesson 3  — Gaussian blur
  Lesson 8  — Adaptive thresholding (Otsu / Gaussian)
"""

import cv2
import numpy as np


def load_image(path: str):
    """
    Load image from disk.  Lesson 1: cv2.imread returns BGR.
    Raises FileNotFoundError if path doesn't exist.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_grayscale(bgr_img):
    """Lesson 1: BGR → Grayscale colour channel conversion."""
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray_img, clip_limit=3.0, tile_grid=(8, 8)):
    """
    Lesson 2: Contrast Limited Adaptive Histogram Equalisation.
    Normalises brightness locally — handles shadows and uneven lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)


def gaussian_blur(gray_img, ksize=9, sigma=0):
    """
    Lesson 3: Gaussian blur to suppress high-frequency noise before
    thresholding.  ksize must be odd.
    """
    return cv2.GaussianBlur(gray_img, (ksize, ksize), sigma)


def adaptive_threshold(blurred_img, block_size=11, C=2):
    """
    Lesson 8: Adaptive (Gaussian-weighted) thresholding.
    Produces a binary image robust to non-uniform illumination.
    Returns a binary-inverted image (white lines on black background).
    """
    return cv2.adaptiveThreshold(
        blurred_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C
    )


def morphological_cleanup(binary_img, kernel_size=3, iterations=1):
    """
    Dilate to close small gaps in grid lines after thresholding.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(binary_img, kernel, iterations=iterations)


def preprocess(path: str):
    """
    Full preprocessing pipeline — returns all intermediate images.

    Returns dict with keys:
        original, gray, clahe, blurred, thresh, morph
    """
    original = load_image(path)
    gray     = to_grayscale(original)
    clahe    = apply_clahe(gray)
    blurred  = gaussian_blur(clahe)
    thresh   = adaptive_threshold(blurred)
    morph    = morphological_cleanup(thresh)

    return {
        "original": original,
        "gray"    : gray,
        "clahe"   : clahe,
        "blurred" : blurred,
        "thresh"  : thresh,
        "morph"   : morph,
    }
