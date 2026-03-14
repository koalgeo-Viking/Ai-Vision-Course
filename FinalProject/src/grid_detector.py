"""
Sudoku grid detection
Uses techniques from HW4 (Hough lines) and HW6 (Harris corners)
"""

import cv2
import numpy as np


def find_sudoku_grid(image, binary):
    """
    Find the sudoku grid in the image
    
    Args:
        image: Original BGR image
        binary: Preprocessed binary image
        
    Returns:
        Corners of the grid (4 points) or None if not found
    """
    # Find contours
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Find the largest contour (likely the sudoku grid)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle area
    area = cv2.contourArea(largest_contour)
    img_area = image.shape[0] * image.shape[1]
    
    # Grid should be significant portion of image (at least 10%)
    if area < img_area * 0.1:
        return None
    
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Should have 4 corners
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    return None


def order_corners(corners):
    """
    Order corners in consistent way: top-left, top-right, bottom-right, bottom-left
    Similar to HW7 perspective transform preparation
    
    Args:
        corners: 4 corner points
        
    Returns:
        Ordered corners
    """
    # Calculate center
    center = corners.mean(axis=0)
    
    # Separate into top and bottom
    top = corners[corners[:, 1] < center[1]]
    bottom = corners[corners[:, 1] >= center[1]]
    
    # Sort top by x (left to right)
    top = top[top[:, 0].argsort()]
    # Sort bottom by x (left to right)  
    bottom = bottom[bottom[:, 0].argsort()]
    
    # Return in order: TL, TR, BR, BL
    if len(top) == 2 and len(bottom) == 2:
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    return corners


if __name__ == "__main__":
    print("Grid detector module ready")
    print("Functions: find_sudoku_grid(), order_corners()")
