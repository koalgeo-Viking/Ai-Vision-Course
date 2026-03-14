"""
Image preprocessing for sudoku detection
Uses techniques from Homework 8 (Otsu thresholding)
"""

import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess input image for grid detection
    
    Args:
        image: Input BGR image
        
    Returns:
        Preprocessed binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding (better than Otsu for varying lighting)
    # Could also use Otsu from HW8
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    return gray, binary


def remove_noise(binary_image):
    """
    Remove small noise using morphological operations
    
    Args:
        binary_image: Binary input image
        
    Returns:
        Cleaned binary image
    """
    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Morphological closing to fill small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned


if __name__ == "__main__":
    # Test preprocessing
    print("Preprocessing module ready")
    print("Functions: preprocess_image(), remove_noise()")
