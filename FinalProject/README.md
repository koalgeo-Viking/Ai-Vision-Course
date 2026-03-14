# Sudoku Solver from Photo 🔢📸

AI-powered system that solves Sudoku puzzles from photographs using classical Computer Vision techniques.

## Project Overview

This project automatically detects, extracts, and solves Sudoku puzzles from images using a multi-stage CV pipeline combining techniques learned throughout the AI Vision course.

## Features

- Automatic grid detection from photos
- Perspective correction for angled images
- Digit recognition using OCR
- Sudoku solving with backtracking algorithm
- Solution overlay on original image

## Pipeline
```
Input Photo → Preprocessing → Grid Detection → 
→ Perspective Transform → Cell Extraction → 
→ Digit Recognition → Solve Algorithm → 
→ Solution Overlay → Output
```

## Techniques Used (from course)

- **HW4**: Hough Transform for line detection
- **HW6**: Harris corner detection for grid corners
- **HW7**: Homography & perspective transformation
- **HW8**: Otsu thresholding for digit extraction
- **Additional**: Morphological operations, contour detection

## Project Structure
```
FinalProject/
  data/
    test_images/       # Test sudoku photos
    solved_results/    # Before/after comparisons
  src/
    preprocessing.py   # Image preprocessing
    grid_detector.py   # Find and extract sudoku grid
    cell_extractor.py  # Extract 81 individual cells
    digit_recognizer.py # OCR for digit recognition
    sudoku_solver.py   # Backtracking algorithm
    visualizer.py      # Overlay solution on image
    pipeline.py        # Complete pipeline
  notebooks/
    01_exploration.ipynb      # Initial experiments
    02_grid_detection.ipynb   # Grid detection testing
    03_digit_recognition.ipynb # OCR testing
    04_full_pipeline.ipynb    # Complete system
  docs/
    presentation.pdf   # Final presentation
    report.md         # Technical report
  requirements.txt
  README.md
```

## Setup
```bash
# Activate conda environment
conda activate cv

# Install additional dependencies
pip install pytesseract pillow
brew install tesseract  # macOS
```

## Usage

Coming soon...

## Development Progress

- [x] Project structure created
- [ ] Phase 1: Grid detection (Week 1)
- [ ] Phase 2: Cell extraction & digit recognition (Week 2)
- [ ] Phase 3: Solving algorithm (Week 3)
- [ ] Phase 4: Solution overlay & visualization (Week 4)
- [ ] Phase 5: Documentation & presentation (Week 5)

## Results

Performance metrics will be measured on:
- Detection accuracy (% of grids correctly detected)
- Recognition accuracy (% of digits correctly recognized)
- Solving success rate (% of puzzles correctly solved)
- Processing time per image

## Author

Oleksii Konakhovych - AI Vision Course Final Project

## License

MIT
