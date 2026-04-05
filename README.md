# AI & Computer Vision Course

Homework assignments for the AI & Computer Vision course.
Built with Python, TensorFlow/Keras, and OpenCV on Apple M5 (macOS).

---

## Environment

| Tool | Version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.16.2 (tensorflow-macos + tensorflow-metal) |
| OpenCV | 4.9.0.80 |
| NumPy | 1.26.4 |
| Conda env | `cv` |

---

## Homeworks

### Homework 1 — Image Collage & Geometric Transformations
**File:** `Lesson1/HomeWork1.ipynb`

First steps with OpenCV. Loaded an image, split it into RGB channels and created
a 2×2 colour collage with swapped channels (RGB, RBG, GRB, BGR).
Also built a geometric collage using horizontal and vertical flips (`fliplr`, `flipud`).

---

### Homework 2 — Colour Balancing
**File:** `Lesson2/HomeWork2.ipynb`

Implemented two colour balancing algorithms based on von Kries' hypothesis:
- **White Patch** — scales channels based on a known white pixel
- **Gray World** — assumes the scene average is gray, computes per-channel coefficients
- **Scale-by-Max** — scales each channel by its maximum value

Experimented with own images and compared the visual effect of each algorithm.

---

### Homework 3 — Unsharp Masking (USM)
**File:** `Lesson3/LineFilterLesson3.ipynb`

Implemented the Unsharp Masking filter:
`sharpened = original + (original − unsharp) × amount`

Used Gaussian blur to create the unsharp version. Handled uint8 overflow/underflow
by casting to float. Answered questions about the effect of the `amount` parameter.

---

### Homework 4 — Lane Line Detection
**File:** `Lesson4/HomeWork4.ipynb`

Built a lane line detector for ADAS using:
- Grayscale conversion + Canny edge detection
- Hough transform for line parametrization
- Horizontal line filtering (±20° from vertical)
- K-means clustering to merge similar lines into 6 final lane lines

Answered questions about Hough resolution and accumulator threshold importance.

---

### Homework 5 — Floyd-Steinberg Dithering
**File:** `Lesson5/HomeWork5.ipynb`

Implemented the Floyd-Steinberg dithering algorithm:
- Defined a 4-colour grayscale palette (black, dark gray, light gray, white)
- Baseline: optimal quantization with average error computation
- FS dithering: error diffusion across neighbouring pixels
- **Bonus:** repeated with K-means optimal 16-colour palette, tested 32 and 256 colours

---

### Homework 6 — Harris Corner Detection & Document Corner Localization
**File:** `Lesson6/HomeWork6.ipynb`

Used Harris corner detector to find the 4 corners of a document:
- Computed cornerness score with Harris detector
- Designed a custom quadrant-based feature descriptor for each corner type
  (top-left, top-right, bottom-left, bottom-right)
- Each descriptor compares brightness of the "paper" quadrant vs background quadrants

Answered questions about camera resolution impact on the algorithm.

---

### Homework 7 — Document Rectification
**File:** `Lesson7/HomeWork7.ipynb`

Used the corners from Homework 6 to rectify a distorted document image:
- Tested **Affine Transform** with first 3 points → poor result
- Tested **Affine Transform** with last 3 points → still poor
- Tested **estimateAffine2D** with all 4 points + RANSAC → marginal improvement
- Applied **Homography** (`getPerspectiveTransform` + `warpPerspective`) → correct result

Answered why affine transform fails for perspective distortion.

---

### Homework 8 — Otsu Thresholding
**File:** `Lesson8/HomeWork8.ipynb`

Implemented the Otsu binarization algorithm from scratch (brute force approach):
- Computed image histogram and identified bimodal distribution
- Iteratively tested all 256 thresholds to minimize within-class variance
- Found optimal threshold that separates document text from background

Answered questions about histogram bimodality and binarization quality.

---

### Homework 9 — Face Detection with dlib
**File:** `Lesson9/HomeWork9.ipynb`

Built a face detector from scratch using dlib:
- Loaded an image with faces
- Applied dlib face predictor
- Drew bounding boxes with different colours per face
- **Optional:** tested on challenging images (glasses, hats, small faces, crowds)

---

### Homework 10 — Object Tracking: KCF vs CSRT
**File:** `Lesson10/HomeWork10.ipynb`

Compared two OpenCV object trackers on a video sequence (~15 frames):
- **KCF** (Kernelized Correlation Filters) — faster, less accurate
- **CSRT** (Discriminative Correlation Filter with Channel and Spatial Reliability) — slower, more accurate

Saved per-frame bounding box results for both trackers and compared performance.
Results saved in `Lesson10/tracking_results/`.

---

### Homework 12 — Binary Traffic Sign Classifier
**File:** `Lesson12/Homework_12_TrafficSigns.ipynb`
**Dataset:** GTSRB subset (2 classes)

- Baseline: single neuron, linear activation → **94.3% val accuracy**
- Improved: `32→16→1` + ReLU + Sigmoid + `lr=0.0001` → **95.75% val accuracy**

**Key lesson:** Smaller architecture + lower learning rate outperforms a larger
model on small datasets.

---

### Homework 13 — GTSRB Dataset Inspection
**File:** `Lesson13/Homework_13_GTSRB.ipynb`
**Dataset:** German Traffic Sign Recognition Benchmark (43 classes, 50k+ images)

- Loaded `Train.csv` and visualised random samples
- Per-class histogram → dataset is **imbalanced** (ratio max/min ~10.7x)
- **Optional:** resolution distribution analysis (most images under 100px)
- **Optional:** brightness per class — brightest: class 0 (118.9), darkest: class 6 (40.8)

**Key lesson:** Real-world datasets are rarely balanced — preprocessing is essential.

---

### Homework 14 — CIFAR-10 Classifier
**File:** `Lesson14/Homework_14_CIFAR10.ipynb`
**Dataset:** CIFAR-10 (10 classes, 60k colour images 32×32)

- Baseline CNN (2 Conv blocks) → **~31% test accuracy**
- Improved (3 Conv blocks + BatchNorm + Dropout + ReduceLROnPlateau) → **82% test accuracy**
- Improvement: **+50 pp**

**Key lesson:** BatchNormalization is critical for stable training on colour images.

---

### Homework 15 — Fashion MNIST: Fighting Overfitting
**File:** `Lesson15/Homework_15_FashionMNIST.ipynb`
**Dataset:** Fashion MNIST (10 clothing classes, 70k grayscale 28×28)

| Model | Train acc | Val acc | Gap |
|---|---|---|---|
| Baseline | 99.3% | 88.3% | ~11% |
| Regularized | 90.8% | **91.77%** | ~0% |

Three attempts to reach >91% val accuracy:
1. `lr=0.001` → unstable, stopped at epoch 8 (~88.7%)
2. `lr=0.0001`, 1 Conv block → plateaued at ~89.9%
3. `lr=0.0001`, 2 Conv blocks + BatchNorm + Dropout → **91.77%** ✅

**Key lesson:** Learning rate and network depth both matter — Fashion MNIST needs
at least two Conv blocks to capture clothing textures properly.

---

## Structure

```
Ai-Vision-Course/
├── Lesson1/
│   └── HomeWork1.ipynb
├── Lesson2/
│   └── HomeWork2.ipynb
├── Lesson3/
│   └── LineFilterLesson3.ipynb
├── Lesson4/
│   └── HomeWork4.ipynb
├── Lesson5/
│   └── HomeWork5.ipynb
├── Lesson6/
│   └── HomeWork6.ipynb
├── Lesson7/
│   └── HomeWork7.ipynb
├── Lesson8/
│   └── HomeWork8.ipynb
├── Lesson9/
│   └── HomeWork9.ipynb
├── Lesson10/
│   ├── HomeWork10.ipynb
│   └── tracking_results/
├── Lesson12/
│   └── Homework_12_TrafficSigns.ipynb
├── Lesson13/
│   └── Homework_13_GTSRB.ipynb
├── Lesson14/
│   └── Homework_14_CIFAR10.ipynb
├── Lesson15/
│   └── Homework_15_FashionMNIST.ipynb
├── FinalProject/
│   ├── notebooks/
│   ├── src/
│   └── requirements.txt
└── README.md
```
