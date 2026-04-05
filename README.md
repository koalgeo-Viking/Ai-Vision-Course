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

### Homework 12 — Binary Traffic Sign Classifier
**Dataset:** GTSRB subset (2 classes)  
**Task:** Build a binary classifier from scratch using a simple neural network.

- Visualised both classes and counted samples per class
- Baseline: single neuron with linear activation → **94.3% val accuracy**
- Improved: `32→16→1` architecture with ReLU + Sigmoid + `lr=0.0001` → **95.75% val accuracy**

**Key lesson:** Smaller architecture + lower learning rate outperforms a larger model on small datasets.

---

### Homework 13 — GTSRB Dataset Inspection
**Dataset:** German Traffic Sign Recognition Benchmark (43 classes, 50k+ images)  
**Task:** Inspect and analyse the dataset.

- Loaded `Train.csv` and visualised random samples
- Computed per-class histogram → dataset is **imbalanced** (ratio max/min ~10.7x)
- **Optional:** Analysed spatial resolution distribution (most images under 100px)
- **Optional:** Computed mean brightness per class — brightest: class 0 (118.9), darkest: class 6 (40.8)

**Key lesson:** Real-world datasets are rarely balanced — preprocessing and augmentation are essential.

---

### Homework 14 — CIFAR-10 Classifier
**Dataset:** CIFAR-10 (10 classes, 60k colour images 32×32)  
**Task:** Build and improve a CNN classifier.

- Baseline CNN (2 Conv blocks) → **~31% test accuracy**
- Improved CNN (3 Conv blocks + BatchNorm + Dropout + ReduceLROnPlateau) → **82% test accuracy**
- Improvement: **+50 pp**

**Key lesson:** BatchNormalization is critical for stable training on colour image datasets.

---

### Homework 15 — Fashion MNIST: Fighting Overfitting
**Dataset:** Fashion MNIST (10 clothing classes, 70k grayscale images 28×28)  
**Task:** Fix a heavily overfitting baseline classifier. Target: val accuracy > 91%.

| Model | Train acc | Val acc | Gap |
|---|---|---|---|
| Baseline | 99.3% | 88.3% | ~11% |
| Regularized | 90.8% | **91.77%** | ~0% |

Three attempts were needed:
1. `lr=0.001` → unstable, stopped at epoch 8 (~88.7%)
2. `lr=0.0001`, 1 Conv block → stable but plateaued at ~89.9%
3. `lr=0.0001`, 2 Conv blocks + BatchNorm + Dropout → **91.77%** ✅

**Key lesson:** Learning rate and network depth both matter — Fashion MNIST needs at least two Conv blocks to capture clothing textures properly.

---

## Structure

```
Ai-Vision-Course/
├── Lesson12/
│   └── Homework_12_TrafficSigns.ipynb
├── Lesson13/
│   └── Homework_13_GTSRB.ipynb
├── Lesson14/
│   └── Homework_14_CIFAR10.ipynb
├── Lesson15/
│   └── Homework_15_FashionMNIST.ipynb
└── README.md
```
