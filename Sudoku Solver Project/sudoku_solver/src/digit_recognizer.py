"""
digit_recognizer.py
───────────────────
Step 5 & 6 of the Sudoku Solver pipeline.

Lessons applied:
  Lesson 8   — Otsu/variance for empty cell detection
  Lesson 2   — CLAHE per cell
  Lessons 12–15 — CNN with BatchNorm, Dropout, augmentation
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path


# ── Step 5: Empty Cell Detection ────────────────────────────────────────────

def detect_empty_cells(cells, cell_size=50, threshold=200):
    """
    Lesson 8: Variance-based empty cell detection.
    Low pixel variance → empty; high variance → digit present.

    Returns:
        variance_map  9×9 float32 array
        empty_map     9×9 bool array (True = empty)
    """
    border       = int(cell_size * 0.10)
    variance_map = np.zeros((9, 9), dtype="float32")
    empty_map    = np.zeros((9, 9), dtype=bool)

    for r in range(9):
        for c in range(9):
            inner             = cells[r][c][border:-border, border:-border]
            var               = float(np.var(inner.astype("float32")))
            variance_map[r,c] = var
            empty_map[r,c]    = var < threshold

    return variance_map, empty_map


# ── Step 6a: Cell Preprocessing ─────────────────────────────────────────────

# Tunable guards — adjust if needed for your puzzle style
MIN_DIGIT_AREA   = 20    # px²  — ignore tiny specks
MIN_ASPECT_RATIO = 0.15  # w/h  — grid-line slivers are ~0.02–0.08
MAX_ASPECT_RATIO = 6.0   # w/h  — nothing legitimately this wide
MIN_FILL_RATIO   = 0.05  # fraction of cell area the component must cover


def preprocess_cell_for_cnn(cell_img):
    """
    Universal cell preprocessing — works for any sudoku photo.

    Key innovations:
    1. Connected Components for digit localisation (not fixed crop)
    2. Aspect-ratio + area guard to reject grid-line slivers
       (root cause of false '1' predictions on empty/border cells)
    3. Auto background detection (light or dark photo)
    4. CLAHE contrast normalisation (Lesson 2)
    """
    h, w = cell_img.shape
    brd  = int(min(h, w) * 0.10)
    cropped = cell_img[brd:-brd, brd:-brd]
    ch, cw  = cropped.shape

    # Lesson 2: CLAHE contrast normalisation
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(cropped)

    # Auto-detect background colour
    mean_brightness = float(np.mean(enhanced))
    if mean_brightness > 127:
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Connected Components with aspect-ratio + area guard
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    cell_area = ch * cw

    best_label = None
    best_area  = 0

    for lbl in range(1, num_labels):
        area   = int(stats[lbl, cv2.CC_STAT_AREA])
        wb     = int(stats[lbl, cv2.CC_STAT_WIDTH])
        hb     = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        if area < MIN_DIGIT_AREA or hb == 0:
            continue
        aspect = wb / hb
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue   # grid-line sliver or absurdly wide blob
        if area / cell_area < MIN_FILL_RATIO:
            continue   # covers < 5% of cell — probably noise

        if area > best_area:
            best_area  = area
            best_label = lbl

    if best_label is not None:
        x_l = int(stats[best_label, cv2.CC_STAT_LEFT])
        y_l = int(stats[best_label, cv2.CC_STAT_TOP])
        w_l = int(stats[best_label, cv2.CC_STAT_WIDTH])
        h_l = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        digit_region = binary[y_l:y_l+h_l, x_l:x_l+w_l]
        pad    = max(w_l, h_l) // 5
        padded = cv2.copyMakeBorder(digit_region, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=0)
    else:
        padded = np.zeros((28, 28), dtype=np.uint8)  # blank → empty cell

    resized    = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    normalised = resized.astype("float32") / 255.0
    return normalised


# ── Step 6b: Dataset Builder ─────────────────────────────────────────────────

def generate_printed_digits():
    """
    Synthetically generate printed digit images with OpenCV fonts.
    4 fonts × 4 scales × 2 thicknesses = 32 variants per digit (1–9).
    """
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]
    images, labels = [], []
    for digit in range(1, 10):
        for font in fonts:
            for scale in [0.8, 1.0, 1.2, 1.5]:
                for thickness in [1, 2]:
                    img = np.zeros((28, 28), dtype=np.uint8)
                    text = str(digit)
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    x = max(0, (28 - tw) // 2)
                    y = min(27, (28 + th) // 2)
                    cv2.putText(img, text, (x, y), font, scale, 255, thickness)
                    images.append(img)
                    labels.append(digit)
    return np.array(images), np.array(labels)


def augment_dataset(images, labels, n_augment=10, seed=42):
    """
    Lesson 15: Data augmentation to prevent overfitting.
    Random rotation, noise, Gaussian blur.
    """
    rng = np.random.default_rng(seed)
    aug_images, aug_labels = [], []
    for img, label in zip(images, labels):
        for _ in range(n_augment):
            aug   = img.copy()
            angle = rng.uniform(-15, 15)
            M     = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            aug   = cv2.warpAffine(aug, M, (28, 28))
            noise = rng.normal(0, 10, aug.shape).astype(np.int16)
            aug   = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if rng.random() > 0.5:
                aug = cv2.GaussianBlur(aug, (3, 3), 0.5)
            aug_images.append(aug)
            aug_labels.append(label)
    return np.array(aug_images), np.array(aug_labels)


def build_combined_dataset():
    """
    Build MNIST + printed-digit dataset for combined training.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    printed_imgs, printed_lbls = generate_printed_digits()
    aug_imgs, aug_lbls = augment_dataset(printed_imgs, printed_lbls, n_augment=10)
    x_combined = np.vstack([x_train, aug_imgs])
    y_combined  = np.concatenate([y_train, aug_lbls])
    return x_combined, y_combined, x_test, y_test


# ── Step 6c: CNN Model ───────────────────────────────────────────────────────

def build_cnn_model():
    """
    Digit recognition CNN.
    Lessons 12–15: Conv2D + BatchNorm + Dropout + softmax.
    """
    model = models.Sequential([
        # Block 1 — Lesson 12: spatial feature extraction
        layers.Conv2D(32, (3,3), padding="same", input_shape=(28,28,1), name="conv1"),
        layers.BatchNormalization(name="bn1"),   # Lesson 13
        layers.Activation("relu"),
        layers.Conv2D(32, (3,3), padding="same", name="conv1b"),
        layers.BatchNormalization(name="bn1b"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25, name="drop1"),      # Lesson 14

        # Block 2 — deeper features
        layers.Conv2D(64, (3,3), padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.Activation("relu"),
        layers.Conv2D(64, (3,3), padding="same", name="conv2b"),
        layers.BatchNormalization(name="bn2b"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25, name="drop2"),

        # Classifier — Lesson 14: Dropout prevents overfitting
        layers.Flatten(),
        layers.Dense(128, name="fc1"),
        layers.BatchNormalization(name="bn3"),
        layers.Activation("relu"),
        layers.Dropout(0.5, name="drop3"),       # Lesson 14: 50%
        layers.Dense(10, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_or_load(model_path: str, force_retrain=False):
    """
    Train CNN (or load from disk if already trained).
    Saves best model to model_path.
    Returns (model, history_or_None).
    """
    mp = Path(model_path)
    if mp.exists() and not force_retrain:
        print(f"Loading saved model: {mp}")
        model = tf.keras.models.load_model(str(mp))
        return model, None

    print("Training CNN on combined dataset …")
    x_comb, y_comb, x_te, y_te = build_combined_dataset()
    x_tr = (x_comb.astype("float32") / 255.0)[..., np.newaxis]
    x_te = (x_te.astype("float32")  / 255.0)[..., np.newaxis]

    model = build_cnn_model()
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(str(mp), save_best_only=True,
                        monitor="val_accuracy", verbose=1),
    ]
    history = model.fit(
        x_tr, y_comb,
        epochs=20, batch_size=256,
        validation_data=(x_te, y_te),
        callbacks=callbacks, verbose=1,
    )
    return model, history


# ── Step 6d: Inference ───────────────────────────────────────────────────────

def predict_digit(cell_img, model, confidence_threshold=0.60):
    """
    Preprocess cell and run CNN inference.
    Returns (digit, confidence).  digit=0 means 'uncertain/empty'.
    """
    preprocessed = preprocess_cell_for_cnn(cell_img)

    # Extra guard: nearly blank → aspect-ratio filter removed everything
    if preprocessed.max() < 0.1:
        return 0, 1.0

    probs = model.predict(preprocessed.reshape(1, 28, 28, 1), verbose=0)[0]
    digit = int(np.argmax(probs))
    conf  = float(probs[digit])
    if conf < confidence_threshold:
        return 0, conf
    return digit, conf


def recognize_board(cells, empty_map, model):
    """
    Run CNN on all non-empty cells and assemble the 9×9 board.
    """
    board      = np.zeros((9, 9), dtype=int)
    conf_map   = np.zeros((9, 9), dtype=float)
    for r in range(9):
        for c in range(9):
            if empty_map[r, c]:
                continue
            digit, conf    = predict_digit(cells[r][c], model)
            board[r, c]    = digit
            conf_map[r, c] = conf
    return board, conf_map
