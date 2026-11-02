"""
Step 2 - Vessel type image classification.

Approach : Transfer learning. A MobileNetV2 network pre-trained on ImageNet is used
           as a frozen feature extractor (1280 features per image); classic
           classifiers (Logistic Regression, SVM) are trained on top.
           This suits a small dataset (100 images) and trains in seconds on CPU.
Evaluation: stratified 5-fold cross-validation + a held-out 80/20 test split.
Outputs  : results/ (metrics.json, confusion_matrix.png, class_distribution.png,
           sample_images.png, cv_scores.png)

Run      : python 02_vessel_classifier.py
"""
import os, json, glob, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
DATA_DIR, RES_DIR, IMG_SIZE = "dataset", "results", 224
os.makedirs(RES_DIR, exist_ok=True)

# ---------- 1. Load data ----------
classes = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(f"{DATA_DIR}/{d}"))
paths, labels = [], []
for i, c in enumerate(classes):
    for p in sorted(glob.glob(f"{DATA_DIR}/{c}/*.jpg")):
        paths.append(p); labels.append(i)
y = np.array(labels)
print(f"{len(paths)} images, {len(classes)} classes: {classes}")

def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32")
X_img = np.stack([load(p) for p in paths])

# ---------- 2. EDA plots ----------
counts = np.bincount(y)
plt.figure(figsize=(6, 3.5)); sns.barplot(x=classes, y=counts, palette="Blues_d")
plt.title("Images per class"); plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(f"{RES_DIR}/class_distribution.png", dpi=150); plt.close()

fig, axs = plt.subplots(len(classes), 4, figsize=(8, 2 * len(classes)))
for r, c in enumerate(range(len(classes))):
    idx = np.where(y == c)[0][:4]
    for k, j in enumerate(idx):
        axs[r, k].imshow(X_img[j].astype("uint8")); axs[r, k].axis("off")
    axs[r, 0].set_title(classes[c], fontsize=8, loc="left")
plt.tight_layout(); plt.savefig(f"{RES_DIR}/sample_images.png", dpi=120); plt.close()

# ---------- 3. Feature extraction (pre-trained CNN) ----------
base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
features = base.predict(preprocess_input(X_img.copy()), batch_size=16, verbose=0)
print("Feature matrix:", features.shape)

# ---------- 4. Models ----------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
metrics, cv_all = {}, {}
for name, m in models.items():
    cv = cross_val_score(m, features, y, cv=skf, scoring="accuracy")
    cv_all[name] = cv
    metrics[name] = {"cv_mean_accuracy": float(cv.mean()), "cv_std": float(cv.std())}
    print(f"{name}: 5-fold CV accuracy = {cv.mean():.3f} +/- {cv.std():.3f}")

# ---------- 5. Hold-out test with the best model ----------
best = max(metrics, key=lambda k: metrics[k]["cv_mean_accuracy"])
Xtr, Xte, ytr, yte = train_test_split(features, y, test_size=0.2, stratify=y, random_state=SEED)
model = models[best].fit(Xtr, ytr)
pred = model.predict(Xte)
metrics["best_model"] = best
metrics["test_accuracy"] = float(accuracy_score(yte, pred))
metrics["test_macro_f1"] = float(f1_score(yte, pred, average="macro"))
metrics["classification_report"] = classification_report(yte, pred, target_names=classes, output_dict=True)
print(f"\nBest model: {best}\nTest accuracy: {metrics['test_accuracy']:.3f}")
print(classification_report(yte, pred, target_names=classes))

cm = confusion_matrix(yte, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best}")
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(f"{RES_DIR}/confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(5, 3.5))
plt.boxplot(list(cv_all.values()), labels=list(cv_all.keys())); plt.ylabel("CV accuracy")
plt.title("5-fold cross-validation"); plt.tight_layout()
plt.savefig(f"{RES_DIR}/cv_scores.png", dpi=150); plt.close()

with open(f"{RES_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved plots and metrics.json in ./{RES_DIR}/")
