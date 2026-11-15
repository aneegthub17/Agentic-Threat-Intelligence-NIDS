
import os
import sys

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

from detection_engine.models.tabular_transformer import TabularTransformer
from detection_engine.training.dataset import create_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "datasets/processed/train_processed.csv"
TEST_PATH = "datasets/processed/test_processed.csv"

MODEL_PATH = "detection_engine/models/best_transformer.pt"

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AGENTIC THREAT INTELLIGENCE NIDS")
print("TRANSFORMER MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

_, test_loader = create_dataloaders(
    TRAIN_PATH,
    TEST_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print(f"Testing samples: {len(test_loader.dataset):,}")


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating Transformer model...")

model = TabularTransformer(
    num_features=147,
    embedding_dim=64,
    num_heads=4,
    num_layers=3,
    feedforward_dim=128,
    dropout=0.1,
    num_classes=2
).to(DEVICE)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading best Transformer checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

print("Model checkpoint loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_targets = []
all_predictions = []
all_attack_probs = []

total_loss = 0.0

criterion = nn.CrossEntropyLoss()

print("\nRunning evaluation...")

with torch.no_grad():

    for features, targets in test_loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(features)

        loss = criterion(outputs, targets)

        # Probability of the Attack class (class 1)
        attack_probs = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        # Default prediction using argmax
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total_loss += loss.item() * features.size(0)

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_attack_probs.extend(
            attack_probs.cpu().numpy()
        )


# Convert collected results to NumPy arrays
all_targets = np.asarray(all_targets)
all_predictions = np.asarray(all_predictions)
all_attack_probs = np.asarray(all_attack_probs)


# ============================================================
# METRICS
# ============================================================

average_loss = (
    total_loss / len(test_loader.dataset)
)

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

precision = precision_score(
    all_targets,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER EVALUATION RESULTS")
print("=" * 70)

print(f"\nTest Loss      : {average_loss:.4f}")
print(f"Test Accuracy  : {accuracy:.4f}")
print(f"Test Precision : {precision:.4f}")
print(f"Test Recall    : {recall:.4f}")
print(f"Test F1        : {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print("\n                 Predicted")
print("              Normal  Attack")

print(
    f"Actual Normal  {tn:7d} {fp:7d}"
)

print(
    f"Actual Attack  {fn:7d} {tp:7d}"
)

# False Positive Rate
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# False Negative Rate
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"\nFalse Positive Rate : {fpr:.4f}")
print(f"False Negative Rate : {fnr:.4f}")
print(f"Specificity         : {specificity:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        all_targets,
        all_predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# ATTACK THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ATTACK THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"\n{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'FPR':>10} "
    f"{'FNR':>10}"
)

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

for threshold in thresholds:

    threshold_predictions = (
        all_attack_probs >= threshold
    ).astype(int)

    threshold_cm = confusion_matrix(
        all_targets,
        threshold_predictions,
        labels=[0, 1]
    )

    tn_t, fp_t, fn_t, tp_t = threshold_cm.ravel()

    threshold_accuracy = accuracy_score(
        all_targets,
        threshold_predictions
    )

    threshold_precision = precision_score(
        all_targets,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        all_targets,
        threshold_predictions,
        zero_division=0
    )

    threshold_f1 = f1_score(
        all_targets,
        threshold_predictions,
        zero_division=0
    )

    threshold_fpr = (
        fp_t / (fp_t + tn_t)
        if (fp_t + tn_t) > 0 else 0.0
    )

    threshold_fnr = (
        fn_t / (fn_t + tp_t)
        if (fn_t + tp_t) > 0 else 0.0
    )

    print(
        f"{threshold:10.2f} "
        f"{threshold_accuracy:10.4f} "
        f"{threshold_precision:10.4f} "
        f"{threshold_recall:10.4f} "
        f"{threshold_f1:10.4f} "
        f"{threshold_fpr:10.4f} "
        f"{threshold_fnr:10.4f}"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER EVALUATION COMPLETED")
print("=" * 70)
