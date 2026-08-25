import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Allow importing project modules
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

# Support both a raw state_dict and a checkpoint dictionary.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

print("✓ Model checkpoint loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_targets = []
all_predictions = []

total_loss = 0.0
criterion = nn.CrossEntropyLoss()

print("\nRunning evaluation...")

with torch.no_grad():

    for features, targets in test_loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(features)

        loss = criterion(outputs, targets)

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
    all_predictions
)

print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

print("\n                 Predicted")
print("              Normal  Attack")
print(
    f"Actual Normal  {cm[0][0]:7d} {cm[0][1]:7d}"
)
print(
    f"Actual Attack  {cm[1][0]:7d} {cm[1][1]:7d}"
)


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
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("=" * 70)
print("TRANSFORMER EVALUATION COMPLETED")
print("=" * 70)