import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "datasets" / "processed" / "train_processed.csv"
TEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "test_processed.csv"

MODEL_DIR = PROJECT_ROOT / "detection_engine" / "models"
MODEL_PATH = MODEL_DIR / "best_transformer.pt"

# Make project root importable when running this file directly.
sys.path.insert(0, str(PROJECT_ROOT))

from detection_engine.training.dataset import create_dataloaders
from detection_engine.models.tabular_transformer import TabularTransformer


# ============================================================
# CONFIGURATION 
# "These are hyperparameters selected to balance computational efficiency, training stability, generalization, and model performance. Some parameters, such as the number of features and classes, are determined by the problem and dataset rather than tuned."
# ============================================================

BATCH_SIZE = 256
NUM_EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_FEATURES = 147
NUM_CLASSES = 2

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_loader):
    """
    Calculate inverse-frequency class weights from
    the training dataset.
    """

    targets = train_loader.dataset.y

    class_counts = torch.bincount(targets, minlength=NUM_CLASSES)

    total_samples = class_counts.sum()

    class_weights = total_samples / (
        NUM_CLASSES * class_counts.float()
    )

    return class_weights


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer
):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            DEVICE,
            non_blocking=True
        )

        batch_y = batch_y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * batch_x.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_targets.extend(
            batch_y.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(train_loader.dataset)
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

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    test_loader,
    criterion
):
    """
    Evaluate the model on the independent
    UNSW-NB15 test dataset.
    """

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            running_loss += (
                loss.item() * batch_x.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.cpu().numpy()
            )

    loss = (
        running_loss /
        len(test_loader.dataset)
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

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("TRANSFORMER TRAINING")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_loader, test_loader = create_dataloaders(
        TRAIN_PATH,
        TEST_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Testing samples: "
        f"{len(test_loader.dataset):,}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nCreating Transformer model...")

    model = TabularTransformer(
        num_features=NUM_FEATURES,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
        feedforward_dim=128,
        dropout=0.1,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader
    ).to(DEVICE)

    print("\nClass distribution:")

    print(
        f"Normal  : "
        f"{int(train_loader.dataset.y.eq(0).sum()):,}"
    )

    print(
        f"Attack  : "
        f"{int(train_loader.dataset.y.eq(1).sum()):,}"
    )

    print(
        "\nClass weights:"
    )

    print(
        f"Normal  : {class_weights[0].item():.4f}"
    )

    print(
        f"Attack  : {class_weights[1].item():.4f}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_f1 = 0.0

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_results = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        (
            train_loss,
            train_accuracy,
            train_precision,
            train_recall,
            train_f1
        ) = train_results

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : {train_accuracy:.4f}"
        )

        print(
            f"Train Precision : {train_precision:.4f}"
        )

        print(
            f"Train Recall    : {train_recall:.4f}"
        )

        print(
            f"Train F1        : {train_f1:.4f}"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        test_results = evaluate(
            model,
            test_loader,
            criterion
        )

        (
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1
        ) = test_results

        print(
            f"\nTest Loss       : {test_loss:.4f}"
        )

        print(
            f"Test Accuracy   : {test_accuracy:.4f}"
        )

        print(
            f"Test Precision  : {test_precision:.4f}"
        )

        print(
            f"Test Recall     : {test_recall:.4f}"
        )

        print(
            f"Test F1         : {test_f1:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if test_f1 > best_f1:

            best_f1 = test_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "num_features": NUM_FEATURES,
                    "num_classes": NUM_CLASSES
                },
                MODEL_PATH
            )

            print(
                f"\n✓ Best model saved."
            )

            print(
                f"  Path: {MODEL_PATH}"
            )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRANSFORMER TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nBest Test F1: {best_f1:.4f}"
    )

    print(
        f"Model saved to:"
    )

    print(MODEL_PATH)


if __name__ == "__main__":
    main()