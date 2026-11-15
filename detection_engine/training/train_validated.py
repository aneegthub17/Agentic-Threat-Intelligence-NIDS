
"""
Validation-based Transformer training for NIDS.

Safeguards:
- Uses only unified_train.csv for training and validation.
- Fits imputation and scaling on the training partition only.
- Selects checkpoints using validation F1, never test F1.
- Saves to best_transformer_validated.pt, preserving the existing model.
- Does not access the independent UNSW test dataset.

Usage:
    python -m detection_engine.training.train_validated --dry-run
    python -m detection_engine.training.train_validated --epochs 5

Warning: Full training can be time-consuming.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from detection_engine.models.tabular_transformer import TabularTransformer
from utfh.preprocessing.preprocess import (
    NUMERIC_FEATURES,
    normalize_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "datasets" / "merged" / "unified_train.csv"
OUTPUT_DIR = ROOT / "detection_engine" / "models"
CHECKPOINT_PATH = OUTPUT_DIR / "best_transformer_validated.pt"
PREPROCESSOR_PATH = OUTPUT_DIR / "validated_preprocessor.joblib"
METRICS_PATH = OUTPUT_DIR / "validated_training_metrics.json"

SEED = 42
BATCH_SIZE = 256
VALIDATION_SIZE = 0.20
FEATURE_COUNT = 147


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_partition(frame, medians, protocol_columns):
    numeric = frame[NUMERIC_FEATURES].copy()

    for column in NUMERIC_FEATURES:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(medians).fillna(0.0)

    protocol = normalize_protocol(frame["protocol"])
    encoded = pd.get_dummies(
        protocol,
        prefix="protocol",
        dtype=np.float32,
    )
    encoded = encoded.reindex(columns=protocol_columns, fill_value=0.0)

    numeric = numeric.reset_index(drop=True)
    encoded = encoded.reset_index(drop=True)

    return pd.concat([numeric, encoded], axis=1)


def build_arrays(train_frame, val_frame):
    numeric_train = train_frame[NUMERIC_FEATURES].copy()

    for column in NUMERIC_FEATURES:
        numeric_train[column] = pd.to_numeric(
            numeric_train[column], errors="coerce"
        )

    numeric_train = numeric_train.replace([np.inf, -np.inf], np.nan)
    medians = numeric_train.median().fillna(0.0)

    train_protocol = normalize_protocol(train_frame["protocol"])
    protocol_columns = list(
        pd.get_dummies(train_protocol, prefix="protocol").columns
    )

    x_train_frame = prepare_partition(
        train_frame, medians, protocol_columns
    )
    x_val_frame = prepare_partition(
        val_frame, medians, protocol_columns
    )

    feature_names = list(x_train_frame.columns)

    if len(feature_names) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, got {len(feature_names)}. "
            "The model configuration must match the actual feature schema."
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_frame).astype(np.float32)
    x_val = scaler.transform(x_val_frame).astype(np.float32)

    y_train = train_frame["target"].to_numpy(dtype=np.int64)
    y_val = val_frame["target"].to_numpy(dtype=np.int64)

    preprocessor = {
        "medians": medians.to_dict(),
        "protocol_columns": protocol_columns,
        "feature_names": feature_names,
        "scaler": scaler,
        "seed": SEED,
        "validation_size": VALIDATION_SIZE,
    }

    return x_train, y_train, x_val, y_val, preprocessor


def make_loader(x, y, batch_size, shuffle, device):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    targets_all = []
    predictions_all = []

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(targets)
        targets_all.extend(targets.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets_all, predictions_all),
        "precision": precision_score(
            targets_all, predictions_all, zero_division=0
        ),
        "recall": recall_score(
            targets_all, predictions_all, zero_division=0
        ),
        "f1": f1_score(targets_all, predictions_all, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("NIDS — VALIDATION-BASED TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Source: {TRAIN_CSV}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("Independent test set: NOT USED")

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    frame = pd.read_csv(
        TRAIN_CSV,
        usecols=["protocol", *NUMERIC_FEATURES, "target"],
    )

    if frame["target"].isna().any():
        raise ValueError("Training targets contain missing values")

    train_frame, val_frame = train_test_split(
        frame,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=frame["target"],
    )

    print(f"Training records: {len(train_frame):,}")
    print(f"Validation records: {len(val_frame):,}")

    x_train, y_train, x_val, y_val, preprocessor = build_arrays(
        train_frame, val_frame
    )

    print(f"Feature count: {x_train.shape[1]}")
    print(f"Training shape: {x_train.shape}")
    print(f"Validation shape: {x_val.shape}")

    if args.dry_run:
        print("\nDRY RUN PASSED")
        print("No checkpoint, scaler, or metrics files were written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        x_train, y_train, args.batch_size, True, device
    )
    val_loader = make_loader(
        x_val, y_val, args.batch_size, False, device
    )

    model = TabularTransformer(
        num_features=x_train.shape[1],
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
        feedforward_dim=128,
        dropout=0.1,
        num_classes=2,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("Both Normal and Attack classes are required")

    class_weights = len(y_train) / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Validation F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "best_validation_f1": best_f1,
                    "num_features": x_train.shape[1],
                    "num_classes": 2,
                },
                CHECKPOINT_PATH,
            )

            print("  Saved new best validation checkpoint.")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_validation_f1": best_f1,
                "training_samples": len(y_train),
                "validation_samples": len(y_val),
                "history": history,
                "independent_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
