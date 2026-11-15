
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "datasets/merged/unified_train.csv"
TEST_PATH = ROOT / "datasets/merged/unified_test.csv"

MODEL_PATH = ROOT / "detection_engine/models/attack_family_rf.joblib"
RESULTS_DIR = ROOT / "evaluation_results"

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "protocol",
    "duration",
    "src_packets",
    "dst_packets",
    "src_bytes",
    "dst_bytes",
    "src_pkt_size_mean",
    "dst_pkt_size_mean",
    "src_iat_mean",
    "dst_iat_mean",
    "src_tcp_window",
    "dst_tcp_window",
    "src_packet_rate",
    "dst_packet_rate",
    "src_bytes_rate",
    "dst_bytes_rate",
]

NUMERIC_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column != "protocol"
]

TARGET_COLUMN = "attack_category"

ATTACK_FAMILIES = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

ALL_CLASSES = ["Normal"] + ATTACK_FAMILIES


def load_data():
    print("Loading datasets...", flush=True)

    columns = (
        FEATURE_COLUMNS
        + ["dataset_source", TARGET_COLUMN]
    )

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=columns,
        low_memory=False,
    )

    test = pd.read_csv(
        TEST_PATH,
        usecols=columns,
        low_memory=False,
    )

    train = train.loc[
        train["dataset_source"] == "UNSW-NB15"
    ].copy()

    test = test.loc[
        test["dataset_source"] == "UNSW-NB15"
    ].copy()

    train = train.loc[
        train[TARGET_COLUMN].isin(ALL_CLASSES)
    ].copy()

    test = test.loc[
        test[TARGET_COLUMN].isin(ALL_CLASSES)
    ].copy()

    if train.empty or test.empty:
        raise RuntimeError(
            "Training or evaluation dataset is empty."
        )

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    print("\nTraining category distribution:")
    print(train[TARGET_COLUMN].value_counts().to_string())

    return train, test


def build_pipeline():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                ["protocol"],
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main():
    start = time.time()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, test = load_data()

    x_train = train[FEATURE_COLUMNS].copy()
    y_train = train[TARGET_COLUMN].astype(str)

    x_test = test[FEATURE_COLUMNS].copy()
    y_test = test[TARGET_COLUMN].astype(str)

    for column in NUMERIC_COLUMNS:
        x_train[column] = pd.to_numeric(
            x_train[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        x_test[column] = pd.to_numeric(
            x_test[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

    x_train["protocol"] = (
        x_train["protocol"]
        .fillna("unknown")
        .astype(str)
    )

    x_test["protocol"] = (
        x_test["protocol"]
        .fillna("unknown")
        .astype(str)
    )

    model = build_pipeline()

    print("\nTraining Random Forest...", flush=True)

    model.fit(x_train, y_train)

    print("Training completed.", flush=True)

    print("\nEvaluating model...", flush=True)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    classes = model.named_steps[
        "classifier"
    ].classes_.tolist()

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=ALL_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=ALL_CLASSES,
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=ALL_CLASSES,
            zero_division=0,
        )
    )

    results = {
        "model": "RandomForestClassifier",
        "dataset": "UNSW-NB15",
        "training_rows": int(len(train)),
        "evaluation_rows": int(len(test)),
        "feature_columns": FEATURE_COLUMNS,
        "classes": classes,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "evaluation_note": (
            "Separate multiclass family classifier. "
            "Test-set evaluation is descriptive. "
            "Per-class performance and uncertainty "
            "must be considered before deployment."
        ),
    }

    metrics_path = (
        RESULTS_DIR / "attack_family_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    matrix_frame = pd.DataFrame(
        matrix,
        index=ALL_CLASSES,
        columns=ALL_CLASSES,
    )

    matrix_frame.to_csv(
        RESULTS_DIR
        / "attack_family_confusion_matrix.csv"
    )

    confidence = probabilities.max(axis=1)

    prediction_frame = pd.DataFrame({
        "actual_family": y_test.to_numpy(),
        "predicted_family": predictions,
        "confidence": confidence,
    })

    prediction_frame.to_csv(
        RESULTS_DIR
        / "attack_family_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": classes,
            "training_dataset": "UNSW-NB15",
        },
        MODEL_PATH,
        compress=3,
    )

    elapsed = time.time() - start

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", metrics_path)
    print(f"Elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
