
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)
from sklearn.base import clone

from train_attack_family import (
    TRAIN_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    ALL_CLASSES,
    RANDOM_STATE,
    build_pipeline,
)


VALIDATION_FRACTION = 0.20
MIN_CONFIDENCE = 0.90
MIN_PRECISION = 0.90
MIN_PREDICTIONS = 30


def prepare_features(frame):
    features = frame[FEATURE_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

    features["protocol"] = (
        features["protocol"]
        .fillna("unknown")
        .astype(str)
    )

    return features


def main():
    started = time.time()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = (
        FEATURE_COLUMNS
        + ["dataset_source", TARGET_COLUMN]
    )

    data = pd.read_csv(
        TRAIN_PATH,
        usecols=columns,
        low_memory=False,
    )

    data = data.loc[
        (data["dataset_source"] == "UNSW-NB15")
        & data[TARGET_COLUMN].isin(ALL_CLASSES)
    ].copy()

    if data.empty:
        raise RuntimeError(
            "No UNSW-NB15 training records found."
        )

    development, validation = train_test_split(
        data,
        test_size=VALIDATION_FRACTION,
        stratify=data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    x_development = prepare_features(development)
    y_development = development[TARGET_COLUMN].astype(str)

    x_validation = prepare_features(validation)
    y_validation = validation[TARGET_COLUMN].astype(str)

    print(
        f"Development records: {len(development):,}",
        flush=True,
    )
    print(
        f"Validation records: {len(validation):,}",
        flush=True,
    )

    validation_model = build_pipeline()

    print(
        "\nTraining validation model...",
        flush=True,
    )

    validation_model.fit(
        x_development,
        y_development,
    )

    predicted = validation_model.predict(
        x_validation
    )

    probabilities = validation_model.predict_proba(
        x_validation
    )

    confidence = probabilities.max(axis=1)

    print(
        "\nValidation accuracy:",
        round(
            accuracy_score(y_validation, predicted),
            4,
        ),
    )

    print("\nValidation classification report:")
    print(
        classification_report(
            y_validation,
            predicted,
            labels=ALL_CLASSES,
            zero_division=0,
        )
    )

    validation_predictions = pd.DataFrame({
        "actual_family": y_validation.to_numpy(),
        "predicted_family": predicted,
        "confidence": confidence,
    })

    validation_predictions["correct"] = (
        validation_predictions["actual_family"]
        == validation_predictions["predicted_family"]
    )

    high_confidence = validation_predictions.loc[
        validation_predictions["confidence"]
        >= MIN_CONFIDENCE
    ].copy()

    family_policy = {}

    print("\nHigh-confidence validation reliability:")
    print(
        f"Minimum confidence: {MIN_CONFIDENCE:.0%}"
    )
    print(
        f"Required precision: {MIN_PRECISION:.0%}"
    )
    print(
        f"Minimum predictions: {MIN_PREDICTIONS}"
    )

    for family in ALL_CLASSES:
        if family == "Normal":
            continue

        subset = high_confidence.loc[
            high_confidence["predicted_family"] == family
        ]

        count = len(subset)

        precision = (
            float(subset["correct"].mean())
            if count > 0
            else 0.0
        )

        eligible = (
            count >= MIN_PREDICTIONS
            and precision >= MIN_PRECISION
        )

        family_policy[family] = {
            "eligible": bool(eligible),
            "validation_predictions": int(count),
            "validation_precision": precision,
        }

        status = (
            "ELIGIBLE"
            if eligible
            else "NOT ELIGIBLE"
        )

        print(
            f"{family:16s} "
            f"count={count:5d} "
            f"precision={precision:7.2%} "
            f"{status}"
        )

    validation_predictions.to_csv(
        RESULTS_DIR
        / "attack_family_validation_predictions.csv",
        index=False,
    )

    policy = {
        "model_type": "RandomForestClassifier",
        "training_source": "UNSW-NB15",
        "validation_fraction": VALIDATION_FRACTION,
        "random_state": RANDOM_STATE,
        "minimum_confidence": MIN_CONFIDENCE,
        "minimum_validation_precision": MIN_PRECISION,
        "minimum_validation_predictions": MIN_PREDICTIONS,
        "families": family_policy,
        "limitations": [
            "Family predictions are hypotheses, not confirmed attacks.",
            "Validation precision is estimated from a single stratified split.",
            "Rare families have limited validation support.",
            "Confidence is not a calibrated probability guarantee.",
            "Row-level validation may contain related network flows.",
        ],
    }

    policy_path = (
        RESULTS_DIR
        / "attack_family_validation_policy.json"
    )

    with open(
        policy_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            policy,
            file,
            indent=2,
        )

    print(
        "\nRetraining final model on all "
        "UNSW-NB15 training records...",
        flush=True,
    )

    final_model = clone(build_pipeline())

    final_model.fit(
        prepare_features(data),
        data[TARGET_COLUMN].astype(str),
    )

    final_classes = final_model.named_steps[
        "classifier"
    ].classes_.tolist()

    artifact = {
        "pipeline": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": final_classes,
        "training_dataset": "UNSW-NB15",
        "validation_policy": policy,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    elapsed = time.time() - started

    print("\nSaved final model:", MODEL_PATH)
    print("Saved validation policy:", policy_path)
    print(
        "Eligible families:",
        [
            family
            for family, details in family_policy.items()
            if details["eligible"]
        ],
    )
    print(
        f"Elapsed time: {elapsed:.1f} seconds"
    )


if __name__ == "__main__":
    main()

