
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detection_engine.inference.predict import (
    load_model,
    predict,
    NUM_FEATURES,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

TEST_PATH = (
    ROOT / "datasets" / "processed" / "test_processed.csv"
)

OUTPUT_DIR = ROOT / "evaluation_results"

BATCH_SIZE = 512
READ_CHUNK_SIZE = 8192

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# METRIC CALCULATION
# ============================================================

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(tp, tn, fp, fn):

    total = tp + tn + fp + fn

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    false_negative_rate = safe_divide(
        fn,
        fn + tp,
    )

    return {
        "total_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 65)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("PHASE 2 - FULL TRANSFORMER EVALUATION")
    print("=" * 65)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    print("\nLoading trained Transformer...")

    model = load_model()

    device = next(
        model.parameters()
    ).device

    print(f"PyTorch version: {torch.__version__}")
    print(f"Execution device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

    print("\nReading test dataset in chunks...")

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    processed_samples = 0

    predictions_output = (
        OUTPUT_DIR / "binary_predictions.csv"
    )

    feature_columns = None

    start_time = time.perf_counter()

    with predictions_output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:

        output_file.write(
            "sample_index,actual,predicted,"
            "confidence,normal_probability,"
            "attack_probability\n"
        )

        reader = pd.read_csv(
            TEST_PATH,
            chunksize=READ_CHUNK_SIZE,
        )

        for chunk_number, chunk in enumerate(
            reader,
            start=1,
        ):

            if "target" not in chunk.columns:
                raise ValueError(
                    "Test dataset is missing target."
                )

            current_features = [
                column
                for column in chunk.columns
                if column != "target"
            ]

            if feature_columns is None:

                feature_columns = current_features

                if len(feature_columns) != NUM_FEATURES:
                    raise ValueError(
                        f"Expected {NUM_FEATURES} features, "
                        f"found {len(feature_columns)}."
                    )

                print(
                    f"Model feature count: "
                    f"{len(feature_columns)}"
                )

            elif current_features != feature_columns:

                raise ValueError(
                    "Feature order changed between chunks."
                )

            actual = pd.to_numeric(
                chunk["target"],
                errors="raise",
            ).to_numpy()

            if not np.isin(
                actual,
                [0, 1],
            ).all():
                raise ValueError(
                    "Test targets must be binary 0/1."
                )

            features = chunk[
                feature_columns
            ].to_numpy(
                dtype=np.float32
            )

            if not np.isfinite(features).all():
                raise ValueError(
                    "Processed test features contain "
                    "non-finite values."
                )

            for start in range(
                0,
                len(features),
                BATCH_SIZE,
            ):

                end = min(
                    start + BATCH_SIZE,
                    len(features),
                )

                batch_features = features[
                    start:end
                ]

                batch_actual = actual[
                    start:end
                ].astype(np.int64)

                predicted, confidence, probabilities = (
                    predict(
                        model,
                        batch_features,
                    )
                )

                predicted = (
                    predicted.numpy().astype(np.int64)
                )

                confidence = (
                    confidence.numpy()
                )

                probabilities = (
                    probabilities.numpy()
                )

                tp += int(
                    np.sum(
                        (batch_actual == 1)
                        & (predicted == 1)
                    )
                )

                tn += int(
                    np.sum(
                        (batch_actual == 0)
                        & (predicted == 0)
                    )
                )

                fp += int(
                    np.sum(
                        (batch_actual == 0)
                        & (predicted == 1)
                    )
                )

                fn += int(
                    np.sum(
                        (batch_actual == 1)
                        & (predicted == 0)
                    )
                )

                batch_result = pd.DataFrame({
                    "sample_index": np.arange(
                        processed_samples,
                        processed_samples + len(predicted),
                    ),
                    "actual": batch_actual,
                    "predicted": predicted,
                    "confidence": confidence,
                    "normal_probability": probabilities[:, 0],
                    "attack_probability": probabilities[:, 1],
                })

                batch_result.to_csv(
                    output_file,
                    index=False,
                    header=False,
                    float_format="%.8f",
                )

                processed_samples += len(predicted)

            print(
                f"Chunk {chunk_number}: "
                f"{processed_samples:,} samples evaluated"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time

    metrics = calculate_metrics(
        tp,
        tn,
        fp,
        fn,
    )

    metrics["execution_device"] = str(device)
    metrics["evaluation_seconds"] = elapsed

    metrics["samples_per_second"] = (
        safe_divide(
            processed_samples,
            elapsed,
        )
    )

    metrics["positive_class"] = "Attack"

    metrics["confusion_matrix"] = [
        [tn, fp],
        [fn, tp],
    ]

    metrics["confusion_matrix_labels"] = [
        "Normal",
        "Attack",
    ]

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("FULL TEST-SET RESULTS")
    print("=" * 65)

    print(
        f"Samples evaluated : "
        f"{metrics['total_samples']:,}"
    )

    print(
        f"Accuracy          : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision (Attack): "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall (Attack)   : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score (Attack) : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print(
        f"Specificity       : "
        f"{metrics['specificity'] * 100:.2f}%"
    )

    print(
        f"False positive rate: "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )

    print(
        f"False negative rate: "
        f"{metrics['false_negative_rate'] * 100:.2f}%"
    )

    print("\nCONFUSION MATRIX")
    print("Rows = actual, columns = predicted")

    print(
        f"{'':14s}"
        f"{'Normal':>12s}"
        f"{'Attack':>12s}"
    )

    print(
        f"{'Actual Normal':14s}"
        f"{tn:12,d}"
        f"{fp:12,d}"
    )

    print(
        f"{'Actual Attack':14s}"
        f"{fn:12,d}"
        f"{tp:12,d}"
    )

    print(
        f"\nEvaluation time: {elapsed:.2f} seconds"
    )

    print(
        f"Throughput: "
        f"{metrics['samples_per_second']:,.1f} "
        f"samples/second"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = (
        OUTPUT_DIR / "evaluation_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    confusion_df = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        index=[
            "Actual Normal",
            "Actual Attack",
        ],
        columns=[
            "Predicted Normal",
            "Predicted Attack",
        ],
    )

    confusion_path = (
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    print("\nResults saved:")

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {confusion_path}"
    )

    print(
        f"  {predictions_output}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
