
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from security_orchestrator.agents.threat_intelligence_agent import (
    ThreatIntelligenceAgent,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    ROOT / "evaluation_results" / "binary_predictions.csv"
)

RAW_PATH = (
    ROOT / "datasets" / "merged" / "unified_test.csv"
)

OUTPUT_DIR = ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 8192

RAW_FIELDS = [
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

METADATA_FIELDS = [
    "target",
    "attack_category",
]

OUTPUT_COLUMNS = [
    "sample_index",
    "actual_binary",
    "predicted_binary",
    "actual_attack_category",
    "heuristic_family",
    "possible_dos",
    "actual_dos",
    "pattern_count",
    "pattern_names",
]


# ============================================================
# HELPERS
# ============================================================

def safe_divide(a, b):
    return a / b if b else 0.0


def clean_numeric(value):
    if pd.isna(value):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def extract_flow(row):
    """
    Extract only traffic observations.

    No ground-truth labels enter the investigation agent.
    """

    features = {}

    for name in RAW_FIELDS:

        value = row[name]

        if name == "protocol":
            features[name] = (
                str(value)
                if pd.notna(value)
                else "unknown"
            )
        else:
            features[name] = clean_numeric(value)

    return features


def calculate_binary_metrics(tp, tn, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": safe_divide(
            tp + tn,
            tp + tn + fp + fn,
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": safe_divide(
            2 * precision * recall,
            precision + recall,
        ),
        "false_positive_rate": safe_divide(
            fp,
            fp + tn,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("PHASE 2 - ATTACK FAMILY HEURISTIC VALIDATION")
    print("=" * 70)

    for path in (PREDICTIONS_PATH, RAW_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    agent = ThreatIntelligenceAgent(retriever=None)

    raw_reader = pd.read_csv(
        RAW_PATH,
        usecols=RAW_FIELDS + METADATA_FIELDS,
        chunksize=CHUNK_SIZE,
    )

    prediction_reader = pd.read_csv(
        PREDICTIONS_PATH,
        chunksize=CHUNK_SIZE,
    )

    output_path = (
        OUTPUT_DIR / "attack_family_predictions.csv"
    )

    total = 0
    actual_dos_count = 0
    predicted_attack_count = 0
    possible_dos_count = 0
    missing_measurements = 0

    tp = tn = fp = fn = 0

    # Counts among records that the binary model
    # already predicted as Attack.
    gated_tp = gated_tn = gated_fp = gated_fn = 0

    family_counts = {}
    pattern_counts = {}
    actual_category_counts = {}

    start_time = time.perf_counter()

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:

        output_file.write(
            ",".join(OUTPUT_COLUMNS) + "\n"
        )

        from itertools import zip_longest

        for chunk_number, pair in enumerate(
            zip_longest(
                raw_reader,
                prediction_reader,
            ),
            start=1,
        ):

            raw_chunk, prediction_chunk = pair

            if (
                raw_chunk is None
                or prediction_chunk is None
            ):
                raise ValueError(
                    "Raw data and prediction files "
                    "have different numbers of chunks."
                )

            if len(raw_chunk) != len(prediction_chunk):
                raise ValueError(
                    "Raw and prediction chunk lengths differ."
                )

            actual = pd.to_numeric(
                raw_chunk["target"],
                errors="raise",
            ).to_numpy()

            saved_actual = pd.to_numeric(
                prediction_chunk["actual"],
                errors="raise",
            ).to_numpy()

            predicted = pd.to_numeric(
                prediction_chunk["predicted"],
                errors="raise",
            ).to_numpy()

            expected_indices = np.arange(
                total,
                total + len(raw_chunk),
            )

            saved_indices = pd.to_numeric(
                prediction_chunk["sample_index"],
                errors="raise",
            ).to_numpy()

            if not np.array_equal(
                actual,
                saved_actual,
            ):
                raise ValueError(
                    f"Target mismatch in chunk {chunk_number}."
                )

            if not np.array_equal(
                saved_indices,
                expected_indices,
            ):
                raise ValueError(
                    f"Sample index mismatch in chunk {chunk_number}."
                )

            if not np.isin(actual, [0, 1]).all():
                raise ValueError(
                    "Ground-truth targets must be binary."
                )

            if not np.isin(predicted, [0, 1]).all():
                raise ValueError(
                    "Model predictions must be binary."
                )

            records = []

            for i in range(len(raw_chunk)):

                row = raw_chunk.iloc[i]

                category = str(
                    row["attack_category"]
                ).strip()

                actual_dos = (
                    category.casefold() == "dos"
                )

                model_attack = (
                    int(predicted[i]) == 1
                )

                # The agent sees only original flow fields.
                flow = extract_flow(row)

                missing_measurements += sum(
                    value is None
                    for key, value in flow.items()
                    if key != "protocol"
                )

                detection_record = {
                    "status": "valid",
                    "prediction": (
                        "Attack"
                        if model_attack
                        else "Normal"
                    ),
                }

                result = agent.process(
                    detection_record,
                    traffic_features=flow,
                )

                family_info = result[
                    "attack_family_analysis"
                ]

                family = family_info.get(
                    "family",
                    "Undetermined",
                )

                patterns = result.get(
                    "traffic_patterns",
                    [],
                )

                pattern_names = [
                    item["pattern"]
                    for item in patterns
                ]

                # Evaluate the complete binary-model +
                # heuristic pipeline.
                possible_dos = (
                    family == "Possible DoS"
                )

                actual_dos_count += int(actual_dos)
                predicted_attack_count += int(model_attack)
                possible_dos_count += int(possible_dos)

                if possible_dos and actual_dos:
                    tp += 1
                elif possible_dos and not actual_dos:
                    fp += 1
                elif not possible_dos and actual_dos:
                    fn += 1
                else:
                    tn += 1

                # Evaluate the same rule within the
                # model-predicted Attack subset.
                if model_attack:
                    if possible_dos and actual_dos:
                        gated_tp += 1
                    elif possible_dos and not actual_dos:
                        gated_fp += 1
                    elif not possible_dos and actual_dos:
                        gated_fn += 1
                    else:
                        gated_tn += 1

                family_counts[family] = (
                    family_counts.get(family, 0) + 1
                )

                actual_category_counts[category] = (
                    actual_category_counts.get(category, 0)
                    + 1
                )

                for name in pattern_names:
                    pattern_counts[name] = (
                        pattern_counts.get(name, 0) + 1
                    )

                records.append({
                    "sample_index": total + i,
                    "actual_binary": int(actual[i]),
                    "predicted_binary": int(predicted[i]),
                    "actual_attack_category": category,
                    "heuristic_family": family,
                    "possible_dos": int(possible_dos),
                    "actual_dos": int(actual_dos),
                    "pattern_count": len(pattern_names),
                    "pattern_names": "|".join(
                        pattern_names
                    ),
                })

            pd.DataFrame(
                records,
                columns=OUTPUT_COLUMNS,
            ).to_csv(
                output_file,
                index=False,
                header=False,
            )

            total += len(raw_chunk)

            print(
                f"Chunk {chunk_number}: "
                f"{total:,} records evaluated"
            )

    elapsed = time.perf_counter() - start_time

    metrics = calculate_binary_metrics(
        tp, tn, fp, fn
    )

    gated_metrics = calculate_binary_metrics(
        gated_tp,
        gated_tn,
        gated_fp,
        gated_fn,
    )

    report = {
        "total_samples": total,
        "actual_dos_samples": actual_dos_count,
        "model_predicted_attack_samples": (
            predicted_attack_count
        ),
        "possible_dos_hypotheses": possible_dos_count,
        "all_samples_pipeline_metrics": metrics,
        "predicted_attack_subset_metrics": (
            gated_metrics
        ),
        "family_counts": family_counts,
        "traffic_pattern_counts": pattern_counts,
        "actual_category_counts": (
            actual_category_counts
        ),
        "missing_numeric_measurements": (
            missing_measurements
        ),
        "evaluation_seconds": elapsed,
        "methodology": (
            "Binary Transformer prediction followed "
            "by unvalidated numerical flow heuristics. "
            "Ground-truth attack_category was used "
            "only after predictions for evaluation."
        ),
        "limitations": [
            "Single-flow heuristics do not confirm DoS.",
            "Screening thresholds are not calibrated.",
            "Duration units require verification.",
            "Binary model false negatives prevent "
            "downstream family hypotheses.",
            "Row alignment relies on preprocessing "
            "preserving the original sample order.",
        ],
    }

    report_path = (
        OUTPUT_DIR / "attack_family_metrics.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("POSSIBLE DoS HEURISTIC RESULTS")
    print("=" * 70)

    print(f"Total records        : {total:,}")
    print(f"Actual DoS records   : {actual_dos_count:,}")
    print(f"Possible DoS flagged : {possible_dos_count:,}")

    print("\nFULL PIPELINE METRICS")
    print(
        f"Precision: {metrics['precision'] * 100:.2f}%"
    )
    print(
        f"Recall   : {metrics['recall'] * 100:.2f}%"
    )
    print(
        f"F1-score : {metrics['f1_score'] * 100:.2f}%"
    )
    print(
        f"FPR      : "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )

    print("\nCONFUSION MATRIX")
    print("Rows: Actual Not-DoS / DoS")
    print("Cols: Predicted Not-DoS / Possible DoS")

    print(f"Not-DoS : {tn:,}  {fp:,}")
    print(f"DoS     : {fn:,}  {tp:,}")

    print("\nFAMILY HYPOTHESES")

    for family, count in sorted(
        family_counts.items(),
        key=lambda item: -item[1],
    ):
        print(f"{family}: {count:,}")

    print("\nTRAFFIC PATTERNS")

    for pattern, count in sorted(
        pattern_counts.items(),
        key=lambda item: -item[1],
    ):
        print(f"{pattern}: {count:,}")

    print(f"\nElapsed: {elapsed:.2f} seconds")
    print(f"\nSaved: {report_path}")
    print(f"Saved: {output_path}")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
