from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "merged"
    / "unified_train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "merged"
    / "unified_test.csv"
)

PROCESSED_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
)

SCALER_PATH = (
    PROCESSED_PATH
    / "standard_scaler.joblib"
)


# ============================================================
# MODEL FEATURES
# ============================================================

# These are the network-flow features that will actually
# enter the Transformer.
#
# Metadata such as dataset_source and attack_category
# are deliberately excluded from model input.

NUMERIC_FEATURES = [
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

PROTOCOL_COLUMN = "protocol"

TARGET_COLUMN = "target"


# ============================================================
# PROTOCOL NORMALIZATION
# ============================================================

def normalize_protocol(series):
    """
    Normalize protocol representations.

    CICIDS2018 commonly represents:
        6  -> TCP
        17 -> UDP

    UNSW-NB15 uses textual protocol names.
    """

    protocol = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    protocol_mapping = {
        "6": "tcp",
        "17": "udp",
        "1": "icmp",
    }

    protocol = protocol.replace(
        protocol_mapping
    )

    return protocol


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():

    print("\nLoading unified training dataset...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print("\nLoading unified test dataset...")

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

    return train_df, test_df


# ============================================================
# VALIDATE INPUT SCHEMA
# ============================================================

def validate_input_schema(df, dataset_name):

    required_columns = (
        NUMERIC_FEATURES
        + [
            PROTOCOL_COLUMN,
            TARGET_COLUMN,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    print(
        f"✓ {dataset_name} schema validated."
    )


# ============================================================
# HANDLE INVALID NUMERIC VALUES
# ============================================================

def clean_numeric_values(df, dataset_name):

    print(
        f"\nCleaning numeric values: "
        f"{dataset_name}"
    )

    # --------------------------------------------------------
    # Convert numeric columns explicitly
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace infinite values with NaN
    # --------------------------------------------------------

    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    missing_before = (
        df[NUMERIC_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing numeric values before "
        f"imputation: {missing_before:,}"
    )

    return df


# ============================================================
# IMPUTATION
# ============================================================

def calculate_training_medians(train_df):

    medians = (
        train_df[NUMERIC_FEATURES]
        .median()
    )

    return medians


def apply_medians(
    df,
    medians,
    dataset_name
):

    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES]
        .fillna(medians)
    )

    remaining = (
        df[NUMERIC_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"{dataset_name} missing numeric values "
        f"after imputation: {remaining:,}"
    )

    return df


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(df):

    # --------------------------------------------------------
    # Normalize protocol
    # --------------------------------------------------------

    df[PROTOCOL_COLUMN] = normalize_protocol(
        df[PROTOCOL_COLUMN]
    )

    # --------------------------------------------------------
    # One-hot encode protocol
    #
    # get_dummies() creates a numerical representation
    # without imposing an artificial ordering.
    # --------------------------------------------------------

    protocol_encoded = pd.get_dummies(
        df[PROTOCOL_COLUMN],
        prefix="protocol",
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Extract numeric features
    # --------------------------------------------------------

    numeric_features = (
        df[NUMERIC_FEATURES]
        .copy()
    )

    # --------------------------------------------------------
    # Combine numeric + encoded protocol
    # --------------------------------------------------------

    features = pd.concat(
        [
            numeric_features,
            protocol_encoded
        ],
        axis=1
    )

    return features


# ============================================================
# ALIGN TRAIN AND TEST FEATURES
# ============================================================

def align_feature_columns(
    train_features,
    test_features
):

    train_features, test_features = (
        train_features.align(
            test_features,
            join="left",
            axis=1,
            fill_value=0
        )
    )

    # Make sure both have exactly the same order.
    test_features = test_features[
        train_features.columns
    ]

    return (
        train_features,
        test_features
    )


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    train_features,
    test_features
):

    print("\nFitting StandardScaler on training data...")

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_features
    )

    print(
        "Transforming test data using "
        "training-fitted scaler..."
    )

    test_scaled = scaler.transform(
        test_features
    )

    return (
        train_scaled,
        test_scaled,
        scaler
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    train_scaled,
    test_scaled,
    train_targets,
    test_targets,
    feature_names
):

    PROCESSED_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Convert arrays to DataFrames
    # --------------------------------------------------------

    train_output = pd.DataFrame(
        train_scaled,
        columns=feature_names
    )

    test_output = pd.DataFrame(
        test_scaled,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Add target
    # --------------------------------------------------------

    train_output[TARGET_COLUMN] = (
        train_targets.values
    )

    test_output[TARGET_COLUMN] = (
        test_targets.values
    )

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    train_output_path = (
        PROCESSED_PATH
        / "train_processed.csv"
    )

    test_output_path = (
        PROCESSED_PATH
        / "test_processed.csv"
    )

    train_output.to_csv(
        train_output_path,
        index=False
    )

    test_output.to_csv(
        test_output_path,
        index=False
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_path = SCALER_PATH

    print(
        f"\nSaving scaler:\n{scaler_path}"
    )

    # The scaler is passed separately in main.
    return (
        train_output_path,
        test_output_path
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("PREPROCESSING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    train_df, test_df = load_datasets()

    # --------------------------------------------------------
    # 2. Validate schemas
    # --------------------------------------------------------

    validate_input_schema(
        train_df,
        "Training dataset"
    )

    validate_input_schema(
        test_df,
        "Test dataset"
    )

    # --------------------------------------------------------
    # 3. Clean numeric values
    # --------------------------------------------------------

    train_df = clean_numeric_values(
        train_df,
        "Training dataset"
    )

    test_df = clean_numeric_values(
        test_df,
        "Test dataset"
    )

    # --------------------------------------------------------
    # 4. Calculate medians ONLY from training data
    # --------------------------------------------------------

    print(
        "\nCalculating training-set medians..."
    )

    training_medians = (
        calculate_training_medians(
            train_df
        )
    )

    # --------------------------------------------------------
    # 5. Apply training medians
    # --------------------------------------------------------

    train_df = apply_medians(
        train_df,
        training_medians,
        "Training dataset"
    )

    test_df = apply_medians(
        test_df,
        training_medians,
        "Test dataset"
    )

    # --------------------------------------------------------
    # 6. Prepare model features
    # --------------------------------------------------------

    print(
        "\nPreparing model features..."
    )

    train_features = prepare_features(
        train_df
    )

    test_features = prepare_features(
        test_df
    )

    # --------------------------------------------------------
    # 7. Align protocol columns
    # --------------------------------------------------------

    (
        train_features,
        test_features
    ) = align_feature_columns(
        train_features,
        test_features
    )

    print(
        f"Model feature count: "
        f"{len(train_features.columns)}"
    )

    # --------------------------------------------------------
    # 8. Scale
    # --------------------------------------------------------

    (
        train_scaled,
        test_scaled,
        scaler
    ) = scale_features(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # 9. Save scaler
    # --------------------------------------------------------

    PROCESSED_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # --------------------------------------------------------
    # 10. Save processed data
    # --------------------------------------------------------

    (
        train_output_path,
        test_output_path
    ) = save_processed_data(
        train_scaled,
        test_scaled,
        train_df[TARGET_COLUMN],
        test_df[TARGET_COLUMN],
        train_features.columns
    )

    # --------------------------------------------------------
    # 11. Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nTraining shape: "
        f"{train_scaled.shape}"
    )

    print(
        f"Testing shape: "
        f"{test_scaled.shape}"
    )

    print(
        f"\nProcessed training file:\n"
        f"{train_output_path}"
    )

    print(
        f"\nProcessed testing file:\n"
        f"{test_output_path}"
    )

    print(
        f"\nSaved scaler:\n"
        f"{SCALER_PATH}"
    )

    print(
        "\nTransformer input preparation is next."
    )