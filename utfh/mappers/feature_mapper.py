from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CICIDS_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "CICIDS2018"
    / "cic.csv"
)

UNSW_TRAIN_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "UNSW-NB15"
    / "UNSW_NB15_training-set.csv"
)

UNSW_TEST_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "UNSW-NB15"
    / "UNSW_NB15_testing-set.csv"
)


# ============================================================
# SEMANTIC FEATURE MAPPING
# ============================================================

# Only features representing the same underlying
# network-flow concept are mapped together.

DIRECT_FEATURE_MAPPING = {

    # Network protocol
    "protocol": {
        "CICIDS2018": "Protocol",
        "UNSW-NB15": "proto"
    },

    # Flow duration
    "duration": {
        "CICIDS2018": "Flow Duration",
        "UNSW-NB15": "dur"
    },

    # Packet counts
    "src_packets": {
        "CICIDS2018": "Tot Fwd Pkts",
        "UNSW-NB15": "spkts"
    },

    "dst_packets": {
        "CICIDS2018": "Tot Bwd Pkts",
        "UNSW-NB15": "dpkts"
    },

    # Byte counts
    "src_bytes": {
        "CICIDS2018": "TotLen Fwd Pkts",
        "UNSW-NB15": "sbytes"
    },

    "dst_bytes": {
        "CICIDS2018": "TotLen Bwd Pkts",
        "UNSW-NB15": "dbytes"
    },

    # Mean packet size
    "src_pkt_size_mean": {
        "CICIDS2018": "Fwd Pkt Len Mean",
        "UNSW-NB15": "smean"
    },

    "dst_pkt_size_mean": {
        "CICIDS2018": "Bwd Pkt Len Mean",
        "UNSW-NB15": "dmean"
    },

    # Inter-arrival time
    "src_iat_mean": {
        "CICIDS2018": "Fwd IAT Mean",
        "UNSW-NB15": "sinpkt"
    },

    "dst_iat_mean": {
        "CICIDS2018": "Bwd IAT Mean",
        "UNSW-NB15": "dinpkt"
    },

    # TCP window
    "src_tcp_window": {
        "CICIDS2018": "Init Fwd Win Byts",
        "UNSW-NB15": "swin"
    },

    "dst_tcp_window": {
        "CICIDS2018": "Init Bwd Win Byts",
        "UNSW-NB15": "dwin"
    }
}


# ============================================================
# DATASET-SPECIFIC FEATURES USED TO DERIVE
# COMMON FEATURES
# ============================================================

DERIVED_FEATURES = [
    "src_packet_rate",
    "dst_packet_rate",
    "src_bytes_rate",
    "dst_bytes_rate"
]


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_columns(df, dataset_name):

    required_columns = []

    for mapping in DIRECT_FEATURE_MAPPING.values():

        required_columns.append(
            mapping[dataset_name]
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(f"\nERROR: Missing columns in {dataset_name}")

        for column in missing_columns:
            print(f"  - {column}")

        return False

    return True


# ============================================================
# NORMALIZE PROTOCOL
# ============================================================

def normalize_protocol(series):
    """
    Convert protocol values to a consistent string
    representation across both datasets.
    """

    return series.astype(str).str.strip().str.lower()


# ============================================================
# CREATE COMMON FEATURE SPACE
# ============================================================

def create_common_features(df, dataset_name):

    print(
        f"\nCreating common feature space for "
        f"{dataset_name}..."
    )

    # --------------------------------------------------------
    # Validate source columns
    # --------------------------------------------------------

    if not validate_columns(df, dataset_name):

        raise ValueError(
            f"Feature validation failed for {dataset_name}"
        )

    common_df = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # Direct semantic mappings
    # --------------------------------------------------------

    for common_name, mapping in DIRECT_FEATURE_MAPPING.items():

        source_column = mapping[dataset_name]

        common_df[common_name] = df[source_column]

    # --------------------------------------------------------
    # Normalize protocol
    # --------------------------------------------------------

    common_df["protocol"] = normalize_protocol(
        common_df["protocol"]
    )

    # --------------------------------------------------------
    # Derived packet rates
    # --------------------------------------------------------

    duration = pd.to_numeric(
        common_df["duration"],
        errors="coerce"
    )

    # CICIDS duration is normally expressed in microseconds,
    # while UNSW duration is expressed in seconds.
    #
    # Convert CICIDS duration to seconds before deriving rates.

    if dataset_name == "CICIDS2018":

        duration_seconds = duration / 1_000_000

    else:

        duration_seconds = duration

    # Avoid division by zero
    duration_seconds = duration_seconds.replace(
        0,
        np.nan
    )

    common_df["src_packet_rate"] = (
        pd.to_numeric(
            common_df["src_packets"],
            errors="coerce"
        )
        / duration_seconds
    )

    common_df["dst_packet_rate"] = (
        pd.to_numeric(
            common_df["dst_packets"],
            errors="coerce"
        )
        / duration_seconds
    )

    # --------------------------------------------------------
    # Derived byte rates
    # --------------------------------------------------------

    # Convert bytes/sec to bits/sec for consistency
    # with UNSW-NB15 Sload/Dload.

    common_df["src_bytes_rate"] = (
        pd.to_numeric(
            common_df["src_bytes"],
            errors="coerce"
        )
        * 8
        / duration_seconds
    )

    common_df["dst_bytes_rate"] = (
        pd.to_numeric(
            common_df["dst_bytes"],
            errors="coerce"
        )
        * 8
        / duration_seconds
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in common_df.columns
        if column != "protocol"
    ]

    for column in numeric_columns:

        common_df[column] = pd.to_numeric(
            common_df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Add dataset source
    # --------------------------------------------------------

    common_df["dataset_source"] = dataset_name

    print(
        f"{dataset_name} common feature space created."
    )

    print(
        f"Rows    : {len(common_df):,}"
    )

    print(
        f"Features: {len(common_df.columns)}"
    )

    return common_df


# ============================================================
# LABEL STANDARDIZATION
# ============================================================

def standardize_cicids_labels(df):

    """
    Convert CICIDS2018 labels into a unified
    binary target:

    0 = Benign
    1 = Attack
    """

    label = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return (
        label
        .ne("benign")
        .astype(int)
    )


def standardize_unsw_labels(df):

    """
    UNSW-NB15 already provides:

    0 = Normal
    1 = Attack
    """

    return pd.to_numeric(
        df["label"],
        errors="coerce"
    ).astype("Int64")


# ============================================================
# ADD STANDARDIZED LABEL
# ============================================================

def add_standardized_label(
    common_df,
    original_df,
    dataset_name
):

    if dataset_name == "CICIDS2018":

        common_df["target"] = (
            standardize_cicids_labels(
                original_df
            )
        )

        # Preserve original attack type
        common_df["attack_category"] = (
            original_df["Label"]
            .astype(str)
            .str.strip()
        )

    elif dataset_name == "UNSW-NB15":

        common_df["target"] = (
            standardize_unsw_labels(
                original_df
            )
        )

        # Preserve UNSW attack category
        common_df["attack_category"] = (
            original_df["attack_cat"]
            .astype(str)
            .str.strip()
        )

    else:

        raise ValueError(
            "Unknown dataset name."
        )

    return common_df


# ============================================================
# VALIDATION REPORT
# ============================================================

def mapping_report(
    cicids_common,
    unsw_train_common,
    unsw_test_common
):

    print("\n")
    print("=" * 70)
    print("COMMON FEATURE SPACE REPORT")
    print("=" * 70)

    print(
        f"CICIDS2018      : "
        f"{cicids_common.shape}"
    )

    print(
        f"UNSW-NB15 Train : "
        f"{unsw_train_common.shape}"
    )

    print(
        f"UNSW-NB15 Test  : "
        f"{unsw_test_common.shape}"
    )

    print("\nCommon feature columns:")

    for column in cicids_common.columns:

        print(f"  - {column}")

    print("\nCICIDS target distribution:")

    print(
        cicids_common["target"]
        .value_counts()
        .sort_index()
    )

    print("\nUNSW-NB15 train target distribution:")

    print(
        unsw_train_common["target"]
        .value_counts()
        .sort_index()
    )

    print("\nUNSW-NB15 test target distribution:")

    print(
        unsw_test_common["target"]
        .value_counts()
        .sort_index()
    )

    print("=" * 70)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("SEMANTIC FEATURE MAPPING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading CICIDS2018...")

    cicids_df = pd.read_csv(
        CICIDS_PATH
    )

    print(
        f"CICIDS2018 loaded: "
        f"{len(cicids_df):,} rows"
    )

    print("\nLoading UNSW-NB15 training...")

    unsw_train_df = pd.read_csv(
        UNSW_TRAIN_PATH
    )

    print(
        f"UNSW-NB15 training loaded: "
        f"{len(unsw_train_df):,} rows"
    )

    print("\nLoading UNSW-NB15 testing...")

    unsw_test_df = pd.read_csv(
        UNSW_TEST_PATH
    )

    print(
        f"UNSW-NB15 testing loaded: "
        f"{len(unsw_test_df):,} rows"
    )

    # --------------------------------------------------------
    # Create common feature spaces
    # --------------------------------------------------------

    cicids_common = create_common_features(
        cicids_df,
        "CICIDS2018"
    )

    unsw_train_common = create_common_features(
        unsw_train_df,
        "UNSW-NB15"
    )

    unsw_test_common = create_common_features(
        unsw_test_df,
        "UNSW-NB15"
    )

    # --------------------------------------------------------
    # Standardize labels
    # --------------------------------------------------------

    cicids_common = add_standardized_label(
        cicids_common,
        cicids_df,
        "CICIDS2018"
    )

    unsw_train_common = add_standardized_label(
        unsw_train_common,
        unsw_train_df,
        "UNSW-NB15"
    )

    unsw_test_common = add_standardized_label(
        unsw_test_common,
        unsw_test_df,
        "UNSW-NB15"
    )

    # --------------------------------------------------------
    # Display mapping report
    # --------------------------------------------------------

    mapping_report(
        cicids_common,
        unsw_train_common,
        unsw_test_common
    )

    print(
        "\nSemantic feature mapping completed successfully."
    )