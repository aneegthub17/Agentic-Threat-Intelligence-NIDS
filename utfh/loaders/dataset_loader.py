from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CICIDS2018
CICIDS_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "CICIDS2018"
    / "cic.csv"
)

# UNSW-NB15
# NOTE: Actual filenames use "_" in UNSW_NB15
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
# DATASET LOADING FUNCTIONS
# ============================================================

def load_cicids2018():
    """Load the CICIDS2018 dataset."""

    print("\nLoading CICIDS2018...")

    if not CICIDS_PATH.exists():
        raise FileNotFoundError(
            f"CICIDS2018 dataset not found:\n{CICIDS_PATH}"
        )

    df = pd.read_csv(CICIDS_PATH)

    print("CICIDS2018 loaded successfully.")

    return df


def load_unsw_nb15():
    """Load the UNSW-NB15 training and testing datasets."""

    print("\nLoading UNSW-NB15...")

    if not UNSW_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 training dataset not found:\n{UNSW_TRAIN_PATH}"
        )

    if not UNSW_TEST_PATH.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 testing dataset not found:\n{UNSW_TEST_PATH}"
        )

    train_df = pd.read_csv(UNSW_TRAIN_PATH)
    test_df = pd.read_csv(UNSW_TEST_PATH)

    print("UNSW-NB15 training set loaded successfully.")
    print("UNSW-NB15 testing set loaded successfully.")

    return train_df, test_df


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(df, dataset_name):
    """Display basic information about a dataset."""

    print("\n" + "=" * 60)
    print(f"{dataset_name} DATASET INSPECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset dimensions
    # --------------------------------------------------------

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    print("\nColumn names:")

    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("  No missing values detected.")
    else:
        print(missing_values)

    # --------------------------------------------------------
    # Label information
    # --------------------------------------------------------

    print("\nLabel information:")

    if "Label" in df.columns:

        print(df["Label"].value_counts(dropna=False))

    elif "label" in df.columns:

        print(df["label"].value_counts(dropna=False))

    else:

        print("  No standard 'Label' column found.")

    print("=" * 60)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("DATASET LOADER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load CICIDS2018
    # --------------------------------------------------------

    cicids_df = load_cicids2018()

    # --------------------------------------------------------
    # Load UNSW-NB15
    # --------------------------------------------------------

    unsw_train_df, unsw_test_df = load_unsw_nb15()

    # --------------------------------------------------------
    # Inspect datasets
    # --------------------------------------------------------

    inspect_dataset(
        cicids_df,
        "CICIDS2018"
    )

    inspect_dataset(
        unsw_train_df,
        "UNSW-NB15 TRAINING"
    )

    inspect_dataset(
        unsw_test_df,
        "UNSW-NB15 TESTING"
    )

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print("\nDataset inspection completed successfully.")