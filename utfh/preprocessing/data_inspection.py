from pathlib import Path
import pandas as pd
import numpy as np


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


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(df, dataset_name):

    print("\n")
    print("=" * 70)
    print(f"{dataset_name} INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\nData types:")

    print(df.dtypes)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:

        print("No missing values detected.")

    else:

        print(missing)

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    print("\nInfinite values:")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite_counts = np.isinf(
        numeric_df
    ).sum()

    infinite_counts = infinite_counts[
        infinite_counts > 0
    ]

    if infinite_counts.empty:

        print("No infinite values detected.")

    else:

        print(infinite_counts)

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    print("\nDuplicate rows:")

    duplicate_count = df.duplicated().sum()

    print(
        f"{duplicate_count:,}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution:")

    print(
        df["target"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------------
    # Dataset source
    # --------------------------------------------------------

    print("\nDataset source distribution:")

    print(
        df["dataset_source"]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    print("\nCategorical columns:")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:

        for column in categorical_columns:

            print(
                f"\n{column}:"
            )

            print(
                df[column]
                .value_counts(dropna=False)
                .head(20)
            )

    else:

        print("No categorical columns found.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Total missing values : "
        f"{df.isnull().sum().sum():,}"
    )

    print(
        f"Total duplicate rows : "
        f"{duplicate_count:,}"
    )

    print(
        f"Total infinite values: "
        f"{infinite_counts.sum():,}"
    )

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("PREPROCESSING INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    print("\nLoading unified training dataset...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    print(
        f"Loaded {len(train_df):,} training records."
    )

    # --------------------------------------------------------
    # Load testing dataset
    # --------------------------------------------------------

    print("\nLoading unified test dataset...")

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Loaded {len(test_df):,} testing records."
    )

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    inspect_dataset(
        train_df,
        "UNIFIED TRAINING DATASET"
    )

    inspect_dataset(
        test_df,
        "UNIFIED TEST DATASET"
    )

    print("\n")
    print("=" * 70)
    print("PREPROCESSING INSPECTION COMPLETED")
    print("=" * 70)