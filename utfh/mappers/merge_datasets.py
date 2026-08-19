from pathlib import Path
import sys
import pandas as pd


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

MERGED_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "merged"
)

PROCESSED_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
)


# ============================================================
# IMPORT FEATURE MAPPING FUNCTIONS
# ============================================================

MAPPER_PATH = (
    PROJECT_ROOT
    / "ut fh"
    / "mappers"
)

sys.path.append(str(MAPPER_PATH))

from feature_mapper import (
    create_common_features,
    add_standardized_label
)


# ============================================================
# LOAD AND MAP ONE DATASET
# ============================================================

def process_dataset(
    file_path,
    dataset_name,
    chunksize=None
):
    """
    Load a dataset and transform it into the
    unified semantic feature space.
    """

    print("\n" + "=" * 70)
    print(f"PROCESSING {dataset_name}")
    print("=" * 70)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    # --------------------------------------------------------
    # Current implementation loads the complete dataset.
    # Chunking will be introduced if memory becomes a problem.
    # --------------------------------------------------------

    print(f"Loading: {file_path.name}")

    df = pd.read_csv(file_path)

    print(
        f"Loaded {len(df):,} records."
    )

    # --------------------------------------------------------
    # Create semantic feature representation
    # --------------------------------------------------------

    common_df = create_common_features(
        df,
        dataset_name
    )

    # --------------------------------------------------------
    # Standardize target and attack category
    # --------------------------------------------------------

    common_df = add_standardized_label(
        common_df,
        df,
        dataset_name
    )

    print(
        f"Mapped {len(common_df):,} records."
    )

    return common_df


# ============================================================
# VALIDATE UNIFIED SCHEMA
# ============================================================

def validate_schema(
    cicids_df,
    unsw_train_df,
    unsw_test_df
):
    """
    Ensure all datasets have exactly the same
    feature schema before concatenation.
    """

    print("\n" + "=" * 70)
    print("SCHEMA VALIDATION")
    print("=" * 70)

    cicids_columns = list(
        cicids_df.columns
    )

    unsw_train_columns = list(
        unsw_train_df.columns
    )

    unsw_test_columns = list(
        unsw_test_df.columns
    )

    if cicids_columns != unsw_train_columns:

        raise ValueError(
            "CICIDS2018 and UNSW-NB15 training "
            "schemas do not match."
        )

    if cicids_columns != unsw_test_columns:

        raise ValueError(
            "CICIDS2018 and UNSW-NB15 testing "
            "schemas do not match."
        )

    print("✓ CICIDS2018 schema validated.")
    print("✓ UNSW-NB15 training schema validated.")
    print("✓ UNSW-NB15 testing schema validated.")
    print("✓ All datasets share the same schema.")


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    df,
    output_path,
    dataset_name
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nSaving {dataset_name}..."
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved to:\n{output_path}"
    )

    print(
        f"Rows saved: {len(df):,}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(
    name,
    df
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print("\nTarget distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    print("\nDataset source:")

    print(
        df["dataset_source"]
        .value_counts()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AGENTIC THREAT INTELLIGENCE NIDS")
    print("UNIFIED DATASET CONSTRUCTION")
    print("=" * 70)

    # ========================================================
    # 1. PROCESS CICIDS2018
    # ========================================================

    cicids_common = process_dataset(
        CICIDS_PATH,
        "CICIDS2018"
    )

    # ========================================================
    # 2. PROCESS UNSW TRAINING
    # ========================================================

    unsw_train_common = process_dataset(
        UNSW_TRAIN_PATH,
        "UNSW-NB15"
    )

    # ========================================================
    # 3. PROCESS UNSW TESTING
    # ========================================================

    unsw_test_common = process_dataset(
        UNSW_TEST_PATH,
        "UNSW-NB15"
    )

    # ========================================================
    # 4. VALIDATE SCHEMAS
    # ========================================================

    validate_schema(
        cicids_common,
        unsw_train_common,
        unsw_test_common
    )

    # ========================================================
    # 5. CREATE UNIFIED TRAINING POOL
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING UNIFIED TRAINING DATASET")
    print("=" * 70)

    unified_train = pd.concat(
        [
            cicids_common,
            unsw_train_common
        ],
        ignore_index=True
    )

    print(
        f"Unified training records: "
        f"{len(unified_train):,}"
    )

    # ========================================================
    # 6. KEEP UNSW TEST SET INDEPENDENT
    # ========================================================

    unified_test = unsw_test_common.copy()

    print(
        f"Independent test records: "
        f"{len(unified_test):,}"
    )

    # ========================================================
    # 7. SHUFFLE TRAINING DATA ONLY
    # ========================================================

    unified_train = unified_train.sample(
        frac=1.0,
        random_state=42
    ).reset_index(drop=True)

    # ========================================================
    # 8. PRINT SUMMARIES
    # ========================================================

    print_summary(
        "UNIFIED TRAINING DATASET",
        unified_train
    )

    print_summary(
        "INDEPENDENT TEST DATASET",
        unified_test
    )

    # ========================================================
    # 9. SAVE TRAINING DATA
    # ========================================================

    save_dataset(
        unified_train,
        MERGED_PATH / "unified_train.csv",
        "Unified Training Dataset"
    )

    # ========================================================
    # 10. SAVE TEST DATA
    # ========================================================

    save_dataset(
        unified_test,
        MERGED_PATH / "unified_test.csv",
        "Independent Test Dataset"
    )

    # ========================================================
    # 11. COMPLETION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("UNIFIED DATASET CONSTRUCTION COMPLETED")
    print("=" * 70)

    print(
        "\nTraining:"
        "\n  CICIDS2018 + UNSW-NB15 Training"
    )

    print(
        "\nTesting:"
        "\n  UNSW-NB15 Testing (kept independent)"
    )

    print(
        "\nLeakage-safe dataset construction completed."
    )