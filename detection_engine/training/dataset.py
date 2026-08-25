import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class NIDSDataset(Dataset):
    """
    PyTorch Dataset for the preprocessed NIDS data.
    """

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        # Separate features and target
        self.X = self.data.drop(columns=["target"]).values
        self.y = self.data["target"].values

        # Convert to PyTorch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataloaders(
    train_path,
    test_path,
    batch_size=256,
    num_workers=0
):
    """
    Create training and testing DataLoaders.
    """

    train_dataset = NIDSDataset(train_path)
    test_dataset = NIDSDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


if __name__ == "__main__":

    TRAIN_PATH = "datasets/processed/train_processed.csv"
    TEST_PATH = "datasets/processed/test_processed.csv"

    train_loader, test_loader = create_dataloaders(
        TRAIN_PATH,
        TEST_PATH
    )

    print("=" * 70)
    print("NIDS TRANSFORMER DATASET PIPELINE")
    print("=" * 70)

    print(f"Training samples : {len(train_loader.dataset):,}")
    print(f"Testing samples  : {len(test_loader.dataset):,}")
    print(f"Batch size       : {train_loader.batch_size}")

    X_batch, y_batch = next(iter(train_loader))

    print(f"\nBatch feature shape : {X_batch.shape}")
    print(f"Batch target shape  : {y_batch.shape}")

    print(f"\nFeature dtype : {X_batch.dtype}")
    print(f"Target dtype  : {y_batch.dtype}")

    print(f"\nCUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    print("\nDataset/DataLoader test completed successfully.")