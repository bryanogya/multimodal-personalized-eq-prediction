import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from configs.paths import AUDIO_METADATA_FILE, SPLITS_DIR
from configs.training import BATCH_SIZE, NUM_WORKERS, RANDOM_SEED
from src.dataset.dataset import AudioEQDataset


def split_metadata_by_group(df):
    """
    Split metadata into train, validation, and test sets by original audio file.

    Args:
        df: Metadata DataFrame containing the original_file column.

    Returns:
        Train, validation, and test DataFrames with no original_file overlap.
    """
    if "original_file" not in df.columns:
        raise ValueError("Column 'original_file' not found in audio metadata.")

    group_split_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_SEED
    )

    train_idx, temp_idx = next(
        group_split_1.split(
            df,
            groups=df["original_file"]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    group_split_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    val_idx, test_idx = next(
        group_split_2.split(
            temp_df,
            groups=temp_df["original_file"]
        )
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def check_group_leakage(train_df, val_df, test_df):
    """
    Check whether the same original audio file appears in multiple splits.

    Args:
        train_df: Training metadata DataFrame.
        val_df: Validation metadata DataFrame.
        test_df: Test metadata DataFrame.

    Raises:
        ValueError: If an original_file appears in more than one split.
    """
    train_files = set(train_df["original_file"])
    val_files = set(val_df["original_file"])
    test_files = set(test_df["original_file"])

    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    print("\nGroup Leakage Check")
    print(f"Train-Val overlap : {len(train_val_overlap)}")
    print(f"Train-Test overlap: {len(train_test_overlap)}")
    print(f"Val-Test overlap  : {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Data leakage detected. Same original_file appears in multiple splits.")


def save_split_metadata(
    train_df,
    val_df,
    test_df,
):
    """
    Save train, validation, and test metadata splits into CSV files.

    Args:
        train_df: Training metadata DataFrame.
        val_df: Validation metadata DataFrame.
        test_df: Test metadata DataFrame.
    """
    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        SPLITS_DIR / "train_metadata.csv",
        index=False
    )

    val_df.to_csv(
        SPLITS_DIR / "val_metadata.csv",
        index=False
    )

    test_df.to_csv(
        SPLITS_DIR / "test_metadata.csv",
        index=False
    )

    # print("\nSplit metadata saved:")
    # print(f"Train: {SPLITS_DIR / 'train_metadata.csv'}")
    # print(f"Val  : {SPLITS_DIR / 'val_metadata.csv'}")
    # print(f"Test : {SPLITS_DIR / 'test_metadata.csv'}")
    

def create_dataloaders():
    """
    Create train, validation, and test DataLoaders from audio metadata.

    Returns:
        Tuple containing train_loader, val_loader, and test_loader.
    """
    df = pd.read_csv(AUDIO_METADATA_FILE)

    train_df, val_df, test_df = split_metadata_by_group(df)

    check_group_leakage(
        train_df,
        val_df,
        test_df
    )
    
    save_split_metadata(
        train_df,
        val_df,
        test_df
    )

    train_dataset = AudioEQDataset(metadata_df=train_df)
    val_dataset = AudioEQDataset(metadata_df=val_df)
    test_dataset = AudioEQDataset(metadata_df=test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("\nDataLoader Ready")
    # print(f"Train samples: {len(train_dataset)}")
    # print(f"Val samples  : {len(val_dataset)}")
    # print(f"Test samples : {len(test_dataset)}")
    # print(f"Train files  : {train_df['original_file'].nunique()}")
    # print(f"Val files    : {val_df['original_file'].nunique()}")
    # print(f"Test files   : {test_df['original_file'].nunique()}")

    return (
        train_loader,
        val_loader,
        test_loader
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    batch = next(iter(train_loader))

    # print("\nDataloader Test")
    # print("audio        :", batch["audio"].shape)
    # print("device       :", batch["device"].shape)
    # print("preference   :", batch["preference"].shape)
    # print("target_eq    :", batch["target_eq"].shape)
    # print("target_curve :", batch["target_curve"].shape)