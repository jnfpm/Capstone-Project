"""Preprocessing and Feature Selection Pipeline.

1. Fetches dataset from Kaggle via kagglehub.
2. Loads transactions, accounts, and patterns.
3. Links accounts and patterns via eda_utils.
4. Calls the preprocessing function (utils.preprocess) to produce X, y, meta.
5. Performs a stratified train/validation split.
6. Saves processed datasets to data/processed/.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure this folder is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import eda_utils as eu
from utils import preprocess, get_kaggle_dataset_dir


def load_and_preprocess(sample_fraction: float = None):
    """Loads transactions, links accounts & patterns, and executes preprocessing.
    
    Args:
        sample_fraction: Optional float (e.g. 0.1 for 10% sample) for fast testing.
        
    Returns:
        X (DataFrame): Engineered features.
        y (Series): Binary target ('Is Laundering').
        meta (DataFrame): Metadata for evaluation and tracking.
    """
    print("=" * 65)
    print("Step 1: Locating dataset via kagglehub...")
    data_dir = get_kaggle_dataset_dir()
    print(f"Dataset path: {data_dir}")

    print("\nStep 2: Loading raw files...")
    trans = eu.load_transactions(data_dir / "HI-Small_Trans.csv")
    accounts = eu.load_accounts(data_dir / "HI-Small_accounts.csv")
    patterns, attempts, malformed = eu.load_patterns(data_dir / "HI-Small_Patterns.txt")
    print(f"Transactions: {len(trans):,} rows")
    print(f"Accounts    : {len(accounts):,} rows")
    print(f"Patterns    : {len(patterns):,} rows across {len(attempts):,} attempts")

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        print(f"\n[Note] Sampling {sample_fraction:.1%} of transactions for quick run...")
        trans = trans.sample(frac=sample_fraction, random_state=42).sort_values("Timestamp").reset_index(drop=True)

    print("\nStep 3: Linking accounts and patterns...")
    trans, _ = eu.link_accounts(trans, accounts)
    trans, _, _ = eu.link_patterns(trans, patterns)
    print(f"Linked DataFrame shape: {trans.shape}")

    print("\nStep 4: Calling preprocess(trans)...")
    X, y, meta = preprocess(trans)
    print(f"Features (X) shape : {X.shape}")
    print(f"Target (y) shape   : {len(y):,} rows ({int(y.sum()):,} positives, {y.mean():.4%} rate)")
    print(f"Metadata shape     : {meta.shape}")

    return X, y, meta


def split_and_save(X, y, meta, test_size: float = 0.20, random_state: int = 42):
    """Splits features into train and validation sets (stratified), and saves to disk."""
    print(f"\nStep 5: Stratified Train / Validation Split ({(1 - test_size):.0%} train / {test_size:.0%} val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"X_train: {X_train.shape[0]:,} rows | Positives: {int(y_train.sum()):,}")
    print(f"X_val  : {X_val.shape[0]:,} rows | Positives: {int(y_val.sum()):,}")

    print("\nStep 6: Saving processed data to data/processed/...")
    out_dir = CURRENT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet (with csv fallback if pyarrow is unavailable)
    try:
        X_train.to_parquet(out_dir / "X_train.parquet")
        X_val.to_parquet(out_dir / "X_val.parquet")
        y_train.to_frame().to_parquet(out_dir / "y_train.parquet")
        y_val.to_frame().to_parquet(out_dir / "y_val.parquet")
        meta.to_parquet(out_dir / "meta.parquet")
        print("Successfully saved Parquet files.")
    except Exception as e:
        print(f"Notice: Parquet export encountered ({e}), saving CSV files instead...")
        X_train.to_csv(out_dir / "X_train.csv", index=False)
        X_val.to_csv(out_dir / "X_val.csv", index=False)
        y_train.to_csv(out_dir / "y_train.csv", index=False)
        y_val.to_csv(out_dir / "y_val.csv", index=False)
        meta.to_csv(out_dir / "meta.csv", index=False)
        print("Successfully saved CSV files.")

    print(f"All files saved in: {out_dir}")
    print("=" * 65)
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Run full preprocessing pipeline
    X, y, meta = load_and_preprocess()
    X_train, X_val, y_train, y_val = split_and_save(X, y, meta)
