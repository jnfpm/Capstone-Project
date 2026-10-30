"""Preprocessing and Feature Selection Pipeline.

1. Fetches dataset from Kaggle via kagglehub.
2. Loads transactions, accounts, and patterns.
3. Links accounts and patterns via eda_utils.
4. Calls the preprocessing function (utils.preprocess) to produce X, y, meta.
5. Performs a chronological day-based train/validation/test split.
6. Saves processed datasets to data/processed/.
"""

import sys
from pathlib import Path
import pandas as pd

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


def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    train_days: list = None,
    val_days: list = None,
    test_days: list = None,
):
    """Splits features into train, validation, and test sets separated by days, and saves to disk.

    By default, days are split chronologically based on `meta['Timestamp']`:
      - Train: Earliest days (~(1 - val_size - test_size) of total days)
      - Validation: Intermediate days (~val_size of total days)
      - Test: Latest days (~test_size of total days)

    Alternatively, explicit lists of days/dates can be passed via `train_days`,
    `val_days`, and `test_days`.
    """
    if "Timestamp" not in meta.columns:
        raise ValueError("Column 'Timestamp' not found in meta DataFrame.")

    # 1. Extract dates
    dates_series = pd.to_datetime(meta["Timestamp"]).dt.date
    unique_dates = sorted(dates_series.unique())
    n_days = len(unique_dates)

    if n_days < 3:
        raise ValueError(
            f"At least 3 distinct days are required for train/val/test split by days, but found {n_days}."
        )

    # 2. Determine day partitions
    if train_days is None and val_days is None and test_days is None:
        n_test = max(1, int(round(n_days * test_size)))
        n_val = max(1, int(round(n_days * val_size)))
        n_train = n_days - n_val - n_test

        if n_train < 1:
            raise ValueError(
                f"Cannot split {n_days} days into train/val/test with val_size={val_size} and test_size={test_size}."
            )

        train_days = unique_dates[:n_train]
        val_days = unique_dates[n_train : n_train + n_val]
        test_days = unique_dates[n_train + n_val :]
    else:
        # Convert provided lists into datetime.date sets
        def _to_date_set(days_list, name):
            if days_list is None:
                raise ValueError(f"When passing explicit days, '{name}' must be provided.")
            return set(pd.to_datetime(list(days_list)).date)

        train_days_set = _to_date_set(train_days, "train_days")
        val_days_set = _to_date_set(val_days, "val_days")
        test_days_set = _to_date_set(test_days, "test_days")

        # Check for overlaps
        if (train_days_set & val_days_set) or (train_days_set & test_days_set) or (val_days_set & test_days_set):
            raise ValueError("train_days, val_days, and test_days must have non-overlapping days.")

        train_days = sorted(train_days_set)
        val_days = sorted(val_days_set)
        test_days = sorted(test_days_set)

    # 3. Create split masks
    train_mask = dates_series.isin(set(train_days))
    val_mask = dates_series.isin(set(val_days))
    test_mask = dates_series.isin(set(test_days))

    X_train, y_train, meta_train = X.loc[train_mask], y.loc[train_mask], meta.loc[train_mask]
    X_val, y_val, meta_val = X.loc[val_mask], y.loc[val_mask], meta.loc[val_mask]
    X_test, y_test, meta_test = X.loc[test_mask], y.loc[test_mask], meta.loc[test_mask]

    print("\nStep 5: Day-Based Temporal Train / Validation / Test Split...")
    print(f"Total days available: {n_days} ({unique_dates[0]} to {unique_dates[-1]})")
    print(
        f"Train : {len(train_days)} days ({train_days[0]} to {train_days[-1]}) | "
        f"{len(X_train):,} rows ({len(X_train)/len(X):.1%}) | Positives: {int(y_train.sum()):,} ({y_train.mean():.4%})"
    )
    print(
        f"Val   : {len(val_days)} days ({val_days[0]} to {val_days[-1]}) | "
        f"{len(X_val):,} rows ({len(X_val)/len(X):.1%}) | Positives: {int(y_val.sum()):,} ({y_val.mean():.4%})"
    )
    print(
        f"Test  : {len(test_days)} days ({test_days[0]} to {test_days[-1]}) | "
        f"{len(X_test):,} rows ({len(X_test)/len(X):.1%}) | Positives: {int(y_test.sum()):,} ({y_test.mean():.4%})"
    )

    # 4. Save splits
    print("\nStep 6: Saving processed data to data/processed/...")
    out_dir = CURRENT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet (with csv fallback if pyarrow is unavailable)
    try:
        X_train.to_parquet(out_dir / "X_train.parquet")
        X_val.to_parquet(out_dir / "X_val.parquet")
        X_test.to_parquet(out_dir / "X_test.parquet")
        y_train.to_frame().to_parquet(out_dir / "y_train.parquet")
        y_val.to_frame().to_parquet(out_dir / "y_val.parquet")
        y_test.to_frame().to_parquet(out_dir / "y_test.parquet")
        meta_train.to_parquet(out_dir / "meta_train.parquet")
        meta_val.to_parquet(out_dir / "meta_val.parquet")
        meta_test.to_parquet(out_dir / "meta_test.parquet")
        meta.to_parquet(out_dir / "meta.parquet")
        print("Successfully saved Parquet files.")
    except Exception as e:
        print(f"Notice: Parquet export encountered ({e}), saving CSV files instead...")
        X_train.to_csv(out_dir / "X_train.csv", index=False)
        X_val.to_csv(out_dir / "X_val.csv", index=False)
        X_test.to_csv(out_dir / "X_test.csv", index=False)
        y_train.to_csv(out_dir / "y_train.csv", index=False)
        y_val.to_csv(out_dir / "y_val.csv", index=False)
        y_test.to_csv(out_dir / "y_test.csv", index=False)
        meta_train.to_csv(out_dir / "meta_train.csv", index=False)
        meta_val.to_csv(out_dir / "meta_val.csv", index=False)
        meta_test.to_csv(out_dir / "meta_test.csv", index=False)
        meta.to_csv(out_dir / "meta.csv", index=False)
        print("Successfully saved CSV files.")

    print(f"All files saved in: {out_dir}")
    print("=" * 65)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Run full preprocessing pipeline
    X, y, meta = load_and_preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(X, y, meta)
