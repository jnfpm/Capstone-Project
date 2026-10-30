"""Data-access helpers (Kaggle download). Preprocessing lives in preprocessing_utils.py."""

import warnings
from pathlib import Path

import kagglehub
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


def get_kaggle_dataset_dir(dataset: str = DATASET) -> Path:
    """Download the dataset with kagglehub (cached after the first run) and return its folder."""
    return Path(kagglehub.dataset_download(dataset))


def load_kaggle_data(dataset: str = DATASET):
    """Download from Kaggle and load the raw transactions and accounts CSVs."""
    data_dir = get_kaggle_dataset_dir(dataset)
    trans = pd.read_csv(data_dir / "HI-Small_Trans.csv")
    accounts = pd.read_csv(data_dir / "HI-Small_accounts.csv")
    return trans, accounts
