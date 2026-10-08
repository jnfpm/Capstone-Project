# Capstone-Project

## File's description 

HI-Small_Trans.csv Transactions

HI-Small_Patterns.txt Laundering Pattern Transactions

HI-Medium_Trans.csv Transactions

HI-Medium_Patterns.txt Laundering Pattern Transactions

HI-Large_Trans.csv Transactions

HI-Large_Patterns.txt Laundering Pattern Transactions

LI-Small_Trans.csv Transactions

LI-Small_Patterns.txt Laundering Pattern Transactions

LI-Medium_Trans.csv Transactions

LI-Medium_Patterns.txt Laundering Pattern Transactions

## Downloading the dataset

`EDA.ipynb` checks for its input files and downloads the dataset into `data/` from Kaggle if any are missing. Install the notebook's dependencies in its Python environment:

```bash
pip install -r requirements.txt
```

Sign in to Kaggle, open the dataset page, and make sure you have access to it. Configure Kaggle credentials using the instructions at [Kaggle API credentials](https://www.kaggle.com/docs/api); KaggleHub can use the legacy `kaggle.json` credential file in `%USERPROFILE%\.kaggle\kaggle.json`. Then run the notebook from the repository root. The transaction CSVs stay on your machine; they are not required to be committed.
