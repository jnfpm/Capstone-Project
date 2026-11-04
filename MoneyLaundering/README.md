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

## Pipeline (run from this folder, in order)

| Step | Script | Output | Check |
|---|---|---|---|
| 1 | `01_join_sql.py` | `data/processed/hi_small_joined.parquet` | `01b_verify_join.py` |
| 3 | `03_features.py` | `data/processed/features/` (train/val/test features + meta, FX table, hubs, data dictionary) | `03b_verify_features.py` (expects `N / N checks passed`) |
| 4 | `04_modelling.ipynb` | `models/txn_scorer.joblib`, `models/test_scores.parquet`, `models/metrics.json` | runs 03 + 03b itself if the feature files are missing |

```bash
pip install -r requirements.txt
python 01_join_sql.py
python 03_features.py            # ~1-3 min on the full file (DuckDB)
python 03b_verify_features.py    # brute-force recompute of 1,000 random rows
jupyter notebook 04_modelling.ipynb
```

**Split:** train 1-6 Sept, validation 7 Sept, test 8-10 Sept. Rows from 11 Sept on are dropped
(generator residue). Rows of a laundering attempt go to the split of the attempt's last row, so
no scheme is cut across splits.

**Features** (`src/features.py`, full list and reasons in `data/processed/features/data_dictionary.md`):
point-in-time only - a transfer at minute t sees transfers in strictly earlier minutes. Account
history uses a fixed 48 h look-back (counts "since 1 Sept" grow with the calendar and drift
between splits); the first 48 h are excluded from model fitting. Self-transfers are not scored
and not counted. Hub list and FX table are learned on train dates only.

**Models** (`src/modeling.py`, `src/evaluation.py`): single-feature rule baseline, logistic
regression, LightGBM; each with all features / without payment format / without payment format
and `conversion_before` (generator mechanics). Model and alert threshold are chosen on
validation, test is scored once. Metrics: PR-AUC next to prevalence, precision/recall at fixed
alert budgets, recall for explained vs unexplained positives, share of schemes with at least
one alert, 95% intervals by resampling sender accounts.

**For the agent layer:**
```python
import sys; sys.path.insert(0, "src")
import modeling as M
bundle = M.load_bundle("models/txn_scorer.joblib")
M.score_transactions(bundle, X)   # X = rows of a *_features.parquet -> score, alert, top reasons
```

**Using the 02 sample instead of the full file:** `python 03_features.py --sample-dir <folder with
train/val/test parquet from 02_subsample_split.py time>` writes `data/processed/features_sample/`
with the sample's split and weights; set `FEATURES_DIR` in the notebook to that folder.
