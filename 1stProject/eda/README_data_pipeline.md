# Data pipeline — how to build the data (HI-Small)

Run these once, in order. Each script prints its own checks. Compare the last line with the
"must say" column below. If anything differs, stop and tell the group.

## 1. Setup (once)

**Python packages**
```
pip install duckdb pyarrow pandas
```
(For the EDA notebook also: `pip install matplotlib scikit-learn ipykernel`.)

**Folder layout** — everything lives under one project folder (ours is `D:\capstone`):
```
capstone\
  data\raw\        HI-Small_Trans.csv, HI-Small_accounts.csv, HI-Small_Patterns.txt
  data\processed\  (created by the scripts)
  notebook\        01_join_sql.py, 01b_verify_join.py,
                   02_subsample_split.py, 02b_verify_sample.py, EDA_HI_Small_v2.ipynb
  src\             eda_utils.py
```
The three raw files come from Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
(only the three **HI-Small** files are needed).

The project folder can be anywhere and have any name. The scripts find it from their own
location (`notebook\` → its parent), so nothing needs editing.

## 2. Easiest: one file does everything

Open `notebook\EDA_HI_Small_v2.ipynb` and click **Run All**. Its **Step 0** cell runs all the
scripts below in order (only the ones whose files are missing), stops with an error if any check
fails, checks the fingerprint, then runs the EDA. First time: several minutes. After that it
skips straight to the EDA.

Settings at the top of that cell: `BUILD = "auto" | "all" | "none"`, `RUN_CHECKS` (set `False` on a
small laptop to skip `01b`, which loads the 1.3 GB CSV), `MAKE_TIME_SPLIT` (builds train/val/test).

## 3. Or run the scripts yourself

Open a terminal in the project folder and run:

| # | Command | Time | What it makes | Last line must say |
|---|---|---|---|---|
| 1 | `python notebook\01_join_sql.py` | ~1 min | `data\processed\hi_small_joined.parquet` (+ `.csv`) | CHECKS block: rows 5,078,336 · positives 5,177 · pattern rows 3,209 · attempts 370 · unmatched 0 / 0 · missing country 0 |
| 2 | `python notebook\01b_verify_join.py` | 1–2 min | nothing (checks only) | `33 / 33 checks passed` and `The joined CSV is correct.` |
| 3 | `python notebook\02_subsample_split.py` | ~30 s | `data\processed\sample\sample\` → `sample.parquet`, `sample.csv`, `accounts.parquet`, `report.txt` | `SAMPLE FINGERPRINT: 4fb17c9f351dbaea0bf766c55f055f60` |
| 4 | `python notebook\02b_verify_sample.py` | ~10 s | nothing | `27 / 27 checks passed` |
| 5 | `python notebook\02_subsample_split.py time` | ~30 s | `data\processed\sample\time\` → `train`, `val`, `test` (.parquet + .csv), `accounts.parquet`, `report.txt` | same fingerprint as step 3 |
| 6 | `python notebook\02b_verify_sample.py time` | ~10 s | nothing | `36 / 36 checks passed` |
| 7 *(optional)* | `python notebook\02_subsample_split.py group` then `python notebook\02b_verify_sample.py group` | ~40 s | `data\processed\sample\group\` | `35 / 35 checks passed` |

**Same fingerprint = same rows.** Everyone running this gets byte-identical files, on any machine.

Step 2 loads the 1.3 GB CSV into pandas — close other programs if your laptop has little memory.

## 4. What each script does (plain words)

- **01_join_sql.py** — glues the three raw files into one table: adds bank name, owner and country
  for sender and receiver, attaches the scheme answer key (`Attempt ID`, `Pattern Type`), removes 9
  duplicate rows, adds 4 yes/no flags, sorts by time.
- **01b_verify_join.py** — re-checks the join with pandas from the raw files (does not trust the
  join's own report).
- **02_subsample_split.py** — drops the broken 11–18 Sept tail, picks **accounts** (all laundering
  accounts + ~7% of the others) and keeps their **complete** history → ~750k rows with every
  laundering row and every scheme. Adds a `weight` per row. With `time` it also splits into
  train (1–6 Sept) / val (7 Sept) / test (8–10 Sept).
- **02b_verify_sample.py** — re-checks every promise of the sample from the joined file.

## 5. Which file to use for what

| Job | File |
|---|---|
| Modelling rows, label, weight, split | `data\processed\sample\time\train / val / test.parquet` |
| Account features (point-in-time) | `data\processed\hi_small_joined.parquet` — **never** compute account counts from the sample |
| Agent lookups ("all transfers of account X") | `data\processed\hi_small_joined.parquet` via DuckDB |
| Exploring / EDA | `notebook\EDA_HI_Small_v2.ipynb` (open → Run All, ~3 min) |

## 6. Loading the files

**Use parquet** (types are kept):
```python
import pandas as pd
train = pd.read_parquet(r"data\processed\sample\time\train.parquet")   # path from the project folder
```
**If you must use the CSV**, pass these or the time stays text:
```python
ID_COLS = ["Account", "Account.1", "acct_from", "acct_to", "Entity ID_from", "Entity ID_to"]
train = pd.read_csv(r"...\train.csv", dtype={c: "string" for c in ID_COLS}, parse_dates=["Timestamp"])
```

## 7. Rules (please read)

- **Every metric uses the weights:** `precision_score(y, pred, sample_weight=df["weight"])`,
  `average_precision_score(y, score, sample_weight=df["weight"])`. Without them precision looks
  ~7x better than it really is. Never report accuracy.
- **Never put these in the model's inputs:** `Attempt ID`, `Pattern Type`, `scheme_truncated`,
  `scheme_moved`, `seed_from`, `seed_to`, `incl_prob`, `weight`, account / entity IDs and names,
  bank names, anything from `accounts.parquet`. They leak the answer or identify accounts.
- **Never split randomly by row** — schemes and accounts would end up in both train and test.
- **Do not use** the old `01_join.py`, `join_accounts.py`, `data\sample_trans.csv`,
  `preprocessing_feature_selection.py` or the old `EDA_HI_Small.ipynb` — all superseded.
- **Do not commit `data\processed\`** to GitHub (too big) — it must be in `.gitignore`.
