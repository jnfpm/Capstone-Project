# AML alert investigation — pipeline

From the raw IBM AML **HI-Small** files to a transaction-scoring model, and the places for what comes
after it (agents, web app, report).

## Layout

```
aml_pipeline/
├── README.md
├── requirements.txt
├── notebooks/                       run in this order (after the two scripts in "Run")
│   ├── EDA_HI_Small_v2.ipynb        EDA: Part A full file (short), Part B the subsample's train split (main)
│   ├── 03_PREPROCESSING.ipynb       subsample -> model-ready train / val / test
│   └── 04_MODELLING.ipynb           rule / logistic regression / LightGBM, evaluation, saved model
├── src/                             all Python files
│   ├── 01_join_sql.py               raw CSVs + patterns -> data/processed/hi_small_joined.parquet
│   ├── 01b_verify_join.py           independent check of the join
│   ├── 02_subsample_split.py        account-based subsample (~750k rows), weights, time / group split
│   ├── 02b_verify_sample.py         independent check of the subsample
│   ├── 03b_verify_preprocessing.py  independent check of the preprocessing (run by its notebook)
│   ├── eda_utils.py                 raw-file loading and parsing, scheme drawings
│   ├── evaluation.py                weighted metrics, alert budgets, bootstrap, drift
│   ├── sample_modeling.py           models, alerts, saved bundle, score_transactions (agent entry point)
│   └── legacy/01_join.py            first pandas join, superseded (reference only)
├── data/
│   ├── raw/                         the three Kaggle files (or kagglehub downloads them)
│   └── processed/                   everything the pipeline writes (not committed)
├── models/                          saved scorers + test scores + metrics (written by 04)
├── reports/                         findings table and figures for the written report (written by 03 / 04)
├── agents/                          NEXT: the investigation agents that use the scorer
└── app/                             NEXT: the deployed assistant web app
```

Scripts and notebooks find the `aml_pipeline` folder themselves (they look upward for
`src/eda_utils.py`), so it works wherever it is unzipped.

## Run

Put `HI-Small_Trans.csv`, `HI-Small_accounts.csv`, `HI-Small_Patterns.txt` in `data/raw/`, then
`pip install -r requirements.txt` (duckdb is for the src/ scripts only) and run the notebooks in order from
`notebooks/`. Each one builds what it is missing (join, subsample, splits) by calling the scripts in `src/`.

1. `EDA_HI_Small_v2.ipynb` — Part A: the full file (what the join, sampler and history depend on). Part B:
   the time-split subsample — integrity, weights, leakage, split, then every signal on the **train split,
   weighted**; each section ends in a rule, B19 lists them. Its features match 03's exactly.
2. `03_PREPROCESSING.ipynb` — builds `02_subsample_split.py time` and `group` itself if missing and stops if
   their check fails; ends with `15 / 15` and `67 / 67 checks passed`.
3. `04_MODELLING.ipynb` — writes `models/` and `reports/model_findings.md`.

**All three notebooks are plain pandas / numpy — no SQL.** Each loads the joined file once into one
DataFrame (text columns as categories). DuckDB is used only inside the scripts in `src/` (the join and
the subsample), which you run from the command line.

On a 2-core / 7 GB laptop (tested on a 4.3M-row synthetic copy): EDA ~1 min (peak ~5 GB, about 5.5 GB
on the real 5M rows), subsample ~1 min per mode, preprocessing ~2 min including the check (peak ~4.6 GB),
modelling ~15–25 min on the real data (rolling folds + 5-seed ensembles; ~1.5 GB). Close other notebooks first; 8 GB of RAM is comfortable.

## What each step does

| Step | In | Out | Key rules |
|---|---|---|---|
| **01 join** (`src/01_join_sql.py`) | `HI-Small_Trans.csv`, `HI-Small_accounts.csv`, `HI-Small_Patterns.txt` | `hi_small_joined.parquet` (27 columns, sorted by time) | 9 exact duplicates dropped; accounts joined on (bank ID, account number) twice; patterns give `Attempt ID` / `Pattern Type` (label leakage — never features) |
| **EDA** | joined file, time-split subsample | — | Part A: cutoff, self-transfers, schemes, bank names, FX, hubs. Part B: subsample checks, weights, leakage, split, history source, signals on train (weighted), stability, rules (B19) |
| **02 sampling** (`src/02_subsample_split.py`) | joined file | `sample/{sample,time,group}/` | drops the 11-18 Sept residue; keeps every laundering account + ~7% of the others with their full history; `weight` = 1 / inclusion probability; time split train 1-6 / val 7 / test 8-10 Sept, schemes kept whole; group split for robustness |
| **03 preprocessing** | the subsample (rows, labels, weights, splits) + the joined file only as account history before each row | `data/processed/preprocessed/` | 40 point-in-time features (48 h windows; incl. 2-step paths, 3-step cycles, fan-in/out bursts, sender behaviour); self-transfers not scored; FX, hubs, caps, category levels learned on train; account counts from the full file because counted inside the sample they are biased by the label (EDA B6, notebook §4.2) |
| **04 modelling** | `preprocessed/` only | `models/`, `reports/` | every metric weighted; warm-up, weighting and tree shape chosen on 3 rolling folds; LightGBM = 5-seed average; test on rows dated 8-10 Sept; alerts = each day's top 0.5% (0.1% / 0.25% also reported); feature sets all / no payment format / no generator / without the B16b graph features; best F1 vs the published HI-Small results; non-ACH share of the alert budget; account-level bootstrap CIs; rows with no seed account left out |

## Never a model input

`Attempt ID`, `Pattern Type`, `scheme_truncated`, `scheme_moved`, `weight`, `incl_prob`, `seed_from`,
`seed_to`, `history_complete`, the split columns, account / entity / bank IDs and names, anything from
`accounts.parquet` (`touches_laundering`, `appearances`, `n_cpty`, `is_hub`, `full_history`). The full
list is `artifacts.json` → `leak_columns`; both notebooks check it.
