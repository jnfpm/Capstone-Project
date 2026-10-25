"""Independent verification of data/processed/hi_small_joined.csv.

The join was built with DuckDB. This script checks it with pandas, re-deriving
answers from the RAW files instead of trusting the join's own report.

Seven checks:
  1. Structure        - row count, columns, label counts
  2. CSV vs Parquet   - the CSV export did not corrupt anything (IDs, dates, sums)
  3. Rows are real    - a random sample of joined rows exists in the raw CSV
  4. Account lookup   - re-join a random sample with pandas, compare every field
  5. Patterns         - per-attempt row counts match the patterns file exactly
  6. Flags            - recompute self_transfer / same_entity / cross_bank / cross_border
  7. EDA numbers      - known EDA results reproduce from the joined file

Run:  python notebook/01b_verify_join.py   (from the project folder)
Takes ~1-2 min (reading the 1.3 GB CSV is the slow part).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project folder: this script lives in <project>/notebook/
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import eda_utils

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CSV = PROC / "hi_small_joined.csv"
PARQUET = PROC / "hi_small_joined.parquet"

SAMPLE_N = 20_000
SEED = 42

t0 = time.time()
results = []


def step(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}")


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))


def first_df(result, must_have):
    if isinstance(result, pd.DataFrame):
        return result
    frames = [x for x in result if isinstance(x, pd.DataFrame)]
    return ([f for f in frames if must_have in f.columns] or frames)[0]


ID_COLS = ["Account", "Account.1", "Entity ID_from", "Entity ID_to"]

# ------------------------------------------------------------------ load
step("reading joined CSV...")
df = pd.read_csv(
    CSV,
    parse_dates=["Timestamp"],
    dtype={c: "string" for c in ID_COLS},
    low_memory=False,
)
step(f"CSV: {df.shape}")

# ================================================================== 1
print("\n1. STRUCTURE")
expected_cols = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
    "Bank Name_from", "Entity ID_from", "Entity Name_from", "Country_from", "Entity Type_from",
    "Bank Name_to", "Entity ID_to", "Entity Name_to", "Country_to", "Entity Type_to",
    "Attempt ID", "Pattern Type",
    "self_transfer", "same_entity", "cross_bank", "cross_border",
]
check("rows = 5,078,336", len(df) == 5_078_336, f"{len(df):,}")
check("27 expected columns", list(df.columns) == expected_cols)
check("positives = 5,177", df["Is Laundering"].sum() == 5_177, f"{int(df['Is Laundering'].sum()):,}")
check("Timestamp parsed as date", pd.api.types.is_datetime64_any_dtype(df["Timestamp"]))
check("sorted by Timestamp", df["Timestamp"].is_monotonic_increasing)
check("no exact duplicate rows", not df.duplicated().any())

nulls = df.isna().sum()
allowed_null = {"Attempt ID", "Pattern Type"}
bad_nulls = nulls[(nulls > 0) & ~nulls.index.isin(allowed_null)]
check("nulls only in Attempt ID / Pattern Type", bad_nulls.empty,
      "" if bad_nulls.empty else bad_nulls.to_dict())

# ================================================================== 2
print("\n2. CSV vs PARQUET")
if PARQUET.exists():
    pq = pd.read_parquet(PARQUET)
    check("same row count", len(pq) == len(df))
    check("same positives", pq["Is Laundering"].sum() == df["Is Laundering"].sum())
    check("same Attempt ID count", pq["Attempt ID"].notna().sum() == df["Attempt ID"].notna().sum())
    check("same amount total",
          np.isclose(pq["Amount Paid"].sum(), df["Amount Paid"].sum(), rtol=1e-9))
    # IDs are the classic CSV casualty: hex-like strings read back as numbers.
    same_ids = all(
        pq[c].astype("string").reset_index(drop=True)
        .equals(df[c].reset_index(drop=True))
        for c in ID_COLS
    )
    check("account / entity IDs identical (no number conversion)", same_ids)
    check("timestamps identical",
          pd.to_datetime(pq["Timestamp"]).reset_index(drop=True)
          .equals(df["Timestamp"].reset_index(drop=True)))
    del pq
else:
    print("  (no parquet found - skipped)")

# ------------------------------------------------------------------ raw files
step("reading raw files for independent re-check...")
raw = pd.read_csv(
    RAW / "HI-Small_Trans.csv",
    dtype={"Account": "string", "Account.1": "string"},
)
raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
acc = pd.read_csv(RAW / "HI-Small_accounts.csv",
                  dtype={"Account Number": "string", "Entity ID": "string"})

# ================================================================== 3
print("\n3. ROWS ARE REAL")
raw_cols = list(raw.columns)
check("raw minus joined = 9 duplicates", len(raw) - len(df) == 9, f"{len(raw) - len(df)}")
check("positives unchanged vs raw", raw["Is Laundering"].sum() == df["Is Laundering"].sum())

rng = np.random.default_rng(SEED)
samp = df.iloc[rng.choice(len(df), size=min(SAMPLE_N, len(df)), replace=False)]
# all laundering rows too - those are the ones that matter most
samp = pd.concat([samp, df[df["Is Laundering"] == 1]]).drop_duplicates()
found = samp[raw_cols].merge(raw.drop_duplicates(), on=raw_cols, how="left", indicator=True)
missing = int((found["_merge"] == "left_only").sum())
check(f"{len(samp):,} sampled rows (incl. all positives) exist in raw file",
      missing == 0, f"{missing} not found")

# ================================================================== 4
print("\n4. ACCOUNT LOOKUP (re-joined with pandas)")
lookup = acc.rename(columns={"Bank ID": "_bank", "Account Number": "_acct"})
for side, bank, acct in (("from", "From Bank", "Account"), ("to", "To Bank", "Account.1")):
    m = samp[[bank, acct, f"Bank Name_{side}", f"Entity ID_{side}", f"Entity Name_{side}"]].merge(
        lookup, left_on=[bank, acct], right_on=["_bank", "_acct"], how="left",
    )
    ok = (
        m["Bank Name"].eq(m[f"Bank Name_{side}"]).all()
        and m["Entity ID"].eq(m[f"Entity ID_{side}"]).all()
        and m["Entity Name"].eq(m[f"Entity Name_{side}"]).all()
    )
    check(f"{side}-side bank / entity match raw accounts file", ok,
          f"{len(m):,} rows compared")

banks = acc["Bank Name"].drop_duplicates()
country_map = pd.Series(eda_utils.extract_bank_country(banks).values, index=banks.values)
ok_c = (
    samp["Bank Name_from"].map(country_map).eq(samp["Country_from"]).all()
    and samp["Bank Name_to"].map(country_map).eq(samp["Country_to"]).all()
)
check("country matches eda_utils.extract_bank_country", ok_c)

# ================================================================== 5
print("\n5. PATTERNS")
pat = first_df(eda_utils.load_patterns(), must_have="Attempt ID")
pat_sizes = pat["Attempt ID"].value_counts().sort_index()
got_sizes = df["Attempt ID"].dropna().astype(int).value_counts().sort_index()
pat_sizes.index = pat_sizes.index.astype(int)
check("370 attempts", got_sizes.size == 370, f"{got_sizes.size}")
check("every attempt has exactly its pattern-file row count",
      got_sizes.equals(pat_sizes),
      f"{int((got_sizes != pat_sizes.reindex(got_sizes.index)).sum())} attempts differ")
check("every Attempt ID row is labelled laundering",
      (df.loc[df["Attempt ID"].notna(), "Is Laundering"] == 1).all())
pt = df.dropna(subset=["Attempt ID"]).groupby("Attempt ID")["Pattern Type"].nunique()
check("one Pattern Type per attempt", (pt == 1).all())
check("1,968 unexplained positives",
      ((df["Is Laundering"] == 1) & df["Attempt ID"].isna()).sum() == 1_968,
      f"{int(((df['Is Laundering'] == 1) & df['Attempt ID'].isna()).sum()):,}")

# ================================================================== 6
print("\n6. FLAGS (recomputed)")
check("self_transfer", df["self_transfer"].eq((df["Account"] == df["Account.1"]).astype(int)).all())
check("same_entity", df["same_entity"].eq((df["Entity ID_from"] == df["Entity ID_to"]).astype(int)).all())
check("cross_bank (by name)", df["cross_bank"].eq((df["Bank Name_from"] != df["Bank Name_to"]).astype(int)).all())
check("cross_border", df["cross_border"].eq((df["Country_from"] != df["Country_to"]).astype(int)).all())
check("self-transfers = 591,212", df["self_transfer"].sum() == 591_212, f"{int(df['self_transfer'].sum()):,}")

# ================================================================== 7
print("\n7. EDA NUMBERS REPRODUCE")
print("   (EDA ran before 9 non-laundering duplicates were removed, so tiny drift is expected)")


def rate(mask):
    return 100 * df.loc[mask, "Is Laundering"].mean()


rows_7 = [
    ("same-entity laundering rate %", rate(df["same_entity"] == 1), 0.0053),
    ("cross-entity laundering rate %", rate(df["same_entity"] == 0), 0.1164),
    ("cross-bank laundering rate %", rate(df["cross_bank"] == 1), 0.1157),
    ("same-bank laundering rate %", rate(df["cross_bank"] == 0), 0.0149),
]
for name, got, exp in rows_7:
    check(f"{name}: {got:.4f} vs EDA {exp:.4f}", abs(got - exp) < 0.0005)

print(f"   cross-border rate: {rate(df['cross_border'] == 1):.4f}%  "
      f"same-country: {rate(df['cross_border'] == 0):.4f}%  "
      f"(EDA: 0.2192 vs 0.0562 - compare by eye; EDA's jurisdiction grouping may differ)")

# ================================================================== summary
n_fail = sum(1 for _, ok in results if not ok)
print("\n" + "=" * 50)
print(f"{len(results) - n_fail} / {len(results)} checks passed")
if n_fail:
    print("FAILED:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
else:
    print("The joined CSV is correct.")
step("done")
