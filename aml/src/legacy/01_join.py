# """Step 1 of preprocessing: build the joined HI-Small table and save it.

# Runs on the FULL 5,078,345 rows. No sampling here.
# Output: D:\\capstone\\data\\processed\\hi_small_joined.parquet

# Everything downstream (time cut, sampling, features, the agent's DuckDB store)
# reads this file instead of re-parsing the CSVs.
# """

# import sys
# import time
# from pathlib import Path

# sys.path.insert(0, r"D:\capstone\src")

# import pandas as pd
# import eda_utils

# OUT_DIR = Path(r"D:\capstone\data\processed")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# OUT_FILE = OUT_DIR / "hi_small_joined.parquet"

# EXPECTED_ROWS = 5_078_345
# EXPECTED_POSITIVES = 5_177
# EXPECTED_PATTERN_ROWS = 3_209

# t0 = time.time()


# def step(msg):
#     print(f"[{time.time() - t0:6.1f}s] {msg}")


# def split_result(result, must_have=None):
#     """eda_utils functions return either a DataFrame or (DataFrame, report).

#     Returns (table, extras): the DataFrame - the one holding `must_have` if
#     several are returned - and everything else for printing.
#     """
#     if isinstance(result, pd.DataFrame):
#         return result, []
#     items = list(result)
#     frames = [x for x in items if isinstance(x, pd.DataFrame)]
#     if must_have:
#         frames = [f for f in frames if must_have in f.columns] or frames
#     if not frames:
#         raise TypeError(f"no DataFrame in result of type {type(result)}")
#     table = frames[0]
#     extras = [x for x in items if x is not table]
#     return table, extras


# # ---------------------------------------------------------------- 1. load
# step("loading transactions (sorted by Timestamp)...")
# trans = eda_utils.load_transactions()
# step(f"transactions: {trans.shape}")

# # ---------------------------------------------------------------- 1b. duplicates
# # Exact duplicates on all 11 columns. Dropped BEFORE the joins so the patterns
# # join cannot match one scheme transfer to two identical rows.
# dup_mask = trans.duplicated(keep=False)
# print(f"\nduplicate rows (all copies): {int(dup_mask.sum())}")
# print(trans.loc[dup_mask].sort_values(list(trans.columns)).to_string())
# print(f"laundering among duplicates: {int(trans.loc[dup_mask, 'Is Laundering'].sum())}\n")

# n_before = len(trans)
# trans = trans.drop_duplicates(keep="first").reset_index(drop=True)
# n_dropped = n_before - len(trans)
# step(f"dropped {n_dropped} duplicate rows -> {len(trans):,}")
# EXPECTED_ROWS = n_before - n_dropped

# step("loading accounts...")
# accounts = eda_utils.load_accounts()
# step(f"accounts: {accounts.shape}")

# # ---------------------------------------------------------------- 2. accounts join
# step("joining accounts onto both sides...")
# df, acct_extras = split_result(eda_utils.link_accounts(trans, accounts), must_have="Timestamp")
# del trans

# for x in acct_extras:
#     print("\naccount join report:")
#     print(x)
#     print()

# assert len(df) == EXPECTED_ROWS, f"row count changed: {len(df):,}"
# step(f"after account join: {df.shape}  (row count unchanged)")

# # ---------------------------------------------------------------- 3. patterns join
# step("joining patterns (Attempt ID, Pattern Type)...")
# patterns, pat_extras = split_result(eda_utils.load_patterns(), must_have="Attempt ID")
# step(f"patterns: {patterns.shape}")
# for x in pat_extras:
#     print("\npatterns load report:")
#     print(x)

# df, link_extras = split_result(eda_utils.link_patterns(df, patterns), must_have="Timestamp")

# assert len(df) == EXPECTED_ROWS, f"row count changed after patterns: {len(df):,}"
# for x in link_extras:
#     print("\npattern reconciliation:")
#     print(x)
#     print()

# # ---------------------------------------------------------------- 4. derived columns
# # Country and entity type for whichever side columns link_accounts produced.
# # Matched by name so this works regardless of the suffixes it uses.
# name_cols = [c for c in df.columns if "Bank Name" in c]
# entity_cols = [c for c in df.columns if "Entity Name" in c]

# if not any("Country" in c for c in df.columns):
#     for c in name_cols:
#         df[c.replace("Bank Name", "Country")] = eda_utils.extract_bank_country(df[c])
#         step(f"added {c.replace('Bank Name', 'Country')}")

# if not any("Entity Type" in c for c in df.columns):
#     for c in entity_cols:
#         df[c.replace("Entity Name", "Entity Type")] = eda_utils.extract_entity_type(df[c])
#         step(f"added {c.replace('Entity Name', 'Entity Type')}")

# df["self_transfer"] = (df["Account"] == df["Account.1"]).astype("int8")

# # ---------------------------------------------------------------- 5. memory
# for c in df.columns:
#     if df[c].dtype == object and c not in ("Account", "Account.1"):
#         df[c] = df[c].astype("category")

# # ---------------------------------------------------------------- 6. checks
# print("\n================ CHECKS ================")
# print(f"rows:          {len(df):>12,}   expected {EXPECTED_ROWS:,}")
# print(f"positives:     {int(df['Is Laundering'].sum()):>12,}   expected {EXPECTED_POSITIVES:,}")

# attempt_col = next((c for c in df.columns if c == "Attempt ID"), None)
# if attempt_col:
#     print(f"pattern rows:  {int(df[attempt_col].notna().sum()):>12,}   expected {EXPECTED_PATTERN_ROWS:,}")
#     print(f"attempts:      {df[attempt_col].nunique():>12,}   expected 370")

# print(f"self-transfers:{int(df['self_transfer'].sum()):>12,}   expected 591,212")

# print("\nnulls in joined columns (should be 0):")
# joined = [c for c in df.columns if any(k in c for k in ("Bank Name", "Entity", "Country"))]
# print(df[joined].isna().sum().to_string())

# print(f"\nmemory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
# print("\ncolumns:")
# print(list(df.columns))

# # ---------------------------------------------------------------- 7. save
# step(f"saving to {OUT_FILE} ...")
# df.to_parquet(OUT_FILE, index=False)
# step(f"done. {OUT_FILE.stat().st_size / 1e6:.0f} MB on disk")

import duckdb

duckdb.sql("""
    WITH t AS (
        SELECT * FROM 'D:/capstone/data/processed/hi_small_joined.parquet'
        WHERE "Timestamp" < TIMESTAMP '2022-09-11'
    ),
    lau AS (
        SELECT "Account" AS a FROM t WHERE "Is Laundering" = 1
        UNION
        SELECT "Account.1" FROM t WHERE "Is Laundering" = 1
    )
    SELECT
        (SELECT COUNT(*) FROM lau)                                   AS laundering_accounts,
        COUNT(*)                                                     AS rows_in_their_histories,
        SUM("Is Laundering")                                         AS positives
    FROM t
    WHERE "Account" IN (SELECT a FROM lau) OR "Account.1" IN (SELECT a FROM lau)
""").show()