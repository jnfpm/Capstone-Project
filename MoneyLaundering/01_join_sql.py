"""Step 1 of preprocessing, SQL version: build the joined HI-Small table.

DuckDB does the joins. Python is used only for the two things SQL cannot do
cleanly here:
  * parsing HI-Small_Patterns.txt (text blocks, not a table) -> eda_utils.load_patterns
  * country / entity type from names -> eda_utils, run on DISTINCT names only

Runs on the FULL file. No sampling.
Output: D:\\capstone\\data\\processed\\hi_small_joined.parquet  (+ .csv)

Needs:  pip install duckdb pyarrow
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import duckdb
import pandas as pd
import eda_utils

RAW = eda_utils.DATA_DIR
OUT_DIR = ROOT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET = OUT_DIR / "hi_small_joined.parquet"
CSV = OUT_DIR / "hi_small_joined.csv"

TRANS_CSV = (RAW / "HI-Small_Trans.csv").as_posix()
ACC_CSV = (RAW / "HI-Small_accounts.csv").as_posix()

EXPECTED_SOURCE_ROWS = 5_078_345
EXPECTED_DUPLICATES = 9
EXPECTED_POSITIVES = 5_177
EXPECTED_PATTERN_ROWS = 3_209
EXPECTED_ATTEMPTS = 370

t0 = time.time()


def step(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}")


def first_df(result, must_have):
    """eda_utils functions return a DataFrame or (DataFrame, report)."""
    if isinstance(result, pd.DataFrame):
        return result
    frames = [x for x in result if isinstance(x, pd.DataFrame)]
    frames = [f for f in frames if must_have in f.columns] or frames
    return frames[0]


con = duckdb.connect()

# ------------------------------------------------------------------ 1. raw tables
# The transactions CSV has TWO columns both called "Account". Names and types are
# set explicitly so the receiver column is "Account.1" (same as pandas) and no
# ID gets guessed as a number.
step("reading transactions...")
con.execute(f"""
    CREATE TABLE trans_raw AS
    SELECT
        strptime("Timestamp", '%Y/%m/%d %H:%M') AS "Timestamp",
        * EXCLUDE ("Timestamp")
    FROM read_csv('{TRANS_CSV}', header = true, columns = {{
        'Timestamp':          'VARCHAR',
        'From Bank':          'BIGINT',
        'Account':            'VARCHAR',
        'To Bank':            'BIGINT',
        'Account.1':          'VARCHAR',
        'Amount Received':    'DOUBLE',
        'Receiving Currency': 'VARCHAR',
        'Amount Paid':        'DOUBLE',
        'Payment Currency':   'VARCHAR',
        'Payment Format':     'VARCHAR',
        'Is Laundering':      'TINYINT'
    }})
""")
n_raw = con.execute("SELECT COUNT(*) FROM trans_raw").fetchone()[0]
step(f"transactions: {n_raw:,} rows")

# 9 exact duplicates: sub-cent Bitcoin rows, none laundering. Dropped before
# joining so the patterns join cannot match one transfer twice.
con.execute("""CREATE TABLE trans AS SELECT DISTINCT * FROM trans_raw""")
n_trans = con.execute("SELECT COUNT(*) FROM trans").fetchone()[0]
step(f"after removing duplicates: {n_trans:,} ({n_raw - n_trans} dropped)")
con.execute("DROP TABLE trans_raw")

step("reading accounts...")
con.execute(f"""
    CREATE TABLE accounts AS
    SELECT * FROM read_csv('{ACC_CSV}', header = true, columns = {{
        'Bank Name':      'VARCHAR',
        'Bank ID':        'BIGINT',
        'Account Number': 'VARCHAR',
        'Entity ID':      'VARCHAR',
        'Entity Name':    'VARCHAR'
    }})
""")
n_acc = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
dup_key = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT "Bank ID", "Account Number" FROM accounts
        GROUP BY 1, 2 HAVING COUNT(*) > 1
    )
""").fetchone()[0]
step(f"accounts: {n_acc:,} rows, duplicate (Bank ID, Account Number) keys: {dup_key}")
assert dup_key == 0, "duplicate account keys would multiply transaction rows"

# ------------------------------------------------------------------ 2. lookups from Python
step("country / entity type on distinct names (eda_utils)...")
banks = con.execute('SELECT DISTINCT "Bank Name" FROM accounts').df()
banks["Country"] = eda_utils.extract_bank_country(banks["Bank Name"]).values
ents = con.execute('SELECT DISTINCT "Entity Name" FROM accounts').df()
ents["Entity Type"] = eda_utils.extract_entity_type(ents["Entity Name"]).values
con.register("bank_country", banks)
con.register("entity_type", ents)
step(f"{len(banks):,} bank names, {len(ents):,} entity names mapped")

step("parsing patterns file (eda_utils)...")
pat = first_df(eda_utils.load_patterns(), must_have="Attempt ID")
pat = pat[["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
           "Amount Paid", "Attempt ID", "Pattern Type"]].copy()
pat["Timestamp"] = pd.to_datetime(pat["Timestamp"])
con.register("patterns_df", pat)
# Bank codes are zero-padded in the .txt ('021174') but not in the CSV ('21174').
# Casting both to BIGINT removes the padding.
con.execute("""
    CREATE TABLE patterns AS
    SELECT
        CAST("Timestamp" AS TIMESTAMP)  AS "Timestamp",
        CAST("From Bank" AS BIGINT)     AS "From Bank",
        CAST("Account"   AS VARCHAR)    AS "Account",
        CAST("To Bank"   AS BIGINT)     AS "To Bank",
        CAST("Account.1" AS VARCHAR)    AS "Account.1",
        CAST("Amount Paid" AS DOUBLE)   AS "Amount Paid",
        "Attempt ID",
        "Pattern Type"
    FROM patterns_df
""")
step(f"patterns: {len(pat):,} rows")

# ------------------------------------------------------------------ 3. the join
# Accounts joined TWICE: once for the sender, once for the receiver.
# Key is (Bank ID, Account Number) - account number alone is not unique
# (518,573 distinct across 518,581 rows). LEFT joins so nothing is dropped.
step("joining...")
con.execute("""
    CREATE TABLE joined AS
    SELECT
        t.*,

        sa."Bank Name"    AS "Bank Name_from",
        sa."Entity ID"    AS "Entity ID_from",
        sa."Entity Name"  AS "Entity Name_from",
        bf."Country"      AS "Country_from",
        ef."Entity Type"  AS "Entity Type_from",

        ra."Bank Name"    AS "Bank Name_to",
        ra."Entity ID"    AS "Entity ID_to",
        ra."Entity Name"  AS "Entity Name_to",
        bt."Country"      AS "Country_to",
        et."Entity Type"  AS "Entity Type_to",

        p."Attempt ID",
        p."Pattern Type",

        CAST(t."Account" = t."Account.1" AS TINYINT)               AS self_transfer,
        CAST(sa."Entity ID" = ra."Entity ID" AS TINYINT)           AS same_entity,
        CAST(sa."Bank Name" <> ra."Bank Name" AS TINYINT)          AS cross_bank,
        CAST(bf."Country" <> bt."Country" AS TINYINT)              AS cross_border

    FROM trans t
    LEFT JOIN accounts sa
           ON t."From Bank" = sa."Bank ID" AND t."Account"   = sa."Account Number"
    LEFT JOIN accounts ra
           ON t."To Bank"   = ra."Bank ID" AND t."Account.1" = ra."Account Number"
    LEFT JOIN bank_country bf ON sa."Bank Name"   = bf."Bank Name"
    LEFT JOIN bank_country bt ON ra."Bank Name"   = bt."Bank Name"
    LEFT JOIN entity_type  ef ON sa."Entity Name" = ef."Entity Name"
    LEFT JOIN entity_type  et ON ra."Entity Name" = et."Entity Name"
    LEFT JOIN patterns p
           ON  t."Timestamp"   = p."Timestamp"
           AND t."From Bank"   = p."From Bank"
           AND t."Account"     = p."Account"
           AND t."To Bank"     = p."To Bank"
           AND t."Account.1"   = p."Account.1"
           AND t."Amount Paid" = p."Amount Paid"
    ORDER BY t."Timestamp"
""")
step("join done")

# ------------------------------------------------------------------ 4. checks
q = lambda sql: con.execute(sql).fetchone()[0]

rows        = q("SELECT COUNT(*) FROM joined")
positives   = q('SELECT SUM("Is Laundering") FROM joined')
pat_rows    = q('SELECT COUNT("Attempt ID") FROM joined')
attempts    = q('SELECT COUNT(DISTINCT "Attempt ID") FROM joined')
unm_from    = q('SELECT COUNT(*) FROM joined WHERE "Bank Name_from" IS NULL')
unm_to      = q('SELECT COUNT(*) FROM joined WHERE "Bank Name_to" IS NULL')
no_country  = q('SELECT COUNT(*) FROM joined WHERE "Country_from" IS NULL OR "Country_to" IS NULL')
pat_not_lau = q('SELECT COUNT(*) FROM joined WHERE "Attempt ID" IS NOT NULL AND "Is Laundering" = 0')
self_tr     = q("SELECT SUM(self_transfer) FROM joined")

print("\n================ CHECKS ================")
print(f"source rows:          {n_raw:>10,}   expected {EXPECTED_SOURCE_ROWS:,}")
print(f"duplicates dropped:   {n_raw - n_trans:>10,}   expected {EXPECTED_DUPLICATES}")
print(f"rows after join:      {rows:>10,}   expected {n_trans:,} (must be unchanged)")
print(f"positives:            {positives:>10,}   expected {EXPECTED_POSITIVES:,}")
print(f"pattern rows matched: {pat_rows:>10,}   expected {EXPECTED_PATTERN_ROWS:,}")
print(f"attempts:             {attempts:>10,}   expected {EXPECTED_ATTEMPTS}")
print(f"unmatched senders:    {unm_from:>10,}   expected 0")
print(f"unmatched receivers:  {unm_to:>10,}   expected 0")
print(f"missing country:      {no_country:>10,}   expected 0")
print(f"pattern row not laundering: {pat_not_lau:>4,}   expected 0")
print(f"self-transfers:       {self_tr:>10,}   (EDA: 591,212 before dedup)")

assert rows == n_trans, "row count changed during the join - a key is not unique"

print("\ncolumns:")
print([r[0] for r in con.execute("DESCRIBE joined").fetchall()])

# ------------------------------------------------------------------ 5. save
step(f"saving {PARQUET} ...")
con.execute(f"COPY joined TO '{PARQUET.as_posix()}' (FORMAT PARQUET)")
step(f"parquet done. {PARQUET.stat().st_size / 1e6:.0f} MB")

# CSV copy for teammates. Excel cannot open it (limit 1,048,576 rows).
step(f"saving {CSV} (takes a minute or two)...")
con.execute(f"COPY joined TO '{CSV.as_posix()}' (HEADER, DELIMITER ',')")
step(f"csv done. {CSV.stat().st_size / 1e6:.0f} MB")
