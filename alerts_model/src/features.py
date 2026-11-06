"""Point-in-time transaction features for HI-Small (preprocessing step 3).

Everything here runs in DuckDB over the FULL joined parquet built by
``01_join_sql.py``. Account counts are never computed from a sample.

Rules (from EDA v2 / PREPROCESSING_PLAN.md):

* Point-in-time: a feature for a transfer at minute t uses only transfers in
  STRICTLY EARLIER minutes. Same-minute transfers are common, so "earlier in the
  same minute" is not used (it would depend on file order). One exception,
  ``conversion_before``, looks at the same minute on purpose (see below).
* Self-transfers (sender account = receiver account: Reinvestment bookings and
  currency conversions) are excluded from every count and are not scored.
  They stay in the base table because ``conversion_before`` needs them.
* Account key is ``"<bank id>:<account number>"`` - account numbers alone are
  not unique across banks.
* ``cross_bank`` compares bank NAMES (468 names map to several bank IDs).
* Rows from 11 Sept onward (generator residue) are dropped.
* Things learned from data (FX table, hub list) are learned on TRAIN dates only.
* Account history uses a FIXED look-back (LOOKBACK_HOURS, default 48 h), not
  "since 1 Sept". Counts since the start of the data grow with the calendar, so
  the same account looks busier on 9 Sept than on 2 Sept and "never seen before"
  is true for almost everyone on 1 Sept - the model would learn the date, and
  train/test distributions differ by construction. With a fixed window the
  features mean the same thing every day once the window is full; rows in the
  first LOOKBACK_HOURS get ``history_complete = False`` (meta, not a feature) and
  are left out of model fitting by default.

Split (same rule as 02_subsample_split.py ``time`` mode): train 1-6 Sept,
val 7 Sept, test 8-10 Sept, by the row's own date - except rows of a laundering
attempt, which all go to the split of that attempt's LAST kept row. That keeps
each scheme in one split; it uses ``Attempt ID`` for the split only, never as
a feature.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
CUTOFF = "2022-09-11"                 # rows from this date on are dropped
SPLIT_DAYS = {                         # inclusive date ranges, by row date
    "train": ("2022-09-01", "2022-09-06"),
    "val": ("2022-09-07", "2022-09-07"),
    "test": ("2022-09-08", "2022-09-10"),
}
TRAIN_END = "2022-09-07"              # exclusive: hub list / FX table use rows before this
HUB_MIN_COUNTERPARTIES = 500
LOOKBACK_HOURS = 48                    # fixed account-history window
CONVERSION_REL_TOL = 1e-6              # "same amount": equal up to float noise (works for sub-cent Bitcoin)

TARGET = "Is Laundering"

# Unique row key (plan step 5). Exact duplicates were removed in step 1.
ROW_KEY = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
           "Amount Paid", "Amount Received", "Payment Format",
           "Payment Currency", "Receiving Currency"]

# ------------------------------------------------------------------ feature lists
COUNT_FEATURES = [               # all over [t - window, t), self-transfers excluded
    "pair_prev_48h",             # sender -> receiver transfers
    "snd_sent_48h",              # sender: transfers sent
    "snd_recv_48h",              # sender: transfers received
    "snd_sent_24h",              # sender: sent, last 24 h
    "snd_sent_60m",              # sender: sent, last 60 min (bursts)
    "rcv_recv_48h",              # receiver: transfers received
    "rcv_sent_48h",              # receiver: transfers sent
    "rcv_recv_24h",              # receiver: received, last 24 h
    "rcv_recv_60m",              # receiver: received, last 60 min
]
FLAG_FEATURES = [
    "new_pair_48h",              # pair_prev_48h == 0
    "reverse_pair_48h",          # receiver paid sender within the window
    "snd_quiet_48h",             # sender had no activity in the window
    "rcv_quiet_48h",             # receiver had no activity in the window
    "conversion_before",         # same-minute same-amount currency conversion by sender
    "snd_hub", "rcv_hub",        # >= 500 counterparties on train dates
    "cross_border", "cross_bank", "same_entity",
    "currency_changes",          # Payment Currency != Receiving Currency
]
AMOUNT_FEATURES = ["log_amount_usd"]
CATEGORICAL_FEATURES = [
    "payment_format", "payment_currency",
    "country_from", "country_to", "etype_from", "etype_to",
]
FEATURES = COUNT_FEATURES + FLAG_FEATURES + AMOUNT_FEATURES + CATEGORICAL_FEATURES

# Feature groups used for the "with / without" model variants.
FORMAT_FEATURES = ["payment_format"]
GENERATOR_FEATURES = ["payment_format", "conversion_before"]   # generator mechanics (EDA v2)

META_COLUMNS = [
    "row_id", "Timestamp", "day", "split", "own_date_split", "scheme_moved",
    "scheme_truncated", "acct_from", "acct_to", "Attempt ID", "Pattern Type",
    "explained", "history_complete", "Payment Format", "amount_usd", "weight",
    "net_48h", "net_24h", "net_60m",
]
# Network volume: all scored (non-self) transfers in the same windows as the count
# features. Meta, not features - used to test whether count drift between days is
# just total volume, and to optionally rescale counts (modeling.normalise_counts).
NET_FOR_COUNT = {
    "pair_prev_48h": "net_48h", "snd_sent_48h": "net_48h", "snd_recv_48h": "net_48h",
    "rcv_recv_48h": "net_48h", "rcv_sent_48h": "net_48h",
    "snd_sent_24h": "net_24h", "rcv_recv_24h": "net_24h",
    "snd_sent_60m": "net_60m", "rcv_recv_60m": "net_60m",
}

# Never allowed in X (checked by 03b and by modeling.assert_no_leak).
LEAK_COLUMNS = [
    TARGET, "Attempt ID", "Pattern Type", "scheme_truncated", "scheme_moved",
    "incl_prob", "weight", "seed_from", "seed_to", "explained",
    "touches_laundering", "appearances", "n_cpty", "is_hub", "full_history",
    "acct_from", "acct_to", "Account", "Account.1", "From Bank", "To Bank",
    "Entity ID_from", "Entity ID_to", "Entity Name_from", "Entity Name_to",
    "Bank Name_from", "Bank Name_to", "row_id", "Timestamp", "split", "day",
    "history_complete", "own_date_split", "net_48h", "net_24h", "net_60m",
]

DATA_DICTIONARY = {
    # feature: (how computed, why - EDA v2 reference, point-in-time?)
    "pair_prev_48h": ("sender->receiver transfers in [t-48h, t)",
                      "first-time pair 20.6x riskier; falls from the 2nd transfer", "yes"),
    "new_pair_48h": ("pair_prev_48h == 0", "77.68% of laundering goes to a never-seen pair", "yes"),
    "reverse_pair_48h": ("receiver sent to sender in [t-48h, t)",
                         "point-in-time rebuild of two-way pairs (48x, but that used the future)", "yes"),
    "snd_sent_48h": ("sender's sends in [t-48h, t)", "laundering senders sent FEWER before (3 vs 18)", "yes"),
    "snd_recv_48h": ("sender's receipts in [t-48h, t)", "account role / pass-through context", "yes"),
    "snd_sent_24h": ("sender's sends in [t-24h, t)", "bursts on a slower scale", "yes"),
    "snd_sent_60m": ("sender's sends in [t-60m, t)", "bursts: same-minute share 18.42% vs 9.84%", "yes"),
    "rcv_recv_48h": ("receiver's receipts in [t-48h, t)", "received 13 vs 2 (whole window); 6 vs 11 point-in-time", "yes"),
    "rcv_sent_48h": ("receiver's sends in [t-48h, t)", "account role", "yes"),
    "rcv_recv_24h": ("receiver's receipts in [t-24h, t)", "gather bursts", "yes"),
    "rcv_recv_60m": ("receiver's receipts in [t-60m, t)", "gather bursts", "yes"),
    "snd_quiet_48h": ("sender: no sends or receipts in [t-48h, t)", "brand-new sender 11.63% vs 4.88%", "yes"),
    "rcv_quiet_48h": ("receiver: no sends or receipts in [t-48h, t)", "symmetry with snd_quiet_48h", "yes"),
    "conversion_before": ("sender made a same-minute self-transfer converting INTO this payment's currency, "
                          "same amount", "61.79% of explained laundering vs 1.68% normal - GENERATOR MECHANIC", "same minute"),
    "snd_hub": (">= 500 counterparties on train dates", "15 hubs, all touch laundering; 32.2% of unexplained", "train-only"),
    "rcv_hub": ("as snd_hub for the receiver", "as above", "train-only"),
    "cross_border": ("country_from != country_to", "3.9x, stable every day", "row"),
    "cross_bank": ("bank NAME differs", "7.5x but flags 86% of rows (AP lift 1.0x)", "row"),
    "same_entity": ("same owner, different accounts", "3.4x safer once self-transfers removed", "row"),
    "currency_changes": ("payment currency != receiving currency", "all positives are same-currency", "row"),
    "log_amount_usd": ("ln(1 + Amount Paid / units per USD), FX from train dates",
                       "USD median 5,878.81 vs 891.06; risk not monotonic", "row"),
    "payment_format": ("raw Payment Format", "ACH 86.59% of laundering - GENERATOR ARTEFACT, train with and without", "row"),
    "payment_currency": ("raw Payment Currency", "Saudi Riyal 0.42% = Saudi Arabia effect", "row"),
    "country_from": ("country parsed from bank name", "route effects (noisy)", "row"),
    "country_to": ("as country_from", "11 of top 15 routes end domestic", "row"),
    "etype_from": ("entity type parsed from owner name", "owner type", "row"),
    "etype_to": ("as etype_from", "owner type", "row"),
}


# ------------------------------------------------------------------ helpers
class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        print(f"[{time.time() - self.t0:6.1f}s] {msg}", flush=True)


def connect(tmp_dir: Path | None = None, memory_limit: str = "4GB", threads: int | None = None):
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    if threads:
        con.execute(f"SET threads = {threads}")
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp_dir.as_posix()}'")
    return con


def _split_case(col: str) -> str:
    """SQL CASE mapping a DATE expression to train / val / test."""
    parts = [f"WHEN {col} BETWEEN DATE '{a}' AND DATE '{b}' THEN '{name}'"
             for name, (a, b) in SPLIT_DAYS.items()]
    return "CASE " + " ".join(parts) + " END"


def build_base(con, joined_parquet: Path):
    """``base``: every kept row (self-transfers included) with keys and split."""
    src = Path(joined_parquet).as_posix()
    key_order = ", ".join(f'"{c}"' for c in ROW_KEY)
    con.execute(f"""
        CREATE OR REPLACE TABLE base AS
        WITH raw AS (
            SELECT *,
                   CAST("From Bank" AS VARCHAR) || ':' || "Account"   AS acct_from,
                   CAST("To Bank"   AS VARCHAR) || ':' || "Account.1" AS acct_to
            FROM read_parquet('{src}')
            WHERE "Timestamp" < TIMESTAMP '{CUTOFF}'
        ),
        att AS (   -- last kept row of each attempt, and whether rows were cut off
            SELECT "Attempt ID", MAX("Timestamp") AS att_last
            FROM raw WHERE "Attempt ID" IS NOT NULL GROUP BY 1
        ),
        cut AS (
            SELECT DISTINCT "Attempt ID" FROM read_parquet('{src}')
            WHERE "Timestamp" >= TIMESTAMP '{CUTOFF}' AND "Attempt ID" IS NOT NULL
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY {key_order}) AS row_id,
            r.*,
            r.acct_from = r.acct_to                         AS is_self,
            CAST(r."Timestamp" AS DATE)                     AS day,
            {_split_case('CAST(r."Timestamp" AS DATE)')}     AS own_date_split,
            COALESCE({_split_case('CAST(a.att_last AS DATE)')},
                     {_split_case('CAST(r."Timestamp" AS DATE)')}) AS split,
            c."Attempt ID" IS NOT NULL                      AS scheme_truncated
        FROM raw r
        LEFT JOIN att a USING ("Attempt ID")
        LEFT JOIN cut c USING ("Attempt ID")
    """)
    con.execute("""CREATE OR REPLACE TABLE base AS
                   SELECT *, split <> own_date_split AS scheme_moved FROM base""")


def build_fx_table(con) -> pd.DataFrame:
    """Units of each currency per 1 USD, from cross-currency rows on train dates.

    Rates in this dataset are fixed (EDA v2: max relative spread 1.06%), so a
    median per currency pair is exact enough. Currencies never paired with USD
    are reached through a currency that is.
    """
    pairs = con.execute(f"""
        SELECT "Payment Currency" AS pay, "Receiving Currency" AS rcv,
               MEDIAN("Amount Received" / "Amount Paid") AS rate,   -- rcv units per pay unit
               COUNT(*) AS n,
               MAX("Amount Received" / "Amount Paid") / MIN("Amount Received" / "Amount Paid") - 1 AS spread
        FROM base
        WHERE "Timestamp" < TIMESTAMP '{TRAIN_END}'
          AND "Payment Currency" <> "Receiving Currency"
          AND "Amount Paid" >= 1 AND "Amount Received" >= 1     -- avoid cent-rounding noise
        GROUP BY 1, 2
    """).df()
    all_cur = sorted(set(con.execute(
        'SELECT DISTINCT "Payment Currency" FROM base UNION SELECT DISTINCT "Receiving Currency" FROM base'
    ).df().iloc[:, 0]))
    per_usd = {"US Dollar": 1.0}
    for _ in range(len(all_cur)):                      # propagate through pairs
        for p in pairs.itertuples():
            if p.pay in per_usd and p.rcv not in per_usd:
                per_usd[p.rcv] = per_usd[p.pay] * p.rate
            elif p.rcv in per_usd and p.pay not in per_usd:
                per_usd[p.pay] = per_usd[p.rcv] / p.rate
    missing = [c for c in all_cur if c not in per_usd]
    if missing:
        raise ValueError(f"no exchange rate path to USD for {missing}")
    fx = pd.DataFrame({"currency": list(per_usd), "units_per_usd": list(per_usd.values())})
    fx = fx.sort_values("currency").reset_index(drop=True)
    fx.attrs["max_pair_spread"] = float(pairs["spread"].max()) if len(pairs) else 0.0
    return fx


def build_hubs(con) -> pd.DataFrame:
    """Accounts with >= HUB_MIN_COUNTERPARTIES distinct counterparties on train dates."""
    return con.execute(f"""
        WITH e AS (
            SELECT acct_from AS acct, acct_to AS cpty FROM base
            WHERE NOT is_self AND "Timestamp" < TIMESTAMP '{TRAIN_END}'
            UNION
            SELECT acct_to, acct_from FROM base
            WHERE NOT is_self AND "Timestamp" < TIMESTAMP '{TRAIN_END}'
        )
        SELECT acct, COUNT(*) AS n_counterparties FROM e
        GROUP BY 1 HAVING COUNT(*) >= {HUB_MIN_COUNTERPARTIES}
        ORDER BY n_counterparties DESC, acct
    """).df()


def _cum_table(con, name: str, key_sql: str, ts_sql: str = '"Timestamp"', source: str = "nonself"):
    """Cumulative event count per (key, minute): ``name(key, ts, cum)``."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT key, ts, SUM(n) OVER (PARTITION BY key ORDER BY ts
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
        FROM (SELECT {key_sql} AS key, {ts_sql} AS ts, COUNT(*) AS n
              FROM {source} GROUP BY 1, 2)
    """)


def _count_before(con, out: str, cum: str, key_sql: str, ts_sql: str = '"Timestamp"'):
    """For every scored row: events of ``key`` in minutes strictly before ``ts``."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {out} AS
        SELECT q.row_id, COALESCE(e.cum, 0) AS v
        FROM (SELECT row_id, {key_sql} AS key, {ts_sql} AS ts FROM nonself) q
        ASOF LEFT JOIN {cum} e ON q.key = e.key AND q.ts > e.ts
    """)


def build_features(con, log=print) -> None:
    """Create table ``feat`` (one row per scored transfer: meta + features + label)."""
    con.execute("CREATE OR REPLACE TABLE nonself AS SELECT * FROM base WHERE NOT is_self")

    # ---- cumulative event tables (self-transfers excluded)
    _cum_table(con, "cum_send", "acct_from")
    _cum_table(con, "cum_recv", "acct_to")
    _cum_table(con, "cum_pair", "acct_from || '>' || acct_to")
    log("cumulative event tables built")

    now = '"Timestamp"'
    h60 = '"Timestamp" - INTERVAL 60 MINUTE'
    h24 = '"Timestamp" - INTERVAL 24 HOUR'
    hlb = f'"Timestamp" - INTERVAL {LOOKBACK_HOURS} HOUR'
    pair, rev = "acct_from || '>' || acct_to", "acct_to || '>' || acct_from"
    jobs = {                    # name: (cumulative table, key, cut-off time)
        "f_pair_now": ("cum_pair", pair, now), "f_pair_lb": ("cum_pair", pair, hlb),
        "f_rev_now": ("cum_pair", rev, now), "f_rev_lb": ("cum_pair", rev, hlb),
        "f_ss_now": ("cum_send", "acct_from", now), "f_ss_lb": ("cum_send", "acct_from", hlb),
        "f_ss_24": ("cum_send", "acct_from", h24), "f_ss_60": ("cum_send", "acct_from", h60),
        "f_sr_now": ("cum_recv", "acct_from", now), "f_sr_lb": ("cum_recv", "acct_from", hlb),
        "f_rr_now": ("cum_recv", "acct_to", now), "f_rr_lb": ("cum_recv", "acct_to", hlb),
        "f_rr_24": ("cum_recv", "acct_to", h24), "f_rr_60": ("cum_recv", "acct_to", h60),
        "f_rs_now": ("cum_send", "acct_to", now), "f_rs_lb": ("cum_send", "acct_to", hlb),
    }
    for out, (cum, key, ts) in jobs.items():
        _count_before(con, out, cum, key, ts)
    log(f"{len(jobs)} point-in-time lookups done")

    # ---- network volume in the same windows (per distinct minute, then joined)
    con.execute("""CREATE OR REPLACE TABLE minutes AS
                   SELECT "Timestamp" AS ts, COUNT(*) AS n FROM nonself GROUP BY 1""")
    con.execute("""CREATE OR REPLACE TABLE cum_net AS
                   SELECT ts, SUM(n) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
                   FROM minutes""")
    con.execute(f"""
        CREATE OR REPLACE TABLE f_net AS
        WITH q AS (SELECT ts,
                          ts - INTERVAL {LOOKBACK_HOURS} HOUR AS t48,
                          ts - INTERVAL 24 HOUR AS t24,
                          ts - INTERVAL 60 MINUTE AS t60 FROM minutes)
        SELECT q.ts,
               COALESCE(a.cum, 0) - COALESCE(b.cum, 0) AS net_48h,
               COALESCE(a.cum, 0) - COALESCE(c.cum, 0) AS net_24h,
               COALESCE(a.cum, 0) - COALESCE(d.cum, 0) AS net_60m
        FROM q
        ASOF LEFT JOIN cum_net a ON q.ts > a.ts
        ASOF LEFT JOIN cum_net b ON q.t48 > b.ts
        ASOF LEFT JOIN cum_net c ON q.t24 > c.ts
        ASOF LEFT JOIN cum_net d ON q.t60 > d.ts
    """)

    # ---- conversion right before the send (same minute, same amount)
    con.execute(f"""
        CREATE OR REPLACE TABLE f_conv AS
        SELECT DISTINCT q.row_id
        FROM nonself q
        JOIN base c
          ON c.is_self
         AND c.acct_from = q.acct_from
         AND c."Timestamp" = q."Timestamp"
         AND c."Payment Currency" <> c."Receiving Currency"
         AND c."Receiving Currency" = q."Payment Currency"
        WHERE ABS(c."Amount Received" - q."Amount Paid")
              <= {CONVERSION_REL_TOL} * GREATEST(ABS(q."Amount Paid"), 1e-12)
    """)

    # ---- assemble
    con.execute(f"""
        CREATE OR REPLACE TABLE feat AS
        SELECT
            n.row_id, n."Timestamp", n.day, n.split, n.own_date_split, n.scheme_moved,
            n.scheme_truncated, n.acct_from, n.acct_to, n."Attempt ID", n."Pattern Type",
            n."Attempt ID" IS NOT NULL                          AS explained,
            n."Payment Format",
            n."Amount Paid" / fx.units_per_usd                  AS amount_usd,
            1.0                                                 AS weight,

            n."Timestamp" >= first_ts.t0 + INTERVAL {LOOKBACK_HOURS} HOUR AS history_complete,
            CAST(nt.net_48h AS BIGINT) AS net_48h,
            CAST(nt.net_24h AS BIGINT) AS net_24h,
            CAST(nt.net_60m AS BIGINT) AS net_60m,

            CAST(pn.v - pl.v AS INTEGER)                        AS pair_prev_48h,
            CAST(ssn.v - ssl.v AS INTEGER)                      AS snd_sent_48h,
            CAST(srn.v - srl.v AS INTEGER)                      AS snd_recv_48h,
            CAST(ssn.v - ss24.v AS INTEGER)                     AS snd_sent_24h,
            CAST(ssn.v - ss60.v AS INTEGER)                     AS snd_sent_60m,
            CAST(rrn.v - rrl.v AS INTEGER)                      AS rcv_recv_48h,
            CAST(rsn.v - rsl.v AS INTEGER)                      AS rcv_sent_48h,
            CAST(rrn.v - rr24.v AS INTEGER)                     AS rcv_recv_24h,
            CAST(rrn.v - rr60.v AS INTEGER)                     AS rcv_recv_60m,

            CAST(pn.v - pl.v = 0 AS TINYINT)                    AS new_pair_48h,
            CAST(rvn.v - rvl.v > 0 AS TINYINT)                  AS reverse_pair_48h,
            CAST(ssn.v - ssl.v = 0 AND srn.v - srl.v = 0 AS TINYINT) AS snd_quiet_48h,
            CAST(rrn.v - rrl.v = 0 AND rsn.v - rsl.v = 0 AS TINYINT) AS rcv_quiet_48h,
            CAST(cv.row_id IS NOT NULL AS TINYINT)              AS conversion_before,
            CAST(hs.acct IS NOT NULL AS TINYINT)                AS snd_hub,
            CAST(hr.acct IS NOT NULL AS TINYINT)                AS rcv_hub,
            CAST(n."Country_from" <> n."Country_to" AS TINYINT) AS cross_border,
            CAST(n."Bank Name_from" <> n."Bank Name_to" AS TINYINT) AS cross_bank,
            CAST(n."Entity ID_from" = n."Entity ID_to" AS TINYINT)  AS same_entity,
            CAST(n."Payment Currency" <> n."Receiving Currency" AS TINYINT) AS currency_changes,

            LN(1 + n."Amount Paid" / fx.units_per_usd)          AS log_amount_usd,

            CAST(n."Payment Format" AS VARCHAR)                 AS payment_format,
            CAST(n."Payment Currency" AS VARCHAR)               AS payment_currency,
            CAST(n."Country_from" AS VARCHAR)                   AS country_from,
            CAST(n."Country_to" AS VARCHAR)                     AS country_to,
            CAST(n."Entity Type_from" AS VARCHAR)               AS etype_from,
            CAST(n."Entity Type_to" AS VARCHAR)                 AS etype_to,

            CAST(n."Is Laundering" AS TINYINT)                  AS "Is Laundering"
        FROM nonself n
        JOIN fx_table fx ON fx.currency = n."Payment Currency"
        JOIN f_pair_now pn USING (row_id)  JOIN f_pair_lb pl USING (row_id)
        JOIN f_rev_now rvn USING (row_id)  JOIN f_rev_lb rvl USING (row_id)
        JOIN f_ss_now ssn USING (row_id)   JOIN f_ss_lb ssl USING (row_id)
        JOIN f_ss_24 ss24 USING (row_id)   JOIN f_ss_60 ss60 USING (row_id)
        JOIN f_sr_now srn USING (row_id)   JOIN f_sr_lb srl USING (row_id)
        JOIN f_rr_now rrn USING (row_id)   JOIN f_rr_lb rrl USING (row_id)
        JOIN f_rr_24 rr24 USING (row_id)   JOIN f_rr_60 rr60 USING (row_id)
        JOIN f_rs_now rsn USING (row_id)   JOIN f_rs_lb rsl USING (row_id)
        CROSS JOIN (SELECT MIN("Timestamp") AS t0 FROM base) first_ts
        JOIN f_net nt ON nt.ts = n."Timestamp"
        LEFT JOIN f_conv cv USING (row_id)
        LEFT JOIN hubs hs ON hs.acct = n.acct_from
        LEFT JOIN hubs hr ON hr.acct = n.acct_to
        ORDER BY n.row_id
    """)
    log("feature table assembled")


def attach_sample(con, sample_dir: Path) -> dict:
    """Restrict ``feat`` to the rows of the 02_subsample_split time-mode sample.

    Expects ``{train,val,test}.parquet`` (or ``.csv``) in ``sample_dir`` with the
    raw key columns and ``weight``. The sample's own split and weights replace
    the full-file ones. Returns row counts for reporting.
    """
    sample_dir = Path(sample_dir)
    parts = []
    for name in SPLIT_DAYS:
        pq, csv = sample_dir / f"{name}.parquet", sample_dir / f"{name}.csv"
        if pq.exists():
            reader = f"read_parquet('{pq.as_posix()}')"
        elif csv.exists():
            reader = (f"read_csv('{csv.as_posix()}', header=true, "
                      "types={'Account':'VARCHAR','Account.1':'VARCHAR','Timestamp':'TIMESTAMP'})")
        else:
            raise FileNotFoundError(
                f"{pq} not found. Run `python 02_subsample_split.py time` first, or point "
                f"--sample-dir at the folder holding train/val/test files.")
        parts.append(f"""SELECT {', '.join(f'CAST("{c}" AS {t})' + f' AS "{c}"' for c, t in [
            ("Timestamp", "TIMESTAMP"), ("From Bank", "BIGINT"), ("Account", "VARCHAR"),
            ("To Bank", "BIGINT"), ("Account.1", "VARCHAR"), ("Amount Paid", "DOUBLE"),
            ("Amount Received", "DOUBLE"), ("Payment Format", "VARCHAR"),
            ("Payment Currency", "VARCHAR"), ("Receiving Currency", "VARCHAR")])},
            CAST(weight AS DOUBLE) AS s_weight, '{name}' AS s_split FROM {reader}""")
    con.execute("CREATE OR REPLACE TABLE sample_rows AS " + " UNION ALL ".join(parts))
    n_sample = con.execute("SELECT COUNT(*) FROM sample_rows").fetchone()[0]
    key = " AND ".join(f's."{c}" = b."{c}"' for c in ROW_KEY)
    con.execute(f"""
        CREATE OR REPLACE TABLE sample_ids AS
        SELECT b.row_id, b.is_self, s.s_weight, s.s_split
        FROM sample_rows s JOIN base b ON {key}
    """)
    n_matched = con.execute("SELECT COUNT(*) FROM sample_ids").fetchone()[0]
    n_self = con.execute("SELECT COUNT(*) FROM sample_ids WHERE is_self").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE feat AS
        SELECT f.* REPLACE (s.s_weight AS weight, s.s_split AS split,
                            s.s_split <> f.own_date_split AS scheme_moved)
        FROM feat f JOIN sample_ids s USING (row_id)
        ORDER BY row_id
    """)
    n_scored = con.execute("SELECT COUNT(*) FROM feat").fetchone()[0]
    return {"sample_rows": n_sample, "matched": n_matched,
            "self_transfers_dropped": n_self, "scored_rows": n_scored}


def export(con, out_dir: Path) -> dict:
    """Write {split}_features.parquet (X + y + row_id) and {split}_meta.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    x_cols = ", ".join(["row_id"] + [f'"{c}"' for c in FEATURES] + [f'"{TARGET}"'])
    m_cols = ", ".join(f'"{c}"' for c in META_COLUMNS)
    for name in SPLIT_DAYS:
        con.execute(f"""COPY (SELECT {x_cols} FROM feat WHERE split = '{name}' ORDER BY row_id)
                        TO '{(out_dir / f'{name}_features.parquet').as_posix()}' (FORMAT PARQUET)""")
        con.execute(f"""COPY (SELECT {m_cols} FROM feat WHERE split = '{name}' ORDER BY row_id)
                        TO '{(out_dir / f'{name}_meta.parquet').as_posix()}' (FORMAT PARQUET)""")
        counts[name] = con.execute(
            f'SELECT COUNT(*), SUM("{TARGET}") FROM feat WHERE split = ?', [name]).fetchone()
    return counts


def write_data_dictionary(path: Path) -> None:
    lines = ["# Data dictionary - model features (03_features.py)", "",
             "Point-in-time = uses only transfers in strictly earlier minutes. "
             "Self-transfers excluded from all counts.", "",
             "| Feature | How computed | Why (EDA v2) | Point-in-time |",
             "|---|---|---|---|"]
    for f in FEATURES:
        how, why, pit = DATA_DICTIONARY[f]
        lines.append(f"| `{f}` | {how} | {why} | {pit} |")
    lines += ["", "Never features: " + ", ".join(f"`{c}`" for c in LEAK_COLUMNS) + "."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
