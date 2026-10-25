"""Step 2 of preprocessing: account-based subsample (+ optional train/val/test split).

Reads   <project>/data/processed/hi_small_joined.parquet   (step 1 output)
Writes  <project>/data/processed/sample/<mode>/     (<mode> = sample, time or group)
            sample mode: sample.parquet (+ sample.csv)   ONE file, not split
            time/group:  train.parquet / val.parquet / test.parquet   (+ .csv copies)
            accounts.parquet       one row per account in the sample
            report.txt             everything printed below

REPRODUCIBILITY: anyone running this on the same joined file gets the SAME rows in
the SAME order, on any machine and any DuckDB version. The random draw uses MD5 (a
fixed standard; DuckDB's own hash() can change between versions) and every output
is sorted on a unique key. Compare `sample.parquet` checksums or the fingerprint
printed in report.txt to confirm two runs match.

DESIGN (one small dataset serves modelling AND the agent)
---------------------------------------------------------
1. Drop the 11-18 Sept residue. The generator stopped background traffic on
   10 Sept but kept writing in-flight schemes (72.7% laundering on 18 Sept).

2. Sample ACCOUNTS, not rows. Every sampled account keeps its COMPLETE history,
   so account features and agent traversal work on it.

3. Seed accounts =
     every account touching laundering                (always kept)
   + a random fraction Q of the other accounts       (Bernoulli, reproducible)
   Hub accounts (>= HUB_MIN_CPTY counterparties) are never seeds: one hub alone
   is 168k rows. Their rows stay wherever the other side is a seed.

4. Sample = every row where the sender OR receiver is a seed.
   All 5,177 positives and all 370 schemes survive by construction.

5. Inverse-probability weights. A row's chance of being in the sample is known
   exactly:
       laundering account on either side   -> 1
       otherwise each eligible side was drawn with probability Q
       p = 1 - (1 - p_from) * (1 - p_to)      (one side only if self-transfer)
   `weight = 1 / p`. Pass it as sample_weight to precision / recall / PR-AUC and
   the metrics estimate what you would get on the full 5M rows. Without it,
   every metric is inflated by the enrichment factor.

6. `seed_from` / `seed_to` = 1 when that account's full history is in the
   sample. The agent's tools must check this: a non-seed counterparty has only
   the rows it shares with seeds, so "no further transfers" may be an artefact.
   `scheme_truncated` = 1 on rows of schemes that continued past 10 Sept and
   lost their later transfers to the residue cut. Exclude those schemes when
   scoring the agent's case reconstruction.

7. Mode, set by SPLIT_MODE (command-line argument, default "sample"):

   "sample" (DEFAULT) - the subsample as ONE file, no split. Split it later.

   "time" (PRIMARY) - train 1-6 Sept, val 7 Sept, test 8-10 Sept.
       A scheme crossing a boundary goes ENTIRELY to the LATER split, so train
       never sees the future. This is the out-of-time holdout and the headline
       result.

   "group" (ROBUSTNESS CHECK) - no dates. Tests generalisation to unseen
       accounts and rings rather than to a later period.
       * Schemes are merged into clusters when they share an account, and each
         cluster goes WHOLE into one split.
       * Every other row follows its owner: the sender if the sender is a seed,
         otherwise the receiver. Each account's own rows are in ONE split.
       * Groups with laundering are cut by cumulative positives (70/15/15 of
         positives); clean groups are cut 70/15/15 at random.
       NEVER replace this with a plain random row split: the same scheme and the
       same account would land in train and test, and the model would score
       well by memorising accounts.

KNOWN LIMITATIONS - state these in the report (also printed in report.txt)
---------------------------------------------------------------------------
* Seed selection uses labels from the whole 1-10 Sept window. An account that
  launders only in the test period is still seeded with its full history, so
  train composition is partly set by test-period labels. The model never sees
  those labels, and the weights correct the marginal distribution, but train
  and test are not fully independent in construction.
* Weighted precision has high variance: each sampled clean row stands for 1/Q
  real rows, each positive for one. Bootstrap confidence intervals over
  ACCOUNTS (not rows) at the modelling step.
* The window is 10 days of synthetic data with no concept drift, so the time
  split is a structural out-of-time holdout, not evidence of robustness to
  drift.

NOT DONE HERE - next step
-------------------------
Account features (degree, counterparties, received-count, pass-through, dwell)
must be computed POINT-IN-TIME: for a transfer at time t, use only that
account's transfers before t, taken from the FULL joined file. Features over
the whole window leak the future into training rows.

LEAKAGE: `Attempt ID` and `Pattern Type` exist only on laundering rows.
They are kept for grouping and agent evaluation. NEVER use them as features.

Run:   python notebook/02_subsample_split.py          (one file, no split)
       python notebook/02_subsample_split.py time     (primary split)
       python notebook/02_subsample_split.py group    (robustness split)
All three modes draw the SAME sample (same SEED and Q); time and group only add a
split on top. Each mode writes to its own folder.
Needs: pip install duckdb pyarrow pandas
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

# ------------------------------------------------------------------ settings
# "sample": one file, no split (default)
# "time"  : train 1-6 Sept, val 7 Sept, test 8-10 Sept (schemes moved whole to later split)
# "group" : no dates. Each scheme cluster and each account goes WHOLE into one split.
SPLIT_MODE = sys.argv[1] if len(sys.argv) > 1 else "sample"
if SPLIT_MODE not in ("sample", "time", "group"):
    raise SystemExit(f"mode must be 'sample', 'time' or 'group', not {SPLIT_MODE!r}")

ROOT = Path(__file__).resolve().parents[1]   # project folder: this script lives in <project>/notebook/
JOINED = ROOT / "data" / "processed" / "hi_small_joined.parquet"
OUT = ROOT / "data" / "processed" / "sample" / SPLIT_MODE   # separate folder per mode

TARGET_ROWS = 750_000    # the script finds the Q that gives this many rows.
                         # Set to None to use the fixed Q below instead.
Q = 0.03                 # share of clean accounts drawn (used only if TARGET_ROWS is None)
Q_MIN = 0.01             # below this, weighted metrics get too noisy
Q_OPTIONS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
SEED = "capstone-2026"   # change -> a different but equally valid draw
HUB_MIN_CPTY = 500       # accounts with this many counterparties are hubs

DATA_END = "2022-09-11"    # 11-18 Sept residue dropped in both modes

TRAIN_END = "2022-09-07"   # time mode only: train = 1-6 Sept
VAL_END = "2022-09-08"     # time mode only: val = 7 Sept, test = 8-10 Sept. Val is small (323 positives): small hyperparameter search + threshold choice only; score test once.

GROUP_FRACTIONS = (0.70, 0.15, 0.15)   # group mode: train / val / test

WRITE_CSV = True

# Unique sort key for every output: identical row order on every machine.
ORDER_KEY = ('"Timestamp", acct_from, acct_to, "Amount Paid", "Amount Received", '
             '"Payment Format", "Payment Currency", "Receiving Currency"')


def unif(expr):
    """SQL: reproducible uniform draw in [0, 1) from MD5 of expr.
    MD5 is a fixed standard, so the draw is identical on every DuckDB version."""
    return f"(CAST(('0x' || substr(md5({expr}), 1, 12)) AS UBIGINT) / 281474976710656.0)"

# ------------------------------------------------------------------ setup
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log_lines = []


def say(msg=""):
    print(msg)
    log_lines.append(msg)


def step(msg):
    say(f"[{time.time() - t0:6.1f}s] {msg}")


con = duckdb.connect()
q1 = lambda sql: con.execute(sql).fetchone()[0]

# ------------------------------------------------------------------ 1. base
step("reading joined file, dropping 11-18 Sept residue...")
# Schemes still running after 10 Sept lose their later transfers to the residue
# cut. They stay in the data (truncated) and are flagged, because the agent's case
# reconstruction cannot be scored against their full pattern-file rows.
con.execute(f"""
    CREATE TABLE base AS
    SELECT
        *,
        CAST("From Bank" AS VARCHAR) || ':' || "Account"   AS acct_from,
        CAST("To Bank"   AS VARCHAR) || ':' || "Account.1" AS acct_to,
        COALESCE("Attempt ID" IN (
            SELECT DISTINCT "Attempt ID" FROM '{JOINED.as_posix()}'
            WHERE "Timestamp" >= TIMESTAMP '{DATA_END}' AND "Attempt ID" IS NOT NULL
        ), FALSE) AS scheme_truncated
    FROM '{JOINED.as_posix()}'
    WHERE "Timestamp" < TIMESTAMP '{DATA_END}'
""")
n_all = q1(f"SELECT COUNT(*) FROM '{JOINED.as_posix()}'")
pos_all = q1(f"""SELECT SUM("Is Laundering") FROM '{JOINED.as_posix()}'""")
n_base = q1("SELECT COUNT(*) FROM base")
pos_base = q1('SELECT SUM("Is Laundering") FROM base')
say(f"   joined file: {n_all:,} rows, {pos_all:,} positives")
say(f"   after residue drop: {n_base:,} rows, {pos_base:,} positives "
    f"({n_all - n_base:,} rows and {pos_all - pos_base:,} positives dropped)")

# ------------------------------------------------------------------ 2. accounts
# Account key is bank + account number: 8 account numbers exist at two banks.
step("profiling accounts...")
con.execute("""
    CREATE TABLE ends AS
    SELECT acct_from AS acct, acct_to   AS cpty, "Is Laundering" AS lau FROM base
    UNION ALL
    SELECT acct_to   AS acct, acct_from AS cpty, "Is Laundering" AS lau FROM base
""")
con.execute(f"""
    CREATE TABLE acct AS
    SELECT
        acct,
        COUNT(*)                         AS appearances,
        COUNT(DISTINCT cpty)             AS n_cpty,
        MAX(lau)                         AS touches_laundering,
        COUNT(DISTINCT cpty) >= {HUB_MIN_CPTY} AS is_hub,
        -- reproducible uniform draw in [0, 1) per account
        {unif(f"acct || '{SEED}'")} AS u
    FROM ends
    GROUP BY acct
""")
con.execute("DROP TABLE ends")

n_acct = q1("SELECT COUNT(*) FROM acct")
n_hub = q1("SELECT COUNT(*) FROM acct WHERE is_hub")
n_lau = q1("SELECT COUNT(*) FROM acct WHERE touches_laundering = 1 AND NOT is_hub")
n_lau_hub = q1("SELECT COUNT(*) FROM acct WHERE touches_laundering = 1 AND is_hub")
hub_rows = q1("SELECT SUM(appearances) FROM acct WHERE is_hub")
say(f"   accounts: {n_acct:,}")
say(f"   hubs (>= {HUB_MIN_CPTY} counterparties): {n_hub:,}, "
    f"{hub_rows:,} row-appearances  (EDA v2: 15 hubs, all at Oasis Thrift)")
say(f"   laundering accounts kept as seeds: {n_lau:,} "
    f"(+ {n_lau_hub} hubs touching laundering, not seeded)")

def in_side(side, q):
    """SQL: is this side's account a seed at fraction q?  side = 'f' or 't'."""
    return f"(NOT hub_{side} AND (lau_{side} = 1 OR u_{side} < {q}))"


def p_side(side, q):
    """SQL: probability this side's account was drawn as a seed."""
    return f"(CASE WHEN hub_{side} THEN 0.0 WHEN lau_{side} = 1 THEN 1.0 ELSE {q} END)"


con.execute("""
    CREATE TABLE rows AS
    SELECT b.*,
           f.is_hub AS hub_f, f.touches_laundering AS lau_f, f.u AS u_f,
           t.is_hub AS hub_t, t.touches_laundering AS lau_t, t.u AS u_t
    FROM base b
    JOIN acct f ON b.acct_from = f.acct
    JOIN acct t ON b.acct_to   = t.acct
""")
assert q1("SELECT COUNT(*) FROM rows") == n_base, "account profile join changed row count"
con.execute("DROP TABLE base")

# ------------------------------------------------------------------ 3. size table
def size_at(q):
    return con.execute(f"""
        SELECT COUNT(*), SUM("Is Laundering") FROM rows
        WHERE {in_side('f', q)} OR {in_side('t', q)} OR "Is Laundering" = 1
    """).fetchone()


base_rate = pos_base / n_base
floor_rows = size_at(0.0)[0]

if TARGET_ROWS is not None:
    step(f"finding Q for ~{TARGET_ROWS:,} rows...")
    if TARGET_ROWS <= floor_rows:
        raise SystemExit(
            f"TARGET_ROWS = {TARGET_ROWS:,} is at or below the floor of {floor_rows:,} rows "
            "(the full histories of the laundering accounts alone). Raise TARGET_ROWS."
        )
    lo, hi = 0.0, 1.0
    for _ in range(25):                       # rows grow with Q, so bisect
        mid = (lo + hi) / 2
        if size_at(mid)[0] < TARGET_ROWS:
            lo = mid
        else:
            hi = mid
    Q = round(hi, 5)
    step(f"Q = {Q} gives {size_at(Q)[0]:,} rows")
    if Q < Q_MIN:
        say(f"   WARNING: Q = {Q} is below {Q_MIN}. Too few clean accounts are drawn and "
            "weighted metrics will be noisy. Raise TARGET_ROWS.")

step("size table for different Q...")
say("")
say(f"   {'Q':>7}  {'rows':>10}  {'positives':>9}  {'rate %':>7}  {'enrichment':>10}  {'% of data':>9}")
for q in sorted(set(Q_OPTIONS + [Q])):
    n, p = size_at(q)
    mark = "  <- chosen" if q == Q else ("  <- floor (not usable)" if q == 0 else "")
    say(f"   {q:>7.4f}  {n:>10,}  {p:>9,}  {100 * p / n:>7.3f}  {p / n / base_rate:>9.1f}x  "
        f"{100 * n / n_base:>8.1f}%{mark}")
say(f"   full (residue dropped): {n_base:,} rows, {100 * base_rate:.4f}% laundering")
say("")

# ------------------------------------------------------------------ 4. build sample
step(f"building sample at Q = {Q} ...")
in_f, in_t = in_side("f", Q), in_side("t", Q)
p_f, p_t = p_side("f", Q), p_side("t", Q)

# Laundering rows are ALWAYS included (probability 1), even between two hubs -
# otherwise a hub-to-hub laundering transfer has no way into the sample.
con.execute(f"""
    CREATE TABLE sample AS
    SELECT
        * EXCLUDE (hub_f, lau_f, u_f, hub_t, lau_t, u_t),
        CAST({in_f} AS TINYINT) AS seed_from,
        CAST({in_t} AS TINYINT) AS seed_to,
        CASE WHEN "Is Laundering" = 1 THEN 1.0
             WHEN acct_from = acct_to THEN {p_f}
             ELSE 1 - (1 - {p_f}) * (1 - {p_t}) END AS incl_prob
    FROM rows
    WHERE {in_f} OR {in_t} OR "Is Laundering" = 1
""")
con.execute("ALTER TABLE sample ADD COLUMN weight DOUBLE")
con.execute("UPDATE sample SET weight = 1.0 / incl_prob")

unreachable = q1("""SELECT COUNT(*) FROM rows
                     WHERE hub_f AND hub_t AND "Is Laundering" = 0""")
n_s = q1("SELECT COUNT(*) FROM sample")
pos_s = q1('SELECT SUM("Is Laundering") FROM sample')
w_rows = q1("SELECT SUM(weight) FROM sample")
w_rate = q1('SELECT SUM(weight * "Is Laundering") / SUM(weight) FROM sample')
step(f"sample: {n_s:,} rows, {pos_s:,} positives")
con.execute("DROP TABLE rows")

# ------------------------------------------------------------------ 5. split
if SPLIT_MODE == "sample":
    step("no split (sample mode)...")
    con.execute(f"""
        CREATE TABLE split AS
        SELECT *, 'all' AS split FROM sample
        ORDER BY {ORDER_KEY}
    """)
    con.execute("DROP TABLE sample")

elif SPLIT_MODE == "time":
    step("time split, schemes kept whole...")
    con.execute("""
        CREATE TABLE scheme_end AS
        SELECT "Attempt ID", MAX("Timestamp") AS last_ts, MIN("Timestamp") AS first_ts
        FROM sample WHERE "Attempt ID" IS NOT NULL
        GROUP BY 1
    """)
    SPLIT_OF = """CASE WHEN {c} < TIMESTAMP '%s' THEN 'train'
                       WHEN {c} < TIMESTAMP '%s' THEN 'val'
                       ELSE 'test' END""" % (TRAIN_END, VAL_END)
    con.execute(f"""
        CREATE TABLE split AS
        SELECT s.*,
               CASE WHEN e.last_ts IS NOT NULL THEN {SPLIT_OF.format(c="e.last_ts")}
                    ELSE {SPLIT_OF.format(c='s."Timestamp"')} END AS split,
               e."Attempt ID" IS NOT NULL
                 AND {SPLIT_OF.format(c="e.first_ts")} <> {SPLIT_OF.format(c="e.last_ts")}
                 AS scheme_moved
        FROM sample s
        LEFT JOIN scheme_end e ON s."Attempt ID" = e."Attempt ID"
        ORDER BY {ORDER_KEY}
    """)
    con.execute("DROP TABLE sample")
    moved = q1('SELECT COUNT(DISTINCT "Attempt ID") FROM split WHERE scheme_moved')
    moved_rows = q1(f"""SELECT COUNT(*) FROM split WHERE scheme_moved
                        AND split <> {SPLIT_OF.format(c='"Timestamp"')}""")

elif SPLIT_MODE == "group":
    step("group split: scheme clusters and accounts kept whole...")

    # (a) Scheme clusters. Schemes sharing an account (100 accounts are in several
    #     schemes) are merged, so no laundering account straddles two splits via
    #     its schemes. Union-find over scheme <-> account links.
    links = con.execute("""
        SELECT DISTINCT CAST("Attempt ID" AS BIGINT), acct_from, acct_to
        FROM sample WHERE "Attempt ID" IS NOT NULL
        ORDER BY 1, 2, 3
    """).fetchall()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        # The smaller name always becomes the root, so a cluster's name is the
        # smallest member's name - the same whatever order the links arrive in.
        ra, rb = find(a), find(b)
        if ra != rb:
            ra, rb = min(ra, rb), max(ra, rb)
            parent[rb] = ra

    for att, a_f, a_t in links:
        node = f"scheme:{int(att)}"
        union(node, a_f)
        union(node, a_t)

    # Arrow tables, not pandas: every DuckDB version can read them.
    accts = sorted(k for k in parent if not k.startswith("scheme:"))
    atts = sorted(int(k.split(":")[1]) for k in parent if k.startswith("scheme:"))
    acct_cluster = pa.table({"acct": accts,
                             "grp": ["cluster:" + find(k) for k in accts]})
    att_cluster = pa.table({"att": pa.array(atts, type=pa.int64()),
                            "grp": ["cluster:" + find(f"scheme:{a}") for a in atts]})
    con.register("acct_cluster", acct_cluster)
    con.register("att_cluster", att_cluster)
    n_clusters = len(set(att_cluster.column("grp").to_pylist()))

    # (b) Each row gets ONE owner group:
    #       scheme row          -> its scheme cluster
    #       otherwise           -> sender if sender is a seed, else receiver
    #       owner in a cluster  -> that cluster, else the owner account itself
    con.execute("""
        CREATE TABLE owned AS
        SELECT s.*,
               COALESCE(ac.grp, oc.grp,
                        'acct:' || CASE WHEN s.seed_from = 1 OR s.seed_to = 0
                                        THEN s.acct_from ELSE s.acct_to END) AS grp
        FROM sample s
        LEFT JOIN att_cluster ac ON CAST(s."Attempt ID" AS BIGINT) = ac.att
        LEFT JOIN acct_cluster oc
               ON oc.acct = CASE WHEN s.seed_from = 1 OR s.seed_to = 0
                                 THEN s.acct_from ELSE s.acct_to END
    """)
    con.execute("DROP TABLE sample")

    # (c) Assign groups to splits.
    #     Groups holding laundering: ordered by a reproducible random number and cut
    #     by cumulative POSITIVES, so each split gets ~70/15/15 of the positives.
    #     Clean groups: by the same random number, cut 70/15/15.
    f_tr, f_va, _ = GROUP_FRACTIONS
    con.execute(f"""
        CREATE TABLE grp_split AS
        WITH g AS (
            SELECT grp, SUM("Is Laundering") AS pos,
                   {unif(f"grp || '{SEED}-split'")} AS u
            FROM owned GROUP BY grp
        ),
        lau AS (
            SELECT grp,
                   SUM(pos) OVER (ORDER BY u, grp ROWS UNBOUNDED PRECEDING)
                     / SUM(pos) OVER () AS cum
            FROM g WHERE pos > 0
        )
        SELECT grp, CASE WHEN cum <= {f_tr} THEN 'train'
                         WHEN cum <= {f_tr + f_va} THEN 'val' ELSE 'test' END AS split
        FROM lau
        UNION ALL
        SELECT grp, CASE WHEN u < {f_tr} THEN 'train'
                         WHEN u < {f_tr + f_va} THEN 'val' ELSE 'test' END
        FROM g WHERE pos = 0
    """)
    con.execute(f"""
        CREATE TABLE split AS
        SELECT o.*, gs.split
        FROM owned o JOIN grp_split gs USING (grp)
        ORDER BY {ORDER_KEY}
    """)
    con.execute("DROP TABLE owned")

    # Leak report: an account owns all its rows in one split by construction, but it
    # can still APPEAR (as the non-owner side) in another split's rows.
    acct_multi = q1("""
        SELECT COUNT(*) FROM (
            SELECT a, COUNT(DISTINCT split) AS k FROM (
                SELECT acct_from AS a, split FROM split WHERE seed_from = 1
                UNION ALL SELECT acct_to, split FROM split WHERE seed_to = 1)
            GROUP BY a) WHERE k > 1
    """)
    lau_acct_multi = q1("""
        SELECT COUNT(*) FROM (
            SELECT a, COUNT(DISTINCT split) AS k FROM (
                SELECT acct_from AS a, split FROM split WHERE "Attempt ID" IS NOT NULL
                UNION ALL SELECT acct_to, split FROM split WHERE "Attempt ID" IS NOT NULL)
            GROUP BY a) WHERE k > 1
    """)
else:
    raise ValueError(f"SPLIT_MODE must be 'time' or 'group', not {SPLIT_MODE!r}")

# ------------------------------------------------------------------ 6. report
say("")
say("================ SAMPLE REPORT ================")
say(f"Q = {Q}, seed = '{SEED}', hub threshold = {HUB_MIN_CPTY} counterparties")
say(f"rows: {n_s:,}  ({100 * n_s / n_base:.1f}% of {n_base:,})")
say(f"positives: {pos_s:,} of {pos_base:,}  (retention {pos_s / pos_base:.4f}, must be 1.0000)")
say(f"sample laundering rate: {100 * pos_s / n_s:.3f}%  vs full {100 * base_rate:.4f}%  "
    f"-> {pos_s / n_s / base_rate:.1f}x enriched")
say(f"WEIGHTED rate: {100 * w_rate:.4f}%   (should be close to {100 * base_rate:.4f}%)")
say(f"weighted row total: {w_rows:,.0f}   (should be close to {n_base - unreachable:,})")
say(f"clean hub-to-hub rows that can never be sampled: {unreachable:,} "
    f"({100 * unreachable / n_base:.2f}% of data) - state this in the report")
say("")
say(f"mode: {SPLIT_MODE}")
if SPLIT_MODE == "sample":
    say("no split - one file. Split later with the time or group mode.")
    dates = {"all": "1-10 Sep"}
elif SPLIT_MODE == "time":
    say(f"schemes crossing a split boundary, moved whole to the later split: {moved} "
        f"({moved_rows:,} rows dated earlier than their split)")
    dates = {"train": "1-6 Sep", "val": "7 Sep", "test": "8-10 Sep"}
else:
    say(f"scheme clusters (schemes sharing accounts merged): {n_clusters}")
    say(f"scheme accounts appearing in more than one split: {lau_acct_multi}  (should be 0)")
    say(f"full-history accounts appearing in more than one split: {acct_multi:,} "
        f"- as the counterparty side only; their own sent rows are all in one split")
    dates = {k: f"{int(100 * f)}% groups" for k, f in zip(("train", "val", "test"), GROUP_FRACTIONS)}
say("")

hdr = (f"{'split':<6} {'':<14} {'rows':>9} {'pos':>6} {'rate %':>7} "
       f"{'w-rate %':>8} {'schemes':>7} {'unexpl.':>7}")
say(hdr)
say("-" * len(hdr))
SPLITS = ("all",) if SPLIT_MODE == "sample" else ("train", "val", "test")
for sp in SPLITS:
    r = con.execute(f"""
        SELECT COUNT(*), SUM("Is Laundering"),
               SUM(weight * "Is Laundering") / SUM(weight),
               COUNT(DISTINCT "Attempt ID"),
               SUM(CASE WHEN "Is Laundering" = 1 AND "Attempt ID" IS NULL THEN 1 ELSE 0 END)
        FROM split WHERE split = '{sp}'
    """).fetchone()
    say(f"{sp:<6} {dates[sp]:<14} {r[0]:>9,} {r[1]:>6,} {100 * r[1] / r[0]:>7.3f} "
        f"{100 * r[2]:>8.4f} {r[3]:>7,} {r[4]:>7,}")

# scheme integrity: every scheme whole, in exactly one split
sch = con.execute("""
    SELECT COUNT(*) AS n,
           SUM(CASE WHEN n_split = 1 THEN 1 ELSE 0 END) AS one_split
    FROM (SELECT "Attempt ID", COUNT(DISTINCT split) AS n_split
          FROM split WHERE "Attempt ID" IS NOT NULL GROUP BY 1)
""").fetchone()
say("")
n_trunc = q1('SELECT COUNT(DISTINCT "Attempt ID") FROM split WHERE scheme_truncated')
n_gone = q1(f"""SELECT COUNT(DISTINCT "Attempt ID") FROM '{JOINED.as_posix()}'
                WHERE "Attempt ID" IS NOT NULL""") - sch[0]
say(f"schemes in sample: {sch[0]}  (of 370; {n_gone} lie wholly in the dropped residue)")
say(f"  of which TRUNCATED by the 11 Sept cut: {n_trunc}  (flag `scheme_truncated`; "
    "exclude from agent case-reconstruction scoring)")
say(f"schemes entirely inside one split: {sch[1]} of {sch[0]}")

n_seed_acct = q1("""
    SELECT COUNT(DISTINCT a) FROM (
        SELECT acct_from AS a FROM split WHERE seed_from = 1
        UNION SELECT acct_to FROM split WHERE seed_to = 1)
""")
n_any_acct = q1("""
    SELECT COUNT(DISTINCT a) FROM (
        SELECT acct_from AS a FROM split UNION SELECT acct_to FROM split)
""")
say(f"accounts in sample: {n_any_acct:,}  -  full history: {n_seed_acct:,}  -  "
    f"partial (counterparty only): {n_any_acct - n_seed_acct:,}")

say("")
say("KNOWN LIMITATIONS (copy into the report)")
say("* Seed selection used labels from the whole 1-10 Sept window, so train and test")
say("  are not fully independent in construction. The model never sees those labels;")
say("  the weights correct the marginal distribution but not this dependence.")
say(f"* Weighted precision is high-variance: each sampled clean row stands for up to "
    f"{1 / Q:,.0f} real rows,")
say("  each positive for one. Bootstrap CIs over ACCOUNTS, not rows, when modelling.")
if SPLIT_MODE == "time":
    say("* 10 days of synthetic data, no concept drift: this is a structural out-of-time")
    say("  holdout, not evidence of robustness to drift.")
    say("* Schemes crossing a boundary go whole to the later split, so some test rows are")
    say("  dated before 8 Sept. Point-in-time features keep this leak-free.")

# Fingerprint: the same value on any machine means the same rows. Built from
# integers and strings only (no float-to-text formatting, which varies by version).
fingerprint = q1(f"""
    SELECT md5(string_agg(
        CAST(epoch("Timestamp") AS BIGINT) || '|' || acct_from || '|' || acct_to || '|' ||
        CAST(round("Amount Paid" * 10000) AS BIGINT) || '|' || "Payment Format",
        ',' ORDER BY {ORDER_KEY}))
    FROM split
""")
say("")
say(f"duckdb version: {duckdb.__version__}")
say(f"SAMPLE FINGERPRINT: {fingerprint}   (identical on every machine = identical rows)")

assert pos_s == pos_base, "positives lost - should be impossible by construction"
assert sch[1] == sch[0], "a scheme is split across train/val/test"

# ------------------------------------------------------------------ 7. save
step("saving...")
# Drop split-construction helpers from the outputs. `scheme_moved` (time mode) is
# kept: it marks scheme rows dated earlier than the split they were placed in.
drop_cols = "split" + (", grp" if SPLIT_MODE == "group" else "")
for sp in SPLITS:
    name = "sample" if sp == "all" else sp
    sel = f"""SELECT * EXCLUDE ({drop_cols}) FROM split WHERE split = '{sp}' ORDER BY {ORDER_KEY}"""
    pq = OUT / f"{name}.parquet"
    con.execute(f"""COPY ({sel}) TO '{pq.as_posix()}' (FORMAT PARQUET)""")
    if WRITE_CSV:
        # Written by pandas, not DuckDB: DuckDB's CSV quoting changes between
        # versions. These settings give a byte-identical CSV on any machine
        # (Windows included - the line ending is fixed to \n).
        pd.read_parquet(pq, dtype_backend="numpy_nullable").to_csv(
            OUT / f"{name}.csv", index=False, lineterminator="\n", na_rep="",
            date_format="%Y-%m-%d %H:%M:%S")

con.execute(f"""
    COPY (
        SELECT a.acct, a.appearances, a.n_cpty, a.touches_laundering, a.is_hub,
               a.acct IN (SELECT acct_from FROM split WHERE seed_from = 1
                          UNION SELECT acct_to FROM split WHERE seed_to = 1) AS full_history
        FROM acct a
        WHERE a.acct IN (SELECT acct_from FROM split UNION SELECT acct_to FROM split)
        ORDER BY a.acct
    ) TO '{(OUT / "accounts.parquet").as_posix()}' (FORMAT PARQUET)
""")
(OUT / "report.txt").write_text("\n".join(log_lines), encoding="utf-8")
step(f"done -> {OUT}")
