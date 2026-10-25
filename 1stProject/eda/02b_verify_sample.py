"""02b - independent validation of 02_subsample_split.py outputs.

Re-derives every guarantee from the joined file instead of trusting the script's
own report: row provenance, positives, scheme completeness, seed full-history,
weights, representativeness vs the full file, split rules, output files.

Run:   python notebook/02b_verify_sample.py          (one-file sample)
       python notebook/02b_verify_sample.py time
       python notebook/02b_verify_sample.py group
The last line must say every check passed.
"""
import sys
from pathlib import Path

import duckdb

MODE = sys.argv[1] if len(sys.argv) > 1 else "sample"
ROOT = Path(__file__).resolve().parents[1]   # project folder: this script lives in <project>/notebook/
JOINED = ROOT / "data" / "processed" / "hi_small_joined.parquet"
OUTD = (ROOT / "data" / "processed" / "sample" / MODE).as_posix()
J = f"'{JOINED.as_posix()}'"
TRAIN_END, VAL_END = "2022-09-07", "2022-09-08"   # must match 02_subsample_split.py
c = duckdb.connect()
q = lambda s: c.execute(s).fetchone()
results = []


def check(name, got, exp, ok=None):
    ok = (got == exp) if ok is None else ok
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}  expected {exp}")


c.execute(f"""CREATE TABLE fulld AS SELECT *,
    CAST("From Bank" AS VARCHAR)||':'||"Account" AS xf, CAST("To Bank" AS VARCHAR)||':'||"Account.1" AS xt
    FROM {J} WHERE Timestamp < TIMESTAMP '2022-09-11'""")
FILES = {"all": "sample"} if MODE == "sample" else {"train": "train", "val": "val", "test": "test"}
c.execute("CREATE TABLE s AS " + " UNION ALL BY NAME ".join(
    f"SELECT *, '{sp}' AS sp FROM '{OUTD}/{name}.parquet'" for sp, name in FILES.items()))
KEY = '"Timestamp","From Bank","Account","To Bank","Account.1","Amount Paid","Amount Received","Payment Format"'

print(f"\n========== VALIDATION: {MODE} ==========")
n_full, pos_full = q("SELECT COUNT(*), SUM(\"Is Laundering\") FROM fulld")
check("residue dropped: rows left", n_full, 5_077_228)
check("residue dropped: positives left", pos_full, 4_522)
check("no rows on/after 11 Sept in sample", q("SELECT COUNT(*) FROM s WHERE Timestamp >= TIMESTAMP '2022-09-11'")[0], 0)

n_s = q("SELECT COUNT(*) FROM s")[0]
check("sample size ~750k (+-0.1%)", n_s, 750_000, abs(n_s - 750_000) < 750)
# every sample row exists in the full file, no row duplicated across splits
c.execute(f"CREATE TABLE fk AS SELECT {KEY}, COUNT(*) k FROM fulld GROUP BY ALL")
c.execute(f"CREATE TABLE sk AS SELECT {KEY}, COUNT(*) k FROM s GROUP BY ALL")
check("sample rows not in full file", q(f"SELECT COUNT(*) FROM sk ANTI JOIN fk USING ({KEY})")[0], 0)
check("sample rows over-represented vs full (duplicates)",
      q(f"SELECT COUNT(*) FROM sk JOIN fk USING ({KEY}) WHERE sk.k > fk.k")[0], 0)
check("positives in sample", q('SELECT SUM("Is Laundering") FROM s')[0], pos_full)
check("unexplained positives in sample (no Attempt ID)",
      q('SELECT COUNT(*) FROM s WHERE "Is Laundering"=1 AND "Attempt ID" IS NULL')[0],
      q('SELECT COUNT(*) FROM fulld WHERE "Is Laundering"=1 AND "Attempt ID" IS NULL')[0])

# ---- schemes
check("every scheme complete (sample rows == full rows, pre-cut)",
      q("""SELECT COUNT(*) FROM (SELECT "Attempt ID", COUNT(*) n FROM fulld WHERE "Attempt ID" IS NOT NULL GROUP BY 1) f
           FULL JOIN (SELECT "Attempt ID", COUNT(*) n FROM s WHERE "Attempt ID" IS NOT NULL GROUP BY 1) x
           USING ("Attempt ID") WHERE f.n IS DISTINCT FROM x.n""")[0], 0)
check("schemes in sample", q('SELECT COUNT(DISTINCT "Attempt ID") FROM s')[0], 363)
check("schemes spanning >1 split", q('SELECT COUNT(*) FROM (SELECT "Attempt ID" FROM s WHERE "Attempt ID" IS NOT NULL GROUP BY 1 HAVING COUNT(DISTINCT sp)>1)')[0], 0)
trunc_true = q(f"""SELECT COUNT(DISTINCT "Attempt ID") FROM {J} WHERE "Attempt ID" IN
    (SELECT "Attempt ID" FROM {J} WHERE Timestamp >= TIMESTAMP '2022-09-11')
    AND Timestamp < TIMESTAMP '2022-09-11'""")[0]
check("scheme_truncated flags the right schemes",
      q('SELECT COUNT(DISTINCT "Attempt ID") FROM s WHERE scheme_truncated')[0], trunc_true)
check("scheme_truncated never set on non-scheme rows",
      q('SELECT COUNT(*) FROM s WHERE scheme_truncated AND "Attempt ID" IS NULL')[0], 0)

# ---- full-history guarantee for seeds (what the agent relies on)
c.execute("""CREATE TABLE seeds AS SELECT acct_from a FROM s WHERE seed_from=1
             UNION SELECT acct_to FROM s WHERE seed_to=1""")
c.execute("""CREATE TABLE e_full AS SELECT xf a FROM fulld UNION ALL SELECT xt FROM fulld WHERE xf<>xt""")
c.execute("""CREATE TABLE e_s AS SELECT acct_from a FROM s UNION ALL SELECT acct_to FROM s WHERE acct_from<>acct_to""")
check("seed accounts whose history is incomplete",
      q("""SELECT COUNT(*) FROM (SELECT a, COUNT(*) n FROM e_full WHERE a IN (SELECT a FROM seeds) GROUP BY a) f
           JOIN (SELECT a, COUNT(*) n FROM e_s WHERE a IN (SELECT a FROM seeds) GROUP BY a) x USING (a)
           WHERE f.n <> x.n""")[0], 0)
check("seed flag consistent per account (seed_from vs seed_to)",
      q("""SELECT COUNT(*) FROM (SELECT acct_from a, seed_from v FROM s UNION SELECT acct_to, seed_to FROM s)
           GROUP BY a HAVING COUNT(DISTINCT v)>1""") is None, True)
check("rows with neither side a seed and not laundering",
      q('SELECT COUNT(*) FROM s WHERE seed_from=0 AND seed_to=0 AND "Is Laundering"=0')[0], 0)
lau_accts = "(SELECT xf FROM fulld WHERE \"Is Laundering\"=1 UNION SELECT xt FROM fulld WHERE \"Is Laundering\"=1)"
hubs = "(SELECT a FROM (SELECT a, COUNT(DISTINCT b) k FROM (SELECT xf a, xt b FROM fulld UNION ALL SELECT xt, xf FROM fulld) GROUP BY a) WHERE k>=500)"
check("non-hub laundering accounts all seeded",
      q(f"SELECT COUNT(*) FROM {lau_accts} l(a) WHERE a NOT IN {hubs} AND a NOT IN (SELECT a FROM seeds)")[0], 0)
check("hubs never seeded", q(f"SELECT COUNT(*) FROM seeds WHERE a IN {hubs}")[0], 0)

# ---- weights: positives weight 1, weights >= 1, weighted totals reproduce the full file
check("positive rows have weight exactly 1", q('SELECT COUNT(*) FROM s WHERE "Is Laundering"=1 AND weight<>1')[0], 0)
check("no weight below 1", q("SELECT COUNT(*) FROM s WHERE weight < 1")[0], 0)
w_n = q("SELECT SUM(weight) FROM s")[0]
check("weighted row total within 2% of full", round(w_n), n_full, abs(w_n / n_full - 1) < 0.02)

print("\n-- representativeness: weighted sample vs full (residue dropped) --")
print(f"{'measure':<44}{'full':>10}{'sample(w)':>11}{'sample(raw)':>12}")
def rate(where, tbl, w):
    wcol = "weight" if w else "1"
    return q(f'SELECT 100*SUM({wcol}*"Is Laundering")/SUM({wcol}) FROM {tbl} WHERE {where}')[0]
def share(where, tbl, w):
    wcol = "weight" if w else "1"
    return q(f"SELECT 100*SUM(CASE WHEN {where} THEN {wcol} ELSE 0 END)/SUM({wcol}) FROM {tbl}")[0]
rows = [("laundering rate %, overall", "TRUE", rate),
        ("laundering rate %, cross-border", "cross_border", rate),
        ("laundering rate %, same jurisdiction", "NOT cross_border", rate),
        ("laundering rate %, cross-bank (name)", "cross_bank", rate),
        ("laundering rate %, same bank (name)", "NOT cross_bank", rate),
        ("laundering rate %, same entity", "same_entity", rate),
        ("laundering rate %, different entity", "NOT same_entity", rate),
        ("laundering rate %, ACH", "\"Payment Format\"='ACH'", rate),
        ("share % rows ACH", "\"Payment Format\"='ACH'", share),
        ("share % rows Reinvestment", "\"Payment Format\"='Reinvestment'", share),
        ("share % rows Wire", "\"Payment Format\"='Wire'", share),
        ("share % rows Cheque", "\"Payment Format\"='Cheque'", share),
        ("share % rows self-transfer", "self_transfer", share),
        ("share % rows cross-border", "cross_border", share),
        ("share % rows cross-currency", "\"Payment Currency\"<>\"Receiving Currency\"", share)]
worst = 0
for name, where, fn in rows:
    f, sw, sr = fn(where, "fulld", False), fn(where, "s", True), fn(where, "s", False)
    rel = abs(sw / f - 1) if f else 0
    worst = max(worst, rel) if "rate" not in name or f > 0.01 else worst
    print(f"{name:<44}{f:>10.4f}{sw:>11.4f}{sr:>12.4f}   ({100*rel:+.1f}% rel)")
fm = q('SELECT MEDIAN("Amount Paid") FROM fulld WHERE "Is Laundering"=0')[0]
sm = q('SELECT MEDIAN("Amount Paid") FROM s WHERE "Is Laundering"=0')[0]
print(f"{'median amount, clean rows (unweighted)':<44}{fm:>10.2f}{sm:>23.2f}")
check("weighted measures within 10% relative of full", f"{100*worst:.1f}%", "<10%", worst < 0.10)

# ---- split-specific
if MODE == "time":
    bounds = {"train": ("2022-09-01", TRAIN_END), "val": (TRAIN_END, VAL_END), "test": (VAL_END, "2022-09-11")}
    bad = 0
    for sp, (lo, hi) in bounds.items():
        bad += q(f"""SELECT COUNT(*) FROM s WHERE sp='{sp}' AND NOT scheme_moved
                     AND NOT (Timestamp >= TIMESTAMP '{lo}' AND Timestamp < TIMESTAMP '{hi}')""")[0]
    check("non-moved rows dated inside their split window", bad, 0)
    order = "CASE {c} WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END"
    own = f"""CASE WHEN Timestamp < TIMESTAMP '{TRAIN_END}' THEN 'train'
                  WHEN Timestamp < TIMESTAMP '{VAL_END}' THEN 'val' ELSE 'test' END"""
    check("moved rows only ever go LATER, never earlier",
          q(f"SELECT COUNT(*) FROM s WHERE {order.format(c='sp')} < {order.format(c=own)}")[0], 0)
    check("train contains no row dated after its window (no future in train)",
          q(f"SELECT COUNT(*) FROM s WHERE sp='train' AND Timestamp >= TIMESTAMP '{TRAIN_END}'")[0], 0)
elif MODE == "group":
    check("scheme accounts in >1 split",
          q("""SELECT COUNT(*) FROM (SELECT a FROM (SELECT acct_from a, sp FROM s WHERE "Attempt ID" IS NOT NULL
               UNION ALL SELECT acct_to, sp FROM s WHERE "Attempt ID" IS NOT NULL) GROUP BY a HAVING COUNT(DISTINCT sp)>1)""")[0], 0)
    check("grp helper column not in outputs", "grp" in [r[0] for r in c.execute("DESCRIBE s").fetchall()], False)

# ---- files
acc = c.execute(f"SELECT COUNT(*), COUNT(DISTINCT acct), SUM(CAST(full_history AS INT)) FROM '{OUTD}/accounts.parquet'").fetchone()
check("accounts.parquet: one row per account", acc[0], acc[1])
check("accounts.parquet: full_history count == seed accounts", acc[2], q("SELECT COUNT(*) FROM seeds")[0])
ORDER_KEY = ('"Timestamp", acct_from, acct_to, "Amount Paid", "Amount Received", '
             '"Payment Format", "Payment Currency", "Receiving Currency"')
for name in FILES.values():
    check(f"{name}.csv rows == {name}.parquet rows",
          q(f"SELECT COUNT(*) FROM read_csv_auto('{OUTD}/{name}.csv')")[0],
          q(f"SELECT COUNT(*) FROM '{OUTD}/{name}.parquet'")[0])
    # deterministic order: key is unique and the file is stored in key order
    check(f"{name}.parquet sort key unique",
          q(f"SELECT COUNT(*) - COUNT(DISTINCT ({ORDER_KEY})) FROM '{OUTD}/{name}.parquet'")[0], 0)
    check(f"{name}.parquet stored in key order (same order on every machine)",
          q(f"""SELECT COUNT(*) FROM (SELECT file_row_number,
                  row_number() OVER (ORDER BY {ORDER_KEY}) - 1 AS r
                  FROM read_parquet('{OUTD}/{name}.parquet', file_row_number=true))
                WHERE file_row_number <> r""")[0], 0)

print("\n-- split table --")
print(c.execute("""SELECT sp, COUNT(*) rows, SUM("Is Laundering") pos,
    ROUND(100*SUM("Is Laundering")/SUM(1.0*1),3) raw_rate, ROUND(100*SUM(weight*"Is Laundering")/SUM(weight),4) w_rate,
    COUNT(DISTINCT "Attempt ID") schemes, COUNT(DISTINCT CASE WHEN scheme_truncated THEN "Attempt ID" END) truncated
    FROM s GROUP BY sp ORDER BY CASE sp WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END""").df().to_string(index=False))
print(f"\n{sum(results)} / {len(results)} checks passed")
