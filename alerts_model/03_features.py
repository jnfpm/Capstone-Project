"""Step 3 of preprocessing: point-in-time features for every scored transfer.

Input : data/processed/hi_small_joined.parquet   (from 01_join_sql.py)
Output: data/processed/features/                  (full file, time split)
          train|val|test_features.parquet   row_id + X + "Is Laundering"
          train|val|test_meta.parquet       row_id, Timestamp, split, accounts, Attempt ID,
                                            Pattern Type, explained, weight, ...
          fx_table.csv, hubs.csv, data_dictionary.md, build_report.json

Usage:
    python 03_features.py                       # full file, time split (train 1-6 / val 7 / test 8-10)
    python 03_features.py --sample-dir data/processed/sample/time
                                                # same features, restricted to the 02 sample rows,
                                                # with the sample's split and weights -> features_sample/

Feature rules are in src/features.py. All counting in DuckDB. ~1-3 min on the full file.
Needs: pip install duckdb pyarrow pandas
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--joined", default=str(ROOT / "data" / "processed" / "hi_small_joined.parquet"))
ap.add_argument("--out", default=None, help="output folder (default data/processed/features[_sample])")
ap.add_argument("--sample-dir", default=None, help="folder with the 02 time-split sample files")
ap.add_argument("--memory-limit", default="2GB",
                help="DuckDB memory limit (leave room for Python and the operating system)")
ap.add_argument("--threads", type=int, default=2,
                help="DuckDB worker threads; fewer threads reduce peak memory use")
args = ap.parse_args()

joined = Path(args.joined)
if not joined.exists():
    sys.exit(f"{joined} not found - run 01_join_sql.py first.")
out = Path(args.out) if args.out else ROOT / "data" / "processed" / (
    "features_sample" if args.sample_dir else "features")

log = F.Timer()
con = F.connect(tmp_dir=ROOT / "data" / "processed" / "duckdb_tmp",
                memory_limit=args.memory_limit, threads=args.threads)

log(f"reading {joined.name} (rows before {F.CUTOFF})...")
F.build_base(con, joined)
q = lambda sql: con.execute(sql).fetchone()
n_base, n_self, n_pos = q('SELECT COUNT(*), SUM(is_self::INT), SUM("Is Laundering") FROM base')
n_drop = q(f"""SELECT COUNT(*) FROM read_parquet('{joined.as_posix()}')
               WHERE "Timestamp" >= TIMESTAMP '{F.CUTOFF}'""")[0]
log(f"base: {n_base:,} rows kept, {n_drop:,} residue rows dropped, {n_self:,} self-transfers, {n_pos:,} positives")

fx = F.build_fx_table(con)
con.register("fx_df", fx)
con.execute("CREATE OR REPLACE TABLE fx_table AS SELECT * FROM fx_df")
log(f"FX table: {len(fx)} currencies, max spread within a pair {fx.attrs['max_pair_spread']:.4%}")

hubs = F.build_hubs(con)
con.register("hubs_df", hubs)
con.execute("CREATE OR REPLACE TABLE hubs AS SELECT * FROM hubs_df")
log(f"hubs (>= {F.HUB_MIN_COUNTERPARTIES} counterparties, train dates): {len(hubs)}")

F.build_features(con, log=log)

sample_report = None
if args.sample_dir:
    sample_report = F.attach_sample(con, Path(args.sample_dir))
    log(f"sample: {sample_report}")

counts = F.export(con, out)
fx.to_csv(out / "fx_table.csv", index=False)
hubs.to_csv(out / "hubs.csv", index=False)
F.write_data_dictionary(out / "data_dictionary.md")

# ------------------------------------------------------------------ checks
n_feat, n_feat_pos = q('SELECT COUNT(*), SUM("Is Laundering") FROM feat')
nan_cols = [c for c in F.COUNT_FEATURES + F.FLAG_FEATURES + F.AMOUNT_FEATURES
            if q(f'SELECT COUNT(*) FROM feat WHERE "{c}" IS NULL OR isnan("{c}"::DOUBLE)')[0]]
split_per_attempt = q("""SELECT COUNT(*) FROM (SELECT "Attempt ID" FROM feat
                         WHERE "Attempt ID" IS NOT NULL GROUP BY 1 HAVING COUNT(DISTINCT split) > 1)""")[0]
unassigned = q("SELECT COUNT(*) FROM feat WHERE split IS NULL")[0]
dup_key = q(f"""SELECT COUNT(*) FROM (SELECT {', '.join(f'"{c}"' for c in F.ROW_KEY)}
               FROM base GROUP BY ALL HAVING COUNT(*) > 1)""")[0]
expected_rows = n_base - n_self if not args.sample_dir else sample_report["matched"] - sample_report["self_transfers_dropped"]

checks = [
    ("scored rows = kept rows minus self-transfers", n_feat == expected_rows, f"{n_feat:,} vs {expected_rows:,}"),
    ("row key unique", dup_key == 0, f"{dup_key} duplicate keys"),
    ("no NULL / NaN in numeric features", not nan_cols, str(nan_cols)),
    ("every row has a split", unassigned == 0, f"{unassigned}"),
    ("each attempt in one split", split_per_attempt == 0, f"{split_per_attempt} attempts split"),
    ("split row counts add up", sum(c[0] for c in counts.values()) == n_feat, str(counts)),
    ("FX pairs consistent (< 2% spread)", fx.attrs["max_pair_spread"] < 0.02, f"{fx.attrs['max_pair_spread']:.4%}"),
]
if args.sample_dir:
    checks.append(("every sample row matched the full file",
                   sample_report["matched"] == sample_report["sample_rows"],
                   f"{sample_report['matched']:,} / {sample_report['sample_rows']:,}"))

print("\n================ CHECKS ================")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
n_ok = sum(ok for _, ok, _ in checks)
print(f"{n_ok} / {len(checks)} checks passed")

print("\nsplit        rows   positives   rate      weighted rate")
for name in F.SPLIT_DAYS:
    r, p, wr, moved = q(f"""SELECT COUNT(*), SUM("Is Laundering"),
                             SUM(weight * "Is Laundering") / SUM(weight), SUM(scheme_moved::INT)
                             FROM feat WHERE split = '{name}'""")
    print(f"{name:<6}{r:>11,}{p:>11,}   {p / r:.4%}   {wr:.4%}   (moved-scheme rows: {moved:,})")
n_burn, p_burn = q('SELECT COUNT(*), SUM("Is Laundering") FROM feat WHERE NOT history_complete')
print(f"burn-in rows (first {F.LOOKBACK_HOURS} h, history window not full): {n_burn:,} rows, {p_burn:,} positives "
      "- excluded from model fitting by default")
print(f"scored positives: {n_feat_pos:,} of {n_pos:,} kept positives "
      f"({n_pos - n_feat_pos:,} are self-transfers or outside the sample)")

report = {
    "joined": str(joined), "sample_dir": args.sample_dir, "cutoff": F.CUTOFF,
    "split_days": F.SPLIT_DAYS, "train_end_for_learned_tables": F.TRAIN_END,
    "base_rows": n_base, "residue_dropped": n_drop, "self_transfers": n_self,
    "base_positives": n_pos, "scored_rows": n_feat, "scored_positives": n_feat_pos,
    "n_hubs": len(hubs), "lookback_hours": F.LOOKBACK_HOURS, "burn_in_rows": n_burn, "features": F.FEATURES,
    "split_counts": {k: {"rows": v[0], "positives": v[1]} for k, v in counts.items()},
    "sample": sample_report, "checks": {n: bool(ok) for n, ok, _ in checks},
}
(out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
log(f"saved to {out}")
if n_ok != len(checks):
    sys.exit(1)
