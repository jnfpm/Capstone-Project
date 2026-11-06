"""Independent check of 03_features.py.

Re-derives everything with plain pandas from hi_small_joined.parquet - no SQL
from src/features.py is reused (only the settings: cutoff, split days, hub
threshold). For N random scored rows every feature is recomputed by brute force
from past rows only and must match exactly.

Usage:
    python 03b_verify_features.py                 # checks data/processed/features
    python 03b_verify_features.py --n 1000 --features-dir data/processed/features_sample
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402  (settings and column lists only)

ap = argparse.ArgumentParser()
ap.add_argument("--joined", default=str(ROOT / "data" / "processed" / "hi_small_joined.parquet"))
ap.add_argument("--features-dir", default=str(ROOT / "data" / "processed" / "features"))
ap.add_argument("--n", type=int, default=1000, help="random rows to recompute")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

joined, fdir = Path(args.joined).as_posix(), Path(args.features_dir)
checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}", flush=True)


# ------------------------------------------------------------------ load outputs
X = {s: pd.read_parquet(fdir / f"{s}_features.parquet") for s in F.SPLIT_DAYS}
M = {s: pd.read_parquet(fdir / f"{s}_meta.parquet") for s in F.SPLIT_DAYS}
fx = pd.read_csv(fdir / "fx_table.csv").set_index("currency")["units_per_usd"]
hubs_out = set(pd.read_csv(fdir / "hubs.csv")["acct"])
feat = pd.concat([X[s].merge(M[s], on="row_id") for s in F.SPLIT_DAYS], ignore_index=True)
print(f"loaded {len(feat):,} scored rows from {fdir}")

# 1. structure
x_cols = set().union(*[set(x.columns) for x in X.values()])
leaks = x_cols & (set(F.LEAK_COLUMNS) - {F.TARGET, "row_id"})   # row_id = join key, dropped before fitting
check("no leak / ID column in X files", not leaks, sorted(leaks))
check("X columns = row_id + FEATURES + target",
      x_cols == set(F.FEATURES) | {"row_id", F.TARGET}, len(x_cols))
num = F.COUNT_FEATURES + F.FLAG_FEATURES + F.AMOUNT_FEATURES
n_nan = int(feat[num].isna().sum().sum()) + int(np.isinf(feat[num].to_numpy(float)).sum())
check("no NaN / inf in numeric features", n_nan == 0, n_nan)
check("no missing categorical values", int(feat[F.CATEGORICAL_FEATURES].isna().sum().sum()) == 0)
check("row_id unique across splits", feat["row_id"].is_unique)
check("split column matches file", all((M[s]["split"] == s).all() for s in F.SPLIT_DAYS))
check("counts never negative", bool((feat[F.COUNT_FEATURES] >= 0).all().all()))
check("windows nested (60m <= 24h <= 48h)",
      bool(((feat.snd_sent_60m <= feat.snd_sent_24h) & (feat.snd_sent_24h <= feat.snd_sent_48h)
            & (feat.rcv_recv_60m <= feat.rcv_recv_24h) & (feat.rcv_recv_24h <= feat.rcv_recv_48h)).all()))

# ------------------------------------------------------------------ the raw file, independently
con = duckdb.connect()
raw = con.execute(f"""
    SELECT "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid",
           "Amount Received", "Payment Currency", "Receiving Currency", "Payment Format",
           "Is Laundering", "Attempt ID", "Bank Name_from", "Bank Name_to",
           "Country_from", "Country_to", "Entity ID_from", "Entity ID_to"
    FROM read_parquet('{joined}')
""").df()
raw["acct_from"] = raw["From Bank"].astype(str) + ":" + raw["Account"].astype(str)
raw["acct_to"] = raw["To Bank"].astype(str) + ":" + raw["Account.1"].astype(str)
cut = raw["Timestamp"] < pd.Timestamp(F.CUTOFF)
truncated = set(raw.loc[~cut, "Attempt ID"].dropna())
raw = raw[cut].reset_index(drop=True)
raw["is_self"] = raw["acct_from"] == raw["acct_to"]
ns = raw[~raw["is_self"]]
print(f"raw: {len(raw):,} rows before cutoff, {int(raw.is_self.sum()):,} self-transfers")

# 2. totals
sample_mode = bool(feat["weight"].ne(1).any())
if not sample_mode:
    check("scored rows = non-self rows", len(feat) == len(ns), f"{len(feat):,} vs {len(ns):,}")
    check("scored positives = non-self positives",
          int(feat[F.TARGET].sum()) == int(ns["Is Laundering"].sum()),
          f"{int(feat[F.TARGET].sum()):,} vs {int(ns['Is Laundering'].sum()):,}")

# 3. split rule, re-derived
def day_split(d):
    for name, (a, b) in F.SPLIT_DAYS.items():
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name


att_last = raw.dropna(subset=["Attempt ID"]).groupby("Attempt ID")["Timestamp"].max().dt.normalize()
day = feat["Timestamp"].dt.normalize()
expected_split = feat["Attempt ID"].map(att_last).fillna(day).map(day_split)
if not sample_mode:
    check("split = own date, or attempt's last date", (expected_split == feat["split"]).all(),
          int((expected_split != feat["split"]).sum()))
t0 = raw["Timestamp"].min()
check("history_complete = at least LOOKBACK_HOURS after the first row",
      (feat["history_complete"] == (feat["Timestamp"] >= t0 + pd.Timedelta(hours=F.LOOKBACK_HOURS))).all())
check("scheme_moved flag correct", (feat["scheme_moved"] == (feat["split"] != day.map(day_split))).all())
check("scheme_truncated flag correct",
      (feat["scheme_truncated"] == feat["Attempt ID"].isin(truncated)).all())
check("each attempt in one split", feat.dropna(subset=["Attempt ID"]).groupby("Attempt ID")["split"].nunique().max() <= 1
      if feat["Attempt ID"].notna().any() else True)
check("train rows never moved (schemes move forward only)", not feat.loc[feat.split == "train", "scheme_moved"].any())

# 4. hubs, re-derived on train dates
tr = ns[ns["Timestamp"] < pd.Timestamp(F.TRAIN_END)]
edges = pd.concat([tr[["acct_from", "acct_to"]].set_axis(["a", "c"], axis=1),
                   tr[["acct_to", "acct_from"]].set_axis(["a", "c"], axis=1)]).drop_duplicates()
deg = edges["a"].value_counts()
hubs_ref = set(deg[deg >= F.HUB_MIN_COUNTERPARTIES].index)
check("hub list matches", hubs_ref == hubs_out, f"{len(hubs_ref)} vs {len(hubs_out)}")
del edges, tr

# 5. FX: direct USD pairs, re-derived
cc = raw[(raw["Payment Currency"] != raw["Receiving Currency"]) & (raw["Amount Paid"] >= 1)
         & (raw["Amount Received"] >= 1) & (raw["Timestamp"] < pd.Timestamp(F.TRAIN_END))]
direct = (cc[cc["Payment Currency"] == "US Dollar"]
          .assign(r=lambda d: d["Amount Received"] / d["Amount Paid"])
          .groupby("Receiving Currency")["r"].median())
fx_err = (fx.reindex(direct.index) / direct - 1).abs().max() if len(direct) else 0.0
check("FX table matches direct USD pairs (< 0.1%)", fx_err < 1e-3, f"max error {fx_err:.2e}")
amt_ref = np.log1p(feat["amount_usd"])
check("log_amount_usd = ln(1 + amount_usd)", np.allclose(amt_ref, feat["log_amount_usd"]))

# ------------------------------------------------------------------ 6. brute-force rows
rng = np.random.default_rng(args.seed)
pick = feat.iloc[np.sort(rng.choice(len(feat), min(args.n, len(feat)), replace=False))].rename(columns={F.TARGET: "y"})
# index the raw rows of the picked accounts only
accts = set(pick.acct_from) | set(pick.acct_to)
ns_from = {k: g for k, g in ns[ns.acct_from.isin(accts)].groupby("acct_from")[["Timestamp", "acct_to"]]}
ns_to = {k: g for k, g in ns[ns.acct_to.isin(accts)].groupby("acct_to")[["Timestamp", "acct_from"]]}
own = ns[ns.acct_from.isin(set(pick.acct_from))]
self_rows = raw[raw["is_self"] & raw.acct_from.isin(accts)]
self_by = {k: g for k, g in self_rows.groupby("acct_from")}
empty_f = pd.DataFrame(columns=["Timestamp", "acct_to"])
empty_t = pd.DataFrame(columns=["Timestamp", "acct_from"])

mismatch = {f: 0 for f in F.COUNT_FEATURES + F.FLAG_FEATURES}
net_bad = 0
ns_ts = np.sort(ns["Timestamp"].to_numpy())
label_bad = 0
for r in pick.itertuples(index=False):
    t, s, v = r.Timestamp, r.acct_from, r.acct_to
    sf = ns_from.get(s, empty_f); st = ns_to.get(s, empty_t)
    vf = ns_from.get(v, empty_f); vt = ns_to.get(v, empty_t)
    lb = t - pd.Timedelta(hours=F.LOOKBACK_HOURS)
    win = lambda d, start: d[(d.Timestamp < t) & (d.Timestamp >= start)]
    sf_w, st_w, vf_w, vt_w = win(sf, lb), win(st, lb), win(vf, lb), win(vt, lb)
    ref = {
        "pair_prev_48h": int((sf_w.acct_to == v).sum()),
        "snd_sent_48h": len(sf_w),
        "snd_recv_48h": len(st_w),
        "snd_sent_24h": len(win(sf, t - pd.Timedelta(hours=24))),
        "snd_sent_60m": len(win(sf, t - pd.Timedelta(minutes=60))),
        "rcv_recv_48h": len(vt_w),
        "rcv_sent_48h": len(vf_w),
        "rcv_recv_24h": len(win(vt, t - pd.Timedelta(hours=24))),
        "rcv_recv_60m": len(win(vt, t - pd.Timedelta(minutes=60))),
        "reverse_pair_48h": int((vf_w.acct_to == s).any()),
        "snd_quiet_48h": int(len(sf_w) == 0 and len(st_w) == 0),
        "rcv_quiet_48h": int(len(vt_w) == 0 and len(vf_w) == 0),
        "snd_hub": int(s in hubs_ref), "rcv_hub": int(v in hubs_ref),
    }
    ref["new_pair_48h"] = int(ref["pair_prev_48h"] == 0)
    for col, delta in (("net_48h", pd.Timedelta(hours=F.LOOKBACK_HOURS)), ("net_24h", pd.Timedelta(hours=24)),
                       ("net_60m", pd.Timedelta(minutes=60))):
        lo, hi = np.searchsorted(ns_ts, np.datetime64(t - delta), "left"), np.searchsorted(ns_ts, np.datetime64(t), "left")
        net_bad += int(getattr(r, col) != hi - lo)
    # the transfer itself, from raw
    me = own[(own.Timestamp == t) & (own.acct_from == s) & (own.acct_to == v)
            & (own["Payment Format"] == r._asdict()["payment_format"])]
    me = me[np.isclose(np.log1p(me["Amount Paid"] / me["Payment Currency"].map(fx)), r.log_amount_usd)]
    row = me.iloc[0]
    label_bad += int(len(me) != 1 or row["Is Laundering"] != r.y)
    ref["cross_border"] = int(row.Country_from != row.Country_to)
    ref["cross_bank"] = int(row["Bank Name_from"] != row["Bank Name_to"])
    ref["same_entity"] = int(row["Entity ID_from"] == row["Entity ID_to"])
    ref["currency_changes"] = int(row["Payment Currency"] != row["Receiving Currency"])
    cv = self_by.get(s)
    if cv is None:
        ref["conversion_before"] = 0
    else:
        cv = cv[(cv.Timestamp == t) & (cv["Payment Currency"] != cv["Receiving Currency"])
                & (cv["Receiving Currency"] == row["Payment Currency"])]
        ref["conversion_before"] = int(np.isclose(cv["Amount Received"], row["Amount Paid"],
                                                  rtol=F.CONVERSION_REL_TOL, atol=0).any())
    rd = r._asdict()
    for f in mismatch:
        mismatch[f] += int(rd[f] != ref[f])

check("label and amount of each picked row match the raw file", label_bad == 0, label_bad)
check("network volume (net_48h / 24h / 60m) matches brute force", net_bad == 0, net_bad)
bad = {f: n for f, n in mismatch.items() if n}
check(f"brute-force recompute of {len(pick):,} random rows: every feature matches", not bad, bad or "")

print(f"\n{sum(checks)} / {len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
