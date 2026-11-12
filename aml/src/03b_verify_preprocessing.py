"""Independent check of 03_PREPROCESSING.ipynb (preprocessing on the subsample).

Re-derives everything with pyarrow / pandas / numpy - none of the notebook's SQL
is reused and DuckDB is not used at all.

  A. rows        re-numbers the full joined file, re-matches every subsample row
                 (time and group folders) to it, checks the scored rows are exactly
                 the subsample's non-self rows, labels, weights, seed flags
  B. splits      split = the subsample's time split and follows the scheme rule,
                 group split = the group folder, moved / truncated / explained /
                 history flags
  C. train-only  FX table (subsample train split), hub list (history, train dates),
                 weighted caps, weighted category levels
  D. brute force every feature of N random rows, plus laundering rows and rows where
                 each flag is 1, recomputed from strictly earlier rows of the history source

Run after the notebook (its last section runs this script), or by hand:
    python src/03b_verify_preprocessing.py [--rows 1000]
Expected last line: "N / N checks passed". Peak memory about 4 GB on the full file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "eda_utils.py").is_file())   # the aml_pipeline folder
KEY = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
       "Amount Paid", "Amount Received", "Payment Format", "Payment Currency", "Receiving Currency"]
ROW_COLS = ["Bank Name_from", "Bank Name_to", "Entity ID_from", "Entity ID_to",
            "Country_from", "Country_to", "Entity Type_from", "Entity Type_to"]

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def read(path, columns=None) -> pd.DataFrame:
    """Parquet -> pandas with strings as categoricals (a fraction of the memory)."""
    return pq.read_table(path, columns=columns).to_pandas(strings_to_categorical=True)


def as_cat(s: pd.Series, categories) -> pd.Series:
    """Categorical with the given (lexically sorted) categories, so codes sort like strings."""
    return s.astype(str).astype(pd.CategoricalDtype(categories, ordered=True))


def date_split(dates: pd.Series, split_days: dict) -> np.ndarray:
    out = np.full(len(dates), None, dtype=object)
    for name, (a, b) in split_days.items():
        out[((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))).to_numpy()] = name
    return out


def wquantile(values: np.ndarray, weights: np.ndarray, q: float):
    """Weighted quantile, inverted CDF: smallest value whose cumulative weight reaches q."""
    o = np.argsort(values, kind="stable")
    v, cw = values[o], np.cumsum(weights[o])
    return v[np.searchsorted(cw, q * cw[-1] - 1e-9 * cw[-1])]


def label(v) -> str:
    return "Unknown" if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ("nan", "None") \
        else str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default=str(ROOT / "data" / "processed" / "hi_small_joined.parquet"))
    ap.add_argument("--sample", default=str(ROOT / "data" / "processed" / "sample"))
    ap.add_argument("--dir", default=str(ROOT / "data" / "processed" / "preprocessed"))
    ap.add_argument("--rows", type=int, default=1000, help="random rows to brute-force")
    ap.add_argument("--positives", type=int, default=300, help="extra laundering rows to brute-force")
    ap.add_argument("--per-flag", type=int, default=100, help="extra rows where each 0/1 feature is 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    d, sdir = Path(args.dir), Path(args.sample)
    art = json.loads((d / "artifacts.json").read_text(encoding="utf-8"))
    report = json.loads((d / "build_report.json").read_text(encoding="utf-8"))
    S = art["settings"]
    target, features = art["target"], art["features"]
    split_days = S["split_days"]
    cutoff, train_end = pd.Timestamp(S["cutoff"]), pd.Timestamp(S["train_end"])
    lb_h = S["lookback_h"]
    fx = art["fx_units_per_usd"]
    hubs = set(art["hubs"])
    from_sample = S["history_from"] == "sample"
    print(f"history source: {S['history_from']}")

    # ------------------------------------------------------------------ full file
    print("loading the joined file ...", flush=True)
    raw = read(args.joined, KEY + [target, "Attempt ID"] + ROW_COLS)
    n_joined = len(raw)
    raw = raw[raw["Timestamp"] < cutoff]
    acc_cats = sorted(set(raw["Account"].astype("category").cat.categories.astype(str))
                      | set(raw["Account.1"].astype("category").cat.categories.astype(str)))
    raw["Account"] = as_cat(raw["Account"], acc_cats)
    raw["Account.1"] = as_cat(raw["Account.1"], acc_cats)
    cats = {}
    for c in ["Payment Format", "Payment Currency", "Receiving Currency"]:
        cats[c] = sorted(raw[c].astype("category").cat.categories.astype(str))
        raw[c] = as_cat(raw[c], cats[c])
    raw = raw.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    raw["row_id"] = np.arange(1, len(raw) + 1, dtype=np.int64)
    mult = len(acc_cats) + 1
    raw["af"] = raw["From Bank"].astype(np.int64) * mult + raw["Account"].cat.codes.astype(np.int64)
    raw["at"] = raw["To Bank"].astype(np.int64) * mult + raw["Account.1"].cat.codes.astype(np.int64)
    raw["is_self"] = raw["af"] == raw["at"]

    def acct_str(i: int) -> str:
        return f"{i // mult}:{acc_cats[i % mult]}"

    # ------------------------------------------------------------------ subsample
    print("matching the subsample ...", flush=True)

    def load_folder(mode, extra):
        parts = []
        for s in split_days:
            df = read(sdir / mode / f"{s}.parquet", KEY + extra)
            df["s_split"] = s
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df["Timestamp"] = df["Timestamp"].astype(raw["Timestamp"].dtype)
        df["Account"] = as_cat(df["Account"], acc_cats)
        df["Account.1"] = as_cat(df["Account.1"], acc_cats)
        for c, cs in cats.items():
            df[c] = as_cat(df[c], cs)
        for c in ["From Bank", "To Bank"]:
            df[c] = df[c].astype(np.int64)
        m = df.merge(raw[KEY + ["row_id"]], on=KEY, how="left")
        return m.drop(columns=KEY)

    st = load_folder("time", ["weight", "seed_from", "seed_to", "scheme_truncated"])
    sg = load_folder("group", [])
    n_s = len(st)

    # ================================================================== A. rows
    print("\nA. rows", flush=True)
    check("every subsample row (time folder) found in the joined file exactly once",
          st["row_id"].notna().all() and st["row_id"].is_unique, f"{n_s:,} rows")
    check("group folder holds the same rows", len(sg) == n_s and sg["row_id"].notna().all()
          and set(sg["row_id"]) == set(st["row_id"]))
    st["row_id"] = st["row_id"].astype(np.int64)
    sg["row_id"] = sg["row_id"].astype(np.int64)
    s_pos = st["row_id"].to_numpy() - 1
    s_self = raw["is_self"].to_numpy()[s_pos]
    check("the subsample holds every laundering row of the full file",
          int(raw[target].to_numpy()[s_pos].sum()) == int(raw[target].sum()), f"{int(raw[target].sum()):,}")

    X = pd.concat([read(d / f"{s}_features.parquet").assign(_split=s) for s in split_days], ignore_index=True)
    M = pd.concat([read(d / f"{s}_meta.parquet") for s in split_days], ignore_index=True)
    exp_x = ["row_id"] + features + [target]
    check("feature files have exactly row_id + features + target",
          all(pq.read_schema(d / f"{s}_features.parquet").names == exp_x for s in split_days))
    leak = set(features) & (set(art["leak_columns"]) | (set(M.columns) - {"row_id"}))
    check("no leak / meta column among features", not leak, ", ".join(sorted(leak)))
    check("no missing values in X", int(X[features].isna().sum().sum()) == 0)
    check("features and meta files have the same rows", X["row_id"].is_unique and
          (X["row_id"].to_numpy() == M["row_id"].to_numpy()).all())
    want = np.sort(st.loc[~s_self, "row_id"].to_numpy())
    same_rows = len(X) == len(want) and np.array_equal(np.sort(X["row_id"].to_numpy()), want)
    check("scored rows = subsample rows minus self-transfers (bank AND account)", same_rows,
          f"{len(X):,} vs {len(want):,}")
    if not same_rows:
        print(f"\n{sum(ok for _, ok, _ in results)} / {len(results)} checks passed")
        return 1
    pos = X["row_id"].to_numpy() - 1
    check("labels equal the raw rows' labels", (raw[target].to_numpy()[pos] == X[target].to_numpy()).all(),
          f"{int(X[target].sum()):,} laundering rows")
    check("rows dropped = rows on or after the cutoff",
          n_joined - len(raw) == report["inputs"]["residue_rows_dropped"], f"{n_joined - len(raw):,}")
    sx = st.set_index("row_id").loc[X["row_id"].to_numpy()]
    check("weight, seed_from, seed_to equal the subsample's",
          np.allclose(M["weight"].to_numpy(dtype=float), sx["weight"].to_numpy(dtype=float))
          and (M["seed_from"].to_numpy() == sx["seed_from"].to_numpy()).all()
          and (M["seed_to"].to_numpy() == sx["seed_to"].to_numpy()).all())
    af, at = raw["af"].to_numpy()[pos], raw["at"].to_numpy()[pos]
    ok_keys = all(M["acct_from"].iat[i] == acct_str(af[i]) and M["acct_to"].iat[i] == acct_str(at[i])
                  for i in np.random.default_rng(1).choice(len(M), min(5000, len(M)), replace=False))
    check("meta account keys = '<bank id>:<account>' of the raw row", ok_keys)

    # ================================================================== B. splits
    print("\nB. splits", flush=True)
    msplit = M["split"].astype(str).to_numpy()
    check("split = the subsample's time split", (msplit == sx["s_split"].to_numpy()).all()
          and (msplit == X["_split"].to_numpy()).all())
    att = raw["Attempt ID"].to_numpy()[pos]
    has = pd.notna(att)
    own = date_split(M["Timestamp"].dt.normalize(), split_days)
    check("own_date_split matches the row date", (own == M["own_date_split"].astype(str).to_numpy()).all())
    ar = raw.dropna(subset=["Attempt ID"])
    last = ar.groupby("Attempt ID")["Timestamp"].max()
    att_split = dict(zip(last.index, date_split(last.dt.normalize(), split_days)))
    exp_split = own.copy()
    exp_split[has] = [att_split[a] for a in att[has]]
    check("split follows the rule: own date, or the date of the attempt's last row", (exp_split == msplit).all())
    gx = sg.set_index("row_id").loc[X["row_id"].to_numpy(), "s_split"].to_numpy()
    gsplit = M["group_split"].astype(str).to_numpy()
    check("group_split = the subsample's group split", (gsplit == gx).all())
    per_att = pd.DataFrame({"a": att[has], "s": msplit[has], "g": gsplit[has]})
    check("every attempt in exactly one time split", int(per_att.groupby("a")["s"].nunique().max()) == 1)
    check("every attempt in exactly one group split", int(per_att.groupby("a")["g"].nunique().max()) == 1)
    check("scheme_moved = split differs from own date", (M["scheme_moved"].to_numpy() == (msplit != own)).all())
    check("scheme_truncated = the subsample's flag",
          (M["scheme_truncated"].to_numpy() == sx["scheme_truncated"].to_numpy().astype(bool)).all())
    check("explained = row belongs to an attempt", (M["explained"].to_numpy() == has).all())
    hc = (M["Timestamp"] >= raw["Timestamp"].min() + pd.Timedelta(hours=lb_h)).to_numpy()
    check("history_complete = at least 48 h after the first timestamp", (hc == M["history_complete"].to_numpy()).all())

    # ================================================================== C. learned on train
    print("\nC. train-only statistics", flush=True)
    tr_pos = st.loc[st["s_split"] == "train", "row_id"].to_numpy() - 1
    tr = raw.iloc[tr_pos]
    pc, rc = tr["Payment Currency"].astype(str), tr["Receiving Currency"].astype(str)
    cc = tr[((pc != rc) & (tr["Amount Paid"] > 0) & (tr["Amount Received"] > 0)).to_numpy()]
    cc = pd.DataFrame({"pay": cc["Payment Currency"].astype(str), "rcv": cc["Receiving Currency"].astype(str),
                       "r": cc["Amount Received"] / cc["Amount Paid"],
                       "big": (cc["Amount Paid"] >= 1) & (cc["Amount Received"] >= 1),
                       # amounts below 0.001 (tiny Bitcoin) carry too few digits: rounding alone moves the ratio
                       "precise": (cc["Amount Paid"] >= 1e-3) & (cc["Amount Received"] >= 1e-3)})
    big = cc[cc["big"]].groupby(["pay", "rcv"])["r"].median()
    small = cc[cc["precise"]].groupby(["pay", "rcv"])["r"].median()
    rates = pd.concat([big, small[~small.index.isin(big.index)]])
    errs = pd.Series({(a, b): abs(fx[b] / fx[a] / r - 1) for (a, b), r in rates.items() if a in fx and b in fx})
    worst = errs.idxmax() if len(errs) else None
    check("FX table agrees with pair medians of the subsample's train split", len(errs) > 0 and errs.max() < 0.02,
          f"max rel. difference {errs.max() if len(errs) else float('nan'):.2e} over {len(errs)} pairs "
          f"(worst: {worst[0]} -> {worst[1]})" if worst else "no pairs")

    # history source: full file, or only the subsample rows
    in_hist = np.ones(len(raw), dtype=bool)
    if from_sample:
        in_hist[:] = False
        in_hist[s_pos] = True
    is_self = raw["is_self"].to_numpy()
    T = raw["Timestamp"].to_numpy()
    AF, AT = raw["af"].to_numpy(), raw["at"].to_numpy()
    hn = np.flatnonzero(in_hist & ~is_self & (T < np.datetime64(train_end)))
    e = np.unique(np.concatenate([np.stack([AF[hn], AT[hn]], 1), np.stack([AT[hn], AF[hn]], 1)]), axis=0)
    acc_ids, deg = np.unique(e[:, 0], return_counts=True)
    my_hubs = {acct_str(i) for i in acc_ids[deg >= S["hub_min_counterparties"]]}
    check("hub list = >= 500 counterparties in the history on train dates", my_hubs == hubs, f"{len(hubs)} hubs")
    del tr, cc, pc, rc, e, hn

    trm = (X["_split"] == "train").to_numpy()
    xt = trm & M["history_complete"].to_numpy()
    wts = M["weight"].to_numpy(dtype=float)
    bad_caps = []
    for f, cap in art["caps"].items():
        qv = wquantile(X[f].to_numpy()[xt], wts[xt], S["cap_quantile"])
        if not (X[f].max() <= cap and (qv == cap or (cap == 1 and qv <= 1))):
            bad_caps.append(f"{f} (cap {cap}, weighted quantile {qv})")
    check("caps = weighted train 99.9th percentile, nothing above them", not bad_caps, ", ".join(bad_caps))
    bad_lv = []
    for c, lv in art["category_levels"].items():
        vol = pd.Series(wts[trm]).groupby(X.loc[trm, c].astype(str).to_numpy()).sum()
        if any(vol.get(v, 0) < S["category_min_weight"][c] for v in lv) or \
                not set(X[c].astype(str).unique()) <= set(lv) | {"Other"}:
            bad_lv.append(c)
    check("category levels: enough weighted train volume, val/test map to them or 'Other'", not bad_lv,
          ", ".join(bad_lv))

    # ================================================================== D. brute force
    print(f"\nD. brute force ({args.rows} random rows + up to {args.positives} laundering rows "
          f"+ up to {args.per_flag} rows per flag)", flush=True)
    rng = np.random.default_rng(args.seed)
    pos_rows = np.flatnonzero(X[target].to_numpy() == 1)
    picks = [rng.choice(len(X), min(args.rows, len(X)), replace=False),
             rng.choice(pos_rows, min(args.positives, len(pos_rows)), replace=False)]
    rare = list(art["feature_groups"]["flag"]) + [f for f in ("sg_paths_48h", "cycle3_48h") if f in X]
    for f in rare:                                   # rare cases (reverse pair, cycles, pass-through ...) always covered
        on = np.flatnonzero(X[f].to_numpy() >= 1)
        picks.append(rng.choice(on, min(args.per_flag, len(on)), replace=False))
    pick = np.unique(np.concatenate(picks))
    smp = raw.iloc[pos[pick]]
    xs = X.iloc[pick].set_index("row_id")
    del X, M, st, sg, sx

    rcv_fx = raw["Receiving Currency"].astype(str).map(fx).to_numpy(dtype=float)
    USD_IN = raw["Amount Received"].to_numpy() / rcv_fx
    nsi = np.flatnonzero(in_hist & ~is_self)
    by_f = nsi[np.argsort(AF[nsi], kind="stable")]
    by_t = nsi[np.argsort(AT[nsi], kind="stable")]
    key_f, key_t = AF[by_f], AT[by_t]
    si = np.flatnonzero(in_hist & is_self)
    self_f = si[np.argsort(AF[si], kind="stable")]
    key_s = AF[self_f]
    PC = raw["Payment Currency"].astype(str).to_numpy()
    RC = raw["Receiving Currency"].astype(str).to_numpy()
    AR = raw["Amount Received"].to_numpy()
    USD_OUT = raw["Amount Paid"].to_numpy() / raw["Payment Currency"].astype(str).map(fx).to_numpy(dtype=float)
    FMT = raw["Payment Format"].astype(str).to_numpy()
    import math

    def rows(order, keys, k):
        lo, hi = np.searchsorted(keys, k, "left"), np.searchsorted(keys, k, "right")
        return order[lo:hi]

    cap, lv = art["caps"], art["category_levels"]
    one_h = np.timedelta64(3600, "s")

    def cat(c, v):
        v = label(v)
        return v if v in lv[c] else "Other"

    mism, exp = {}, {}
    for r in smp.to_dict("records"):
        t, A, B = np.datetime64(r["Timestamp"]), r["af"], r["at"]
        out, inn = rows(by_f, key_f, A), rows(by_t, key_t, A)
        bout, binn = rows(by_f, key_f, B), rows(by_t, key_t, B)

        def w(ix, h):
            ts = T[ix]
            return ix[(ts < t) & (ts >= t - h * one_h)]

        o48, i48, bo48, bi48 = w(out, lb_h), w(inn, lb_h), w(bout, lb_h), w(binn, lb_h)
        pay_cur, rcv_cur, paid = str(r["Payment Currency"]), str(r["Receiving Currency"]), r["Amount Paid"]
        usd = paid / fx[pay_cur]
        prev_in = inn[T[inn] < t]
        last_in = T[prev_in].max() if len(prev_in) else None
        pt = 0
        if last_in is not None and last_in >= t - S["passthrough_h"] * one_h:
            ins = USD_IN[prev_in[T[prev_in] == last_in]]
            pt = int(np.any(np.abs(usd / ins - 1) <= S["passthrough_tol"]))
        sv = rows(self_f, key_s, A)
        sv = sv[(T[sv] == t) & (PC[sv] != RC[sv]) & (RC[sv] == pay_cur)]
        conv = int(np.any(np.abs(AR[sv] - paid) <= S["conversion_rel_tol"] * max(abs(paid), 1e-12)))
        pair = int((AT[o48] == B).sum())
        cf, ct = label(r["Country_from"]), label(r["Country_to"])
        exp = {
            "snd_sent_48h": len(o48), "snd_recv_48h": len(i48),
            "rcv_recv_48h": len(bi48), "rcv_sent_48h": len(bo48),
            "snd_cpty_48h": len(np.unique(AT[o48])), "rcv_cpty_48h": len(np.unique(AF[bi48])),
            "pair_prev_48h": pair,
            "snd_sent_24h": len(w(out, 24)), "snd_sent_60m": len(w(out, 1)),
            "rcv_recv_24h": len(w(binn, 24)), "rcv_recv_60m": len(w(binn, 1)),
            "snd_dwell_h": min((t - last_in) / one_h, S["passthrough_h"])
            if last_in is not None else S["passthrough_h"],
            "new_pair_48h": int(pair == 0),
            "reverse_pair_48h": int((AT[bo48] == A).any()),
            "snd_quiet_48h": int(len(o48) == 0 and len(i48) == 0),
            "rcv_quiet_48h": int(len(bi48) == 0 and len(bo48) == 0),
            "passthrough": pt, "conversion_before": conv,
            "snd_hub": int(acct_str(A) in hubs), "rcv_hub": int(acct_str(B) in hubs),
            "cross_bank": int(label(r["Bank Name_from"]) != label(r["Bank Name_to"])),
            "cross_border": int(cf != ct),
            "same_entity": int(pd.notna(r["Entity ID_from"]) and str(r["Entity ID_from"]) == str(r["Entity ID_to"])),
            "currency_changes": int(pay_cur != rcv_cur),
            "log_amount_usd": float(np.log1p(usd)),
            # graph shape and sender behaviour (EDA B16b), straight from the definitions
            "snd_cpty_6h": len(np.unique(AT[w(out, 6)])), "rcv_cpty_6h": len(np.unique(AF[w(binn, 6)])),
            "snd_in_cpty_48h": len(np.unique(AF[i48])), "rcv_out_cpty_48h": len(np.unique(AT[bo48])),
            "sg_paths_48h": len((set(AT[o48].tolist()) & set(AF[bi48].tolist())) - {A, B}),
            "cycle3_48h": len((set(AT[bo48].tolist()) & set(AF[i48].tolist())) - {A, B}),
            "snd_recv_usd_24h": float(np.log1p(math.fsum(USD_IN[w(inn, 24)].tolist()))),
            "snd_amt_vs_mean_48h": float(np.log1p(usd) - np.log1p(math.fsum(USD_OUT[o48].tolist()) / len(o48)))
            if len(o48) else 0.0,
            "snd_new_format_48h": int(len(o48) > 0 and not (FMT[o48] == str(r["Payment Format"])).any()),
            "payment_format": cat("payment_format", r["Payment Format"]),
            "payment_currency": cat("payment_currency", pay_cur),
            "country_from": cat("country_from", cf), "country_to": cat("country_to", ct),
            "etype_from": cat("etype_from", r["Entity Type_from"]),
            "etype_to": cat("etype_to", r["Entity Type_to"]),
        }
        for f in exp:
            if f in cap:
                exp[f] = min(exp[f], cap[f])
        got = xs.loc[r["row_id"]]
        for f in features:
            if f not in exp:
                continue
            v, g = exp[f], got[f]
            tol = 1e-6 if f in ("snd_recv_usd_24h", "snd_amt_vs_mean_48h") else 1e-9     # running sums vs fsum
            same = (str(g) == v) if isinstance(v, str) else abs(float(g) - float(v)) <= tol * max(1.0, abs(float(v)))
            if not same:
                mism.setdefault(f, []).append((int(r["row_id"]), g, v))
    missing_f = set(features) - set(exp)
    check("brute force covers every feature", not missing_f, ", ".join(sorted(missing_f)))
    for f in features:
        bad = mism.get(f, [])
        check(f"brute force: {f}", not bad,
              f"{len(bad)} of {len(smp)} differ, e.g. row {bad[0][0]}: got {bad[0][1]}, expected {bad[0][2]}"
              if bad else f"{len(smp)} rows")

    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok} / {len(results)} checks passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
