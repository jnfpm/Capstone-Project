"""Evaluation for the transaction-scoring models.

Accuracy is meaningless at ~0.1% prevalence. What we report:

* PR-AUC (average precision), always next to the split's prevalence, and ROC-AUC.
* Alert-budget tables: alert the top X% of transfers (by weighted volume) and
  measure precision / recall. The threshold is chosen on VAL and carried to TEST
  unchanged - that is what a deployed system would do.
* Breakdowns of recall by pattern type, explained vs unexplained positives and
  payment format.
* Scheme-level catch rate: the agent starts an investigation from an alert, so
  a scheme counts as caught if at least one of its transfers is alerted.
* Confidence intervals by resampling sender ACCOUNTS (rows of one account are
  not independent).
* Drift: population stability index (PSI) of every feature, train vs val/test.

Every function takes ``w`` (sample weights); pass ones on the full file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


# ------------------------------------------------------------------ headline metrics
def prevalence(y, w) -> float:
    return float(np.average(y, weights=w))


def headline(y, s, w) -> dict:
    ap = average_precision_score(y, s, sample_weight=w)
    p = prevalence(y, w)
    return {"rows": len(y), "positives": int(np.sum(y)), "prevalence": p,
            "pr_auc": ap, "pr_auc_lift": ap / p, "roc_auc": roc_auc_score(y, s, sample_weight=w)}


def compare(models_scores: dict, y, w) -> pd.DataFrame:
    """{name: scores} -> one row per model."""
    return pd.DataFrame([{"model": k, **headline(y, s, w)} for k, s in models_scores.items()]) \
        .sort_values("pr_auc", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ alert budgets
def threshold_for_budget(s, w, budget: float) -> float:
    """Score threshold that alerts ``budget`` (fraction) of the weighted volume."""
    order = np.argsort(-s, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cw, budget, side="left"))
    return float(s[order][min(k, len(s) - 1)])


def at_threshold(y, s, w, thr: float, n_days: float | None = None) -> dict:
    a = s >= thr
    tp = float(w[a & (y == 1)].sum())
    alerts_w = float(w[a].sum())
    pos_w = float(w[y == 1].sum())
    out = {"threshold": thr, "alerts": int(a.sum()), "alerts_weighted": alerts_w,
           "alert_rate": alerts_w / w.sum(),
           "precision": tp / alerts_w if alerts_w else np.nan,
           "recall": tp / pos_w if pos_w else np.nan,
           "true_positives": int((a & (y == 1)).sum())}
    out["lift"] = out["precision"] / prevalence(y, w) if alerts_w else np.nan
    if n_days:
        out["alerts_per_day"] = alerts_w / n_days
    return out


def budget_table(y, s, w, budgets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05),
                 thresholds_from: tuple | None = None, n_days: float | None = None) -> pd.DataFrame:
    """Precision / recall when alerting the top ``budget`` share of transfers.

    ``thresholds_from=(s_val, w_val)`` -> thresholds are set on that (validation)
    data and applied here; otherwise they are set on this data (optimistic).
    """
    rows = []
    for b in budgets:
        thr = threshold_for_budget(*(thresholds_from or (s, w)), b)
        rows.append({"budget": b, **at_threshold(y, s, w, thr, n_days)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ breakdowns
def recall_breakdown(meta: pd.DataFrame, alert: np.ndarray, by: str) -> pd.DataFrame:
    """Recall among positives, grouped by a meta column (positives have weight 1)."""
    pos = meta["y"].to_numpy() == 1
    d = pd.DataFrame({by: meta.loc[pos, by].astype("object").fillna("(none)").to_numpy(),
                      "caught": alert[pos]})
    t = d.groupby(by)["caught"].agg(positives="size", caught="sum")
    t["recall"] = t["caught"] / t["positives"]
    return t.sort_values("positives", ascending=False)


def scheme_catch(meta: pd.DataFrame, alert: np.ndarray) -> pd.DataFrame:
    """One row per attempt: rows, rows alerted, caught (>= 1 alert), first alert position."""
    m = meta.assign(alert=alert)
    m = m[m["Attempt ID"].notna()].sort_values(["Attempt ID", "Timestamp", "row_id"])
    m["pos"] = m.groupby("Attempt ID").cumcount()
    g = m.groupby("Attempt ID")
    out = pd.DataFrame({
        "pattern": g["Pattern Type"].first(),
        "truncated": g["scheme_truncated"].first(),
        "rows": g.size(),
        "rows_alerted": g["alert"].sum(),
        "first_alert_step": m[m.alert].groupby("Attempt ID")["pos"].min(),
    })
    out["caught"] = out["rows_alerted"] > 0
    return out


def scheme_summary(sc: pd.DataFrame) -> pd.DataFrame:
    """Catch rate by pattern type, complete schemes and truncated ones separately."""
    t = sc.groupby(["pattern", "truncated"]).agg(
        schemes=("caught", "size"), caught=("caught", "sum"),
        row_recall=("rows_alerted", "sum"), rows=("rows", "sum"),
        median_first_alert_step=("first_alert_step", "median"))
    t["catch_rate"] = t["caught"] / t["schemes"]
    t["row_recall"] = t["row_recall"] / t["rows"]
    return t.drop(columns="rows")


# ------------------------------------------------------------------ uncertainty
def cluster_bootstrap(y, s, w, groups, metric="pr_auc", B: int = 100, seed: int = 0,
                      budget_thr: float | None = None) -> dict:
    """95% interval by resampling groups (sender accounts) with replacement.

    Implemented by giving every row the multiplicity of its group in the draw,
    so no data is copied. ``metric`` in {"pr_auc", "recall", "precision"}
    (the last two need ``budget_thr``).
    """
    codes, uniq = pd.factorize(pd.Series(groups))
    G = len(uniq)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        mult = np.bincount(rng.integers(0, G, G), minlength=G)[codes]
        ww = w * mult
        keep = ww > 0
        yy, ss, ww = y[keep], s[keep], ww[keep]
        if yy.sum() == 0:
            continue
        if metric == "pr_auc":
            vals.append(average_precision_score(yy, ss, sample_weight=ww))
        else:
            r = at_threshold(yy, ss, ww, budget_thr)
            vals.append(r[metric])
    vals = np.array(vals, float)
    return {"metric": metric, "B": len(vals), "mean": float(np.nanmean(vals)),
            "lo": float(np.nanpercentile(vals, 2.5)), "hi": float(np.nanpercentile(vals, 97.5))}


# ------------------------------------------------------------------ drift
def _psi(p, q, eps=1e-6):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def psi_table(train: pd.DataFrame, w_train, others: dict, numeric, categorical, bins: int = 10) -> pd.DataFrame:
    """PSI of each feature: train vs each other split (weighted).

    Numeric: train quantile bins (ties collapse, so binary flags get 2 bins).
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 big shift.
    """
    rows = []
    for c in numeric:
        x = train[c].to_numpy(float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        edges = np.r_[-np.inf, edges[1:-1], np.inf] if len(edges) > 2 else np.array([-np.inf, edges[0], np.inf])

        def dist(v, ww):
            h = np.histogram(v, bins=edges, weights=ww)[0]
            return h / h.sum()
        p = dist(x, w_train)
        rows.append({"feature": c, **{k: _psi(p, dist(d[c].to_numpy(float), ww)) for k, (d, ww) in others.items()}})
    for c in categorical:
        p = pd.Series(w_train, index=train[c].astype(str)).groupby(level=0).sum()
        p = p / p.sum()
        r = {"feature": c}
        for k, (d, ww) in others.items():
            q = pd.Series(ww, index=d[c].astype(str)).groupby(level=0).sum()
            q = q / q.sum()
            idx = p.index.union(q.index)
            r[k] = _psi(p.reindex(idx, fill_value=0).to_numpy(), q.reindex(idx, fill_value=0).to_numpy())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("feature")
    t["max_psi"] = t.max(axis=1)
    t["flag"] = pd.cut(t["max_psi"], [-1, 0.1, 0.25, np.inf], labels=["stable", "moderate", "big"])
    return t.sort_values("max_psi", ascending=False)


# ------------------------------------------------------------------ plotting
def plot_pr(ax, curves: dict, y, w, title=""):
    """curves = {name: scores}. Draws PR curves with the prevalence line."""
    for name, s in curves.items():
        p, r, _ = precision_recall_curve(y, s, sample_weight=w)
        ap = average_precision_score(y, s, sample_weight=w)
        ax.plot(r, p, label=f"{name} (AP {ap:.4f})", lw=1.5)
    ax.axhline(prevalence(y, w), ls="--", c="grey", lw=1, label=f"prevalence {prevalence(y, w):.4%}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_yscale("log")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.3)
