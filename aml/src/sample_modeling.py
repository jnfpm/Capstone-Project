"""Models for transaction scoring on the SUBSAMPLE (03_PREPROCESSING.ipynb outputs).

Used by ``04_MODELLING.ipynb``; the saved bundle is loaded by the agent layer, so
everything that gets pickled lives here, not in the notebook.

Data rules (EDA_HI_Small_v2, D2-D6, D10)
---------------------------------------
* Rows: the subsample (02_subsample_split.py), self-transfers removed (03).
* ``weight`` (meta) = 1 / inclusion probability. Every metric uses it, so numbers
  estimate the full 1-10 Sept population. Fitting may or may not use it - chosen
  on validation (``WEIGHTINGS``).
* Fit rows: train split, ``history_complete`` (the 48 h windows are full) and
  *reachable* (below).
* Evaluation rows: rows dated inside the split (``own_date_split``; earlier rows of
  schemes moved into the split are all positives with no negatives from their days)
  and reachable.
* Reachable: at least one side is a seed account. Hub accounts are never seeds, so
  a NORMAL transfer between two hubs can never enter the subsample, while a
  LAUNDERING one always does (laundering rows are kept with probability 1). In the
  subsample, "both sides not seeds" is therefore 100% laundering - a pure artefact
  of the sampling. Those rows are left out of fitting and of the weighted metrics
  (their normal counterparts have weight 0, i.e. are missing) and reported apart.

Models
------
* ``RuleModel``   best single feature, direction chosen on train.
* ``LogRegModel`` logistic regression: log counts, flags, amount buckets, one-hot.
* ``LGBMModel``   LightGBM with native categoricals, early-stopped on weighted
                  validation PR-AUC.
Alerts: each day, the top ``budget`` share of that day's (weighted) transfers.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

SPLITS = ("train", "val", "test")
_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
_KBINS_KW = {"quantile_method": "averaged_inverted_cdf"} if _SK >= (1, 7) else {}


# ================================================================== data
@dataclass
class Data:
    X: pd.DataFrame
    y: np.ndarray
    w: np.ndarray
    meta: pd.DataFrame

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "Data":
        m = np.asarray(mask, bool)
        return Data(self.X[m].reset_index(drop=True), self.y[m], self.w[m], self.meta[m].reset_index(drop=True))


def load_artifacts(pre_dir: Path) -> dict:
    return json.loads((Path(pre_dir) / "artifacts.json").read_text(encoding="utf-8"))


def load_splits(pre_dir: Path, art: dict) -> dict:
    """{split: Data} from 03's {split}_features / {split}_meta parquet files."""
    pre_dir = Path(pre_dir)
    out = {}
    for s in SPLITS:
        xf = pd.read_parquet(pre_dir / f"{s}_features.parquet")
        mf = pd.read_parquet(pre_dir / f"{s}_meta.parquet")
        if not (xf["row_id"].to_numpy() == mf["row_id"].to_numpy()).all():
            raise ValueError(f"{s}: features and meta are not row-aligned")
        y = xf.pop(art["target"]).to_numpy().astype(np.int8)
        X = cast_categories(xf.drop(columns="row_id"), art)
        assert_no_leak(X, art)
        mf["y"] = y
        mf["reachable"] = (mf["seed_from"].to_numpy() == 1) | (mf["seed_to"].to_numpy() == 1)
        out[s] = Data(X=X, y=y, w=mf["weight"].to_numpy(float), meta=mf)
    return out


def cast_categories(X: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Text categories -> pandas Categorical with the TRAIN levels + 'Other' (same codes in every split)."""
    X = X.copy()
    for c, levels in art["category_levels"].items():
        if c in X:
            v = X[c].astype(str)
            cats = list(levels) + ["Other"]
            X[c] = pd.Categorical(v.where(v.isin(levels), "Other"), categories=cats)
    return X


def assert_no_leak(X: pd.DataFrame, art: dict) -> None:
    bad = sorted(set(X.columns) & set(art["leak_columns"]))
    if bad:
        raise ValueError(f"leak / ID columns in X: {bad}")


def fit_mask(d: Data, warmup_h: int = 48, t_start=None) -> np.ndarray:
    """Rows a model is FITTED on: reachable, and at least ``warmup_h`` hours after the start of the data
    (48 = full 48 h history = ``history_complete``; less keeps more rows whose windows are partly empty)."""
    reach = d.meta["reachable"].to_numpy(bool)
    if warmup_h >= 48 or t_start is None:
        return d.meta["history_complete"].to_numpy(bool) & reach
    return (d.meta["Timestamp"] >= pd.Timestamp(t_start) + pd.Timedelta(hours=warmup_h)).to_numpy() & reach


def eval_mask(d: Data, split: str) -> np.ndarray:
    """Rows a model is SCORED on: dated inside the split (own_date_split) and reachable."""
    return (d.meta["own_date_split"].astype(str).to_numpy() == split) & d.meta["reachable"].to_numpy(bool)


def feature_sets(art: dict) -> dict:
    f = art["features"]
    g = art["feature_groups"]
    return {"all": list(f),
            "no_format": [x for x in f if x not in g["format"]],
            "no_generator": [x for x in f if x not in g["generator"]]}


def weighted_ap(y, s, w) -> float:
    return float(average_precision_score(y, s, sample_weight=w))


# ================================================================== rule baseline
def single_feature_table(train: Data, val: Data, cols) -> pd.DataFrame:
    """Weighted PR-AUC of each numeric feature used alone; direction chosen on train."""
    base_tr, base_va = np.average(train.y, weights=train.w), np.average(val.y, weights=val.w)
    rows = []
    for c in cols:
        s_tr = train.X[c].to_numpy(float)
        up, down = weighted_ap(train.y, s_tr, train.w), weighted_ap(train.y, -s_tr, train.w)
        sign = 1 if up >= down else -1
        ap_va = weighted_ap(val.y, sign * val.X[c].to_numpy(float), val.w)
        rows.append({"feature": c, "direction": "higher" if sign > 0 else "lower",
                     "train_ap": max(up, down), "train_lift": max(up, down) / base_tr,
                     "val_ap": ap_va, "val_lift": ap_va / base_va})
    return pd.DataFrame(rows).sort_values("train_ap", ascending=False).reset_index(drop=True)


@dataclass
class RuleModel:
    feature: str
    sign: int
    features: list = field(default_factory=list)
    name: str = "rule"

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        tb = X["log_amount_usd"].to_numpy(float) * 1e-6 if "log_amount_usd" in X else 0
        return self.sign * X[self.feature].to_numpy(float) + tb


def fit_rule(table: pd.DataFrame) -> RuleModel:
    """Best feature on TRAIN (the table is sorted by train AP)."""
    best = table.iloc[0]
    return RuleModel(best["feature"], 1 if best["direction"] == "higher" else -1, [best["feature"]])


# ================================================================== logistic regression
def correlated_to_drop(X: pd.DataFrame, cols, threshold: float = 0.85, max_rows: int = 300_000,
                       seed: int = 0) -> tuple:
    """Greedy: from the most correlated pair down, drop the second column (Spearman on train)."""
    Xs = X[cols].sample(min(len(X), max_rows), random_state=seed)
    c = Xs.rank().corr().abs()
    pairs = (c.where(np.triu(np.ones(c.shape, bool), 1)).stack().rename("abs_spearman").reset_index()
             .set_axis(["a", "b", "abs_spearman"], axis=1).sort_values("abs_spearman", ascending=False))
    drop = []
    for r in pairs[pairs.abs_spearman > threshold].itertuples():
        if r.a not in drop and r.b not in drop:
            drop.append(r.b)
    return drop, pairs.reset_index(drop=True)


@dataclass
class LogRegModel:
    features: list
    pipeline: Pipeline
    name: str = "logreg"

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]


def fit_logreg(train: Data, features: list, art: dict, drop: list, weighting: str = "sample",
               C: float = 1.0, seed: int = 0) -> LogRegModel:
    g = art["feature_groups"]
    feats = [f for f in features if f not in drop]
    counts = [f for f in feats if f in g["count"]]
    times = [f for f in feats if f in g["time"]]
    flags = [f for f in feats if f in g["flag"]]
    amount = [f for f in feats if f in g["amount"]]
    cats = [f for f in feats if f in g["categorical"]]
    pre = ColumnTransformer([
        ("counts", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), counts),
        ("time", StandardScaler(), times),
        ("flags", "passthrough", flags),
        # EDA B6b: risk by amount is not monotonic -> buckets, not a slope
        ("amount", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile", **_KBINS_KW), amount),
        ("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cats),
    ], sparse_threshold=1.0)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                                              random_state=seed))])
    w = train.w if weighting != "none" else np.ones(len(train))
    pipe.fit(train.X[feats], train.y, clf__sample_weight=w)
    return LogRegModel(feats, pipe)


def logreg_coefficients(model: LogRegModel) -> pd.DataFrame:
    names = model.pipeline.named_steps["pre"].get_feature_names_out()
    coef = model.pipeline.named_steps["clf"].coef_.ravel()
    return (pd.DataFrame({"term": names, "coef": coef}).assign(a=lambda d: d.coef.abs())
            .sort_values("a", ascending=False).drop(columns="a").reset_index(drop=True))


# ================================================================== LightGBM
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=20,
    min_child_weight=1e-3, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, cat_smooth=20, min_data_per_group=50, max_cat_threshold=32,
    n_estimators=3000, verbose=-1,
)
# How the TRAINING rows are weighted (metrics are always weighted):
#   none        - every subsample row counts once. The subsample is enriched for laundering
#                 (about 1 in 165 rows instead of 1 in 1,100), which acts as undersampling.
#   sample      - the sample weights: the fit sees the true class balance (~1:1,100).
#   sample_sqrt - sample weights, positives up-weighted by sqrt(weighted negatives / positives).
WEIGHTINGS = ("none", "sample", "sample_sqrt")


@dataclass
class LGBMModel:
    features: list
    booster: lgb.LGBMClassifier
    weighting: str
    best_iteration: int = 0
    params: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    name: str = "lgbm"

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X[self.features], num_iteration=self.best_iteration or None)[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP values in log-odds (LightGBM TreeSHAP)."""
        c = self.booster.predict_proba(X[self.features], pred_contrib=True, num_iteration=self.best_iteration or None)
        return pd.DataFrame(c, columns=self.features + ["bias"], index=X.index)

    def importance(self) -> pd.DataFrame:
        b = self.booster.booster_
        return (pd.DataFrame({"feature": b.feature_name(),
                              "gain": b.feature_importance("gain", iteration=self.best_iteration or None)})
                .assign(gain_share=lambda d: d.gain / d.gain.sum())
                .sort_values("gain", ascending=False).reset_index(drop=True))


def fit_lgbm(train: Data, val: Data, features: list, art: dict, weighting: str = "none",
             params: dict | None = None, early_stopping_rounds: int = 200, seed: int = 0) -> LGBMModel:
    """LightGBM early-stopped on WEIGHTED validation PR-AUC (val = own-date, reachable rows)."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    p = {**LGBM_PARAMS, **(params or {}), "random_state": seed}
    w = np.ones(len(train)) if weighting == "none" else train.w
    if weighting == "sample_sqrt":
        p["scale_pos_weight"] = float(np.sqrt(w[train.y == 0].sum() / w[train.y == 1].sum()))
    cats = [f for f in features if f in art["feature_groups"]["categorical"]]
    clf = lgb.LGBMClassifier(**p)
    hist = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        clf.fit(train.X[features], train.y, sample_weight=w,
                eval_set=[(val.X[features], val.y)], eval_sample_weight=[val.w], eval_names=["val"],
                eval_metric=["average_precision"], categorical_feature=cats,
                callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                           lgb.record_evaluation(hist)])
    best = int(clf.best_iteration_ or p["n_estimators"])
    return LGBMModel(list(features), clf, weighting, best, {k: v for k, v in p.items() if k != "verbose"}, hist)


@dataclass
class SeedEnsemble:
    """Average of LightGBM models that differ only in the random seed (row / column subsampling).
    Scores, SHAP contributions and importances are means over the members."""
    members: list
    name: str = "lgbm_seeds"

    @property
    def features(self) -> list:
        return self.members[0].features

    @property
    def weighting(self) -> str:
        return self.members[0].weighting

    @property
    def params(self) -> dict:
        return self.members[0].params

    @property
    def best_iteration(self) -> int:
        return int(round(np.mean([m.best_iteration for m in self.members])))

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_score(X) for m in self.members], axis=0)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        return sum(m.contributions(X) for m in self.members) / len(self.members)

    def importance(self) -> pd.DataFrame:
        imp = pd.concat([m.importance().set_index("feature")["gain"] for m in self.members], axis=1).mean(axis=1)
        return (imp.rename("gain").reset_index().assign(gain_share=lambda d: d.gain / d.gain.sum())
                .sort_values("gain", ascending=False).reset_index(drop=True))


def fit_lgbm_seeds(train: Data, val: Data, features: list, art: dict, weighting: str = "none",
                   params: dict | None = None, seeds=(0, 1, 2, 3, 4)) -> SeedEnsemble:
    return SeedEnsemble([fit_lgbm(train, val, features, art, weighting=weighting, params=params, seed=s)
                         for s in seeds])


def best_f1(y, s, w) -> dict:
    """Weighted F1 at the best threshold (the way the IBM paper reports minority-class F1)."""
    from sklearn.metrics import precision_recall_curve
    p, r, thr = precision_recall_curve(y, s, sample_weight=w)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    i = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    return {"f1": float(f1[i]), "precision": float(p[i]), "recall": float(r[i]), "threshold": float(thr[i])}


def split_budget_alerts(s, day, w, group, budget: float, group_share: float) -> np.ndarray:
    """Daily top-k alerts with part of the budget reserved for one group of rows (e.g. non-ACH):
    each day, ``group_share`` of the alert volume goes to the top-scored rows of the group and the
    rest to the top-scored rows outside it. Volume is weighted (= real transfers)."""
    s, w, day, group = np.asarray(s, float), np.asarray(w, float), np.asarray(day), np.asarray(group, bool)
    alert = np.zeros(len(s), bool)
    for d in np.unique(day):
        on_day = day == d
        total = w[on_day].sum() * budget
        for mask, share in ((on_day & group, group_share), (on_day & ~group, 1 - group_share)):
            idx = np.flatnonzero(mask)
            if not len(idx) or share <= 0:
                continue
            order = idx[np.argsort(-s[idx], kind="mergesort")]
            k = int(np.searchsorted(np.cumsum(w[order]), share * total, side="right"))
            alert[order[:max(k, 1)]] = True
    return alert


# ================================================================== alerts
def daily_thresholds(s, day, w, budget: float) -> dict:
    """Per day: the score of the last row inside the top ``budget`` share of that day's weighted volume."""
    s, w, day = np.asarray(s, float), np.asarray(w, float), pd.Series(np.asarray(day))
    out = {}
    for d, idx in day.groupby(day).groups.items():
        idx = np.asarray(idx)
        order = idx[np.argsort(-s[idx], kind="mergesort")]
        cw = np.cumsum(w[order]) / w[idx].sum()
        k = min(int(np.searchsorted(cw, budget, side="left")), len(order) - 1)
        out[d] = float(s[order[k]])
    return out


def apply_thresholds(s, day, thr: dict, fallback: float) -> np.ndarray:
    t = pd.Series(np.asarray(day)).map(thr).fillna(fallback).to_numpy(float)
    return np.asarray(s, float) >= t


# ================================================================== bundle for the agent layer
def reason_text(contrib: pd.DataFrame, X: pd.DataFrame, top: int = 3) -> list:
    """'new_pair_48h=1 (+1.20), conversion_before=1 (+0.95), ...' per row (log-odds contributions)."""
    c = contrib.drop(columns="bias")
    out = []
    for i in range(len(c)):
        row = c.iloc[i]
        idx = row.abs().sort_values(ascending=False).index[:top]
        vals = [X.iloc[i][f] for f in idx]
        vals = [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in vals]
        out.append(", ".join(f"{f}={v} ({row[f]:+.2f})" for f, v in zip(idx, vals)))
    return out


def save_bundle(path: Path, model, art: dict, budget: float, threshold: float, info: dict) -> None:
    """model + feature list + category levels + alert rule (daily top-k share, val threshold as fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(model.features), "category_levels": art["category_levels"],
                 "alert_budget": float(budget), "threshold": float(threshold), "info": info}, path)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def score_transactions(bundle: dict, X: pd.DataFrame, meta: pd.DataFrame | None = None,
                       explain: bool = True, top: int = 3) -> pd.DataFrame:
    """Score rows of the 03 feature table -> score, alert, top reasons.

    ``X`` may hold the categories as text or as Categorical. With ``meta`` (rows of
    the matching *_meta file: Timestamp and weight), alerts are each day's top
    ``alert_budget`` share of weighted volume; without it the validation threshold.
    """
    model = bundle["model"]
    X = cast_categories(X, {"category_levels": bundle["category_levels"]})
    s = model.predict_score(X)
    if meta is not None:
        day = pd.to_datetime(meta["Timestamp"]).dt.date.to_numpy()
        w = meta["weight"].to_numpy(float) if "weight" in meta else np.ones(len(s))
        alert = apply_thresholds(s, day, daily_thresholds(s, day, w, bundle["alert_budget"]), bundle["threshold"])
    else:
        alert = s >= bundle["threshold"]
    out = pd.DataFrame({"score": s, "alert": alert}, index=X.index)
    if explain and hasattr(model, "contributions"):
        out["reasons"] = reason_text(model.contributions(X), X[model.features], top=top)
    return out
