"""Models for transaction scoring (the detection tool inside the agent loop).

Inputs are the files written by ``03_features.py``. Everything learned from
data (category levels, amount bins, scaling, early stopping) is fitted on
TRAIN, tuned on VAL, and only then applied to TEST.

Models
------
* ``rule``     - best single feature (direction chosen on train): the bar to beat.
* ``logreg``   - logistic regression on log counts, flags, amount buckets and
                 one-hot categories, class_weight='balanced'.
* ``lgbm``     - LightGBM with native categoricals, scale_pos_weight, early
                 stopping on validation PR-AUC.

Each can be trained on three feature sets (EDA v2: some signals are generator
mechanics, so we report with and without them):
* ``all``          - every feature
* ``no_format``    - without payment format
* ``no_generator`` - without payment format and conversion_before

All fitting and every metric takes ``sample_weight`` (1 on the full file,
inverse inclusion probability on the 02 sample).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

import features as F

SPLITS = ("train", "val", "test")

FEATURE_SETS = {
    "all": list(F.FEATURES),
    "no_format": [f for f in F.FEATURES if f not in F.FORMAT_FEATURES],
    "no_generator": [f for f in F.FEATURES if f not in F.GENERATOR_FEATURES],
}

# Logistic regression: one of each pair of numeric features correlated above
# this (Spearman, on TRAIN) is dropped - see correlated_to_drop().
CORR_THRESHOLD = 0.85


# ------------------------------------------------------------------ data
@dataclass
class Split:
    X: pd.DataFrame
    y: np.ndarray
    w: np.ndarray
    meta: pd.DataFrame

    def __len__(self):
        return len(self.y)


def load_splits(features_dir: Path, columns: list[str] | None = None) -> dict[str, Split]:
    """Read {split}_features / {split}_meta written by 03_features.py."""
    features_dir = Path(features_dir)
    out = {}
    for s in SPLITS:
        xf = pd.read_parquet(features_dir / f"{s}_features.parquet")
        mf = pd.read_parquet(features_dir / f"{s}_meta.parquet")
        if not (xf["row_id"].to_numpy() == mf["row_id"].to_numpy()).all():
            raise ValueError(f"{s}: features and meta are not row-aligned")
        y = xf.pop(F.TARGET).to_numpy().astype(np.int8)
        X = xf.drop(columns="row_id")
        if columns is not None:
            X = X[columns]
        mf["y"] = y
        out[s] = Split(X=X, y=y, w=mf["weight"].to_numpy(float), meta=mf)
    return out


def assert_no_leak(X: pd.DataFrame) -> None:
    bad = sorted(set(X.columns) & set(F.LEAK_COLUMNS))
    if bad:
        raise ValueError(f"leak / ID columns in X: {bad}")


def fit_rows(split: Split, burn_in: bool = True) -> Split:
    """Rows used to FIT a model: drop the burn-in (account-history window not full yet)."""
    if not burn_in or "history_complete" not in split.meta:
        return split
    k = split.meta["history_complete"].to_numpy(bool)
    return Split(split.X[k].reset_index(drop=True), split.y[k], split.w[k],
                 split.meta[k].reset_index(drop=True))


def correlated_to_drop(X: pd.DataFrame, cols, threshold: float = CORR_THRESHOLD, max_rows: int = 500_000,
                       seed: int = 0) -> tuple[list[str], pd.DataFrame]:
    """Greedy: walk pairs from most to least correlated, drop the later column of each pair.

    Spearman on (a random subset of) TRAIN. Returns (columns to drop, the pairs found).
    """
    Xs = X[cols]
    if len(Xs) > max_rows:
        Xs = Xs.sample(max_rows, random_state=seed)
    c = Xs.rank().corr().abs()
    pairs = (c.where(np.triu(np.ones(c.shape, bool), 1)).stack()
             .rename("abs_spearman").reset_index().set_axis(["a", "b", "abs_spearman"], axis=1)
             .sort_values("abs_spearman", ascending=False))
    drop = []
    for r in pairs[pairs.abs_spearman > threshold].itertuples():
        if r.a not in drop and r.b not in drop:
            drop.append(r.b)
    return drop, pairs.reset_index(drop=True)


def downsample_negatives(split: Split, frac: float, seed: int = 0) -> Split:
    """Keep every positive and ``frac`` of negatives; negatives' weight /= frac.

    Training-speed option only. Weighted totals stay unbiased. Never use on
    val / test.
    """
    if frac >= 1:
        return split
    rng = np.random.default_rng(seed)
    keep = (split.y == 1) | (rng.random(len(split)) < frac)
    w = split.w * np.where(split.y == 1, 1.0, 1.0 / frac)
    return Split(split.X[keep].reset_index(drop=True), split.y[keep], w[keep],
                 split.meta[keep].reset_index(drop=True))


def scale_pos_weight(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted negatives / weighted positives."""
    return float(w[y == 0].sum() / w[y == 1].sum())


# ------------------------------------------------------------------ categories (train-fitted)
class CategoryLevels:
    """Category levels seen at least ``min_count`` times in TRAIN; the rest -> 'Other'."""

    def __init__(self, min_count: int = 100):
        self.min_count = min_count
        self.levels: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame, cols):
        for c in cols:
            vc = X[c].astype(str).value_counts()
            self.levels[c] = sorted(v for v in vc[vc >= self.min_count].index if v != "Other")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, lev in self.levels.items():
            if c in X:
                v = X[c].astype(str)
                X[c] = pd.Categorical(v.where(v.isin(lev), "Other"), categories=lev + ["Other"])
        return X


# ------------------------------------------------------------------ rule baseline
def single_feature_table(train: Split, val: Split, cols=None) -> pd.DataFrame:
    """PR-AUC of each numeric feature used alone as a score.

    Direction (higher or lower = riskier) is chosen on train, then applied to val.
    """
    cols = cols or (F.COUNT_FEATURES + F.FLAG_FEATURES + F.AMOUNT_FEATURES)
    base_tr = np.average(train.y, weights=train.w)
    base_va = np.average(val.y, weights=val.w)
    rows = []
    for c in cols:
        if c not in train.X:
            continue
        s_tr = train.X[c].to_numpy(float)
        up = average_precision_score(train.y, s_tr, sample_weight=train.w)
        down = average_precision_score(train.y, -s_tr, sample_weight=train.w)
        sign = 1 if up >= down else -1
        ap_va = average_precision_score(val.y, sign * val.X[c].to_numpy(float), sample_weight=val.w)
        rows.append({"feature": c, "direction": "higher" if sign > 0 else "lower",
                     "train_ap": max(up, down), "train_lift": max(up, down) / base_tr,
                     "val_ap": ap_va, "val_lift": ap_va / base_va})
    return pd.DataFrame(rows).sort_values("val_ap", ascending=False).reset_index(drop=True)


@dataclass
class RuleModel:
    feature: str
    sign: int
    name: str = "rule"

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        # tiny amount tie-breaker so equal rule values are ordered by amount
        tb = X["log_amount_usd"].to_numpy(float) * 1e-6 if "log_amount_usd" in X else 0
        return self.sign * X[self.feature].to_numpy(float) + tb


def fit_rule(train: Split, val: Split) -> RuleModel:
    tab = single_feature_table(train, val)
    best = tab.iloc[0]
    return RuleModel(best["feature"], 1 if best["direction"] == "higher" else -1)


# ------------------------------------------------------------------ logistic regression
import sklearn  # noqa: E402
_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
# sklearn >= 1.7 warns unless the quantile method is named; older versions reject the argument
_KBINS_KW = {"quantile_method": "averaged_inverted_cdf"} if _SK >= (1, 7) else {}

@dataclass
class LogRegModel:
    features: list[str]
    pipeline: Pipeline
    name: str = "logreg"

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]


def fit_logreg(train: Split, features: list[str], C: float = 1.0, min_count: int = 100,
               amount_bins: int = 10, drop: list[str] | None = None, seed: int = 0) -> LogRegModel:
    """``drop``: columns to leave out (default: correlated_to_drop on train)."""
    if drop is None:
        num = [f for f in features if f in F.COUNT_FEATURES + F.FLAG_FEATURES]
        drop, _ = correlated_to_drop(train.X, num)
    feats = [f for f in features if f not in drop]
    counts = [f for f in feats if f in F.COUNT_FEATURES]
    flags = [f for f in feats if f in F.FLAG_FEATURES]
    amount = [f for f in feats if f in F.AMOUNT_FEATURES]
    cats = [f for f in feats if f in F.CATEGORICAL_FEATURES]
    pre = ColumnTransformer([
        ("counts", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), counts),
        ("flags", "passthrough", flags),
        # risk by amount is not monotonic (EDA v2) -> buckets, not a slope
        ("amount", KBinsDiscretizer(n_bins=amount_bins, encode="onehot", strategy="quantile",
                                    **_KBINS_KW), amount),
        ("cats", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_count,
                               sparse_output=True), cats),
    ], sparse_threshold=1.0)
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                                solver="lbfgs", random_state=seed))])
    X = train.X[feats].copy()
    X[cats] = X[cats].astype(str)
    pipe.fit(X, train.y, clf__sample_weight=train.w)
    return LogRegModel(feats, pipe)


def logreg_coefficients(model: LogRegModel) -> pd.DataFrame:
    names = model.pipeline.named_steps["pre"].get_feature_names_out()
    coef = model.pipeline.named_steps["clf"].coef_.ravel()
    return (pd.DataFrame({"term": names, "coef": coef})
            .assign(abs_coef=lambda d: d.coef.abs())
            .sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True))


# ------------------------------------------------------------------ LightGBM
LGBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    # small on purpose: a rare but pure signal (e.g. conversion_before on a few
    # hundred rows) must be splittable; overfitting is held by early stopping + reg
    min_child_samples=20,
    min_child_weight=1e-3,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    cat_smooth=20, min_data_per_group=50, max_cat_threshold=32,
    n_estimators=3000,
    verbose=-1,
)


@dataclass
class LGBMModel:
    features: list[str]
    levels: CategoryLevels
    booster: lgb.LGBMClassifier
    best_iteration: int = 0
    history: dict = field(default_factory=dict)
    name: str = "lgbm"

    def _X(self, X):
        return self.levels.transform(X[self.features])

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(self._X(X), num_iteration=self.best_iteration or None)[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP values (log-odds), LightGBM's exact TreeSHAP."""
        c = self.booster.predict_proba(self._X(X), pred_contrib=True,
                                       num_iteration=self.best_iteration or None)
        return pd.DataFrame(c, columns=self.features + ["bias"], index=X.index)

    def importance(self) -> pd.DataFrame:
        b = self.booster.booster_
        return (pd.DataFrame({"feature": b.feature_name(),
                              "gain": b.feature_importance("gain", iteration=self.best_iteration or None),
                              "splits": b.feature_importance("split", iteration=self.best_iteration or None)})
                .assign(gain_share=lambda d: d.gain / d.gain.sum())
                .sort_values("gain", ascending=False).reset_index(drop=True))


def fit_lgbm(train: Split, val: Split, features: list[str], params: dict | None = None,
             early_stopping_rounds: int = 200, spw: float | str = "sqrt", seed: int = 0,
             min_count: int = 100, log_every: int = 0) -> LGBMModel:
    """LightGBM, early-stopped on weighted validation PR-AUC.

    ``spw`` (scale_pos_weight): "sqrt" (default) = sqrt(neg/pos), "full" = neg/pos,
    or a number. Full re-weighting at ~1:1000 makes every leaf near-pure-positive
    after a few trees, so validation PR-AUC peaks almost immediately and scores
    pile up (ties) - tune this on validation, never on test.
    """
    p = {**LGBM_PARAMS, **(params or {}), "random_state": seed}
    ratio = scale_pos_weight(train.y, train.w)
    p["scale_pos_weight"] = {"sqrt": np.sqrt(ratio), "full": ratio}.get(spw, None) if isinstance(spw, str) else float(spw)
    cats = [f for f in features if f in F.CATEGORICAL_FEATURES]
    levels = CategoryLevels(min_count).fit(train.X, cats)
    Xtr, Xva = levels.transform(train.X[features]), levels.transform(val.X[features])
    clf = lgb.LGBMClassifier(**p)
    hist = {}
    callbacks = [lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                 lgb.record_evaluation(hist)]
    if log_every:
        callbacks.append(lgb.log_evaluation(log_every))
    with warnings.catch_warnings():   # LightGBM >= 4.7 renames eval_set; older versions need it
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        clf.fit(Xtr, train.y, sample_weight=train.w,
                eval_set=[(Xva, val.y)], eval_sample_weight=[val.w], eval_names=["val"],
                eval_metric=["average_precision"], categorical_feature=cats, callbacks=callbacks)
    return LGBMModel(features, levels, clf, int(clf.best_iteration_ or p["n_estimators"]), hist)


# ------------------------------------------------------------------ bundle for the agent layer
def reason_text(contrib: pd.DataFrame, X: pd.DataFrame, top: int = 3) -> list[str]:
    """'new_pair=1 (+1.20), conversion_before=1 (+0.95), ...' for each row."""
    c = contrib.drop(columns="bias")
    out = []
    for i in range(len(c)):
        row = c.iloc[i]
        idx = row.abs().sort_values(ascending=False).index[:top]
        vals = [X.iloc[i][f] for f in idx]
        vals = [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in vals]
        out.append(", ".join(f"{f}={v} ({row[f]:+.2f})" for f, v in zip(idx, vals)))
    return out


def save_bundle(path: Path, model, threshold: float, budget: float, info: dict) -> None:
    """Everything the agent's scoring tool needs: model, feature list, threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": model.features if hasattr(model, "features") else None,
                 "threshold": float(threshold), "alert_budget": float(budget), "info": info}, path)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def score_transactions(bundle: dict, X: pd.DataFrame, explain: bool = True, top: int = 3) -> pd.DataFrame:
    """Score rows of the 03 feature table. Returns score, alert flag and top reasons."""
    model = bundle["model"]
    s = model.predict_score(X)
    out = pd.DataFrame({"score": s, "alert": s >= bundle["threshold"]}, index=X.index)
    if explain and hasattr(model, "contributions"):
        out["reasons"] = reason_text(model.contributions(X), X[model.features], top=top)
    return out
