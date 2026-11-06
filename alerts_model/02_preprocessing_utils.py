"""Preprocessing functions for the HI-Small AML project.

Order (called from preprocessing_feature_selection.py):
    clean_transactions -> split_by_days -> preprocess_splits

The split happens FIRST. Only after that do we build features, and anything
that is "learned" from data (rare categories, caps, constant columns) is
learned on the TRAIN split only and then applied to val and test.
"""

import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
TARGET = "Is Laundering"
CUTOFF_DATE = "2022-09-11"      # after this day the generator only writes laundering residue
PASS_THROUGH_MINUTES = 60       # money forwarded within this time after arriving = pass-through
CAP_QUANTILE = 0.999            # extreme values are capped at this train quantile

# Columns that must NEVER be in X:
# 1) target and anything derived from the label (or from the patterns file)
LEAKAGE_COLUMNS = ["Is Laundering", "Attempt ID", "Pattern Type", "Header Meta",
                   "In Patterns File", "Sent_Laundering", "Recv_Laundering",
                   "Laundering Rows", "Is Laundering Account"]
# 2) raw identifiers the model could memorise (their info is kept in derived features)
ID_COLUMNS = ["From Bank", "To Bank", "Account", "Account.1",
              "Sender Entity ID", "Receiver Entity ID",
              "Sender Bank Name", "Receiver Bank Name",
              "Sender Entity Name", "Receiver Entity Name"]
# 3) redundant / replaced columns (Amount Received -> amount_ratio + currency_mismatch)
REDUNDANT_COLUMNS = ["Amount Received", "Amount Paid", "Timestamp"]
DROP_COLUMNS = LEAKAGE_COLUMNS + ID_COLUMNS + REDUNDANT_COLUMNS

# Raw columns we keep as features (they get encoded after the split)
KEEP_RAW_COLUMNS = ["Payment Format", "Payment Currency", "Receiving Currency"]

# Text columns: levels seen fewer times than this in TRAIN become "Other"
CATEGORICAL_MIN_COUNT = {
    "corridor": 1000, "sender_country": 100, "receiver_country": 100,
    "sender_entity_type": 100, "receiver_entity_type": 100,
    "Payment Format": 100, "Payment Currency": 100, "Receiving Currency": 100,
}
# Numeric columns with extreme values that get capped
CAP_COLUMNS = ["sender_prior_tx_count", "receiver_prior_recv_count", "amount_ratio"]

# The only columns needed from earlier splits to compute account history
HISTORY_COLUMNS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1"]


# ------------------------------------------------------------ 1. cleaning
def clean_transactions(trans):
    """Sort by time and drop the rows from the cutoff date onward."""
    trans = trans.sort_values("Timestamp", kind="mergesort")
    trans = trans[trans["Timestamp"] < CUTOFF_DATE]
    # Fresh index 0..N-1: it works as a row id across all splits
    return trans.reset_index(drop=True)


# --------------------------------------------------------------- 2. split
def split_by_days(trans, val_size=0.2, test_size=0.2):
    """Chronological split by whole days: earliest days = train, latest = test."""
    day = trans["Timestamp"].dt.normalize()
    days = sorted(day.unique())
    n_test = max(1, round(len(days) * test_size))
    n_val = max(1, round(len(days) * val_size))
    n_train = len(days) - n_val - n_test

    day_groups = {
        "train": days[:n_train],
        "val": days[n_train:n_train + n_val],
        "test": days[n_train + n_val:],
    }
    # Raw rows only - no feature has been computed yet
    splits = {name: trans[day.isin(d)] for name, d in day_groups.items()}
    return splits, day_groups


# ------------------------------------------------- 3. features (per split)
def row_features(split):
    """Features that only need the row itself."""
    X = pd.DataFrame(index=split.index)

    # Amount: log of amount paid, ratio received/paid, currency mismatch flag
    X["log_amount"] = np.log1p(split["Amount Paid"])
    X["amount_ratio"] = split["Amount Received"] / split["Amount Paid"].replace(0, np.nan)
    X["currency_mismatch"] = (split["Payment Currency"].astype(str)
                              != split["Receiving Currency"].astype(str)).astype(int)

    # Who talks to whom
    sender = split["From Bank"].astype(str) + "_" + split["Account"].astype(str)
    receiver = split["To Bank"].astype(str) + "_" + split["Account.1"].astype(str)
    X["same_bank"] = (split["From Bank"] == split["To Bank"]).astype(int)
    X["self_transfer"] = (sender == receiver).astype(int)
    X["same_entity"] = (split["Sender Entity ID"] == split["Receiver Entity ID"]).astype(int)

    # Entity type = name without the number ("Corporation #41" -> "Corporation")
    X["sender_entity_type"] = split["Sender Entity Name"].str.replace(r"\s*#\d+\s*$", "", regex=True)
    X["receiver_entity_type"] = split["Receiver Entity Name"].str.replace(r"\s*#\d+\s*$", "", regex=True)

    # Country: foreign banks are named "<Country> Bank #<id>", the rest is domestic
    s_country = split["Sender Bank Name"].str.extract(r"^(.*?)\s+Bank\s+#\d+$")[0].fillna("Domestic")
    r_country = split["Receiver Bank Name"].str.extract(r"^(.*?)\s+Bank\s+#\d+$")[0].fillna("Domestic")
    # "Crytpo" is a typo in the source data
    s_country = s_country.replace({"Crytpo": "Crypto"})
    r_country = r_country.replace({"Crytpo": "Crypto"})
    X["sender_country"] = s_country
    X["receiver_country"] = r_country
    X["cross_border"] = (s_country != r_country).astype(int)
    X["corridor"] = s_country + ">" + r_country

    # 00:00 is the default when no time was recorded
    X["is_midnight"] = (split["Timestamp"].dt.hour == 0).astype(int)
    return X


def history_features(all_rows, n_history):
    """Account behaviour BEFORE each transaction.

    all_rows = rows from earlier splits (n_history of them) followed by the
    current split, in time order. An account's past does not reset at a split
    boundary, so val/test must see train's rows. Only the current split's
    rows are returned. Uses only the past and never the label -> no leakage.
    """
    n = len(all_rows)
    sender = all_rows["From Bank"].astype(str) + "_" + all_rows["Account"].astype(str)
    receiver = all_rows["To Bank"].astype(str) + "_" + all_rows["Account.1"].astype(str)
    # Turn account names into integer codes (much faster to group by)
    codes, _ = pd.factorize(pd.concat([sender, receiver], ignore_index=True))
    s_code, r_code = codes[:n], codes[n:]
    ts = pd.Series(all_rows["Timestamp"].to_numpy())

    out = pd.DataFrame(index=range(n))
    # How many transactions the sender made / the receiver got before this one
    out["sender_prior_tx_count"] = pd.Series(s_code).groupby(s_code).cumcount()
    out["receiver_prior_recv_count"] = pd.Series(r_code).groupby(r_code).cumcount()
    # Minutes since the sender's previous transaction
    out["sender_gap_min"] = ts.groupby(s_code).diff().dt.total_seconds() / 60

    # Last time the sender RECEIVED money strictly before this transaction
    sent = pd.DataFrame({"acct": s_code, "ts": ts, "pos": np.arange(n)})
    received = pd.DataFrame({"acct": r_code, "last_received": ts})
    last = pd.merge_asof(sent, received, left_on="ts", right_on="last_received",
                         by="acct", allow_exact_matches=False).sort_values("pos")
    # Minutes the money stayed in the account before being sent on
    dwell = ((last["ts"] - last["last_received"]).dt.total_seconds() / 60).to_numpy()
    out["sender_dwell_min"] = dwell
    # 1 = forwarded quickly after arriving (NaN, i.e. never received, gives 0)
    out["sender_pass_through"] = (dwell <= PASS_THROUGH_MINUTES).astype(int)

    return out.iloc[n_history:]


def build_features(split, history=None):
    """Raw split -> X, y, meta. Nothing is learned from data in this step."""
    # Row-level features + the raw columns we keep
    X = pd.concat([row_features(split), split[KEEP_RAW_COLUMNS]], axis=1)

    # Account history: earlier splits first, then this split
    if history is not None:
        all_rows = pd.concat([history, split[HISTORY_COLUMNS]], ignore_index=True)
        n_history = len(history)
    else:
        all_rows = split[HISTORY_COLUMNS].reset_index(drop=True)
        n_history = 0
    hist = history_features(all_rows, n_history)
    hist.index = split.index
    X = pd.concat([X, hist], axis=1)

    # Safety check: no leak / ID / redundant column may be in X
    bad = [c for c in X.columns if c in DROP_COLUMNS]
    assert not bad, f"forbidden columns in X: {bad}"

    y = split[TARGET]
    # meta is for evaluation and tracking only, never used as model input
    meta = pd.DataFrame({
        "Timestamp": split["Timestamp"],
        "sender": split["From Bank"].astype(str) + "_" + split["Account"].astype(str),
        "receiver": split["To Bank"].astype(str) + "_" + split["Account.1"].astype(str),
        "Attempt ID": split["Attempt ID"],
        "Pattern Type": split["Pattern Type"],
        "In Patterns File": split["In Patterns File"],
    })
    return X, y, meta


# ---------------------------------------- 4. learned from TRAIN, applied to all
def apply_state(X, state):
    """Cap extreme values, group rare categories, drop constant columns."""
    X = X.copy()
    for col, cap in state["caps"].items():
        X[col] = X[col].clip(upper=cap)
    for col, kept in state["categories"].items():
        values = X[col].astype(str)
        values = values.where(values.isin(kept), "Other")   # rare or unseen -> "Other"
        # Same category list in every split, so encodings always match
        X[col] = pd.Categorical(values, categories=kept + ["Other"])
    return X.drop(columns=state["constant_columns"])


def fit_state(X_train):
    """Learn everything from the TRAIN split only."""
    state = {"categories": {}, "caps": {}, "constant_columns": []}
    # Which category levels are frequent enough to keep
    for col, min_count in CATEGORICAL_MIN_COUNT.items():
        counts = X_train[col].astype(str).value_counts()
        state["categories"][col] = sorted(c for c in counts[counts >= min_count].index if c != "Other")
    # Upper cap for extreme values
    for col in CAP_COLUMNS:
        state["caps"][col] = float(X_train[col].quantile(CAP_QUANTILE))
    # Constant columns carry no information: find them after capping/grouping
    X_tmp = apply_state(X_train, state)
    state["constant_columns"] = [c for c in X_tmp.columns if X_tmp[c].nunique(dropna=False) <= 1]
    return state


def save_state(state, path):
    """Store the learned values so they can be reused on new data."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------ 5. everything
def preprocess_splits(splits, drop_columns=()):
    """Splits (raw) -> {name: (X, y, meta)} plus the state learned on train.

    drop_columns: optional extra columns to remove, e.g. ["Payment Format"].
    """
    # Step A: features per split. Each split's history = all earlier splits.
    built = {}
    for name, df in splits.items():
        earlier = [d for other, d in splits.items()
                   if other != name and d["Timestamp"].max() < df["Timestamp"].min()]
        history = (pd.concat([d[HISTORY_COLUMNS] for d in sorted(earlier, key=lambda d: d["Timestamp"].min())])
                   if earlier else None)
        built[name] = build_features(df, history)

    # Step B: learn from train only
    state = fit_state(built["train"][0])

    # Step C: apply to every split
    results = {}
    for name, (X, y, meta) in built.items():
        X = apply_state(X, state).drop(columns=list(drop_columns))
        results[name] = (X, y, meta)
    return results, state
