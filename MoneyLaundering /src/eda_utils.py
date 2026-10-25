"""Helpers for the HI-Small EDA.

Scope: HI-Small only, full 5,078,345 rows. No subsampling anywhere in this module.

The functions here are kept out of the notebook so they can be unit-tested and
reused by the preprocessing stage. Nothing in this module encodes, scales or
splits — that belongs downstream.

Expected layout::

    capstone/
      data/raw/   HI-Small_Trans.csv, HI-Small_accounts.csv, HI-Small_Patterns.txt
      notebook/   EDA_HI_Small.ipynb
      src/        eda_utils.py   <- this file
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start=None):
    """Locate the project root — the directory containing ``data/raw``.

    Resolved rather than hard-coded because Jupyter runs with the notebook's
    own directory as the working directory while VS Code may use the workspace
    root, and a relative ``data/raw`` breaks under one of the two.

    Tries this module's location first (``src/eda_utils.py`` -> root is its
    parent), then walks up from the working directory.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "raw").is_dir() or (candidate / "data").is_dir():
            return candidate

    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "raw").is_dir() or (candidate / "data").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate a 'data' directory above either "
        f"{here} or {start}. Pass paths explicitly if the layout differs."
    )


try:
    PROJECT_ROOT = find_project_root()
except FileNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent

if (PROJECT_ROOT / "data" / "raw").is_dir():
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
elif (PROJECT_ROOT / "data").is_dir():
    DATA_DIR = PROJECT_ROOT / "data"
else:
    DATA_DIR = PROJECT_ROOT / "data"

TRANS_DTYPES = {
    "From Bank": "int64",
    "To Bank": "int64",
    "Account": "string",
    "Account.1": "string",
    "Amount Received": "float64",
    "Amount Paid": "float64",
    "Receiving Currency": "category",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}

PATTERN_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]

# Join key between the patterns file and the transaction log.
# Account/bank pair alone is NOT unique - the same pair transacts repeatedly -
# so timestamp and amount are both required.
PATTERN_JOIN_KEYS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid",
]

_HEADER_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+)\s*:?\s*(.*)$")


# ---------------------------------------------------------------- loading


def load_transactions(path=DATA_DIR / "HI-Small_Trans.csv", sort=True):
    """Load the HI-Small transaction log, sorted by Timestamp.

    The file ships in account order, not time order. Every temporal statement in
    the EDA depends on this sort, so it happens at load time and nowhere else.
    """
    df = pd.read_csv(path, dtype=TRANS_DTYPES)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    if sort:
        df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    return df


def load_accounts(path=DATA_DIR / "HI-Small_accounts.csv"):
    """Load the HI-Small account reference table.

    Account Number and Entity ID stay strings: they are hex-like identifiers and
    casting them to a numeric type silently corrupts them.
    """
    return pd.read_csv(
        path,
        dtype={
            "Bank Name": "string",
            "Bank ID": "int64",
            "Account Number": "string",
            "Entity ID": "string",
            "Entity Name": "string",
        },
    )


def load_patterns(path=DATA_DIR / "HI-Small_Patterns.txt"):
    """Parse the patterns file into one row per laundering transfer.

    Differs from the original parser in two ways that matter:

    * Each BEGIN/END block gets an ``Attempt ID``. Without it a 14-hop
      GATHER-SCATTER is 14 unrelated rows and the ground-truth case structure is
      lost - which is the only thing in this dataset an investigation agent can
      be scored against.
    * The block header text ("Max 16-degree Fan-Out", "Max 10 hops") is kept as
      ``Header Meta``. It is the generator's own description of the scheme.

    Bank codes are written zero-padded in this file ("021174") and unpadded in
    the CSV (21174). Both are cast to int here so the join cannot silently miss.
    """
    rows, attempts = [], []
    attempt_id = -1
    ptype = meta = None
    malformed = []

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                match = _HEADER_RE.match(line)
                ptype = match.group(1).strip() if match else "UNKNOWN"
                meta = (match.group(2).strip() if match else "") or None
                attempt_id += 1
                attempts.append(
                    {"Attempt ID": attempt_id, "Pattern Type": ptype, "Header Meta": meta}
                )
            elif line.startswith("END LAUNDERING ATTEMPT"):
                continue
            else:
                fields = line.split(",")
                if len(fields) != len(PATTERN_COLUMNS):
                    malformed.append(lineno)
                    continue
                record = dict(zip(PATTERN_COLUMNS, fields))
                record["Attempt ID"] = attempt_id
                record["Pattern Type"] = ptype
                record["Header Meta"] = meta
                rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No transfer rows parsed from {path}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    for col in ("From Bank", "To Bank"):
        df[col] = df[col].astype(int)
    for col in ("Amount Received", "Amount Paid"):
        df[col] = pd.to_numeric(df[col])
    df["Is Laundering"] = df["Is Laundering"].astype(int)
    for col in ("Account", "Account.1", "Receiving Currency",
                "Payment Currency", "Payment Format"):
        df[col] = df[col].astype("string")

    return df, pd.DataFrame(attempts), malformed


# ------------------------------------------------------- quality inspection


def inspect(df, name="Dataset"):
    """Column-level quality report. Returns a frame; prints nothing."""
    return pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing": df.isna().sum(),
        "Missing %": (df.isna().sum() / len(df) * 100).round(4),
        "Unique": df.nunique(),
        "Sample": [
            df[c].dropna().iloc[0] if df[c].notna().any() else pd.NA
            for c in df.columns
        ],
    })


def quality_summary(df, name):
    """One-line scalar summary: rows, cols, missing cells, duplicate rows."""
    return {
        "Dataset": name,
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
    }


def compare_sources(data_dir=DATA_DIR, files=None):
    """Laundering density across the four source files.

    Reads only the ``Is Laundering`` column, so this costs a fraction of a full
    load. This table is the entire justification for choosing HI-Small; the
    original notebook spent a 7.6 GB download and four univariate passes to
    reach a conclusion this answers directly - and reached the wrong one.
    """
    if files is None:
        files = {
            "HI-Small": "HI-Small_Trans.csv",
            "HI-Medium": "HI-Medium_Trans.csv",
            "LI-Small": "LI-Small_Trans.csv",
            "LI-Medium": "LI-Medium_Trans.csv",
        }
    records = []
    for label, filename in files.items():
        path = Path(data_dir) / filename
        if not path.is_file():
            records.append({"Source": label, "Rows": pd.NA,
                            "Laundering": pd.NA, "Rate %": pd.NA})
            continue
        col = pd.read_csv(path, usecols=["Is Laundering"], dtype={"Is Laundering": "int8"})
        records.append({
            "Source": label,
            "Rows": len(col),
            "Laundering": int(col["Is Laundering"].sum()),
            "Rate %": round(col["Is Laundering"].mean() * 100, 4),
        })
    return pd.DataFrame(records).sort_values("Rate %", ascending=False, na_position="last")


# ------------------------------------------------------------ account link


def link_accounts(trans, accounts):
    """Attach sender and receiver metadata without changing the row count.

    Sender matches on (From Bank, Account), receiver on (To Bank, Account.1),
    against (Bank ID, Account Number). Left joins, so unmatched rows survive as
    nulls and get counted rather than silently dropped.
    """
    lookup = accounts[["Bank ID", "Account Number", "Bank Name", "Entity ID", "Entity Name"]]

    before = len(trans)
    out = trans.merge(
        lookup.rename(columns={
            "Bank ID": "From Bank", "Account Number": "Account",
            "Bank Name": "Sender Bank Name", "Entity ID": "Sender Entity ID",
            "Entity Name": "Sender Entity Name",
        }),
        on=["From Bank", "Account"], how="left", validate="many_to_one",
    ).merge(
        lookup.rename(columns={
            "Bank ID": "To Bank", "Account Number": "Account.1",
            "Bank Name": "Receiver Bank Name", "Entity ID": "Receiver Entity ID",
            "Entity Name": "Receiver Entity Name",
        }),
        on=["To Bank", "Account.1"], how="left", validate="many_to_one",
    )

    if len(out) != before:
        raise AssertionError(f"Account join changed row count: {before} -> {len(out)}")

    stats = {
        "Rows": len(out),
        "Sender unmatched": int(out["Sender Entity ID"].isna().sum()),
        "Receiver unmatched": int(out["Receiver Entity ID"].isna().sum()),
    }
    return out, stats


# --------------------------------------------------------- pattern linking


def link_patterns(trans, patterns, keys=None):
    """Attach Attempt ID / Pattern Type to the transaction log.

    Returns (trans_with_pattern, reconciliation).

    The reconciliation is the point of this function. The transaction log flags
    5,177 rows as laundering; the patterns file explains 3,209 of them. Anything
    that fails to match is either a generator artefact or a bug in the join, and
    the EDA has no business claiming the labels are verified until the gap is
    quantified.
    """
    keys = keys or PATTERN_JOIN_KEYS

    right = patterns[keys + ["Attempt ID", "Pattern Type", "Header Meta"]].copy()
    dup_keys = int(right.duplicated(subset=keys).sum())
    right = right.drop_duplicates(subset=keys)

    before = len(trans)
    out = trans.merge(right, on=keys, how="left", validate="many_to_one")
    if len(out) != before:
        raise AssertionError(f"Pattern join changed row count: {before} -> {len(out)}")

    out["In Patterns File"] = out["Attempt ID"].notna()

    flagged = out["Is Laundering"].eq(1)
    in_patterns = out["In Patterns File"]

    reconciliation = pd.DataFrame([
        {"Category": "Flagged and in patterns file",
         "Rows": int((flagged & in_patterns).sum()),
         "Note": "Label corroborated by a named laundering attempt"},
        {"Category": "Flagged, not in patterns file",
         "Rows": int((flagged & ~in_patterns).sum()),
         "Note": "Flagged with no corroborating attempt - investigate"},
        {"Category": "In patterns file, not flagged",
         "Rows": int((~flagged & in_patterns).sum()),
         "Note": "Join error or label inconsistency - must be 0"},
        {"Category": "Unflagged background",
         "Rows": int((~flagged & ~in_patterns).sum()),
         "Note": "Legitimate traffic"},
    ])

    coverage = {
        "Pattern rows in file": len(patterns),
        "Pattern rows matched": int(in_patterns.sum()),
        "Pattern rows unmatched": len(patterns) - int(in_patterns.sum()),
        "Duplicate join keys dropped": dup_keys,
    }
    return out, reconciliation, coverage


def attempt_profile(patterns):
    """Per-attempt shape: rows, time span, accounts, banks."""
    grp = patterns.groupby("Attempt ID")
    profile = pd.DataFrame({
        "Pattern Type": grp["Pattern Type"].first(),
        "Header Meta": grp["Header Meta"].first(),
        "Rows": grp.size(),
        "Start": grp["Timestamp"].min(),
        "End": grp["Timestamp"].max(),
        "Total Amount": grp["Amount Paid"].sum(),
        "Distinct Banks": grp.apply(
            lambda g: len(set(g["From Bank"]) | set(g["To Bank"])), include_groups=False
        ),
        "Distinct Accounts": grp.apply(
            lambda g: len(set(g["Account"]) | set(g["Account.1"])), include_groups=False
        ),
    })
    profile["Span (days)"] = (profile["End"] - profile["Start"]).dt.total_seconds() / 86400
    return profile.reset_index()


def pattern_accounts(patterns):
    """Every account appearing in any laundering attempt, with its attempt count."""
    stacked = pd.concat([
        patterns[["Account", "Attempt ID"]].rename(columns={"Account": "Account ID"}),
        patterns[["Account.1", "Attempt ID"]].rename(columns={"Account.1": "Account ID"}),
    ]).drop_duplicates()
    return (
        stacked.groupby("Account ID")["Attempt ID"]
        .nunique().rename("Attempts").reset_index()
        .sort_values("Attempts", ascending=False)
    )


# --------------------------------------------------------- risk breakdowns


def risk_by(df, column, target="Is Laundering"):
    """Volume and laundering rate per category, sorted by rate."""
    out = df.groupby(column, observed=True)[target].agg(
        Transactions="size", Laundering="sum"
    )
    out["Rate %"] = (out["Laundering"] / out["Transactions"] * 100).round(4)
    out["Share of Laundering %"] = (
        out["Laundering"] / out["Laundering"].sum() * 100
    ).round(2)
    out["Share of Volume %"] = (out["Transactions"] / len(df) * 100).round(2)
    return out.sort_values("Rate %", ascending=False)


def bank_relationship(df):
    """Intra-bank vs inter-bank laundering split."""
    rel = np.where(df["From Bank"].eq(df["To Bank"]), "Intra-Bank", "Inter-Bank")
    return risk_by(df.assign(Relationship=rel), "Relationship")


def currency_consistency(df):
    """How often paid and received amounts agree, split by currency match.

    The original notebook checked this on same-currency rows only, found 100%
    agreement, and recommended keeping both amount columns anyway. Splitting by
    currency match shows whether the columns ever carry distinct information.
    """
    mismatch = df["Payment Currency"].ne(df["Receiving Currency"])
    amounts_equal = df["Amount Paid"].eq(df["Amount Received"])
    table = pd.crosstab(
        mismatch.map({False: "Same currency", True: "Cross currency"}),
        amounts_equal.map({False: "Amounts differ", True: "Amounts identical"}),
    )
    table["Laundering"] = df.groupby(
        mismatch.map({False: "Same currency", True: "Cross currency"})
    )["Is Laundering"].sum()
    return table


# ----------------------------------------------------------- graph / network


def account_activity(df):
    """Per-account behavioural profile.

    This is the feature family the whole project rests on - an investigation
    agent reasons about accounts and their counterparties, not about isolated
    rows - and the original EDA computed none of it.
    """
    sent = df.groupby("Account", observed=True).agg(
        Sent_Count=("Amount Paid", "size"),
        Sent_Total=("Amount Paid", "sum"),
        Sent_Median=("Amount Paid", "median"),
        Out_Counterparties=("Account.1", "nunique"),
        Sent_Laundering=("Is Laundering", "sum"),
    )
    received = df.groupby("Account.1", observed=True).agg(
        Recv_Count=("Amount Received", "size"),
        Recv_Total=("Amount Received", "sum"),
        Recv_Median=("Amount Received", "median"),
        In_Counterparties=("Account", "nunique"),
        Recv_Laundering=("Is Laundering", "sum"),
    )
    sent.index.name = "Account ID"
    received.index.name = "Account ID"

    acc = sent.join(received, how="outer").fillna(0)
    acc["Degree"] = acc["Sent_Count"] + acc["Recv_Count"]
    acc["Counterparties"] = acc["Out_Counterparties"] + acc["In_Counterparties"]
    acc["Laundering Rows"] = acc["Sent_Laundering"] + acc["Recv_Laundering"]
    acc["Is Laundering Account"] = acc["Laundering Rows"] > 0
    acc["Net Flow"] = acc["Recv_Total"] - acc["Sent_Total"]
    acc["Flow Ratio"] = acc["Sent_Total"] / acc["Recv_Total"].replace(0, np.nan)
    return acc.reset_index()


def activity_contrast(account_df):
    """Compare laundering-involved accounts against the rest."""
    metrics = ["Degree", "Counterparties", "Sent_Count", "Recv_Count",
               "Sent_Total", "Recv_Total", "Flow Ratio"]
    return (
        account_df.groupby("Is Laundering Account")[metrics]
        .agg(["median", "mean"]).round(2).T
    )


def interarrival(df, by="Account", limit=None):
    """Gap in minutes between consecutive transactions of the same account.

    Velocity is cited repeatedly in the original notebook's 'future steps' and
    never computed. Requires a time-sorted frame.
    """
    work = df[[by, "Timestamp", "Is Laundering"]]
    if limit is not None:
        keep = work[by].value_counts().head(limit).index
        work = work[work[by].isin(keep)]
    work = work.sort_values([by, "Timestamp"], kind="mergesort")
    gap = work.groupby(by, observed=True)["Timestamp"].diff().dt.total_seconds() / 60
    return work.assign(**{"Gap (min)": gap}).dropna(subset=["Gap (min)"])


# ------------------------------------------------------------- structuring


def round_amount_profile(df, column="Amount Paid"):
    """Share of amounts landing on round values, overall and by class.

    Structuring - breaking a large sum into round sub-threshold transfers - is a
    core laundering typology, so round-number density is a signal worth checking
    before any feature engineering.
    """
    amounts = df[column]
    checks = {
        "Multiple of 10": amounts.mod(10).eq(0),
        "Multiple of 100": amounts.mod(100).eq(0),
        "Multiple of 1000": amounts.mod(1000).eq(0),
        "Whole number": amounts.mod(1).eq(0),
    }
    rows = []
    for label, mask in checks.items():
        rows.append({
            "Check": label,
            "All %": round(mask.mean() * 100, 3),
            "Legitimate %": round(mask[df["Is Laundering"].eq(0)].mean() * 100, 3),
            "Laundering %": round(mask[df["Is Laundering"].eq(1)].mean() * 100, 3),
        })
    return pd.DataFrame(rows)


def repeated_amounts(df, min_repeats=3):
    """Accounts sending the same exact amount repeatedly - a STACK/layering tell."""
    grp = df.groupby(["Account", "Amount Paid"], observed=True).agg(
        Repeats=("Timestamp", "size"),
        Laundering=("Is Laundering", "sum"),
        Counterparties=("Account.1", "nunique"),
    )
    return grp[grp["Repeats"] >= min_repeats].sort_values("Repeats", ascending=False).reset_index()


# ------------------------------------------------------------------ leakage


def account_overlap_across_time(df, cutoff):
    """Accounts appearing on both sides of a temporal cutoff.

    A random train/validation split shares laundering rings between the two
    sides, so validation recall measures memorisation. This quantifies how much
    overlap a time-based cut leaves behind.
    """
    cutoff = pd.Timestamp(cutoff)
    before = df[df["Timestamp"] < cutoff]
    after = df[df["Timestamp"] >= cutoff]

    def accs(frame):
        return set(frame["Account"].dropna()) | set(frame["Account.1"].dropna())

    a, b = accs(before), accs(after)
    shared = a & b
    return {
        "Cutoff": cutoff,
        "Rows before": len(before),
        "Rows after": len(after),
        "Laundering before": int(before["Is Laundering"].sum()),
        "Laundering after": int(after["Is Laundering"].sum()),
        "Accounts before": len(a),
        "Accounts after": len(b),
        "Shared accounts": len(shared),
        "Shared % of after": round(len(shared) / len(b) * 100, 2) if b else 0.0,
    }


def attempts_straddling(attempt_profile_df, cutoff):
    """Laundering attempts split by a temporal cutoff."""
    cutoff = pd.Timestamp(cutoff)
    straddle = attempt_profile_df["Start"].lt(cutoff) & attempt_profile_df["End"].ge(cutoff)
    return {
        "Cutoff": cutoff,
        "Attempts total": len(attempt_profile_df),
        "Attempts straddling": int(straddle.sum()),
        "Attempts fully before": int(attempt_profile_df["End"].lt(cutoff).sum()),
        "Attempts fully after": int(attempt_profile_df["Start"].ge(cutoff).sum()),
    }


# =====================================================================
# Extensions: hubs, jurisdiction, entities, pass-through, lifetime,
# integrity, topology examples, feature summary.
# =====================================================================


# ------------------------------------------------------------- 1. hubs


def identify_hubs(account_df, degree_quantile=0.9999, min_counterparties=500):
    """Flag accounts whose activity is orders of magnitude above normal.

    A handful of accounts in HI-Small transact hundreds of thousands of times
    with tens of thousands of counterparties. They behave like clearing or
    aggregator nodes, not customers, and because they also touch laundering
    they can single-handedly create an apparent "high degree => laundering"
    effect. Flagging them lets the network comparison be re-run without them.
    """
    threshold = account_df["Degree"].quantile(degree_quantile)
    is_hub = (
        account_df["Degree"].ge(threshold)
        | account_df["Counterparties"].ge(min_counterparties)
    )
    return account_df.assign(**{"Is Hub": is_hub}), {
        "Degree threshold": float(threshold),
        "Quantile used": degree_quantile,
        "Min counterparties": min_counterparties,
        "Hub accounts": int(is_hub.sum()),
        "Hub share of accounts %": round(is_hub.mean() * 100, 5),
    }


def hub_impact(account_df):
    """Network contrast with and without hub accounts.

    If the laundering-vs-clean gap survives hub removal the signal is real; if
    it collapses, the finding was a handful of rows.
    """
    metrics = ["Degree", "Counterparties", "Sent_Count", "Recv_Count"]
    with_hubs = account_df.groupby("Is Laundering Account")[metrics].median()
    without = (account_df[~account_df["Is Hub"]]
               .groupby("Is Laundering Account")[metrics].median())

    out = pd.concat(
        {"All accounts": with_hubs.T, "Hubs excluded": without.T}, axis=1
    )
    out.columns = [f"{a} | {'laundering' if b else 'clean'}" for a, b in out.columns]
    return out


def hub_detail(account_df, trans, top_n=20):
    """Who the hubs actually are: entity, bank, and share of total volume."""
    hubs = account_df.nlargest(top_n, "Degree")[
        ["Account ID", "Degree", "Sent_Count", "Recv_Count",
         "Counterparties", "Laundering Rows", "Is Laundering Account"]
    ].copy()

    names = (trans[["Account", "Sender Bank Name", "Sender Entity Name"]]
             .drop_duplicates(subset="Account")
             .rename(columns={"Account": "Account ID",
                              "Sender Bank Name": "Bank",
                              "Sender Entity Name": "Entity"}))
    hubs = hubs.merge(names, on="Account ID", how="left")
    hubs["Share of all rows %"] = (hubs["Degree"] / (2 * len(trans)) * 100).round(3)
    return hubs


# ----------------------------------------------------- 2. jurisdiction


COUNTRY_HINTS = {
    "Australia", "Austria", "Belgium", "Brazil", "Bulgaria", "Canada", "China",
    "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hong Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Korea", "Latvia",
    "Lithuania", "Luxembourg", "Malaysia", "Malta", "Mexico", "Netherlands",
    "New Zealand", "Norway", "Pakistan", "Poland", "Portugal", "Romania",
    "Russia", "Saudi Arabia", "Singapore", "Slovakia", "Slovenia",
    "South Africa", "Spain", "Sweden", "Switzerland", "Taiwan", "Thailand",
    "Turkey", "UAE", "UK", "Ukraine", "USA", "Vietnam",
}

# Not a country. The dataset labels crypto exchanges "Crytpo Bank #n" - the
# misspelling is in the source data, kept verbatim so the match works. These
# are given their own label rather than being lumped in with domestic banks,
# because a crypto venue is a distinct jurisdiction-like category in AML terms.
NON_COUNTRY_LABELS = {"Crytpo": "Crypto exchange", "Crypto": "Crypto exchange"}

_BANK_PREFIX_RE = re.compile(r"^(.+?)\s+Bank\s+#\d+$")


def extract_bank_country(series, hints=None):
    """Pull a country from bank names shaped '<Country> Bank #1234'.

    Foreign banks in this dataset are named by country; domestic ones carry US
    city or brand names ('National Bank of Laramie', 'Golden Bancorp'). Anything
    that does not resolve to a known country is labelled Domestic/Other rather
    than guessed at.
    """
    hints = hints or COUNTRY_HINTS
    # Resolve on distinct names (tens of thousands) and map back, rather than
    # running a regex over every one of five million rows.
    uniques = pd.Index(series.dropna().unique())
    mapping = {}
    for name in uniques:
        match = _BANK_PREFIX_RE.match(str(name))
        prefix = match.group(1).strip() if match else None
        if prefix in hints:
            mapping[name] = prefix
        elif prefix in NON_COUNTRY_LABELS:
            mapping[name] = NON_COUNTRY_LABELS[prefix]
        else:
            mapping[name] = "Domestic/Other"
    return series.map(mapping).fillna("Domestic/Other").astype("category")


def bank_prefix_census(series, top=30):
    """Every '<X> Bank #n' prefix found, so the country list can be checked."""
    prefix = series.astype("string").str.extract(_BANK_PREFIX_RE, expand=False)
    census = prefix.value_counts().rename("Accounts").to_frame().head(top)
    census["Recognised as country"] = census.index.isin(COUNTRY_HINTS)
    census["Mapped to"] = [
        name if name in COUNTRY_HINTS
        else NON_COUNTRY_LABELS.get(name, "Domestic/Other")
        for name in census.index
    ]
    return census


def jurisdiction_analysis(trans, top_pairs=20):
    """Laundering rate by sender country, receiver country and country pair.

    Cross-border movement is the most widely used risk indicator in real AML
    practice, so a financial-crime audience will look for it first.
    """
    work = pd.DataFrame({
        "Sender Country": extract_bank_country(trans["Sender Bank Name"]),
        "Receiver Country": extract_bank_country(trans["Receiver Bank Name"]),
        "Is Laundering": trans["Is Laundering"],
    })
    work["Cross Border"] = work["Sender Country"].astype(str).ne(
        work["Receiver Country"].astype(str))

    by_sender = risk_by(work, "Sender Country").head(20)
    by_receiver = risk_by(work, "Receiver Country").head(20)
    by_cross = risk_by(work.assign(
        Flow=np.where(work["Cross Border"], "Cross-border", "Same jurisdiction")
    ), "Flow")

    pair = work.groupby(
        [work["Sender Country"].astype(str), work["Receiver Country"].astype(str)],
        observed=True,
    )["Is Laundering"].agg(Transactions="size", Laundering="sum")
    pair["Rate %"] = (pair["Laundering"] / pair["Transactions"] * 100).round(4)
    pair = pair[pair["Transactions"] >= 1000].sort_values("Rate %", ascending=False)

    return {
        "by_sender": by_sender,
        "by_receiver": by_receiver,
        "by_cross_border": by_cross,
        "riskiest_pairs": pair.head(top_pairs),
        "columns": work[["Sender Country", "Receiver Country", "Cross Border"]],
    }


# ---------------------------------------------------------- 3. entities


_ENTITY_TYPE_RE = re.compile(r"^(.+?)\s+#\d+$")


def extract_entity_type(series):
    """'Corporation #34854' -> 'Corporation'.

    Resolved on distinct entity names and mapped back: there are ~166k distinct
    names against five million rows, so this is roughly thirty times less work
    than a regex per row.
    """
    uniques = pd.Index(series.dropna().unique())
    mapping = {}
    for name in uniques:
        match = _ENTITY_TYPE_RE.match(str(name))
        mapping[name] = match.group(1).strip() if match else "Unknown"
    return series.map(mapping).fillna("Unknown").astype("category")


def entity_analysis(trans):
    """Entity type risk, and transfers that stay inside a single entity.

    Moving funds between accounts you already control is a standard layering
    step, so same-entity transfers are worth measuring directly.
    """
    # Built column by column rather than copying the name columns, which are
    # five million strings each.
    work = pd.DataFrame({
        "Sender Entity Type": extract_entity_type(trans["Sender Entity Name"]),
        "Receiver Entity Type": extract_entity_type(trans["Receiver Entity Name"]),
        "Same Entity": trans["Sender Entity ID"].eq(trans["Receiver Entity ID"]),
        "Is Laundering": trans["Is Laundering"],
    })

    same = risk_by(work.assign(
        Relationship=np.where(work["Same Entity"], "Same entity", "Different entity")
    ), "Relationship")

    pair = pd.crosstab(work["Sender Entity Type"], work["Receiver Entity Type"])
    pair_rate = pd.crosstab(work["Sender Entity Type"], work["Receiver Entity Type"],
                            values=work["Is Laundering"], aggfunc="mean").mul(100).round(4)

    return {
        "by_sender_type": risk_by(work, "Sender Entity Type"),
        "by_receiver_type": risk_by(work, "Receiver Entity Type"),
        "same_entity": same,
        "type_pair_volume": pair,
        "type_pair_rate": pair_rate,
        "columns": work[["Sender Entity Type", "Receiver Entity Type", "Same Entity"]],
    }


def accounts_per_entity(accounts):
    """How many accounts each entity controls."""
    per = accounts.groupby("Entity ID")["Account Number"].nunique().rename("Accounts")
    return per.describe().round(2), per.value_counts().sort_index().head(15)


# ------------------------------------------------------- 4. pass-through


def passthrough(trans, window_hours=24, tolerance=0.10):
    """Money in, money out: funds forwarded shortly after arrival.

    For every outgoing transfer, find that account's most recent *incoming*
    transfer within ``window_hours`` and compare the amounts. An outgoing
    transfer whose value is within ``tolerance`` of what just arrived is a
    pass-through - the behaviour that distinguishes a conduit account from one
    that simply holds money.

    Implemented with ``merge_asof`` so it stays linear rather than comparing
    every pair of transfers.
    """
    # Group on integer codes rather than account strings: merge_asof with a
    # string `by` key over 5M rows and ~500k groups is markedly slower, and the
    # codes only have to be consistent between the two frames.
    codes, _ = pd.factorize(
        pd.concat([trans["Account"], trans["Account.1"]], ignore_index=True)
    )
    n = len(trans)
    out_codes, in_codes = codes[:n], codes[n:]

    outgoing = (pd.DataFrame({
        "Acct": out_codes,
        "t_out": trans["Timestamp"].to_numpy(),
        "amt_out": trans["Amount Paid"].to_numpy(),
        "Is Laundering": trans["Is Laundering"].to_numpy(),
    }).sort_values("t_out", kind="mergesort"))

    incoming = (pd.DataFrame({
        "Acct": in_codes,
        "t_in": trans["Timestamp"].to_numpy(),
        "amt_in": trans["Amount Received"].to_numpy(),
    }).sort_values("t_in", kind="mergesort"))

    merged = pd.merge_asof(
        outgoing, incoming,
        left_on="t_out", right_on="t_in", by="Acct",
        direction="backward", tolerance=pd.Timedelta(hours=window_hours),
        allow_exact_matches=True,
    )

    merged["Dwell (hours)"] = (
        (merged["t_out"] - merged["t_in"]).dt.total_seconds() / 3600
    )
    ratio = merged["amt_out"] / merged["amt_in"].replace(0, np.nan)
    merged["Amount Ratio"] = ratio
    merged["Is Passthrough"] = (
        merged["t_in"].notna() & ratio.sub(1).abs().le(tolerance)
    )

    summary = merged.groupby("Is Laundering")["Is Passthrough"].agg(
        Transfers="size", Passthroughs="sum"
    )
    summary["Passthrough %"] = (
        summary["Passthroughs"] / summary["Transfers"] * 100
    ).round(3)

    dwell = merged.dropna(subset=["Dwell (hours)"]).groupby("Is Laundering")[
        "Dwell (hours)"
    ].describe().round(3)

    return merged, summary, dwell


# -------------------------------------------------------- 5. lifetime


def account_lifetime(trans):
    """First seen, last seen and active span per account.

    Conduit accounts are often short-lived: they appear, move funds and stop.
    Span is measured across both sending and receiving roles.
    """
    events = pd.concat([
        trans[["Account", "Timestamp", "Is Laundering"]]
        .rename(columns={"Account": "Account ID"}),
        trans[["Account.1", "Timestamp", "Is Laundering"]]
        .rename(columns={"Account.1": "Account ID"}),
    ], ignore_index=True)

    life = events.groupby("Account ID", observed=True).agg(
        First_Seen=("Timestamp", "min"),
        Last_Seen=("Timestamp", "max"),
        Events=("Timestamp", "size"),
        Laundering_Rows=("Is Laundering", "sum"),
    )
    life["Span (days)"] = (
        (life["Last_Seen"] - life["First_Seen"]).dt.total_seconds() / 86400
    )
    life["Is Laundering Account"] = life["Laundering_Rows"] > 0
    return life.reset_index()


def lifetime_contrast(life):
    return (life.groupby("Is Laundering Account")[["Span (days)", "Events"]]
            .describe().round(2).T)


# ------------------------------------------------------- 6. integrity


def integrity_checks(trans, accounts):
    """Checks the previous EDA never ran, each with a preprocessing consequence."""
    self_loop = trans["Account"].eq(trans["Account.1"])
    zero_paid = trans["Amount Paid"].eq(0)
    zero_recv = trans["Amount Received"].eq(0)

    checks = pd.DataFrame([
        {"Check": "Self-transfers (Account == Account.1)",
         "Rows": int(self_loop.sum()),
         "% of file": round(self_loop.mean() * 100, 4),
         "Laundering": int(trans.loc[self_loop, "Is Laundering"].sum())},
        {"Check": "Amount Paid == 0",
         "Rows": int(zero_paid.sum()),
         "% of file": round(zero_paid.mean() * 100, 4),
         "Laundering": int(trans.loc[zero_paid, "Is Laundering"].sum())},
        {"Check": "Amount Received == 0",
         "Rows": int(zero_recv.sum()),
         "% of file": round(zero_recv.mean() * 100, 4),
         "Laundering": int(trans.loc[zero_recv, "Is Laundering"].sum())},
        {"Check": "Exact duplicate rows",
         "Rows": int(trans.duplicated().sum()),
         "% of file": round(trans.duplicated().mean() * 100, 4),
         "Laundering": pd.NA},
    ])

    by_format = (trans.assign(Self=self_loop)
                 .groupby("Payment Format", observed=True)["Self"]
                 .agg(Rows="size", Self_Transfers="sum"))
    by_format["Self %"] = (
        by_format["Self_Transfers"] / by_format["Rows"] * 100
    ).round(2)

    pairs = accounts[["Bank Name", "Bank ID"]].drop_duplicates()
    ids_per_name = pairs.groupby("Bank Name")["Bank ID"].nunique()
    names_per_id = pairs.groupby("Bank ID")["Bank Name"].nunique()
    bank_consistency = {
        "Distinct bank names": int(pairs["Bank Name"].nunique()),
        "Distinct bank IDs": int(pairs["Bank ID"].nunique()),
        "Names mapping to >1 ID": int(ids_per_name.gt(1).sum()),
        "Max IDs for one name": int(ids_per_name.max()),
        "IDs mapping to >1 name": int(names_per_id.gt(1).sum()),
    }
    return checks, by_format, bank_consistency, ids_per_name.nlargest(10)


# ------------------------------------------------- 7. topology examples


def pick_example_attempt(patterns, profile, pattern_type, target_rows=8):
    """One representative attempt of a given topology, as an ordered transfer list."""
    candidates = profile[profile["Pattern Type"].eq(pattern_type)]
    if candidates.empty:
        return None, None
    best = (candidates.assign(_d=(candidates["Rows"] - target_rows).abs())
            .sort_values("_d").iloc[0])
    rows = (patterns[patterns["Attempt ID"].eq(best["Attempt ID"])]
            .sort_values("Timestamp"))
    return best, rows


def draw_attempt(rows, title, ax, node_colour="#9947dc", edge_colour="#6b21a8"):
    """Circular node-link diagram of one laundering attempt.

    Hand-rolled rather than pulled from networkx: this is the only graph drawing
    in the notebook and it is not worth an extra dependency the group then has
    to install.
    """
    nodes = list(dict.fromkeys(
        list(rows["Account"]) + list(rows["Account.1"])
    ))
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    counts = {}
    for _, r in rows.iterrows():
        a, b = pos[r["Account"]], pos[r["Account.1"]]
        key = (r["Account"], r["Account.1"])
        counts[key] = counts.get(key, 0) + 1
        curve = 0.12 * counts[key]
        ax.annotate(
            "", xy=b, xytext=a,
            arrowprops=dict(arrowstyle="-|>", color=edge_colour, alpha=0.75,
                            lw=1.2, shrinkA=9, shrinkB=9,
                            connectionstyle=f"arc3,rad={curve}"),
        )

    xs = [pos[nd][0] for nd in nodes]
    ys = [pos[nd][1] for nd in nodes]
    ax.scatter(xs, ys, s=320, c=node_colour, zorder=3, edgecolors="white", linewidths=1.5)
    for nd in nodes:
        ax.annotate(nd[-4:], pos[nd], ha="center", va="center",
                    fontsize=6, color="white", zorder=4, fontweight="bold")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.axis("off")


# -------------------------------------------- 8. candidate feature table


def candidate_features(trans, account_df, life):
    """One table summarising every candidate signal found in this EDA.

    The handoff artefact for preprocessing: what each feature is, how the two
    classes differ on it, and whether it is worth carrying forward.
    """
    rows = []

    def add(name, legit, laund, note):
        try:
            ratio = round(float(laund) / float(legit), 2) if float(legit) else np.nan
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = np.nan
        rows.append({"Feature": name, "Legitimate": legit, "Laundering": laund,
                     "Ratio": ratio, "Note": note})

    g = trans.groupby("Is Laundering", observed=True)
    add("Amount Paid (median)",
        round(g["Amount Paid"].median().get(0, np.nan), 2),
        round(g["Amount Paid"].median().get(1, np.nan), 2),
        "Monetary scale")

    cross = trans["From Bank"].ne(trans["To Bank"])
    add("Cross-bank share %",
        round(cross[trans["Is Laundering"].eq(0)].mean() * 100, 2),
        round(cross[trans["Is Laundering"].eq(1)].mean() * 100, 2),
        "Network routing")

    ach = trans["Payment Format"].astype(str).eq("ACH")
    add("ACH share %",
        round(ach[trans["Is Laundering"].eq(0)].mean() * 100, 2),
        round(ach[trans["Is Laundering"].eq(1)].mean() * 100, 2),
        "Channel - check for generator artefact")

    a = account_df.groupby("Is Laundering Account")
    add("Account degree (median)",
        a["Degree"].median().get(False, np.nan),
        a["Degree"].median().get(True, np.nan), "Account level")
    add("Counterparties (median)",
        a["Counterparties"].median().get(False, np.nan),
        a["Counterparties"].median().get(True, np.nan), "Account level")

    l = life.groupby("Is Laundering Account")
    add("Active span days (median)",
        round(l["Span (days)"].median().get(False, np.nan), 2),
        round(l["Span (days)"].median().get(True, np.nan), 2), "Account level")

    return pd.DataFrame(rows)
