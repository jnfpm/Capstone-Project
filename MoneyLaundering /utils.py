import numpy as np
import pandas as pd
import kagglehub
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# Column names created by eu.link_accounts - check them with print(trans.columns) and adjust if needed
SENDER_BANK_NAME = "Sender Bank Name"
RECEIVER_BANK_NAME = "Receiver Bank Name"
SENDER_ENTITY_NAME = "Sender Entity Name"
RECEIVER_ENTITY_NAME = "Receiver Entity Name"
PASS_THROUGH_MINUTES = 60   # money forwarded within this window after arriving counts as pass-through


def get_kaggle_dataset_dir(dataset="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"):
    """Download dataset directly from Kaggle using kagglehub and return folder Path."""
    from pathlib import Path
    return Path(kagglehub.dataset_download(dataset))


def load_kaggle_data(dataset="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"):
    """Download from Kaggle and load CSVs directly into DataFrames."""
    from pathlib import Path
    data_dir = get_kaggle_dataset_dir(dataset)
    trans = pd.read_csv(data_dir / "HI-Small_Trans.csv")
    accounts = pd.read_csv(data_dir / "HI-Small_accounts.csv")
    return trans, accounts



def preprocess(trans):
    """trans: transactions (already linked with the accounts and patterns files)
    Return X (features), y (target) and meta (necessary columns for split)."""

    # CLEANING
    # 1. Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(trans["Timestamp"]):
        trans = trans.copy()
        trans["Timestamp"] = pd.to_datetime(trans["Timestamp"])

    # 2. Ensure chronological order (critical for point-in-time features: cumcount, diff, dwell)
    if not trans["Timestamp"].is_monotonic_increasing:
        trans = trans.sort_values("Timestamp")

    # Keep only data until 11th of September and index from 0 to N
    # The 9 exact duplicate rows are kept (a repeated row can be a real repeated payment)
    df = trans[trans["Timestamp"] < "2022-09-11"].reset_index(drop=True)

    # UNIQUE IDENTIFIER
    # Each account, built from two parts: the bank code and the account number.
    sender = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    receiver = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)

    # FEATURES
    # Empty table with the same row labels as df, so every feature we add lands on the same transaction as in y and meta
    X = pd.DataFrame(index=df.index)

    # AMOUNT
    # Log transformation
    X["log_amount"] = np.log1p(df["Amount Paid"])

    # Ratio between amount received and paid - exchange 0 for NaN
    X["amount_ratio"] = df["Amount Received"] / df["Amount Paid"].replace(0, np.nan)

    # 1 - different currency in transfer; 0 - same currency in transfer
    X["currency_mismatch"] = (df["Payment Currency"] != df["Receiving Currency"]).astype(int)

    # Who is talking to whom 
    # 1 - intrabank; 0 - interbank transfer
    X["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(int)

    # 1 - same sender and receiver; 0 - different sender and receiver
    X["self_transfer"] = (sender == receiver).astype(int)

    # 1 - both accounts belong to the same entity (22x safer in the EDA); 0 - different entities
    X["same_entity"] = (df["Sender Entity ID"] == df["Receiver Entity ID"]).astype(int)

    # Entity type = entity name without the number (e.g. "Sole Proprietorship #41" -> "Sole Proprietorship")
    X["sender_entity_type"] = df[SENDER_ENTITY_NAME].str.replace(r"\s*#\d+\s*$", "", regex=True)
    X["receiver_entity_type"] = df[RECEIVER_ENTITY_NAME].str.replace(r"\s*#\d+\s*$", "", regex=True)

    # GEOGRAPHY
    # Foreign banks are named "<Country> Bank #<id>"; names that don't follow this pattern are domestic
    sender_country = df[SENDER_BANK_NAME].str.extract(r"^(.*?)\s+Bank\s+#\d+$")[0].fillna("Domestic")
    receiver_country = df[RECEIVER_BANK_NAME].str.extract(r"^(.*?)\s+Bank\s+#\d+$")[0].fillna("Domestic")
    # The source data spells crypto exchanges as "Crytpo"
    sender_country = sender_country.replace({"Crytpo": "Crypto"})
    receiver_country = receiver_country.replace({"Crytpo": "Crypto"})

    X["sender_country"] = sender_country
    X["receiver_country"] = receiver_country

    # 1 - sender and receiver in different countries (4x laundering rate in the EDA); 0 - same country
    X["cross_border"] = (sender_country != receiver_country).astype(int)

    # Country route (e.g. "Spain>Finland"); rare routes are grouped in "Other" after the split
    X["corridor"] = sender_country + ">" + receiver_country

    # TIME 
    # No recorded time transactions are marked as 00:00;
    # 1 - time between 00:00 and 00:59; else - 0
    X["is_midnight"] = (df["Timestamp"].dt.hour == 0).astype(int)

    # Account history (point-in-time: only looks at what happened BEFORE each transaction) 
    # How many transactions the sender already made before this one
    X["sender_prior_tx_count"] = df.groupby(sender).cumcount()

    # How many transactions the receiver already received before this one
    X["receiver_prior_recv_count"] = df.groupby(receiver).cumcount()

    # Minutes since the sender's previous transaction (NaN on its first one)
    X["sender_gap_min"] = df.groupby(sender)["Timestamp"].diff().dt.total_seconds() / 60

    # For each transaction, find the last time the sender RECEIVED money strictly before it
    sent = pd.DataFrame({"account": sender, "Timestamp": df["Timestamp"], "row": df.index})
    received = pd.DataFrame({"account": receiver, "last_received": df["Timestamp"]})
    # merge_asof needs both tables sorted by time
    sent = sent.sort_values("Timestamp")
    received = received.sort_values("last_received")
    last = pd.merge_asof(sent, received, left_on="Timestamp", right_on="last_received",
                         by="account", allow_exact_matches=False)
    last = last.set_index("row").sort_index()

    # Minutes the money stayed in the sender's account before being sent (NaN if it never received before)
    dwell = (last["Timestamp"] - last["last_received"]).dt.total_seconds() / 60
    X["sender_dwell_min"] = dwell

    # 1 - sender forwards money that just arrived (strongest signal in the EDA); 0 - otherwise
    X["sender_pass_through"] = (dwell <= PASS_THROUGH_MINUTES).astype(int)

    # Remaining columns 
    for c in ["Payment Format", "Payment Currency", "Receiving Currency"]:
        # copy column for new df, encoding in the split phase
        X[c] = df[c]

    # NOT done here (they learn from data, so they go after the split, fitted on train only):
    # hub flag, Amount_USD, amount percentile within currency, capping the count features,
    # encoding the text columns

    # TARGET:
    # 1 - laundering; 0 - no laundering
    y = df["Is Laundering"]

    # META: columns that won't be in our model but will be required for the splitting phase
    meta = pd.DataFrame({
        "Timestamp": df["Timestamp"],               # datetime, good for a split based on time
        "sender": sender,                           # sender account, used to link accounts in the split
        "receiver": receiver,                       # receiver account, used to link accounts in the split
        "Attempt ID": df["Attempt ID"],             # which laundering scheme is it part of?
        "Pattern Type": df["Pattern Type"],         # scheme topology, to report recall by topology
        "In Patterns File": df["In Patterns File"], # False = positive with no scheme behind it
    })
    # return final df
    return X, y, meta


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure this folder is in sys.path so eda_utils can be imported
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    import eda_utils as eu

    # 1. Download files directly via kagglehub
    data_dir = get_kaggle_dataset_dir()
    print("Path to dataset files:", data_dir)

    trans_path = data_dir / "HI-Small_Trans.csv"
    accounts_path = data_dir / "HI-Small_accounts.csv"
    patterns_path = data_dir / "HI-Small_Patterns.txt"

    # 2. Load data
    print("Loading datasets...")
    trans = eu.load_transactions(trans_path)
    accounts = eu.load_accounts(accounts_path)
    patterns, _, _ = eu.load_patterns(patterns_path)

    # 3. Connect accounts and patterns
    print("Linking accounts and patterns...")
    trans, _ = eu.link_accounts(trans, accounts)
    trans, _, _ = eu.link_patterns(trans, patterns)

    # 4. Preprocessing
    print("Running preprocessing...")
    X, y, meta = preprocess(trans)

    # 5. Save processed results
    out = current_dir / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out / "X.parquet")
    y.to_frame().to_parquet(out / "y.parquet")
    meta.to_parquet(out / "meta.parquet")

    print(f"Done! X: {X.shape}, y: {int(y.sum())} positives")



