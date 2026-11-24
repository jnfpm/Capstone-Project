# Load source transactions for one prepared alert without reading training labels.  # Explain the module purpose.

from __future__ import annotations  # Keep type annotations compatible with Python 3.12.

import hashlib  # Create stable source identifiers for transaction records.
import json  # Serialize selected source facts consistently before hashing them.
from dataclasses import dataclass  # Return retrieved rows and coverage as one result.
from pathlib import Path  # Handle the joined dataset and FX table paths.
from typing import Mapping  # Describe the prepared alert metadata accepted by the loader.

import duckdb  # Query the large Parquet source without loading it all into memory.
import pandas as pd  # Normalize timestamps and values returned by DuckDB.

from agents.case_context import TRANSACTION_FIELDS  # Reuse the context's source-field allowlist.

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Find the repository root from this file.
AML_ROOT = PROJECT_ROOT / "aml"  # Use the current AML pipeline, not the legacy alerts_model copy.
DEFAULT_JOINED_PATH = AML_ROOT / "data" / "processed" / "hi_small_joined.parquet"  # Locate source rows.
DEFAULT_FX_PATH = AML_ROOT / "data" / "processed" / "preprocessed" / "fx_table.csv"  # Locate USD rates.
HISTORY_WINDOW = pd.Timedelta(hours=48)  # Match the AML pipeline's fixed history look-back.
MAX_HISTORY_ROWS = 20  # Bound prompt size while keeping the most recent related transfers.
SOURCE_COLUMNS = (  # Select only transaction and descriptive fields that agents may inspect.
    '"Timestamp"',  # Preserve the source transaction time.
    '"From Bank"',  # Preserve the sender bank identifier.
    '"To Bank"',  # Preserve the receiver bank identifier.
    '"Account"',  # Preserve the sender account identifier.
    '"Account.1"',  # Preserve the receiver account identifier.
    '"Amount Paid"',  # Preserve the source payment amount.
    '"Amount Received"',  # Preserve the source received amount.
    '"Payment Format"',  # Preserve the source payment format.
    '"Payment Currency"',  # Preserve the currency paid.
    '"Receiving Currency"',  # Preserve the currency received.
    '"Bank Name_from"',  # Preserve the sender bank name when available.
    '"Bank Name_to"',  # Preserve the receiver bank name when available.
    '"Country_from"',  # Preserve the sender country when available.
    '"Country_to"',  # Preserve the receiver country when available.
)  # End the explicit source-column list and exclude all label columns.
OUTPUT_COLUMNS = (  # Name the selected fields exposed to the agent context.
    "Timestamp",  # Keep the source transaction time.
    "From Bank",  # Keep the sender bank identifier.
    "To Bank",  # Keep the receiver bank identifier.
    "Account",  # Keep the sender account identifier.
    "Account.1",  # Keep the receiver account identifier.
    "Amount Paid",  # Keep the source payment amount.
    "Amount Received",  # Keep the source received amount.
    "Payment Format",  # Keep the source payment format.
    "Payment Currency",  # Keep the currency paid.
    "Receiving Currency",  # Keep the currency received.
    "Bank Name_from",  # Keep the sender bank name when available.
    "Bank Name_to",  # Keep the receiver bank name when available.
    "Country_from",  # Keep the sender country when available.
    "Country_to",  # Keep the receiver country when available.
)  # End the source allowlist returned by this module.


@dataclass(frozen=True)  # Keep the lookup result immutable after construction.
class CaseDataResult:  # Bundle retrieved transactions with a clear coverage note.
    transactions: tuple[Mapping[str, object], ...]  # Hold the target and related source rows.
    coverage_note: str  # Tell the agents the lookup window and whether results were capped.


def _sql_path(path: Path) -> str:  # Safely quote a local path for DuckDB's Parquet reader.
    return str(path.resolve()).replace("'", "''")  # Escape SQL quote characters in the path.


def _stable_source_id(row: Mapping[str, object]) -> str:  # Build a reproducible ID from source facts.
    facts = {  # Keep the ID independent of labels and unrelated joined columns.
        key: str(row.get(key, ""))  # Normalize source values into stable JSON strings.
        for key in OUTPUT_COLUMNS  # Hash only the fields explicitly approved for review.
    }  # Finish the stable source-fact mapping.
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"))  # Canonicalize the facts.
    return "txn-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]  # Return a stable ID.


def _fx_rates(fx_path: Path) -> dict[str, float]:  # Load the pipeline's training-derived FX table.
    if not fx_path.is_file():  # Do not guess conversion rates when the artifact is absent.
        return {}  # Leave disambiguation to the exact account, time, and payment fields.
    rates = pd.read_csv(fx_path)  # Read the small, generated FX lookup table.
    required = {"currency", "units_per_usd"}  # Require the columns written by preprocessing.
    if not required.issubset(rates.columns):  # Reject incompatible FX artifacts clearly.
        raise ValueError(f"FX table must contain columns {sorted(required)}.")  # Explain the mismatch.
    return {str(row.currency): float(row.units_per_usd) for row in rates.itertuples()}  # Index rates.


def _prepare_record(row: Mapping[str, object]) -> dict[str, object]:  # Convert a source row for context.
    record = {key: row.get(key) for key in OUTPUT_COLUMNS}  # Copy only the approved source fields.
    record["acct_from"] = f"{row['From Bank']}:{row['Account']}"  # Add the stable sender account key.
    record["acct_to"] = f"{row['To Bank']}:{row['Account.1']}"  # Add the stable receiver account key.
    record["country_from"] = row.get("Country_from")  # Map the sender country to the context field name.
    record["country_to"] = row.get("Country_to")  # Map the receiver country to the context field name.
    record["transaction_id"] = _stable_source_id(row)  # Attach a reproducible citation source ID.
    return {  # Return only facts accepted by the evidence builder.
        key: value  # Preserve each selected fact.
        for key, value in record.items()  # Inspect the explicitly selected record fields.
        if key in TRANSACTION_FIELDS or key == "transaction_id"  # Exclude everything outside the allowlist.
    }  # Finish the safe context record.


def _select_target(  # Resolve the exact alert to one row in the source dataset.
    candidates: pd.DataFrame, alert: Mapping[str, object], fx_rates: Mapping[str, float]  # Accept candidates and prepared alert fields.
) -> pd.Series:  # Return the unique matching source transaction.
    if candidates.empty:  # Detect a source row missing from the joined dataset.
        raise LookupError("The selected alert transaction was not found in the joined source data.")  # Fail clearly.
    matches = candidates  # Start with exact timestamp, accounts, and payment format matches.
    target_usd = alert.get("amount_usd")  # Use the prepared USD amount to disambiguate collisions.
    if target_usd is not None and fx_rates:  # Verify the prepared USD amount when training rates exist.
        target_value = float(target_usd)  # Normalize the prepared amount for numeric comparison.
        keep = []  # Track candidate rows with a matching training-derived FX conversion.
        for index, row in matches.iterrows():  # Compare each candidate's paid amount in USD.
            currency = str(row["Payment Currency"])  # Read the source payment currency.
            units_per_usd = fx_rates.get(currency)  # Look up the pipeline's learned conversion rate.
            if units_per_usd and abs(float(row["Amount Paid"]) / units_per_usd - target_value) <= max(1e-6, abs(target_value) * 1e-6):  # Compare with small numeric tolerance.
                keep.append(index)  # Retain candidates matching the prepared USD value.
        matches = matches.loc[keep]  # Keep only disambiguated source rows.
    if len(matches) != 1:  # Prevent attaching ambiguous records to the alert.
        raise LookupError(f"Expected one source row for alert, found {len(matches)} matching rows.")  # Fail closed.
    return matches.iloc[0]  # Return the single source row.


def load_case_data(  # Retrieve the target transaction and recent account/counterparty history.
    row_id: int,  # Identify the prepared alert row to resolve.
    alert: Mapping[str, object],  # Supply safe alert metadata from the prepared files.
    *,  # Require source paths to be explicit keyword options.
    joined_path: Path = DEFAULT_JOINED_PATH,  # Use the current joined AML dataset by default.
    fx_path: Path = DEFAULT_FX_PATH,  # Use the preprocessing FX table when available.
    max_history_rows: int = MAX_HISTORY_ROWS,  # Bound the number of prior rows placed in context.
) -> CaseDataResult:  # Return source records plus a coverage statement.
    joined_path = Path(joined_path)  # Normalize the joined dataset path.
    if not joined_path.is_file():  # Do not silently review an alert without its source data.
        raise FileNotFoundError(f"Joined transaction data is missing: {joined_path}")  # Name the required artifact.
    if max_history_rows < 1:  # Ensure at least one history row can be considered.
        raise ValueError("max_history_rows must be at least 1.")  # Reject invalid retrieval limits.
    timestamp = pd.Timestamp(alert["Timestamp"])  # Use the alert's prepared timestamp as cutoff.
    acct_from = str(alert["acct_from"])  # Read the sender account key from prepared metadata.
    acct_to = str(alert["acct_to"])  # Read the receiver account key from prepared metadata.
    payment_format = str(alert["Payment Format"])  # Read the alert payment format for matching.
    from_bank, from_account = acct_from.split(":", 1)  # Split the sender bank/account key.
    to_bank, to_account = acct_to.split(":", 1)  # Split the receiver bank/account key.
    source_sql = ", ".join(SOURCE_COLUMNS)  # Build an explicit projection excluding labels.
    parquet = _sql_path(joined_path)  # Escape the source path before embedding it in SQL.
    target_query = (  # Match the selected alert on stable source fields.
        f"SELECT {source_sql} FROM read_parquet('{parquet}') "  # Read only the approved source columns.
        'WHERE "Timestamp" = ? AND CAST("From Bank" AS VARCHAR) = ? '  # Match time and sender identity.
        'AND "Account" = ? AND CAST("To Bank" AS VARCHAR) = ? '  # Match sender and receiver accounts.
        'AND "Account.1" = ? AND "Payment Format" = ?'  # Finish the target transaction key.
    )  # Finish the source lookup query.
    history_query = (  # Fetch recent rows involving either alert account.
        f"SELECT {source_sql} FROM read_parquet('{parquet}') "  # Read only the approved source columns.
        'WHERE "Timestamp" >= ? AND "Timestamp" < ? '  # Enforce the 48-hour lookback and alert cutoff.
        "AND ((CAST(\"From Bank\" AS VARCHAR) || ':' || \"Account\") IN (?, ?) "  # Match sender accounts.
        "OR (CAST(\"To Bank\" AS VARCHAR) || ':' || \"Account.1\") IN (?, ?)) "  # Match receiver accounts.
        'ORDER BY "Timestamp" DESC, "From Bank", "Account", "To Bank", "Account.1", '  # Sort ties deterministically.
        '"Amount Paid", "Amount Received", "Payment Format", "Payment Currency", '  # Continue stable ordering.
        '"Receiving Currency" LIMIT ?'  # Bound the result after stable sorting.
    )  # Finish the history lookup query.
    with duckdb.connect(database=":memory:") as connection:  # Use a temporary query connection.
        candidates = connection.execute(  # Query candidate rows for the selected alert.
            target_query,  # Use the explicit target lookup query.
            [timestamp.to_pydatetime(), from_bank, from_account, to_bank, to_account, payment_format],  # Bind safe target values.
        ).fetchdf()  # Retrieve candidates as a DataFrame.
        fx_rates = _fx_rates(Path(fx_path))  # Load rates for resolving candidate collisions.
        target = _select_target(candidates, alert, fx_rates)  # Require one exact source transaction.
        history = connection.execute(  # Query recent account history before the alert timestamp.
            history_query,  # Use the bounded history lookup query.
            [  # Bind the time window and both alert account IDs.
                (timestamp - HISTORY_WINDOW).to_pydatetime(),  # Set the inclusive lookback start.
                timestamp.to_pydatetime(),  # Set the exclusive alert-time cutoff.
                acct_from,  # Match sender-side history for the alert's sender.
                acct_to,  # Match sender-side history for the alert's receiver.
                acct_from,  # Match receiver-side history for the alert's sender.
                acct_to,  # Match receiver-side history for the alert's receiver.
                max_history_rows + 1,  # Fetch one extra row to detect truncation.
            ],  # Finish the bound history parameters.
        ).fetchdf()  # Retrieve recent history as a DataFrame.
    truncated = len(history) > max_history_rows  # Detect whether the retrieval limit clipped history.
    history = history.iloc[:max_history_rows]  # Keep only the newest bounded history rows.
    records = [_prepare_record(target.to_dict())]  # Put the selected alert transaction first.
    records.extend(  # Append prior transfers after the selected alert transaction.
        _prepare_record(row)  # Convert each source row to the safe evidence shape.
        for row in history.to_dict(orient="records")  # Process the bounded prior-history result.
    )  # Finish the retrieved transaction list.
    coverage = (  # Describe what the history lookup did and did not retrieve.
        f"Source: joined AML transactions; target row_id={row_id}; "  # Identify the source and target alert.
        "prior-history window: 48 hours before the alert timestamp; "  # State the lookback window.
        f"returned {len(history)} prior transfers involving either alert account; "  # State the result size.
        f"history truncated: {'yes' if truncated else 'no'}."  # State whether the row cap applied.
    )  # Finish the retrieval coverage note.
    return CaseDataResult(transactions=tuple(records), coverage_note=coverage)  # Return facts and coverage.
