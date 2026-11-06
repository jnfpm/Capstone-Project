"""Prepare one alert's facts and model results for the agents."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

from agents.contracts import Evidence, EvidenceSource

# These are useful case details. Training labels are intentionally not included.
ALERT_FIELDS = (
    "Timestamp",
    "acct_from",
    "acct_to",
    "amount_usd",
    "Payment Format",
    "country_from",
    "country_to",
)
TRANSACTION_FIELDS = (
    "Timestamp",
    "From Bank",
    "To Bank",
    "Account",
    "Account.1",
    "Amount Paid",
    "Amount Received",
    "Payment Format",
    "Payment Currency",
    "amount_usd",
    "acct_from",
    "acct_to",
    "country_from",
    "country_to",
)


@dataclass(frozen=True)
class CaseContext:
    """The evidence prepared for one alert review."""

    alert_id: str
    evidence: tuple[Evidence, ...]


def _slug(value: str) -> str:
    """Make an ID part from a value without changing its meaning elsewhere."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "unknown"


def _stable_key(value: str) -> str:
    """Keep IDs readable while distinguishing values with the same slug."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(value)}-{digest}"


def _display(value: object) -> str | None:
    """Turn a simple data value into readable text, skipping missing numbers."""
    if value is None:
        return None
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return None
    text = str(value).strip()
    if text.lower() in ("", "nan", "nat", "<na>"):
        return None
    return text


def _transaction_id(transaction: Mapping[str, object]) -> str:
    """Use a stable row ID, or hash the allowed transaction facts."""
    for key in ("row_id", "transaction_id", "id"):
        value = _display(transaction.get(key))
        if value:
            return _stable_key(value)

    facts = {
        key: _display(transaction[key])
        for key in TRANSACTION_FIELDS
        if key in transaction and _display(transaction[key]) is not None
    }
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _add_facts(
    evidence: list[Evidence],
    *,
    evidence_id: str,
    source: EvidenceSource,
    reference: str,
    facts: Mapping[str, object],
    allowed_fields: Sequence[str],
) -> None:
    """Add only the selected fields as separately citeable evidence."""
    for field in allowed_fields:
        if field not in facts:
            continue
        value = _display(facts[field])
        if value is None:
            continue
        evidence.append(
            Evidence(
                id=f"{evidence_id}-{_slug(field)}",
                source=source,
                reference=reference,
                description=f"{field}: {value}",
            )
        )


def build_case_context(
    alert_id: str,
    alert: Mapping[str, object],
    transactions: Sequence[Mapping[str, object]],
    model_output: Mapping[str, object],
) -> CaseContext:
    """Build stable, citeable evidence without exposing training labels."""
    if not alert_id.strip():
        raise ValueError("alert_id cannot be empty.")

    alert_key = _stable_key(alert_id)
    evidence: list[Evidence] = []

    # Keep alert facts and transaction facts separate for clear citations.
    _add_facts(
        evidence,
        evidence_id=f"alert-{alert_key}",
        source="case_record",
        reference=f"alert {alert_id}",
        facts=alert,
        allowed_fields=ALERT_FIELDS,
    )

    # Duplicate row IDs would make citations ambiguous, so reject them.
    seen_transactions: set[str] = set()
    duplicate_counts: Counter[str] = Counter()
    for transaction in transactions:
        transaction_key = _transaction_id(transaction)
        if transaction_key in seen_transactions and any(
            _display(transaction.get(key)) for key in ("row_id", "transaction_id", "id")
        ):
            raise ValueError(f"duplicate transaction ID: {transaction_key!r}.")
        seen_transactions.add(transaction_key)
        duplicate_counts[transaction_key] += 1
        suffix = (
            f"-{duplicate_counts[transaction_key]}"
            if duplicate_counts[transaction_key] > 1
            else ""
        )
        _add_facts(
            evidence,
            evidence_id=f"transaction-{transaction_key}{suffix}",
            source="transaction",
            reference=f"transaction {transaction_key}",
            facts=transaction,
            allowed_fields=TRANSACTION_FIELDS,
        )

    # Scores and explanations remain labeled as model output, not factual proof.
    for field in ("score", "alert"):
        value = _display(model_output.get(field))
        if value is not None:
            evidence.append(
                Evidence(
                    id=f"model-{alert_key}-{field}",
                    source="model_output",
                    reference=f"scoring model output for alert {alert_id}",
                    description=f"Model output {field}: {value}",
                )
            )

    reasons = model_output.get("reasons")
    if isinstance(reasons, str):
        reason_values = [reasons]
    elif isinstance(reasons, Sequence) and not isinstance(reasons, (bytes, bytearray)):
        reason_values = list(reasons)
    else:
        reason_values = []
    for index, reason in enumerate(reason_values, start=1):
        value = _display(reason)
        if value is not None:
            evidence.append(
                Evidence(
                    id=f"model-{alert_key}-reason-{index}",
                    source="model_output",
                    reference=f"scoring model explanation for alert {alert_id}",
                    description=f"Model output reason: {value}",
                )
            )

    return CaseContext(alert_id=alert_id, evidence=tuple(evidence))
