
"""Shared evidence and response types used by the agents."""

from dataclasses import dataclass
from typing import Literal, Sequence


# Keep evidence sources consistent across all agent responses.
EvidenceSource = Literal["transaction", "case_record", "regulation", "model_output"]
AgentName = Literal["prosecution", "defense"]


@dataclass(frozen=True)
class Evidence:
    """One fact the agents may use to support their findings."""

    id: str
    source: EvidenceSource
    reference: str
    description: str


@dataclass(frozen=True)
class Finding:
    """One claim and the evidence IDs that support it."""

    claim: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class AgentReport:
    """The shared response format for prosecution and defense."""

    agent: AgentName
    findings: list[Finding]


def validate_report(report: AgentReport, evidence: Sequence[Evidence]) -> None:
    """Raise ValueError if a report is incomplete or cites unknown evidence."""
    if report.agent not in ("prosecution", "defense"):
        raise ValueError("agent must be 'prosecution' or 'defense'.")

    # Build the set of valid evidence IDs and reject ambiguous IDs.
    evidence_ids = set()
    for item in evidence:
        if not item.id.strip():
            raise ValueError("evidence IDs cannot be empty.")
        if item.id in evidence_ids:
            raise ValueError(f"duplicate evidence ID: {item.id!r}.")
        if item.source not in (
            "transaction",
            "case_record",
            "regulation",
            "model_output",
        ):
            raise ValueError(f"unsupported evidence source: {item.source!r}.")
        if not item.reference.strip() or not item.description.strip():
            raise ValueError(f"evidence {item.id!r} needs a reference and description.")
        evidence_ids.add(item.id)

    # Every finding must make a claim that points to supplied evidence.
    if not report.findings:
        raise ValueError("report must contain at least one finding.")
    for number, finding in enumerate(report.findings, start=1):
        if not finding.claim.strip():
            raise ValueError(f"finding {number} needs a claim.")
        if not finding.evidence_ids:
            raise ValueError(f"finding {number} must cite at least one evidence ID.")
        unknown_ids = set(finding.evidence_ids) - evidence_ids
        if unknown_ids:
            raise ValueError(
                f"finding {number} cites unknown evidence IDs: {sorted(unknown_ids)!r}."
            )
