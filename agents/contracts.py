"""Shared evidence and response types used by the agents."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# Keep evidence sources consistent across all agent responses.
EvidenceSource = Literal["transaction", "case_record", "regulation", "model_output"]
AgentName = Literal["prosecution", "defense"]
MediatorOutcome = Literal["supported", "partially_supported", "not_supported", "inconclusive"]

# This is the answer format we send to Anthropic. Think of it as a blank form:
# the model must fill in the agent name, conclusions and evidence IDs.
# minItems=1 means "at least one". additionalProperties=False means "no extra fields".
# A correct form can still contain a wrong claim, so we also read the reports.
REPORT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "enum": ["prosecution", "defense"]},
        "findings": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    # Mirror the existing non-empty citation rule at the API boundary.
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["claim", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["agent", "findings"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Evidence:
    """One fact the agents may use to support their findings."""

    # id: the label a report cites. description: the actual supplied fact.
    # source says what kind of record it is; reference points to its origin.
    id: str
    source: EvidenceSource
    reference: str
    description: str


@dataclass(frozen=True)
class Finding:
    """One claim and the evidence IDs that support it."""

    # Keep the sentence and its references together. Each conclusion must
    # have its own references, even if another conclusion used them already.
    claim: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class AgentReport:
    """The shared response format for prosecution and defense."""

    agent: AgentName
    findings: list[Finding]


@dataclass(frozen=True)
class ReplyFinding: # Nós estamos a criar isto porque vamos ter  de responder a varias conclusoes
    """One response to a numbered conclusion in the prosecution report."""

    # Number 1 means the first prosecution finding, not an evidence ID.
    prosecution_finding: int
    position: Literal["agree", "qualify", "disagree"]
    claim: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class DefenseReply:
    """The defense's single reply, kept separate from its original report."""

    agent: Literal["defense"]
    replies: list[ReplyFinding]


@dataclass(frozen=True)
class MediatorFinding:
    """The mediator's evidence-based assessment of one prosecution finding."""

    # This points to a conclusion, not an evidence item; factual claims cite IDs below.
    prosecution_finding: int
    outcome: MediatorOutcome
    assessment: str
    evidence_ids: list[str]
    open_questions: list[str]


@dataclass(frozen=True)
class MediatorResolution:
    """A bounded synthesis that does not make a guilt or innocence determination."""

    agent: Literal["mediator"]
    findings: list[MediatorFinding]


# The reply needs a different form from an independent report: every point
# must say which prosecution conclusion it addresses. Evidence stays separate.
REPLY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "enum": ["defense"]},
        "replies": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "prosecution_finding": {"type": "integer", "minimum": 1},
                    "position": {"type": "string", "enum": ["agree", "qualify", "disagree"]},
                    "claim": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": ["prosecution_finding", "position", "claim", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["agent", "replies"],
    "additionalProperties": False,
}


MEDIATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "enum": ["mediator"]},
        "findings": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "prosecution_finding": {"type": "integer", "minimum": 1},
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "partially_supported",
                            "not_supported",
                            "inconclusive",
                        ],
                    },
                    "assessment": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "open_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "prosecution_finding",
                    "outcome",
                    "assessment",
                    "evidence_ids",
                    "open_questions",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["agent", "findings"],
    "additionalProperties": False,
}


def validate_reply(
    reply: DefenseReply, prosecution: AgentReport, evidence: Sequence[Evidence]
) -> None:
    """Check response targets and citations, not whether the reasoning is true."""
    if reply.agent != "defense":
        raise ValueError("reply agent must be 'defense'.")
    if prosecution.agent != "prosecution":
        raise ValueError("reply requires a prosecution report.")
    validate_report(prosecution, evidence)
    seen_targets = set()
    for item in reply.replies:
        target = item.prosecution_finding
        # bool is a subclass of int in Python; True must not mean conclusion 1.
        if type(target) is not int or not 1 <= target <= len(prosecution.findings):
            raise ValueError("reply targets an unknown prosecution finding.")
        if target in seen_targets:
            raise ValueError("combine replies to the same prosecution finding.")
        seen_targets.add(target)
        if item.position not in ("agree", "qualify", "disagree"):
            raise ValueError("reply position must be agree, qualify or disagree.")

    # Reuse the existing checks for non-empty claims and original evidence IDs.
    # A prosecution conclusion number is never accepted as a new evidence fact.
    validate_report(
        AgentReport("defense", [Finding(item.claim, item.evidence_ids) for item in reply.replies]),
        evidence,
    )


def validate_mediation(
    resolution: MediatorResolution,
    prosecution: AgentReport,
    defense: AgentReport,
    reply: DefenseReply,
    evidence: Sequence[Evidence],
) -> None:
    """Validate targets and evidence while leaving the truth of conclusions to review."""
    if resolution.agent != "mediator":
        raise ValueError("resolution agent must be 'mediator'.")
    if defense.agent != "defense":
        raise ValueError("mediation requires a defense report.")

    # All inputs are checked before the mediator's output is trusted.
    validate_report(prosecution, evidence)
    validate_report(defense, evidence)
    validate_reply(reply, prosecution, evidence)

    expected_targets = set(range(1, len(prosecution.findings) + 1))
    seen_targets = set()
    for item in resolution.findings:
        target = item.prosecution_finding
        # bool is an int subclass, but cannot identify a valid finding number.
        if type(target) is not int or target not in expected_targets:
            raise ValueError("mediation targets an unknown prosecution finding.")
        if target in seen_targets:
            raise ValueError("mediation must assess each prosecution finding once.")
        seen_targets.add(target)
        if item.outcome not in (
            "supported",
            "partially_supported",
            "not_supported",
            "inconclusive",
        ):
            raise ValueError("unsupported mediation outcome.")
        if not isinstance(item.assessment, str) or not item.assessment.strip():
            raise ValueError(f"mediation finding {target} needs an assessment.")
        if (
            not isinstance(item.evidence_ids, list)
            or not item.evidence_ids
            or not all(isinstance(evidence_id, str) for evidence_id in item.evidence_ids)
        ):
            raise ValueError(f"mediation finding {target} must cite original evidence.")
        if not isinstance(item.open_questions, list) or any(
            not isinstance(question, str) or not question.strip()
            for question in item.open_questions
        ):
            raise ValueError(f"mediation finding {target} has an invalid open question.")
        if item.outcome == "inconclusive" and not item.open_questions:
            raise ValueError(
                f"inconclusive mediation finding {target} must identify an open question."
            )

        # Reuse report validation so mediator citations obey the same evidence-ID rules.
        validate_report(
            AgentReport(
                "prosecution",
                [Finding(claim=item.assessment, evidence_ids=item.evidence_ids)],
            ),
            evidence,
        )

    if seen_targets != expected_targets:
        raise ValueError("mediation must assess every prosecution finding.")


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
        # This asks "does the cited ID exist?", not "does it prove the sentence?"
        # Checking what the sentence means is a separate evaluation task.
        unknown_ids = set(finding.evidence_ids) - evidence_ids
        if unknown_ids:
            raise ValueError(
                f"finding {number} cites unknown evidence IDs: {sorted(unknown_ids)!r}."
            )
