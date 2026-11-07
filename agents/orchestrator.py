"""Run the prosecution and defense drafts for one case."""

from dataclasses import dataclass

from agents.case_context import CaseContext
from agents.contracts import AgentReport, validate_report
from agents.defense import create_report as create_defense_report
from agents.llm import TextGenerator
from agents.prosecution import create_report as create_prosecution_report


@dataclass(frozen=True)
class DebateResult:
    """The two sides' reports for one case."""

    prosecution: AgentReport
    defense: AgentReport


def review_case(
    context: CaseContext, *, text_generator: TextGenerator | None = None
) -> DebateResult:
    """Create both reports from the same evidence and validate their citations."""
    evidence = list(context.evidence)
    prosecution = create_prosecution_report(context, text_generator=text_generator)
    defense = create_defense_report(context, text_generator=text_generator)

    # Check both reports again at the point where they are combined.
    validate_report(prosecution, evidence)
    validate_report(defense, evidence)
    return DebateResult(prosecution=prosecution, defense=defense)
