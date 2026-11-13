"""Run independent prosecution and defense reviews for one case."""

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
    # Stop if a role or a validation raises an error. Let the caller see it.
    # We deliberately do not catch errors and return a partial result here.
    # If prosecution raises, defense is never called. If defense raises,
    # prosecution has finished, but this function still returns no combined result.
    # Each create_report call does two steps inside llm.py: draft, then edit.
    # Run prosecution first, then defense. Pass the same case to both.
    prosecution = create_prosecution_report(context, text_generator=text_generator)
    # Do not pass prosecution here: defense must make its own assessment.
    defense = create_defense_report(context, text_generator=text_generator)

    # Check both reports again at the point where they are combined.
    validate_report(prosecution, evidence)
    validate_report(defense, evidence)
    # Despite its name, DebateResult only holds the two independent reports.
    # No back-and-forth discussion takes place here yet.
    return DebateResult(prosecution=prosecution, defense=defense)
