"""Run independent reviews, optionally followed by one defense reply."""

from dataclasses import dataclass

from agents.case_context import CaseContext
from agents.contracts import AgentReport, DefenseReply, validate_report, validate_reply
from agents.defense import create_report as create_defense_report
from agents.defense import create_reply as create_defense_reply
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
    # Each create_report call does two steps inside llm.py: draft, then edit.
    # Run prosecution first, then defense. Pass the same case to both.
    prosecution = create_prosecution_report(context, text_generator=text_generator)
    # Do not pass prosecution here: defense must make its own assessment.
    defense = create_defense_report(context, text_generator=text_generator)

    # Check both reports again at the point where they are combined.
    validate_report(prosecution, evidence)
    validate_report(defense, evidence)
    # Keep this existing entry point independent. The optional debate has its
    # own function below, so old scripts do not silently spend another API call.
    return DebateResult(prosecution=prosecution, defense=defense)


@dataclass(frozen=True)
class SingleRoundDebateResult:
    """Keep both original reports and the later reply as separate outputs."""

    prosecution: AgentReport
    defense: AgentReport
    defense_reply: DefenseReply


def run_debate(
    context: CaseContext, *, text_generator: TextGenerator | None = None
) -> SingleRoundDebateResult:
    """Create independent reports, then ask defense to respond exactly once."""
    # review_case completes and validates BOTH reports before the reply begins.
    # The original defense report therefore cannot read prosecution's arguments.
    initial = review_case(context, text_generator=text_generator)
    reply = create_defense_reply(
        context, initial.prosecution, initial.defense, text_generator=text_generator
    )
    validate_reply(reply, initial.prosecution, context.evidence)
    # There is no loop, judge or second rebuttal. If any stage fails, its error
    # reaches the caller; an incomplete debate is not returned as a success.
    return SingleRoundDebateResult(initial.prosecution, initial.defense, reply)
