"""Create an independent defense report or one later reply to prosecution."""

from agents.case_context import CaseContext
from agents.contracts import AgentReport, DefenseReply
from agents.llm import TextGenerator, generate_report, generate_defense_reply


def _require_flagged_alert(context: CaseContext) -> None:
    """Both the independent report and the reply require a selected alert."""
    evidence = list(context.evidence)
    # Defense reviews the same flagged cases as prosecution. Its role is to
    # examine the strength of the evidence, not to clear unflagged transactions.
    # The flag currently depends on the text produced by build_case_context;
    # this check must stay consistent with that builder and the prosecution check.
    alert_item = next(
        (
            item
            for item in evidence
            if item.source == "model_output"
            and item.description.startswith("Model output alert: ")
        ),
        None,
    )
    if alert_item is None or not alert_item.description.lower().endswith(": true"):
        raise ValueError("defense review requires an alert flagged by the model.")


def create_report(
    context: CaseContext, *, text_generator: TextGenerator | None = None
) -> AgentReport:
    """Assess uncertainty and alternative interpretations of a flagged alert."""
    _require_flagged_alert(context)
    # Only case evidence is supplied here, not prosecution's report. This is an
    # independent review, not a rebuttal. The role selects the defense prompt.
    # A supplied text_generator replaces the provider call for offline tests.
    return generate_report("defense", list(context.evidence), text_generator=text_generator)


def create_reply(
    context: CaseContext,
    prosecution: AgentReport,
    defense: AgentReport,
    *,
    text_generator: TextGenerator | None = None,
) -> DefenseReply:
    """Respond once, after both independent reports have been produced."""
    _require_flagged_alert(context)
    # Reports are passed as arguments to discuss, never added to case evidence.
    return generate_defense_reply(
        context.evidence, prosecution, defense, text_generator=text_generator
    )
