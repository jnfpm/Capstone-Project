"""Create a simple, evidence-cited defense review draft."""

from agents.case_context import CaseContext
from agents.contracts import AgentReport
from agents.llm import TextGenerator, generate_report


def create_report(
    context: CaseContext, *, text_generator: TextGenerator | None = None
) -> AgentReport:
    """Assess uncertainty and alternative interpretations of a flagged alert."""
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

    # Only case evidence is supplied here, not prosecution's report. This is an
    # independent review, not a rebuttal. The role selects the defense prompt.
    # A supplied text_generator replaces the provider call for offline tests.
    return generate_report("defense", evidence, text_generator=text_generator)
