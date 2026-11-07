"""Create a simple, evidence-cited prosecution review draft."""

from agents.case_context import CaseContext
from agents.contracts import AgentReport
from agents.llm import TextGenerator, generate_report


def create_report(
    context: CaseContext, *, text_generator: TextGenerator | None = None
) -> AgentReport:
    """Ask the prosecution prompt to analyze this case's evidence."""
    evidence = list(context.evidence)
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
        raise ValueError("prosecution review requires an alert flagged by the model.")

    return generate_report(
        "prosecution", evidence, text_generator=text_generator
    )
