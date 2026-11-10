"""Create a simple, evidence-cited prosecution review draft."""

from agents.case_context import CaseContext
from agents.contracts import AgentReport
from agents.llm import TextGenerator, generate_report


def create_report(
    context: CaseContext, *, text_generator: TextGenerator | None = None
) -> AgentReport:
    """Identify evidence-supported reasons to investigate a flagged alert."""
    evidence = list(context.evidence)
    # Only review alerts already selected upstream. A high score alone does not
    # authorize this review; the context must contain an explicit true alert flag.
    # The current flag is encoded in descriptive text by build_case_context.
    # Changing that text requires updating this check and the defense check too.
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

    # The role selects the prompt, not a separate model. A supplied generator
    # allows offline tests; otherwise the configured provider is called.
    return generate_report(
        "prosecution", evidence, text_generator=text_generator
    )
