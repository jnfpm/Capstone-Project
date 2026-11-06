"""Create a simple, evidence-cited prosecution review draft."""

from agents.case_context import CaseContext
from agents.contracts import AgentReport, Finding, validate_report


def create_report(context: CaseContext) -> AgentReport:
    """Summarize why the scoring model sent this case for review."""
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

    score_evidence = next(
        (
            item.id
            for item in evidence
            if item.source == "model_output"
            and item.description.startswith("Model output score: ")
        ),
        None,
    )
    reason_ids = [
        item.id
        for item in evidence
        if item.source == "model_output"
        and item.description.startswith("Model output reason: ")
    ]

    # Keep the score and reasons linked to the exact output used for this report.
    cited_ids = [alert_item.id]
    if score_evidence is not None:
        cited_ids.append(score_evidence)
    cited_ids.extend(reason_ids)

    # This first draft only summarizes model output; it does not decide guilt.
    report = AgentReport(
        agent="prosecution",
        findings=[
            Finding(
                claim=(
                    "The scoring model flagged this case for review. Its score and "
                    "reasons are model output, not proof of money laundering."
                ),
                evidence_ids=cited_ids,
            )
        ],
    )
    validate_report(report, evidence)
    return report
