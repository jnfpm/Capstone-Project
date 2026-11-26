"""Create an evidence-based synthesis after the bounded debate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.case_context import CaseContext
from agents.contracts import MediatorResolution
from agents.llm import TextGenerator, generate_mediator_resolution

if TYPE_CHECKING:
    from agents.orchestrator import SingleRoundDebateResult


def create_resolution(
    context: CaseContext,
    debate: SingleRoundDebateResult,
    *,
    text_generator: TextGenerator | None = None,
) -> MediatorResolution:
    """Assess debate conclusions without treating either side as evidence."""
    # The generator validates each report, reply and final citation against this case.
    return generate_mediator_resolution(
        context.evidence,
        debate.prosecution,
        debate.defense,
        debate.defense_reply,
        text_generator=text_generator,
    )
