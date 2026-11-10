"""Synthetic cases for developing reviewers without preprocessing data."""

import json
from pathlib import Path

from agents.case_context import CaseContext, build_case_context

CASE_NAMES = (
    "01_concentrated_transfers",
    "02_regular_payments",
    "03_insufficient_information",
)


def load_reference_case(name: str) -> dict:
    """Read the input data and evaluation criteria for one of the three cases."""
    if name not in CASE_NAMES:
        raise ValueError(f"Unknown reference case: {name!r}")
    path = Path(__file__).with_name(f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_reference_context(name: str) -> CaseContext:
    """Pass only facts and simulated model outputs to the existing builder."""
    # Criteria belong to the evaluator. Sending them to the LLM would reveal
    # expected answers and undermine independent evaluation of the reports.
    data = load_reference_case(name)["input"]
    return build_case_context(
        alert_id=data["alert_id"],
        alert=data["alert"],
        transactions=data["transactions"],
        model_output=data["model_output"],
    )
