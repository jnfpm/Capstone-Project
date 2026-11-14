"""Check successful reviews and failures without calling a real model."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.case_context import build_case_context
from agents.contracts import AgentReport, Finding
from agents.orchestrator import review_case


class ReviewCaseTests(unittest.TestCase):
    """A complete result needs two valid reports; failures must stay visible."""

    def setUp(self):
        # unittest runs this before each test, giving it a fresh fictional case.
        self.context = build_case_context(
            "alert-1",
            {"Timestamp": "2022-09-01 11:05:00"},
            [
                {
                    "row_id": 681848,
                    "Timestamp": "2022-09-01 11:05:00",
                    "Amount Paid": 250.0,
                    "Payment Currency": "USD",
                }
            ],
            {"score": 0.91, "alert": True, "reasons": ["new_pair=1"]},
        )

        # These small reports let us test the coordinator, not the model's prose.
        fact = self.context.evidence[0]
        self.prosecution_report = AgentReport(
            agent="prosecution",
            findings=[Finding(claim=fact.description, evidence_ids=[fact.id])],
        )
        self.defense_report = AgentReport(
            agent="defense",
            findings=[Finding(claim=fact.description, evidence_ids=[fact.id])],
        )

    def test_returns_cited_prosecution_and_defense_reports(self):
        # Both reviewers receive the same model and transaction evidence.
        context = self.context

        def fake_generator(system_prompt, user_prompt):
            evidence_text = user_prompt.split("Evidence:\n", maxsplit=1)[1]
            evidence = json.loads(evidence_text)
            agent = "defense" if "defense-side" in system_prompt else "prosecution"
            return json.dumps(
                {
                    "agent": agent,
                    "findings": [
                        {
                            "claim": f"A {agent} review finding.",
                            "evidence_ids": [evidence[0]["id"]],
                        }
                    ],
                }
            )

        result = review_case(context, text_generator=fake_generator)

        self.assertEqual(result.prosecution.agent, "prosecution")
        self.assertEqual(result.defense.agent, "defense")
        evidence_ids = {item.id for item in context.evidence}
        for report in (result.prosecution, result.defense):
            self.assertTrue(report.findings)
            for finding in report.findings:
                self.assertTrue(finding.evidence_ids)
                self.assertTrue(set(finding.evidence_ids) <= evidence_ids)

    def test_prosecution_failure_stops_before_defense(self):
        # patch temporarily replaces a function with a pretend version. THIS IS IMPORTANT!!!!!
        # side_effect makes that version raise an error instead of returning.
        # Both role functions are replaced, so this test cannot call the API.
        failure = RuntimeError("Prosecution is unavailable.")
        with (
            patch("agents.orchestrator.create_prosecution_report", side_effect=failure) as prosecution,
            patch("agents.orchestrator.create_defense_report") as defense,
        ):
            # An error is the EXPECTED result here. Returning any result fails the test.
            with self.assertRaises(RuntimeError) as caught:
                review_case(self.context)

            self.assertIs(caught.exception, failure)
            prosecution.assert_called_once_with(self.context, text_generator=None)
            defense.assert_not_called()

    def test_defense_failure_does_not_return_a_partial_review(self):
        # Prosecution succeeds, but defense fails. The caller must receive the
        # error, not a result that looks complete or silently contains one side.
        failure = RuntimeError("Defense is unavailable.")
        with (
            patch(
                "agents.orchestrator.create_prosecution_report",
                return_value=self.prosecution_report,
            ) as prosecution,
            patch("agents.orchestrator.create_defense_report", side_effect=failure) as defense,
        ):
            with self.assertRaises(RuntimeError) as caught:
                review_case(self.context)

            self.assertIs(caught.exception, failure)
            prosecution.assert_called_once_with(self.context, text_generator=None)
            defense.assert_called_once_with(self.context, text_generator=None)

    def test_final_validation_rejects_an_unknown_citation_from_either_role(self):
        # Pretend a role returns a report with an invented reference. Normally
        # llm.py rejects this earlier. Here we check the coordinator's own check.
        # Keep validate_report real: replacing it would hide a missing safeguard.
        for role in ("prosecution", "defense"):
            with self.subTest(role=role):
                invalid = AgentReport(
                    agent=role,
                    findings=[Finding(claim="An unsupported claim.", evidence_ids=["invented-id"])],
                )
                with (
                    patch(
                        "agents.orchestrator.create_prosecution_report",
                        return_value=invalid if role == "prosecution" else self.prosecution_report,
                    ) as prosecution,
                    patch(
                        "agents.orchestrator.create_defense_report",
                        return_value=invalid if role == "defense" else self.defense_report,
                    ) as defense,
                ):
                    with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
                        review_case(self.context)

                    # This last validation happens AFTER both roles return.
                    # It is different from a role raising an error during generation.
                    prosecution.assert_called_once()
                    defense.assert_called_once()


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
