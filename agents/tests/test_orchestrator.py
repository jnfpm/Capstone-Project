"""Tests for the basic two-sided case review."""

import json
import sys
import unittest
from pathlib import Path

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.case_context import build_case_context
from agents.orchestrator import review_case


class ReviewCaseTests(unittest.TestCase):
    """Check that both sides use the same case evidence."""

    def test_returns_cited_prosecution_and_defense_reports(self):
        # Both drafts receive the same model and transaction evidence.
        context = build_case_context(
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


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
