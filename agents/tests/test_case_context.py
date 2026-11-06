"""Tests for preparing and reviewing a small sample alert."""

import unittest

from agents.case_context import build_case_context
from agents.prosecution import create_report


class CaseContextTests(unittest.TestCase):
    """Use a small synthetic alert to check the full evidence flow."""

    def setUp(self):
        # This sample has transaction facts and model output, not training labels.
        self.alert = {
            "Timestamp": "2022-09-01 11:05:00",
            "acct_from": "bank-a:account-1",
            "acct_to": "bank-b:account-2",
            "amount_usd": 250.0,
            "Attempt ID": 42,
            "Pattern Type": "GATHER-SCATTER",
            "Is Laundering": 1,
        }
        self.transactions = [
            {
                "row_id": 681848,
                "Timestamp": "2022-09-01 11:05:00",
                "Amount Paid": 250.0,
                "Payment Currency": "USD",
                "Is Laundering": 1,
            }
        ]
        self.model_output = {
            "score": 0.91,
            "alert": True,
            "reasons": "new_pair=1 (+1.20), cross_border=1 (+0.62)",
        }

    def test_builds_stable_evidence_without_training_labels(self):
        # Rebuilding the same case should produce the same citation IDs.
        first = build_case_context(
            "alert-1", self.alert, self.transactions, self.model_output
        )
        second = build_case_context(
            "alert-1", self.alert, self.transactions, self.model_output
        )

        self.assertEqual(
            [item.id for item in first.evidence],
            [item.id for item in second.evidence],
        )
        descriptions = [item.description for item in first.evidence]
        self.assertFalse(any("Is Laundering" in item for item in descriptions))
        self.assertFalse(any("Attempt ID" in item for item in descriptions))
        self.assertFalse(any("Pattern Type" in item for item in descriptions))
        self.assertTrue(
            all(
                "Model output" in item.description
                for item in first.evidence
                if item.source == "model_output"
            )
        )

    def test_prosecution_report_cites_model_output(self):
        # The draft report uses the shared contract and validates its citations.
        context = build_case_context(
            "alert-1", self.alert, self.transactions, self.model_output
        )

        report = create_report(context)

        self.assertEqual(report.agent, "prosecution")
        self.assertTrue(report.findings[0].evidence_ids)
        self.assertIn("not proof", report.findings[0].claim)

    def test_prosecution_requires_a_flagged_alert(self):
        # A non-alert should not be turned into a prosecution review.
        context = build_case_context(
            "alert-1",
            self.alert,
            self.transactions,
            {**self.model_output, "alert": False},
        )

        with self.assertRaisesRegex(ValueError, "requires an alert flagged"):
            create_report(context)


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
