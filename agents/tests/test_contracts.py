"""Small tests for the shared agent response contract."""

import unittest

from agents.contracts import AgentReport, Evidence, Finding, validate_report


class ValidateReportTests(unittest.TestCase):
    """Check that reports only contain claims supported by supplied evidence."""

    def setUp(self):
        # Reuse one evidence item across the valid and invalid report tests.
        self.evidence = [
            Evidence(
                id="tx-1",
                source="transaction",
                reference="transaction 1 on 2026-09-01",
                description="The transfer was cross-border.",
            )
        ]

    def test_accepts_a_finding_with_known_evidence(self):
        # A report is valid when each finding cites evidence it was given.
        report = AgentReport(
            agent="prosecution",
            findings=[
                Finding(
                    claim="The transfer crossed a border.",
                    evidence_ids=["tx-1"],
                )
            ],
        )

        validate_report(report, self.evidence)

    def test_rejects_a_finding_without_evidence(self):
        # Unsupported claims must not pass validation.
        report = AgentReport(
            agent="defense",
            findings=[Finding(claim="This claim has no support.", evidence_ids=[])],
        )

        with self.assertRaisesRegex(ValueError, "must cite at least one evidence ID"):
            validate_report(report, self.evidence)

    def test_rejects_unknown_evidence_ids(self):
        # Every citation must refer to evidence in the current case.
        report = AgentReport(
            agent="defense",
            findings=[
                Finding(claim="The transfer was unusual.", evidence_ids=["missing"])
            ],
        )

        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            validate_report(report, self.evidence)


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
