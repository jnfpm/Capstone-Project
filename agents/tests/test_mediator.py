"""Mediator contract and prompt tests that do not call a real model."""

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.case_context import CaseContext
from agents.contracts import (
    AgentReport,
    DefenseReply,
    Evidence,
    Finding,
    MediatorFinding,
    MediatorResolution,
    ReplyFinding,
    validate_mediation,
)
from agents.llm import generate_mediator_resolution
from agents.orchestrator import (
    SingleRoundDebateResult,
    run_mediated_debate,
)


class MediatorTests(unittest.TestCase):
    """Keep mediation bounded to numbered findings and original evidence."""

    def setUp(self):
        self.evidence = [
            Evidence(
                id="tx-1",
                source="transaction",
                reference="transaction 1",
                description="The transfer was cross-border.",
            )
        ]
        self.prosecution = AgentReport(
            "prosecution",
            [Finding("The transfer crossed a border.", ["tx-1"])],
        )
        self.defense = AgentReport(
            "defense",
            [Finding("The transfer alone does not establish wrongdoing.", ["tx-1"])],
        )
        self.reply = DefenseReply(
            "defense",
            [
                ReplyFinding(
                    prosecution_finding=1,
                    position="qualify",
                    claim="The transfer alone does not establish wrongdoing.",
                    evidence_ids=["tx-1"],
                )
            ],
        )
        self.resolution = MediatorResolution(
            "mediator",
            [
                MediatorFinding(
                    prosecution_finding=1,
                    outcome="inconclusive",
                    assessment="The transfer is cross-border, but the supplied fact "
                    "does not resolve the broader allegation.",
                    evidence_ids=["tx-1"],
                    open_questions=["What other transaction context is available?"],
                )
            ],
        )

    def test_accepts_inconclusive_outcome_with_an_open_question(self):
        validate_mediation(
            self.resolution,
            self.prosecution,
            self.defense,
            self.reply,
            self.evidence,
        )

    def test_rejects_inconclusive_outcome_without_open_questions(self):
        finding = self.resolution.findings[0]
        resolution = MediatorResolution(
            agent="mediator",
            findings=[replace(finding, open_questions=[])],
        )

        with self.assertRaisesRegex(ValueError, "must identify an open question"):
            validate_mediation(
                resolution,
                self.prosecution,
                self.defense,
                self.reply,
                self.evidence,
            )

    def test_requires_exactly_one_assessment_for_every_prosecution_finding(self):
        with self.assertRaisesRegex(ValueError, "assess every prosecution finding"):
            validate_mediation(
                MediatorResolution("mediator", []),
                self.prosecution,
                self.defense,
                self.reply,
                self.evidence,
            )

    def test_rejects_mediator_citations_outside_the_original_evidence(self):
        finding = self.resolution.findings[0]
        invalid = MediatorResolution(
            agent="mediator",
            findings=[replace(finding, evidence_ids=["invented"])],
        )

        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            validate_mediation(
                invalid, self.prosecution, self.defense, self.reply, self.evidence
            )

    def test_generates_resolution_from_arguments_separated_from_evidence(self):
        captured = {}

        def fake_generator(system_prompt, user_prompt):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return json.dumps(
                {
                    "agent": "mediator",
                    "findings": [
                        {
                            "prosecution_finding": 1,
                            "outcome": "inconclusive",
                            "assessment": "The original fact supports the transfer "
                            "observation but not the broader allegation.",
                            "evidence_ids": ["tx-1"],
                            "open_questions": ["What was the payment's documented purpose?"],
                        }
                    ],
                }
            )

        result = generate_mediator_resolution(
            self.evidence,
            self.prosecution,
            self.defense,
            self.reply,
            text_generator=fake_generator,
        )

        self.assertEqual(result.findings[0].outcome, "inconclusive")
        self.assertIn("Do not decide guilt", captured["system"])
        self.assertIn("Prosecution arguments (not evidence)", captured["user"])
        self.assertIn("Original evidence:", captured["user"])

    def test_mediated_entrypoint_keeps_debate_and_resolution_separate(self):
        debate = SingleRoundDebateResult(
            self.prosecution, self.defense, self.reply
        )
        context = CaseContext(alert_id="case-1", evidence=tuple(self.evidence))

        with (
            patch("agents.orchestrator.run_debate", return_value=debate) as run_debate,
            patch(
                "agents.orchestrator.create_mediator_resolution",
                return_value=self.resolution,
            ) as create_resolution,
        ):
            result = run_mediated_debate(context)

        run_debate.assert_called_once_with(context, text_generator=None)
        create_resolution.assert_called_once_with(
            context, debate, text_generator=None
        )
        self.assertIs(result.debate, debate)
        self.assertIs(result.mediation, self.resolution)


if __name__ == "__main__":
    unittest.main()
