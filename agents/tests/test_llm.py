"""Tests for structured LLM output without calling an external provider."""

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.contracts import Evidence
from agents.llm import generate_report


class GenerateReportTests(unittest.TestCase):
    """Check parsing, prompts, and citation validation using mocked responses."""

    def setUp(self):
        # The fake model can cite only this evidence ID.
        self.evidence = [
            Evidence(
                id="transaction-1",
                source="transaction",
                reference="transaction 1",
                description="The transfer was cross-border.",
            )
        ]

    def test_builds_report_from_mocked_json_response(self):
        # The fake generator captures prompts instead of contacting an API.
        captured = {}

        def fake_generator(system_prompt, user_prompt):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return json.dumps(
                {
                    "agent": "prosecution",
                    "findings": [
                        {
                            "claim": "This transfer may merit review.",
                            "evidence_ids": ["transaction-1"],
                        }
                    ],
                }
            )

        report = generate_report("prosecution", self.evidence, text_generator=fake_generator)

        self.assertEqual(report.agent, "prosecution")
        self.assertEqual(report.findings[0].evidence_ids, ["transaction-1"])
        self.assertIn("not proof", captured["system"])
        self.assertIn("transaction-1", captured["user"])

    def test_each_role_receives_a_valid_json_example_and_unchanged_evidence(self):
        # Check the actual prompt passed through generate_report. The old
        # "prosecution" or "defense" example fails JSON decoding here.
        for role in ("prosecution", "defense"):
            with self.subTest(role=role):
                captured = []

                def fake_generator(system_prompt, user_prompt, role=role, captured=captured):
                    captured.append(user_prompt)
                    return json.dumps(
                        {
                            "agent": role,
                            "findings": [
                                {
                                    "claim": "The supplied evidence describes a cross-border transfer.",
                                    "evidence_ids": ["transaction-1"],
                                }
                            ],
                        }
                    )

                report = generate_report(role, self.evidence, text_generator=fake_generator)
                self.assertEqual(report.agent, role)
                self.assertEqual(len(captured), 1)
                instructions, evidence_json = captured[0].split("Evidence:\n", 1)
                example_start = instructions.index("{")
                example, _ = json.JSONDecoder().raw_decode(instructions[example_start:])
                self.assertEqual(set(example), {"agent", "findings"})
                self.assertEqual(example["agent"], role)
                self.assertEqual(len(example["findings"]), 1)
                finding = example["findings"][0]
                self.assertEqual(set(finding), {"claim", "evidence_ids"})
                self.assertIsInstance(finding["claim"], str)
                self.assertTrue(finding["claim"].strip())
                self.assertIsInstance(finding["evidence_ids"], list)
                self.assertTrue(finding["evidence_ids"])
                self.assertTrue(all(isinstance(item, str) for item in finding["evidence_ids"]))
                # The format example must not become additional case evidence.
                self.assertEqual(json.loads(evidence_json), [asdict(e) for e in self.evidence])

    def test_rejects_citation_to_unknown_evidence(self):
        # A well-formed response still fails if its citation is not in this case.
        def fake_generator(system_prompt, user_prompt):
            return json.dumps(
                {
                    "agent": "defense",
                    "findings": [
                        {
                            "claim": "This claim cites missing evidence.",
                            "evidence_ids": ["not-in-case"],
                        }
                    ],
                }
            )

        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            generate_report("defense", self.evidence, text_generator=fake_generator)

    def test_rejects_malformed_response_shape(self):
        # The report parser requires the exact shared report fields.
        def fake_generator(system_prompt, user_prompt):
            return '{"agent":"defense","findings":[],"extra":"not allowed"}'

        with self.assertRaisesRegex(ValueError, "agent and findings"):
            generate_report("defense", self.evidence, text_generator=fake_generator)


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
