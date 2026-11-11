"""Tests for structured LLM output without calling an external provider."""

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents import config
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
                self.assertEqual(len(captured), 2)
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
                self.assertEqual(
                    json.loads(captured[1].split("Evidence:\n", 1)[1]),
                    [asdict(e) for e in self.evidence],
                )

    def test_returns_the_revision_and_supplies_only_its_own_draft(self):
        draft = {
            "agent": "defense",
            "findings": [{"claim": "Initial wording.", "evidence_ids": ["transaction-1"]}],
        }
        revised = {
            "agent": "defense",
            "findings": [{"claim": "Revised wording.", "evidence_ids": ["transaction-1"]}],
        }
        generator = Mock(side_effect=[json.dumps(draft), json.dumps(revised)])
        report = generate_report("defense", self.evidence, text_generator=generator)
        self.assertEqual(report.findings[0].claim, "Revised wording.")
        self.assertEqual(generator.call_count, 2)
        revision_prompt = generator.call_args_list[1].args[1]
        supplied_draft = json.loads(
            revision_prompt.split("<draft>\n", 1)[1].split("\n</draft>", 1)[0]
        )
        self.assertEqual(supplied_draft, draft)

    def test_rejects_invalid_revision_instead_of_returning_the_draft(self):
        draft = {
            "agent": "defense",
            "findings": [{"claim": "Valid draft.", "evidence_ids": ["transaction-1"]}],
        }
        invalid_revision = {
            "agent": "defense",
            "findings": [{"claim": "Invalid revision.", "evidence_ids": ["invented-id"]}],
        }
        generator = Mock(side_effect=[json.dumps(draft), json.dumps(invalid_revision)])
        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            generate_report("defense", self.evidence, text_generator=generator)
        self.assertEqual(generator.call_count, 2)

    def test_revision_failure_propagates_without_retry(self):
        draft = {
            "agent": "defense",
            "findings": [{"claim": "Valid draft.", "evidence_ids": ["transaction-1"]}],
        }
        generator = Mock(side_effect=[json.dumps(draft), RuntimeError("provider unavailable")])
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            generate_report("defense", self.evidence, text_generator=generator)
        self.assertEqual(generator.call_count, 2)

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

    def test_rejects_token_limited_responses_even_with_valid_json(self):
        # Check our acceptance policy through generate_report, without network
        # access. A valid-looking fragment must not bypass the completion check.
        valid_text = json.dumps(
            {
                "agent": "prosecution",
                "findings": [
                    {
                        "claim": "A cross-border transfer is supplied.",
                        "evidence_ids": ["transaction-1"],
                    }
                ],
            }
        )
        for text in ('{"agent":', valid_text):
            with self.subTest(text=text):
                response = SimpleNamespace(
                    stop_reason="max_tokens",
                    content=[SimpleNamespace(type="text", text=text)],
                )
                client = SimpleNamespace(
                    messages=SimpleNamespace(create=Mock(return_value=response))
                )
                with (
                    patch.multiple(
                        config, PROVIDER="anthropic", API_KEY="test-key", ANTHROPIC_MAX_TOKENS=4096
                    ),
                    patch("anthropic.Anthropic", return_value=client),
                    self.assertRaisesRegex(RuntimeError, "report is incomplete"),
                ):
                    generate_report("prosecution", self.evidence)

    def test_accepts_a_completed_anthropic_report(self):
        # The new completion check must preserve the normal successful path.
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {
                            "agent": "defense",
                            "findings": [
                                {
                                    "claim": "The evidence describes a cross-border transfer.",
                                    "evidence_ids": ["transaction-1"],
                                }
                            ],
                        }
                    ),
                )
            ],
        )
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))
        with (
            patch.multiple(config, PROVIDER="anthropic", API_KEY="test-key"),
            patch("anthropic.Anthropic", return_value=client),
        ):
            report = generate_report("defense", self.evidence)
        self.assertEqual(report.agent, "defense")
        self.assertEqual(report.findings[0].evidence_ids, ["transaction-1"])
        # Both calls must ask the provider for non-empty findings and citations.
        # Local validation remains necessary even when the API constrains shape.
        self.assertEqual(client.messages.create.call_count, 2)
        for call in client.messages.create.call_args_list:
            schema = call.kwargs["output_config"]["format"]["schema"]
            findings = schema["properties"]["findings"]
            self.assertEqual(findings["minItems"], 1)
            self.assertEqual(findings["items"]["properties"]["evidence_ids"]["minItems"], 1)


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
