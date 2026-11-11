"""Offline tests for case context preparation and the three reference cases."""

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.case_context import build_case_context
from agents.orchestrator import review_case
from agents.prompts import REVISION_SYSTEM_PROMPTS, SYSTEM_PROMPTS
from agents.prosecution import create_report
from agents.reference_cases import (
    CASE_NAMES,
    build_reference_context,
    load_reference_case,
)


class CaseContextTests(unittest.TestCase):
    """Use a small synthetic alert to check the full evidence flow."""

    def setUp(self):
# Include training labels in the test input on purpose. The test below
        # checks that these answer-revealing fields do NOT reach the agents.
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
        first = build_case_context("alert-1", self.alert, self.transactions, self.model_output)
        second = build_case_context("alert-1", self.alert, self.transactions, self.model_output)

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
        # A fake text generator keeps this test independent of an API key.
        context = build_case_context("alert-1", self.alert, self.transactions, self.model_output)
        evidence_id = next(item.id for item in context.evidence if item.source == "model_output")

        def fake_generator(system_prompt, user_prompt):
            return (
                '{"agent":"prosecution","findings":[{"claim":"The model flagged '
                'the case for review, not as proof.","evidence_ids":["' + evidence_id + '"]}]}'
            )

        report = create_report(context, text_generator=fake_generator)

        self.assertEqual(report.agent, "prosecution")
        self.assertTrue(report.findings[0].evidence_ids)
        self.assertEqual(report.findings[0].evidence_ids, [evidence_id])

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


# These tests use our three fictional teaching cases, not the full dataset.
class ReferenceCaseTests(unittest.TestCase):
    """Check fixture consistency and evidence flow, not real LLM reasoning quality."""

    def test_case_facts_and_evaluation_criteria_are_present(self):
        for name in CASE_NAMES:
            with self.subTest(case=name):
                case = load_reference_case(name)
                self.assertTrue(case["synthetic"])
                self.assertIs(case["input"]["model_output"]["alert"], True)
                rows = case["input"]["transactions"]
                self.assertEqual(len(rows), len({row["row_id"] for row in rows}))
                for key in ("prosecution", "defense", "forbidden", "limitations"):
                    self.assertTrue(case["evaluation"][key])

        # These values define the teaching scenario. If they change, review the
        # criteria too, so they do not require conclusions unsupported by the data.
        concentrated = load_reference_case(CASE_NAMES[0])["input"]["transactions"]
        self.assertEqual(sum(row["amount_usd"] for row in concentrated[:3]), 15000)
        self.assertEqual(concentrated[3]["amount_usd"], 14900)
        self.assertTrue(
            all(row["acct_to"] == concentrated[3]["acct_from"] for row in concentrated[:3])
        )
        regular = load_reference_case(CASE_NAMES[1])["input"]["transactions"]
        self.assertEqual(len(regular), 3)
        self.assertEqual(len({(r["acct_from"], r["acct_to"], r["amount_usd"]) for r in regular}), 1)

    def test_both_reviewers_receive_only_the_same_input_evidence(self):
        for name in CASE_NAMES:
            with self.subTest(case=name):
                context = build_reference_context(name)
                expected = [asdict(item) for item in context.evidence]
                # Record who is called and compare the facts received each time.
                calls = []

                def fake_generator(system_prompt, user_prompt, expected=expected, calls=calls):
                    role = next(
                        role
                        for role, prompt in SYSTEM_PROMPTS.items()
                        if system_prompt in (prompt, REVISION_SYSTEM_PROMPTS[role])
                    )
                    supplied = json.loads(user_prompt.split("Evidence:\n", 1)[1])
                    self.assertEqual(supplied, expected)
                    # A simple factual statement is enough to test the connection.
                    # This stub does not evaluate the quality of model reasoning.
                    calls.append(role)
                    return json.dumps(
                        {
                            "agent": role,
                            "findings": [
                                {
                                    "claim": supplied[0]["description"],
                                    "evidence_ids": [supplied[0]["id"]],
                                }
                            ],
                        }
                    )

                # Block the provider entry point to prevent accidental API calls.
                with patch(
                    "agents.llm._generate_text",
                    side_effect=AssertionError("API calls are disabled in this test"),
                ):
                    result = review_case(context, text_generator=fake_generator)
                # Four calls: prosecution draft/edit, then defense draft/edit.
                # This checks the order, not the quality of the model's writing.
                self.assertEqual(calls, ["prosecution", "prosecution", "defense", "defense"])
                self.assertEqual(result.prosecution.agent, "prosecution")
                self.assertEqual(result.defense.agent, "defense")
                serialized = json.dumps(expected, ensure_ascii=False)
                case = load_reference_case(name)
                for role in ("prosecution", "defense"):
                    for criterion in case["evaluation"][role]:
                        self.assertNotIn(criterion, serialized)
                self.assertEqual(context, build_reference_context(name))

    def test_missing_data_does_not_turn_into_transaction_facts(self):
        context = build_reference_context("03_insufficient_information")
        self.assertFalse(any(item.source == "transaction" for item in context.evidence))
        self.assertEqual(
            {item.description for item in context.evidence},
            {
                "Timestamp: 2022-09-08 12:00:00",
                "Model output score: 0.88",
                "Model output alert: True",
            },
        )

    def test_unknown_case_is_rejected(self):
        with self.assertRaises(ValueError):
            load_reference_case("../config")


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
