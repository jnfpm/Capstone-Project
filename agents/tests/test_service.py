"""Tests for connecting prepared feature files to the agent review."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

# Add the repository root when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.service import (
    AML_ROOT, DEFAULT_FEATURES_DIR, DEFAULT_MODEL_PATH, _load_day, review_alert,
)


class ReviewAlertTests(unittest.TestCase):
    """Use tiny parquet files and a fake scorer instead of loading the real model."""

    def setUp(self):
        # Two same-day rows ensure scoring uses the daily batch, not one row alone.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.features_dir = Path(self.temp_dir.name)
        timestamp = pd.Timestamp("2022-09-08 10:00:00")
        pd.DataFrame(
            {
                "row_id": [10, 11],
                "Timestamp": [timestamp, timestamp + pd.Timedelta(minutes=1)],
                "acct_from": ["bank-a:sender", "bank-a:sender"],
                "acct_to": ["bank-b:receiver", "bank-c:receiver"],
                "amount_usd": [100.0, 200.0],
                "Payment Format": ["ACH", "Wire"],
                "weight": [1.0, 1.0],
                # These fields exist in the new metadata, but must not reach the LLM.
                "Attempt ID": ["hidden-attempt", "hidden-attempt"],
                "Pattern Type": ["hidden-pattern", "hidden-pattern"],
                "seed_from": [1, 1],
                "seed_to": [0, 0],
                "own_date_split": ["test", "test"],
            }
        ).to_parquet(self.features_dir / "test_meta.parquet", index=False)
        pd.DataFrame(
            {
                "row_id": [10, 11],
                "cross_border": [0, 1],
                "country_from": ["US", "US"],
                "country_to": ["US", "UK"],
                "Is Laundering": [0, 1],
            }
        ).to_parquet(self.features_dir / "test_features.parquet", index=False)
        empty_meta = pd.DataFrame(
            {
                "row_id": pd.Series([], dtype="int64"),
                "Timestamp": pd.Series([], dtype="datetime64[ns]"),
            }
        )
        empty_meta.to_parquet(
            self.features_dir / "train_meta.parquet", index=False
        )
        empty_meta.to_parquet(
            self.features_dir / "val_meta.parquet", index=False
        )
        self.model_path = self.features_dir / "model.joblib"
        self.model_path.touch()

    def tearDown(self):
        # Remove the temporary parquet data created by this test.
        self.temp_dir.cleanup()

    def test_scores_daily_rows_then_runs_both_agents(self):
        # The mocked scorer marks row 11 as an alert after scoring both rows.
        observed = {}

        class FakeModel:
            features = ["cross_border", "country_from", "country_to"]

        def fake_score_with_explain(bundle, X, meta, explain=True):
            observed.setdefault("calls", []).append((len(X), len(meta), explain))
            if explain:
                return pd.DataFrame(
                    {"score": [0.9], "alert": [True], "reasons": ["cross_border=1"]}
                )
            return pd.DataFrame(
                {"score": [0.1, 0.9], "alert": [False, True]}
            )

        fake_modeling = SimpleNamespace(
            load_bundle=lambda path: {"model": FakeModel(), "features": FakeModel.features},
            score_transactions=fake_score_with_explain,
        )

        def fake_text_generator(system_prompt, user_prompt):
            observed.setdefault("prompts", []).append(user_prompt)
            evidence_text = user_prompt.split("Evidence:\n", maxsplit=1)[1]
            evidence = json.loads(evidence_text)
            agent = "defense" if "defense-side" in system_prompt else "prosecution"
            return json.dumps(
                {
                    "agent": agent,
                    "findings": [
                        {
                            "claim": "This is a test review.",
                            "evidence_ids": [evidence[0]["id"]],
                        }
                    ],
                }
            )

        with patch("agents.service._get_modeling_module", return_value=fake_modeling):
            result = review_alert(
                11,
                features_dir=self.features_dir,
                model_path=self.model_path,
                text_generator=fake_text_generator,
            )

        self.assertEqual(observed["calls"], [(2, 2, False), (1, 1, True)])
        self.assertEqual(result.alert_id, "11")
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.reasons, ("cross_border=1",))
        self.assertEqual(result.debate.prosecution.agent, "prosecution")
        self.assertEqual(result.debate.defense.agent, "defense")
        for prompt in observed["prompts"]:
            for forbidden in ("hidden-attempt", "hidden-pattern", "seed_from", "Is Laundering"):
                self.assertNotIn(forbidden, prompt)
            self.assertIn("bank-c:receiver", prompt)
            self.assertIn("200.0", prompt)

    def test_defaults_use_the_new_pipeline(self):
        # A new review must not silently fall back to the old preprocessing files.
        self.assertEqual(AML_ROOT.name, "aml")
        self.assertEqual(DEFAULT_FEATURES_DIR, AML_ROOT / "data/processed/preprocessed")
        self.assertEqual(DEFAULT_MODEL_PATH, AML_ROOT / "models/txn_scorer.joblib")

    def test_aligns_shuffled_features_with_metadata(self):
        # Parquet row order can change. Match by transaction ID, not by position.
        path = self.features_dir / "test_features.parquet"
        pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)
        features, meta = _load_day(
            "test", pd.Timestamp("2022-09-08"), self.features_dir, ["cross_border"]
        )
        self.assertEqual(meta["row_id"].tolist(), [10, 11])
        self.assertEqual(features["cross_border"].tolist(), [0, 1])

    def test_rejects_duplicate_transaction_ids(self):
        # Duplicates must stop the review before facts can be assigned to a wrong row.
        for filename in ("test_features.parquet", "test_meta.parquet"):
            with self.subTest(filename=filename):
                path = self.features_dir / filename
                original = pd.read_parquet(path)
                try:
                    pd.concat([original, original.iloc[[0]]]).to_parquet(path, index=False)
                    with self.assertRaisesRegex(ValueError, "Duplicate row_id"):
                        _load_day("test", pd.Timestamp("2022-09-08"), self.features_dir, ["cross_border"])
                finally:
                    original.to_parquet(path, index=False)

    def test_daily_batch_excludes_ineligible_rows(self):
        # Match the modelling notebook: do not change its comparison population.
        path = self.features_dir / "test_meta.parquet"
        original = pd.read_parquet(path)
        for column, value in (("seed_from", 0), ("own_date_split", "train")):
            with self.subTest(column=column):
                changed = original.copy()
                changed.loc[0, column] = value
                changed.to_parquet(path, index=False)
                features, meta = _load_day(
                    "test", pd.Timestamp("2022-09-08"), self.features_dir, ["cross_border"]
                )
                self.assertEqual(meta["row_id"].tolist(), [11])
                self.assertEqual(features["cross_border"].tolist(), [1])

    def test_rejects_row_that_is_not_an_alert(self):
        # The service must not run a case review for a non-alert.
        class FakeModel:
            features = ["cross_border", "country_from", "country_to"]

        fake_modeling = SimpleNamespace(
            load_bundle=lambda path: {"model": FakeModel(), "features": FakeModel.features},
            score_transactions=lambda bundle, X, meta, explain=True: pd.DataFrame(
                {"score": [0.1, 0.9], "alert": [False, False]}
            ),
        )

        with patch("agents.service._get_modeling_module", return_value=fake_modeling):
            with self.assertRaisesRegex(ValueError, "not in the model's daily alert set"):
                review_alert(
                    11,
                    features_dir=self.features_dir,
                    model_path=self.model_path,
                    text_generator=lambda system, user: "",
                )


if __name__ == "__main__":
    # Let this test file run directly as well as through unittest discovery.
    unittest.main()
