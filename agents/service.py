"""Connect saved transaction scores to the agent review flow."""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from agents.case_context import ALERT_FIELDS, CaseContext, build_case_context
from agents.case_data import DEFAULT_JOINED_PATH, load_case_data  # Retrieve source-backed case history.
from agents.contracts import AgentReport
from agents.llm import TextGenerator
from agents.orchestrator import DebateResult, review_case
from agents.regulation_retriever import retrieve_regulations  # Retrieve local official AML passages.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The current preprocessing pipeline lives in aml; alerts_model is the old version.
AML_ROOT = PROJECT_ROOT / "aml"
DEFAULT_FEATURES_DIR = AML_ROOT / "data" / "processed" / "preprocessed"
DEFAULT_MODEL_PATH = AML_ROOT / "models" / "txn_scorer.joblib"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AlertReview:
    """Model output and both agent reports for one alert."""

    alert_id: str
    score: float
    reasons: tuple[str, ...]
    debate: DebateResult


def _get_modeling_module():
    """Import the scorer under the name used inside the saved model file."""
    # joblib needs this exact module name to rebuild the saved Python objects.
    source_dir = AML_ROOT / "src"
    source_path = str(source_dir.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    modeling = importlib.import_module("sample_modeling")
    expected_file = (source_dir / "sample_modeling.py").resolve()
    if Path(modeling.__file__).resolve() != expected_file:
        raise ImportError(
            "A different module named 'sample_modeling' is already imported; "
            f"expected {expected_file}."
        )
    return modeling


def _find_alert(
    row_id: int, features_dir: Path
) -> tuple[str, pd.Timestamp]:
    """Find which split contains the alert and return its timestamp."""
    matches = []
    for split in SPLITS:
        meta_path = features_dir / f"{split}_meta.parquet"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing split metadata: {meta_path}")
        rows = pd.read_parquet(
            meta_path,
            columns=["row_id", "Timestamp"],
            filters=[("row_id", "==", row_id)],
        )
        if not rows.empty:
            matches.append((split, pd.Timestamp(rows.iloc[0]["Timestamp"])))

    if not matches:
        raise KeyError(f"No prepared feature row found for row_id={row_id}.")
    if len(matches) > 1:
        raise ValueError(f"row_id={row_id} appears in multiple data splits.")
    return matches[0]


def _load_day(
    split: str,
    timestamp: pd.Timestamp,
    features_dir: Path,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one day's model features and metadata in matching row order."""
    start = timestamp.normalize()
    end = start + pd.Timedelta(days=1)
    meta_path = features_dir / f"{split}_meta.parquet"
    feature_path = features_dir / f"{split}_features.parquet"

    meta = pd.read_parquet(
        meta_path,
        filters=[("Timestamp", ">=", start), ("Timestamp", "<", end)],
    )
    if meta.empty:
        raise ValueError(f"No rows found for {start.date()} in the {split} split.")

    if meta["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id in metadata rows.")
    # Match sample_modeling.eval_mask: moved rows and unreachable sampled rows
    # do not belong in the notebook's daily alert budget. These are sampling
    # controls only; they are never evidence of suspicious behaviour.
    required = {"own_date_split", "seed_from", "seed_to", "weight"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Missing AML metadata columns: {sorted(missing)}")
    eligible = (meta["own_date_split"].astype(str) == split) & (
        (meta["seed_from"] == 1) | (meta["seed_to"] == 1)
    )
    meta = meta.loc[eligible]
    if meta.empty:
        raise ValueError("No eligible rows for the daily alert budget.")

    meta = meta.sort_values("row_id").reset_index(drop=True)
    row_ids = meta["row_id"].tolist()
    features = pd.read_parquet(
        feature_path,
        columns=["row_id", *feature_names],
        filters=[("row_id", "in", row_ids)],
    )
    # An ID must identify one transaction, otherwise evidence could be mixed up.
    if meta["row_id"].duplicated().any() or features["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id in feature rows or metadata rows.")
    if len(features) != len(meta) or set(features["row_id"]) != set(row_ids):
        raise ValueError("Feature rows and metadata rows do not match for this day.")

    features = features.set_index("row_id").loc[row_ids, list(feature_names)]
    return features.reset_index(drop=True), meta


def review_alert(
    row_id: int,
    *,
    features_dir: Path = DEFAULT_FEATURES_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    related_transactions: Sequence[Mapping[str, object]] | None = None,  # None enables automatic source retrieval.
    joined_path: Path = DEFAULT_JOINED_PATH,  # Allow a caller to select the joined source dataset.
    text_generator: TextGenerator | None = None,
) -> AlertReview:
    """Score one prepared transaction and create its two-sided agent review.

    The scorer evaluates the target alongside all prepared rows from that day,
    because alert selection uses a daily top-k budget.
    """
    if isinstance(row_id, bool) or not isinstance(row_id, int):
        raise TypeError("row_id must be an integer.")

    features_dir = Path(features_dir)
    model_path = Path(model_path)
    split, timestamp = _find_alert(row_id, features_dir)

    modeling = _get_modeling_module()
    bundle = modeling.load_bundle(model_path)
    feature_names = bundle.get("features") or getattr(bundle["model"], "features", None)
    if not feature_names:
        raise ValueError("The saved scorer does not list its input features.")

    X_day, meta_day = _load_day(split, timestamp, features_dir, feature_names)
    scored = modeling.score_transactions(bundle, X_day, meta_day, explain=False)

    target_rows = meta_day.index[meta_day["row_id"] == row_id].tolist()
    if len(target_rows) != 1:
        raise ValueError(f"row_id={row_id} is not eligible for the daily alert budget.")
    target_index = target_rows[0]
    model_output = scored.iloc[target_index].to_dict()
    if not bool(model_output.get("alert", False)):
        raise ValueError(f"row_id={row_id} is not in the model's daily alert set.")

    # Feature explanations are expensive, so compute them only for this alert.
    explained = modeling.score_transactions(
        bundle,
        X_day.iloc[[target_index]],
        meta_day.iloc[[target_index]],
        explain=True,
    )
    if "reasons" in explained.columns:
        model_output["reasons"] = explained.iloc[0]["reasons"]

    # Give case_context only fields intended for review, not training labels.
    alert_row = meta_day.iloc[target_index].to_dict()
    alert_row.update(X_day.iloc[target_index].to_dict())
    alert = {field: alert_row[field] for field in ALERT_FIELDS if field in alert_row}
    history_note = "Related transaction history was supplied by the caller."  # Describe caller-provided data.
    if related_transactions is None:  # Retrieve history automatically unless the caller supplied it.
        try:  # Keep a missing local source visible while allowing synthetic offline reviews.
            case_data = load_case_data(row_id, alert, joined_path=joined_path)  # Fetch target and recent history.
            related_transactions = case_data.transactions  # Pass retrieved source rows into the case context.
            history_note = case_data.coverage_note  # Preserve the lookup window and truncation status.
        except FileNotFoundError:  # Distinguish unavailable source data from an empty history result.
            related_transactions = ()  # Do not invent transaction history when the source file is absent.
            history_note = (  # State why the agents have no source history.
                "Related transaction history was not retrieved because the source file is missing: "  # Explain the lookup failure.
                f"{joined_path}."  # Identify the required source path.
            )  # Finish the missing-history note.
    # Search topics connected to the actual alert fields. The retriever expands
    # these plain-language terms into AML topic aliases and ranks local passages.
    explanation = model_output.get("reasons", "")
    if isinstance(explanation, Sequence) and not isinstance(explanation, str):
        explanation = " ".join(str(reason) for reason in explanation)
    elif not isinstance(explanation, str):
        explanation = ""
    case_terms = [
        str(alert.get(field, ""))
        for field in ("Payment Format", "country_from", "country_to")
    ]
    for transaction in related_transactions:
        if isinstance(transaction, Mapping):
            case_terms.extend(
                str(transaction.get(field, ""))
                for field in ("Payment Format", "Payment Currency", "country_from", "country_to")
            )
    regulation_query = " ".join(
        [
            *case_terms,
            explanation,
            "suspicious transaction monitoring payment transfer payer payee risk factors",
        ]
    )
    regulation_result = retrieve_regulations(
        regulation_query,
        case_date=timestamp.date(),
    )
    context: CaseContext = build_case_context(
        alert_id=str(row_id),
        alert=alert,
        transactions=related_transactions,  # Supply retrieved or explicitly provided transaction facts.
        model_output=model_output,
        additional_evidence=[  # Record retrieval coverage and add source-backed legal passages.
            {  # Represent the retrieval result as a case note, never as a transaction fact.
                "id": f"history-{row_id}-coverage",  # Give the coverage note a stable case-specific ID.
                "source": "case_record",  # Mark this as retrieval metadata rather than transaction evidence.
                "reference": f"history lookup for alert {row_id}",  # Identify the lookup described by the note.
                "description": history_note,  # Explain the time window, result count, or missing source.
            },  # Finish the retrieval-coverage evidence record.
            {  # Report which legal sources were available for this review.
                "id": f"regulation-search-{row_id}-coverage",  # Give the legal-search note a stable ID.
                "source": "case_record",  # Mark coverage as retrieval metadata, not a legal rule.
                "reference": f"regulation search for alert {row_id}",  # Identify the source-search operation.
                "description": regulation_result.coverage_note,  # State available and missing legal sources.
            },  # Finish the regulation-search coverage record.
            {
                "id": f"regulation-search-{row_id}-topics",
                "source": "case_record",
                "reference": f"topics used to retrieve regulations for alert {row_id}",
                "description": (
                    "Retrieved AML topics: "
                    + (", ".join(regulation_result.query_topics) or "no specific topic identified")
                ),
            },
        ] + [passage.as_evidence() for passage in regulation_result.passages],  # Add the retrieved legal passages.
    )
    debate = review_case(context, text_generator=text_generator)

    reasons = model_output.get("reasons")
    if isinstance(reasons, str):
        reason_list = (reasons,)
    elif isinstance(reasons, Sequence):
        reason_list = tuple(str(reason) for reason in reasons)
    else:
        reason_list = ()

    return AlertReview(
        alert_id=str(row_id),
        score=float(model_output["score"]),
        reasons=reason_list,
        debate=debate,
    )
