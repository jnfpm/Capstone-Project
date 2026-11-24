"""Run a prosecution/defense review for one AML model alert.

From the repository root:
    python aml/run_agent_review.py --list-alerts
    python aml/run_agent_review.py --row-id 123456

The first command lists eligible alerts from the saved test scores. The second
re-scores that alert with the saved bundle, retrieves its source transactions,
and sends the evidence to the two configured review agents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the repository root importable when this file is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AML_ROOT = PROJECT_ROOT / "aml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCORES_PATH = AML_ROOT / "models" / "test_scores.parquet"


def list_alerts(limit: int) -> None:
    """Print model-generated test alerts without showing their ground-truth labels."""
    if not SCORES_PATH.is_file():
        raise FileNotFoundError(
            f"Saved test scores are missing: {SCORES_PATH}. Run the AML modelling notebook first."
        )

    # Read only safe fields for choosing an alert. The parquet also contains
    # evaluation labels, but this script never prints or sends those labels.
    scores = pd.read_parquet(
        SCORES_PATH,
        columns=["row_id", "Timestamp", "score_lgbm_all", "alert_lgbm_all"],
    )
    alerts = scores.loc[scores["alert_lgbm_all"].astype(bool)].copy()
    alerts = alerts.sort_values(
        ["Timestamp", "score_lgbm_all"], ascending=[True, False]
    ).head(limit)

    if alerts.empty:
        print("No model alerts were found in the saved test scores.")
        return

    print("Model alerts you can review (copy a row_id for the next command):")
    print(
        alerts[["row_id", "Timestamp", "score_lgbm_all"]]
        .rename(columns={"score_lgbm_all": "score"})
        .to_string(index=False)
    )


def print_report(title: str, report: object) -> None:
    """Print one agent's findings and their evidence IDs in a readable form."""
    print(f"\n{title}")
    for number, finding in enumerate(report.findings, start=1):
        print(f"{number}. {finding.claim}")
        print(f"   Evidence: {', '.join(finding.evidence_ids)}")


def review(row_id: int) -> None:
    """Score one row and pass its supported case evidence to both agents."""
    # Import here so --help and --list-alerts do not need the provider SDKs.
    from agents.service import review_alert

    # The two agents each draft and revise a report, so allow time for the
    # provider responses and show that the command has started successfully.
    print(
        f"Preparing alert {row_id}; requesting prosecution and defense reviews...",
        flush=True,
    )
    result = review_alert(row_id)
    print(f"Alert {result.alert_id} | model score: {result.score:.6f}")
    if result.reasons:
        print("Model explanation (signals, not proof):")
        for reason in result.reasons:
            print(f"- {reason}")

    print_report("PROSECUTION REVIEW", result.debate.prosecution)
    print_report("DEFENSE REVIEW", result.debate.defense)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select an AML model alert and send its evidence for two-sided review."
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--list-alerts",
        action="store_true",
        help="list saved test alerts; does not call an external language model",
    )
    choice.add_argument(
        "--row-id",
        type=int,
        help="review this prepared transaction row (makes agent API calls)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="maximum number of alerts shown with --list-alerts (default: 10)",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.list_alerts:
        list_alerts(args.limit)
    else:
        review(args.row_id)


if __name__ == "__main__":
    main()
