# CLEAR - Collaborative Laundering Evidence & Alert Review

CLEAR is a decision-support platform for AML (Anti-Money Laundering) compliance analysts.
A transaction-scoring machine learning model filters daily alerts, and a collaborative multi-agent LLM layer reviews flagged transactions, debates opposing perspectives (prosecution vs. defense), and validates evidence citations.

## How it works

1. `alerts_model` processes IBM AML transaction data and scores daily alerts with a trained LightGBM model.
2. `agents.service.review_alert(row_id)` retrieves the alert, computes feature contributions, and builds a sanitized `CaseContext`.
3. `agents.orchestrator.review_case` runs the prosecution and defense reviewers on the evidence.
4. `agents.contracts.validate_report` verifies that every finding cites exact, valid evidence IDs.
5. Structured reports are delivered to the analyst interface.

## Quickstart

```powershell
# 1. Install dependencies with uv
uv sync

# 2. Configure environment
# Copy .env.example to .env and insert your API key
copy .env.example .env

# 3. Run tests
uv run pytest
```

