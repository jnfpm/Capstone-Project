# agents/ — the investigation agents (next step)

The multi-agent layer that takes an alert from the scorer and investigates it. Team decisions so far:
Gemini (free tier), built from scratch without an agent framework. The team's current agent code sits in the
repository's top-level `agents/` folder; it can move here once it reads from this pipeline.

## What the agents get from this pipeline

| Need | Use |
|---|---|
| Score transfers and decide alerts | `src/sample_modeling.py`: `load_bundle("models/txn_scorer.joblib")`, then `score_transactions(bundle, X, meta)` → `score`, `alert` (each day's top 0.5% of weighted volume), `reasons` (top three feature contributions, as text) |
| Feature rows to score | `data/processed/preprocessed/{split}_features.parquet` + matching `{split}_meta.parquet` (same `row_id` order) |
| Look up an account's transfers | `data/processed/hi_small_joined.parquet` through DuckDB, **explicit columns only** (never `Is Laundering`, `Attempt ID`, `Pattern Type`) and always capped at the alert's timestamp (`WHERE "Timestamp" <= :as_of`) |
| Know whether an account's history is complete in the subsample | meta `seed_from` / `seed_to` (1 = full history kept) |

## Evaluating the agents

`models/test_scores.parquet` has every test row with its score, alert flag, `Attempt ID`, `Pattern Type` and
`scheme_truncated` (ground truth — for scoring the agents only, never shown to them):
- **case reconstruction:** schemes with `scheme_truncated == False` (complete) — did the agent find the
  scheme's accounts and transfers?
- **detection only:** truncated schemes (cut on 10 Sept);
- **false alerts:** alerted rows with `y == 0` — does the agent close them?
