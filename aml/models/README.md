# models

Written by `notebooks/04_MODELLING.ipynb`:

- `txn_scorer.joblib` - LightGBM on all features (main scorer for the agents)
- `txn_scorer_no_generator.joblib` - without payment format and conversion_before
- `test_scores.parquet` - every test row with each model's score and alert flag
- `metrics.json` - all numbers behind `reports/model_findings.md`

Load with `src/sample_modeling.load_bundle`, score with `src/sample_modeling.score_transactions`.
