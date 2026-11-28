# Connect model alerts to the investigation agents

The model produces a ranked alert. The investigation agents receive the alert
and a small set of source-backed facts, then write independent prosecution and
defense reviews. The agents do not receive the dataset's answer key.

## Run one review

Run these commands from the repository root, using the same Conda environment
used to train the model:

```bash
conda activate aml-clean
python -m pip install python-dotenv anthropic google-genai
python aml/run_agent_review.py --list-alerts
```

Copy one `row_id` from the list, then run:

```bash
python aml/run_agent_review.py --row-id 123456
```

Replace `123456` with a listed row ID. Listing alerts is local and makes no
language-model calls. Reviewing an alert makes four provider calls: each agent
creates a report and edits it once. Provider usage may incur cost.

Configure one provider in the repository-root `.env` file. For Google:

```dotenv
PROVIDER=google
GOOGLE_API_KEY=your_key
```

For Anthropic, set `PROVIDER=anthropic` and `ANTHROPIC_API_KEY=your_key`
instead. Keep `.env` private and out of version control.

## What happens

1. The command loads the saved AML model and finds the selected row in the
   prepared train, validation, or test data.
2. It scores the other eligible transactions from that same day too, because
   alerts are selected using the daily weighted alert budget. It stops if the
   selected row is not an alert under the saved model.
3. It retrieves the alert transaction and up to 20 related transfers from the
   previous 48 hours. The lookup stops before the alert timestamp and reads
   explicit transaction fields only.
4. The case includes the model score and top reasons, clearly labeled as model
   output rather than proof. Local regulatory passages and notes about data
   coverage are added when available.
5. Prosecution and defense receive the same evidence independently. Each report
   cites evidence IDs, which the program checks against the case record.

The terminal prints the score, model reasons, and both reports. Treat these as
investigation support: a valid evidence citation does not guarantee that a
claim is correct, and a model score is not proof of laundering.

## Required project files

The run needs the saved scorer, the prepared feature and metadata Parquet files,
and the joined transaction Parquet file. The AML notebooks create these under
`aml/models/` and `aml/data/processed/`. Use the environment that created the
saved scorer so its Python and LightGBM versions match the model artifact.

The integration code lives in:

- `aml/run_agent_review.py` — short command-line entry point and readable output.
- `agents/service.py` — scoring, source retrieval, case assembly, and handoff.
- `agents/case_data.py` — bounded source lookup without training labels.
- `agents/case_context.py` and `agents/orchestrator.py` — evidence IDs and the
  independent two-sided review flow.
