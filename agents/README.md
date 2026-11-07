# Agent review flow

The model creates alerts; the agents review them. The agent modules currently use
the evidence and model output supplied in a `CaseContext`.

## Files

- `contracts.py` defines evidence and report shapes and checks citations.
- `case_context.py` prepares facts and model output for one alert.
- `prompts.py` gives prosecution and defense separate instructions.
- `llm.py` calls the provider selected in `config.py`, parses its JSON, and
  validates the report and its evidence IDs.
- `prosecution.py` and `defense.py` request their respective reports.
- `orchestrator.py` runs both sides on the same case context.
- `service.py` loads and scores a prepared alert before starting the review.

## Running a review

Install the requirements with `pip install -r alerts_model/requirements.txt`,
set `PROVIDER` and the matching API key in the project-root `.env` file, then
review an alert by its `row_id`:

```python
from agents.service import review_alert

result = review_alert(681848)
print(result.debate.prosecution)
print(result.debate.defense)
```

`review_alert` scores all prepared rows from that day so the daily top-k alert
rule is applied correctly. It then builds the case context and asks the
orchestrator to make one provider call for each side. Transaction history beyond
the selected row can be supplied with `related_transactions`; loading that
history from the raw transaction store is not implemented yet.

Tests pass fake scorers and text generators, so they do not need credentials or
network access:

```powershell
python -m unittest discover -s agents\tests -v
```

The end-to-end loader that reads a selected alert, scores it with the saved
transaction model, and builds its `CaseContext` is not implemented yet. A small
`service.py` can own that handoff, leaving the agents independent of whether
case data comes from Parquet files or a database.
