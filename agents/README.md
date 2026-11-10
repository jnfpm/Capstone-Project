# Agent review flow

## Getting started with reference cases

The [three synthetic cases and evaluation guide](reference_cases/README.md)
support independent report development while preprocessing is being updated.
They include role-specific criteria, explanatory comments and offline tests
in `tests/test_case_context.py`. These tests do not measure real LLM quality.

The model creates alerts; the agents review them. The agent modules currently use
the evidence and model output supplied in a `CaseContext`.

## Weekly checklist progress

### A1 - Review both role instructions: completed

- **Prosecution:** identify evidence-supported reasons for further investigation
  without treating an alert as proof of wrongdoing.
- **Defense:** assess limitations, uncertainty and possible alternative
  interpretations without inventing facts or asserting innocence.

Both modules require a true model alert flag and call the same report-generation
pipeline with different role instructions. Defense does not receive prosecution's
report. Comments in `prompts.py`, `prosecution.py` and `defense.py` explain these
boundaries. A1 changes comments and documentation only; prompt text and runtime
behavior remain unchanged.

The existing prompts already require supplied evidence, exact citation IDs,
separation of evidence from instructions, and no invented facts or regulations.
The following are the next prompt improvements to consider in **A2**:

1. Replace the `"prosecution" or "defense"` example with the requested role so
   the example itself is valid JSON.
2. Explicitly distinguish observed facts from hypotheses. Defense alternatives
   must be framed as possibilities to verify, not established explanations.
3. Explain how to report insufficient evidence without inventing a suspicious
   pattern or treating missing history as proof of no activity. Keep the existing
   requirement for at least one cited finding in mind.
4. Require citations that support the actual claim, including the relevant facts
   for comparisons or calculations. A known evidence ID alone is not sufficient.
5. Ask defense to acknowledge observed patterns and prosecution to acknowledge
   limits, rather than forcing either side toward a predetermined conclusion.

These are general instructions to review, not case-specific expected answers.
Do not copy the reference-case evaluation criteria into prompts. Their effect
on real report quality remains unverified until the baseline evaluation.

**Next task:** A2. After changing prompts, complete the offline checks in A3
before making the real-model calls planned in B1.

## Module overview

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
