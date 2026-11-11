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

### A2 - Clarify prompts and fix the JSON example: completed

`prompts.py` now has three clearly separated parts:

1. **Shared evidence rules** apply to both reviewers: use supplied facts, label
   hypotheses, cite supporting facts for comparisons and calculations, and avoid
   interpreting missing history as proof of no activity.
2. **Role instructions** define the two perspectives. Prosecution acknowledges
   weak or insufficient grounds for suspicion. Defense acknowledges observed
   patterns and frames alternative explanations as hypotheses to verify.
3. **User prompt construction** uses `json.dumps` to produce a valid JSON example
   with the requested role. The example placeholders must be replaced with real
   findings and supplied evidence IDs.

Both roles are told how to express insufficient evidence within the existing
report contract: return at least one finding explaining a limitation and cite
what was actually supplied. Citing an alert or score does not prove that external
history is absent. No report fields, provider settings or orchestration behavior
were changed. Reference-case answers remain outside the prompts.

### A3 - Offline prompt checks: completed with A2

The regression test in `tests/test_llm.py` checks the actual user prompt sent
through `generate_report` for both roles. It parses the JSON example, checks its
role and fields, and verifies that the supplied evidence is unchanged. It uses a
fake generator and makes no API calls. Existing context and orchestration tests
also exercise the updated prompts.

Prompt clarity and valid JSON examples do not prove real-model quality. Human
review against the reference-case criteria remains pending in B1-B2 and C1.

**Next task:** A4 - confirm provider configuration and inspect the evidence before
starting the real-model baseline in B1.

## Citation coverage and repetition experiment (11 November 2026)

The shared prompts now ask for evidence supporting every factual detail within
each finding, both sides of comparisons and each record covered by an "all" claim.
They also ask for distinct analytical points and consolidated missing-data requests.
No response schema or case-specific expected answer was added.

All 19 offline tests passed. Six new live reports were generated with the same
configured Anthropic model and 4096-token budget. Original reports were preserved.
The local text files are `output/pdf/output1stcase_v2.txt`,
`output/pdf/output2ndcase_v2.txt` and `output/pdf/output3rdcase_v2.txt`.
The detailed qualitative comparison is `output/pdf/comparison_v1_v2.txt`.

Results are mixed: total findings decreased from 33 to 26, but shorter reports
still repeated some limitations. Some comparisons had fuller citations; others
still omitted fields. Case 2 defense introduced an unsupported assertion that a
PT/USD/ACH combination is unusual. Every cited ID exists, but this does not prove
that the citations support the entire claim or its interpretation.

These prompts remain an evaluated iteration, not a demonstrated fix. One run per
role and case cannot establish repeatability; the full verdict records regressions
as well as improvements. Further prompt changes and runs require a separate
iteration rather than overwriting these outputs.

## Module reference

- `contracts.py` defines evidence and report shapes and checks citations.
- `case_context.py` prepares facts and model output for one alert.
- `prompts.py` gives prosecution and defense separate instructions.
- `llm.py` calls the provider selected in `config.py`, parses its JSON, and
  validates the report and its evidence IDs.
- `prosecution.py` and `defense.py` request their respective reports.
- `orchestrator.py` runs both sides on the same case context.
- `service.py` loads and scores a prepared alert before starting the review.

## Running a review

### Anthropic response budget

`agents/config.py` sets `ANTHROPIC_MAX_TOKENS` to 4096 by default. To override
it locally, add a positive integer to the project-root `.env`, for example:

```dotenv
ANTHROPIC_MAX_TOKENS=4096
```

The first live prosecution review produced incomplete JSON. A diagnostic repeat
at the original 1200-token limit returned `stop_reason=max_tokens`. The larger
budget gives reports and their citation IDs more room, but does not guarantee
completion. It is a maximum, not a requested report length; longer responses can
increase cost. This setting currently applies only to Anthropic. Restart an
existing Python session after changing configuration.

The Anthropic adapter rejects `stop_reason=max_tokens` before parsing the text,
with a clear error naming the configured limit. It does not repair partial JSON
or retry automatically. Offline tests cover incomplete responses (including
valid-looking JSON) and a completed report. All 19 offline tests passed.

On 11 November 2026, a live rerun of `01_concentrated_transfers` at 4096 tokens
completed both reports with `stop_reason=end_turn`. Prosecution used 2219 output
tokens and defense used 1989; each returned six findings. Both passed the existing
JSON and citation-ID checks. Semantic assessment is still pending, and this result
does not establish that the budget is sufficient for every case.

The initial local report and diagnostic files were removed at the user's request.
For now, inspect one report at a time in the terminal. Cases 2 and 3 still need
baseline runs, and semantic assessment remains pending.

### Review a prepared alert

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
