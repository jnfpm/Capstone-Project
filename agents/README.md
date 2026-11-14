# Independent agent reviews

## Simple reading guide

Read the flow as: facts -> first draft -> checks -> one edit -> checks -> final report.
A check of the format is not a check of whether the conclusion is true.
The two agents work separately; neither reads the other report.

The model creates alerts; prosecution and defense review the supplied evidence.
The three [fictional reference cases](reference_cases/README.md) let us develop
this flow while preprocessing changes. They do not prove production readiness.

## Current flow

1. `case_context.py` turns the supplied alert, related transactions and model
   output into individually identified evidence facts.
2. `orchestrator.py` gives both reviewers the same context. Neither sees the
   other reviewer's report.
3. For each role, `llm.py` requests a draft and validates its shape and IDs.
4. The same model edits that draft once against the same evidence, using a
   separate editing instruction. The revision is validated and returned.

That is two model calls per report, four per complete case, when successful.
This is editing within each independent review, not debate between agents.
There is no application-level retry or fallback to an unreviewed draft when
validation fails. Provider SDK transport retries are separate behavior.

`prompts.py` asks for material observations, supported calculations and account
links, followed by one qualified assessment and missing-information request.
The editing pass removes unsupported details and repeated analytical points.
It does not guarantee correctness: each factual detail still needs assessment
against the cited evidence. An existing ID is not proof of semantic support.

## Configuration and execution

Install dependencies with `python -m pip install -r agents/requirements.txt`. The project-root
`.env` holds configuration; never commit credentials. The evaluated Anthropic
configuration is:

```dotenv
PROVIDER=anthropic
MODEL=claude-opus-5-5
ANTHROPIC_MAX_TOKENS=8192
```

Set `ANTHROPIC_API_KEY` locally as well. `config.py` reads these settings.
If MODEL is omitted, the Anthropic code default remains `claude-haiku-5-5`.
Specify the evaluated model explicitly to reproduce that configuration.
Restart existing Python sessions after changing settings.

Anthropic receives the JSON schema from `contracts.py`. Shape constraints do
not verify claims. Responses stopped at the token limit are rejected before
parsing. The 8192-token cap is a maximum per call, not a target length or a
completion guarantee. Editing adds calls and may increase cost and latency;
this experiment does not measure costs. Google was not live-evaluated here.

From the repository root, run offline tests without API credentials:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s agents\tests -v
```

Tests cover context, contracts, independent orchestration, prompt construction,
revision validation and incomplete responses. They use simulated text generation;
passing them does not establish real-model report quality.

For a live synthetic review (four provider calls):

```python
from agents.orchestrator import review_case
from agents.reference_cases import build_reference_context

result = review_case(build_reference_context("01_concentrated_transfers"))
print(result.prosecution)
print(result.defense)
```

For prepared data, `service.py` uses `aml/src/sample_modeling.py` and
`aml/models/txn_scorer.joblib`. It loads the selected alert's day from
`aml/data/processed/preprocessed/{train,val,test}_{features,meta}.parquet`.
It keeps rows from their own date split with at least one seed account, matching
the modelling notebook's evaluation population. It then applies the weighted
daily alert rule, builds the context and starts review:

```python
from agents.service import review_alert

# Replace this example ID with an eligible alert from your prepared data.
result = review_alert(681848)
print(result.debate.prosecution)
print(result.debate.defense)
```

This prepared-data example requires compatible preprocessing/model artifacts;
its real-data path was not exercised by the synthetic report experiment.
Related history can be supplied via `related_transactions`. Automatic retrieval
of broader history from the raw transaction store is not implemented.

### Integration checkpoint (14 November 2026)

The six prepared parquet files are not present in this checkout. They must come
from the new preprocessing notebook; the old `alerts_model` files are not a
substitute. `aml/models/test_scores.parquet` contains evaluation results, not
the complete feature and history input required by this service.

Offline tests use small parquet files and simulated reports. They check daily
scoring, row alignment, duplicate IDs, eligibility and exclusion of training
labels from prompts. They do not measure real report quality.

The saved model was successfully loaded locally, but scikit-learn reported a
version mismatch: saved estimator 1.5.2, installed 1.9.1. Loading alone does not
prove equivalent predictions. Reproduce the training environment with the
modelling author before validating real scores; the dependency lists are not
a fully pinned environment. No notebook retraining or live API run was done
for this integration.

Next: obtain the six parquet files, align library versions, verify one complete
eligible day's scores against the notebook, and connect account history with
a cutoff at the alert timestamp. Only then evaluate a real case with both agents.

## Evaluation records

Local plain-text outputs under `output/output/` preserve each experimental version.
Each output contains both roles. The three final confirmation outputs and the
consolidated report are in that folder; the 36 older outputs are in its
`archive_experiences/` subfolder. These are local artifacts, not repository fixtures.

Prompt-only revisions were insufficient. A bounded editing pass, Anthropic JSON
schema and larger response cap addressed separate writing/format problems, but
the Haiku runs still contained substantive errors. Sonnet improved coverage but
had remaining omissions. With identical final prompts, two consecutive Opus sets
(six reports each) met the development criteria in qualitative assistant review.
All 22 offline tests passed. The final reports contain 10 and 11 findings across
six reports, versus 33 in the original baseline; counts alone are not quality.
See local `output/output/final_iteration_report.txt` for failures, evidence and
limits. It explains each experiment in plain English. Final outputs
are `output{1st,2nd,3rd}case_v10_opus.txt` and the corresponding `_check.txt` files.
These artifacts remain local. Runtime code for both sets is commit `dd0fbbd`.
The local .env explicitly selects Opus; credentials and .env are not committed.

Only three development cases are used. A repeat on these cases is a consistency
check, not an independent evaluation set. Before broader use, add unseen cases
and human assessment, including the larger evaluation required by the handbook.
Expected answers stay in fixture evaluation fields, outside the model prompts.

## Module responsibilities

- `config.py`: provider, model, credentials and response budget.
- `contracts.py`: shared data shapes, JSON schema and structural validation.
- `case_context.py`: evidence construction and stable IDs.
- `prompts.py`: role, draft and editing instructions.
- `llm.py`: provider calls, parsing and validation of both stages.
- `prosecution.py` / `defense.py`: role entry points requiring a true alert.
- `orchestrator.py`: independent reviews on the same context.
- `service.py`: prepared-data scoring and handoff to the agents.

## Running a new experiment

`agents/reference_cases/run_agent_iteration.py` runs the same three fictional cases and saves
one TXT per case, with both role reports in each file. It does not train a model,
change prompts, grade prose automatically, or generate new cases. Errors are
recorded as NOT GRADED. Each successful run currently makes 12 model calls.

Read instructions without calling the API:

```powershell
.\.venv\Scripts\python.exe -B agents/reference_cases/run_agent_iteration.py --help
```

Only when intentionally starting a new paid experiment:

```powershell
.\.venv\Scripts\python.exe -B agents/reference_cases/run_agent_iteration.py new_experiment_name
```

Choose a new name each time. Existing outputs are never overwritten. Use
`--output-dir` to choose a different folder. Assess the results afterwards against
the fixture evaluation criteria; creating a file does not mean the report passed.
