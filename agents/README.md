# Independent agent reviews

## Optional single-round debate (implemented, not executed)

`orchestrator.run_debate(context)` first creates both independent reports,
then requests one defense reply to numbered prosecution findings. It returns
`prosecution`, `defense` and `defense_reply` separately. The successful path
uses five application-level model calls: two per initial report and one reply.
`review_case`, the prepared-data service and the existing experiment runner
still use independent reviews only. They do not silently start a debate.

See [AGENT_PROJECT_STATUS.pdf](AGENT_PROJECT_STATUS.pdf) for the updated project
status, Mariana's agent work and remaining tasks. This implementation has not been executed or tested, at Mariana's
request. Earlier passing tests and report evaluations do not validate it.

## Simple reading guide

Read the flow as: facts -> first draft -> checks -> one edit -> checks -> final report.
A check of the format is not a check of whether the conclusion is true.
The two agents work separately; neither reads the other report.

The model creates alerts; prosecution and defense review the supplied evidence.
The [fictional reference cases](reference_cases/README.md) let us develop this
flow while preprocessing changes. They do not prove production readiness.

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
Related history is retrieved by `agents/case_data.py` from
`aml/data/processed/hi_small_joined.parquet` when `review_alert` is called
without an explicit `related_transactions` argument. The loader includes the
selected transaction and up to 20 most recent transactions from the prior 48
hours that involve either alert account. It selects explicit source columns,
excludes training labels, assigns stable transaction IDs, and adds a coverage
note that says whether the history was capped. When the joined source file is
missing, the review records that history could not be retrieved instead of
presenting the empty result as proof that no transactions occurred. Callers can
still pass `related_transactions` to provide their own source rows.

Regulatory passages are searched locally by `agents/regulation_retriever.py`.
From the repository root, download the official EUR-Lex HTML sources once with
`python -m agents.regulation_retriever --download`. The downloader contacts
only the registered official URLs and does not run automatically during an
alert review. The local text files for the EBA guideline and Directive (EU)
2018/1673 are searchable without network access. Search results cite the
document title, section, version, applicability note and official URL. The
corpus includes Regulation (EU) 2023/1113 as a separate transfers and
crypto-assets source; it is not the AMLR (Regulation (EU) 2024/1624). A
coverage note is added to each service case so missing local documents remain
visible to the agents.

The corpus also includes the actual AMLR (Regulation (EU) 2024/1624), retrieved
as official English XHTML from the EU Publications Office CELLAR endpoint and
cited through EUR-Lex. Its general application date is 10 July 2027, so the
2022 reference debate includes AMLR passages only as future-framework context.
The old file named `duplicate_regulation_eu_2023_1113_misnamed_2024_1624.pdf`
remains Regulation (EU) 2023/1113; it is not used as the AMLR source. The
retriever also includes a 2021 consolidated Directive (EU) 2015/849 for the
2022 example date, alongside the 2024 consolidation for current reference.

`agents/reference_cases/show_simulated_debate.py` attaches retrieved passages
from Directive (EU) 2015/849 Articles 1 and 33 and AMLR Articles 2 and 69 before
calling `run_debate()`. It adds an explicit date and jurisdiction limit: the
case is dated 8 September 2022, AMLR is not applied retroactively, and no
national statute or regulated-entity status is supplied. The prosecution,
defense and reply text is prewritten for demonstration; only the retrieval,
evidence-ID checks and orchestration are executed. Run it from the repository
root in the `clear-aml` environment with `python -m agents.reference_cases.show_simulated_debate`.
No model API calls are made.

### New debate tests

`agents/tests/test_orchestrator.py` now contains four tests for
`run_debate()`:

1. `test_run_debate_keeps_initial_reports_independent_and_replies_after_both`
   replaces model calls with helpers named `create_prosecution`,
   `create_defense` and `create_reply`. The prosecution helper records that
   its report finished first and returns a simulated prosecution report. The
   defense helper receives only the original case context and returns a
   simulated independent defense report. The reply helper receives the
   original context and both completed reports, then returns a simulated
   defense reply. The test checks the order `prosecution -> defense -> reply`
   and confirms that all three outputs are returned separately.
2. `test_run_debate_rejects_reply_to_unknown_prosecution_finding` supplies a
   reply aimed at a prosecution finding that does not exist. It checks that
   reply validation rejects the unknown target.
3. `test_run_debate_does_not_start_reply_if_an_initial_report_fails` simulates
   a failure while creating the initial defense report. It checks that the
   error propagates and the reply helper is never called.
4. `test_run_debate_does_not_return_success_if_reply_generation_fails`
   simulates a failure while creating the final defense reply. It checks that
   the error propagates instead of returning a debate with only the two
   initial reports.

All model calls in these tests are replaced with simulated functions, so they
do not require provider credentials or incur API usage. The helpers in the
first test are local test functions; they do not change production behavior.

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

The paid runner still uses only the original three development cases. Case 04 is
now available for offline context checks, but has not been run live. A repeat on
the original cases is a consistency check, not an independent evaluation set.
Before broader use, add unseen cases and human assessment, including the larger
evaluation required by the handbook. Expected answers stay in fixture evaluation
fields, outside the model prompts.

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
