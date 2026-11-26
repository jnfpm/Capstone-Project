# CLEAR Agent Layer — Project status

**Editable project status report**  
Updated 26 November 2026

This Markdown report can be edited directly in any text editor or Markdown
editor. It refreshes the status described in `AGENT_PROJECT_STATUS.pdf` using
the current code and documentation; the original PDF remains unchanged.

## Purpose and decision boundary

CLEAR helps an analyst review anti-money-laundering alerts. A transaction model
selects alerts; language-model agents review the supplied evidence from
prosecution and defense perspectives. Reports, scores and legal excerpts
support an analyst. They do not establish that money laundering occurred and
must not make an automatic decision.

## Current agent flow

1. `service.py` finds the prepared alert, scores the eligible daily population,
   loads related history when available, and gathers local legal excerpts.
2. `case_context.py` creates individually identified facts from approved
   transaction, model and document fields.
3. `orchestrator.py` requests independent prosecution and defense reports.
   Each report is drafted, checked, edited once, and checked again.
4. The optional `run_debate()` flow requests one defense reply after both
   independent reports are complete. The service continues to use independent
   reports by default.
5. The optional `run_mediated_debate()` flow adds one mediator assessment after
   that reply. It does not decide guilt or whether to close an investigation.

Structural checks verify output shape, evidence IDs and reply targets. They do
not verify whether a cited passage semantically supports a claim. Human review
and broader evaluation are still needed.

## Regulations agent

`regulation_retriever.py` searches official EU acts and EBA guidance saved under
`agents/legislations/`. The service passes the alert date and builds query
terms from alert fields, supplied related transactions, and the model's
available explanation text. Results are ranked with BM25-style term weighting and a small
AML topic map covering reporting, customer due diligence, monitoring, risk
assessment, record keeping, transfers and criminal law. Topic aliases help
match ordinary wording differences. This is transparent lexical search, not
legal interpretation.

Each excerpt includes its document title, article or paragraph, version,
application information, official URL and a note about its timing relative to
the case. The note flags post-dated source versions and rules whose general
application begins later. For directives and guidelines, national
implementation or supervisory adoption still needs to be checked. Future
passages are retained as context and marked as such; they are not treated as
retroactive rules.

The case context records which documents were available, which were missing,
and which AML topics informed retrieval. A no-match result is explicit. The
retriever does not download sources during a case review. To refresh registered
EUR-Lex files, run `python -m agents.regulation_retriever --download` from the
repository root.

### Known limits

- Ranking depends on query wording and the hand-maintained AML topic aliases.
- It does not infer a legal issue from all case facts or establish jurisdiction,
  regulated-entity status, national transposition, or the correct law for a
  specific case.
- It marks temporal status but does not filter out future or post-dated texts;
  the reviewing agents must respect the status note.
- Retrieval quality and legal citation accuracy have not yet been evaluated
  across a representative, independently reviewed case set.

## Prepared-data integration

The service now points to the newer `aml/` pipeline rather than the older
`alerts_model/` copy. It aligns features and metadata by `row_id`, scores the
eligible daily population, and keeps training labels and sampling controls out
of suspicious-activity evidence. However, full real-data integration still
depends on obtaining the six prepared parquet files, aligning the saved-model
environment with the training environment, checking daily scores against the
modelling notebook, and validating a real case end to end.

Account history is loaded from the configured joined history file when
available, with a documented time window and coverage note. If the source is
missing, the case says that history was unavailable instead of claiming there
was no activity. History cutoff and source provenance should be verified as
part of real-data integration.

## Evaluation and remaining work

Offline tests use simulated provider responses and small data fixtures. They
check contracts, orchestration, context construction, retrieval metadata and
service handoff. Passing offline checks does not show that live reports are
factually correct or legally sound. Previously recorded live evaluations used
fictional development cases and need broader unseen cases and independent human
assessment.

Recommended next steps:

1. Evaluate regulation retrieval on labelled case questions and have a reviewer
   check each passage, version, legal status and case-date treatment.
2. Expand retrieval queries from the full allowed case facts and measure missed
   relevant provisions as well as irrelevant excerpts.
3. Obtain the prepared data, confirm model/library versions, and reproduce one
   complete daily score population against the modelling notebook.
4. Validate source-backed account history with a strict cutoff at the alert
   timestamp.
5. Evaluate debate and mediation with unseen cases, repeated runs and
   independent human reviewers.
6. Add durable audit records for source evidence, model/prompt versions,
   reports, replies, analyst actions, usage, latency and failures.
7. Connect the reviewed flow to the analyst interface and persistence layer.

## Main files

- `regulation_retriever.py` — local legal source loading, ranking and citations.
- `service.py` — prepared-data scoring and case assembly.
- `case_context.py` — allowed evidence fields and evidence IDs.
- `contracts.py` — report, reply and mediation structures and validation.
- `prompts.py`, `llm.py` — model instructions, provider calls and parsing.
- `orchestrator.py` — independent reports, optional debate and mediation.
- `tests/` — offline checks; they do not evaluate live legal correctness.

