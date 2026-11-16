"""Rebuild the project status PDF; this script never imports or runs agents."""
from pathlib import Path
from shutil import copy2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader
import pypdfium2 as pdfium

original = Path('agents/AGENT_PROJECT_STATUS.pdf')
backup = Path('tmp/pdfs/AGENT_PROJECT_STATUS_before_20261116.pdf')
if not backup.exists():
    copy2(original, backup)
output = Path('output/pdf/AGENT_PROJECT_STATUS.pdf')
output.parent.mkdir(parents=True, exist_ok=True)
navy, teal, grey = '#193650', '#087F83', '#52616D'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Cover', fontName='Helvetica-Bold', fontSize=27, leading=32, textColor=colors.HexColor(navy), spaceAfter=14))
styles.add(ParagraphStyle(name='PageTitle', fontName='Helvetica-Bold', fontSize=21, leading=26, textColor=colors.HexColor(navy), spaceAfter=15))
styles.add(ParagraphStyle(name='Section', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor(teal), spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name='Text', fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor(navy), spaceAfter=8))
styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica', fontSize=8.3, leading=11.3, textColor=colors.HexColor(grey), spaceAfter=6))
styles.add(ParagraphStyle(name='Cell', fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor(navy)))
story = []
def para(text, style='Text'):
    return Paragraph(text, styles[style])
def add(text, style='Text'):
    story.append(para(text, style))
def title(text):
    add(text, 'PageTitle')
def section(text):
    add(text, 'Section')
def table(headers, rows, widths):
    data = [[para('<b>'+x+'</b>', 'Cell') for x in headers]]
    data += [[para(x, 'Cell') for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4EEF5')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor('#B8CDD9')),
    ]))
    story.append(t)
    story.append(Spacer(1,9))
def footer(c,d):
    c.setStrokeColor(colors.HexColor('#D2DDE5'))
    c.line(44,38,551,38)
    c.setFont('Helvetica',8)
    c.setFillColor(colors.HexColor(grey))
    c.drawString(44,25,'CLEAR Agent Layer | Updated 16 November 2026')
    c.drawRightString(551,25,f'{d.page} / 6')

# Page 1: keep the purpose of the original guide, with an explicit status date.
add('CLEAR Agent Layer', 'Cover')
add('Project status, Mariana\'s work, and what remains', 'PageTitle')
add('Updated 16 November 2026 | Replaces the 8 November status snapshot', 'SmallText')
add('CLEAR supports an analyst reviewing anti-money-laundering alerts. The transaction model selects alerts. The language-model agents explain the supplied evidence from prosecution and defense perspectives. Neither the score nor a report establishes that laundering occurred.')
section('What Mariana has progressed')
add('The original agent skeleton already existed. Mariana\'s subsequent agent workstream developed reference cases, improved report instructions and citations, evaluated live reports, strengthened failure checks, adapted the connection to the new preprocessing pipeline, and added a first bounded debate. The sections below distinguish these additions from the remaining team work.')
table(['Area','Status on 16 November'],[
('Independent reports','Implemented; live development evaluations exist on four fictional report cases. Each role drafts and revises once.'),
('Evidence and tests','Stable evidence references, document excerpts and rejection checks. <b>32 offline tests passed before the debate change.</b>'),
('New preprocessing','Adapter updated to <b>aml/</b> and checked with small fake data. Real prepared files are absent from this checkout; full integration is not verified.'),
('One-round debate','Implemented on 16 November: one defense reply after the independent reports. <b>Not executed or tested.</b>'),
('Wider application','Automatic account-history retrieval, regulatory retrieval, source verification, audit storage and analyst UI remain pending in the reviewed agent flow.'),
], [121,386])
section('How to read this update')
add('<b>Implemented</b> means code exists. <b>Offline checked</b> means simulated inputs or responses were used. <b>Live evaluated</b> means real model reports were inspected on the named development cases. None of these means production readiness.')
add('The proposal originally assigned Mariana full-stack/evaluation work; her current responsibility in this workstream is prosecution, defense and orchestration. Preprocessing is handled by a teammate. This records the current working arrangement, not a revision of the whole team\'s ownership. [S2]', 'SmallText')
story.append(PageBreak())

# Page 2: update the old file guide instead of claiming a new architecture.
title('1. Structure and current flow')
section('Independent reviews')
add('<b>Prepared rows + saved scorer</b> → service.py → case_context.py → review_case → prosecution report + independent defense report.<br/>Each report follows: draft → format/reference checks → one edit → checks → final report.')
section('Optional debate')
add('<b>run_debate(context)</b> first completes those independent reports, then requests one defense reply. The original reports stay separate. The prepared-data service and existing experiment runner still use independent reviews; they do not start debate automatically.')
table(['File in agents/','Purpose and current contribution'],[
('service.py','Find a row, load its eligible daily comparison group, align features by row_id, score it and build the case. Updated from alerts_model to aml.'),
('case_context.py','Build CaseContext from allowed alert/transaction fields, model output and optional document excerpts. Assign evidence IDs; exclude training labels. It does not fetch history.'),
('prosecution.py','Require an explicitly flagged alert, then request an evidence-supported prosecution report. It must not force an accusation.'),
('defense.py','Create the independent defense report. New create_reply receives both initial reports and responds once; both entry points require a true alert.'),
('orchestrator.py','review_case returns the two reports. New run_debate returns SingleRoundDebateResult: prosecution, defense and defense_reply. Errors remain visible.'),
('prompts.py','Independent role instructions, editing instructions and new reply instructions. Report text is an argument to discuss, never additional evidence.'),
('llm.py','The provider-call layer: draft/edit for independent reports; one call for a reply. Parse JSON and validate output. Supports simulated generators.'),
('contracts.py','Evidence, Finding and AgentReport; new ReplyFinding and DefenseReply. Validate format, response targets and allowed evidence IDs, not factual truth.'),
('config.py / __init__.py','Read provider/model settings from local .env / make agents importable. Anthropic is used in the recorded experiments; credentials are never documentation.'),
('reference_cases/ / tests/','Fictional inputs and evaluation criteria; runner for saving reports; offline checks for contexts, contracts, LLM handling, orchestration and service.'),
], [137,370])
add('Why references matter: knowing that a cited ID exists is only a structural check. A reviewer must still read that evidence and decide whether it supports every detail of the claim. [S3]', 'SmallText')
story.append(PageBreak())

# Page 3: distinguish actual evaluation from new, untested code.
title('2. Evidence, cases and evaluations')
section('Cases developed in Mariana\'s workstream')
table(['Case','Purpose and verified scope'],[
('01 - Concentrated transfers','Three incoming payments of 5,000 USD; one outgoing payment of 14,900 USD. Check totals, timing, account links and cautious interpretations.'),
('02 - Regular payments','Three payments of 1,200 USD, 31 days apart, between the same accounts. Regularity can support a hypothesis, not prove a legitimate purpose.'),
('03 - Limited information','Timestamp and model signal, but no transactions. Reports explain the limits without inventing a payment pattern.'),
('04 - Documented payment','One 1,200 USD payment plus fictional agreement/invoice excerpts. Compare terms with the transfer; do not claim authentication or verified service delivery.'),
('05 - Empty evidence','No facts or true alert. Offline checks verify rejection before text generation. This is not a live-report case.'),
], [136,371])
section('What was improved, and why')
add('Prompts now require support for each finding, including the final assessment. A claim that a payment matches an invoice needs both document and transaction references. A hypothesis that the transfer pays the documented fee also needs the amount, accounts and date. Claims about different senders need the source-account references, not just a shared destination.')
add('The writing was made more focused: combine the transfer/document comparison, avoid repeating conclusions and keep verification limits together. Missing verification is an information gap, not a suspicious indicator. An agreement stating a monthly fee does not demonstrate recurring transfers.')
section('Observed results, not guarantees')
add('On 15 November, case 04 went through a baseline and successive prompt revisions. Version v6 and its unchanged confirmation both produced satisfactory prosecution and defense reports in assistant review. Cases 01-03 were checked with v5; they were not live-rerun after the document-specific v6 addition. One connection failure was recorded separately as not graded.')
add('That session produced 28 final reports across 14 successful case executions. There are seven current TXT files and ten archived intermediate files under output/output/cases_04_05/. Each live file contains both reports; it is not one file per API call. The full sequence and limitations are in final_report.txt. [S4]')
section('What the 32 passing tests establish')
add('Before debate was added, offline checks covered evidence construction, JSON/ID validation, simulated report generation, failures in orchestration, daily-data alignment and case 05 rejection. A list-versus-string error in a case 04 test was corrected. These checks do not establish real-model reasoning quality, and their earlier success does not validate the new debate code.')
add('The development assessments were performed by the same assistant that changed the prompts. Unseen cases and independent human review remain necessary. No new agent execution was performed to prepare this PDF.', 'SmallText')
story.append(PageBreak())

title('3. The first debate: implemented only')
add('<b>Status: not run, not tested and not live-evaluated.</b> Mariana requested implementation only; colleagues will test it. Earlier report evaluations concern independent reviews.')
section('What was added on 16 November')
add('The new run_debate(context) completes the independent prosecution and defense reports first. Only then does defense receive both reports and the original evidence. It writes one reply addressing selected prosecution conclusions. There is no second round or judge, and neither original report is overwritten.')
table(['Reply field','Plain-language meaning'],[
('prosecution_finding','The conclusion being answered: 1 refers to the first prosecution finding. This number is not an evidence ID.'),
('position','agree, qualify (add a qualification), or disagree. Agreement is valid; the defense should not invent a disagreement.'),
('claim','The specific response and why it follows from the evidence.'),
('evidence_ids','Original evidence supporting that response. Earlier arguments do not become new facts.'),
], [139,368])
section('Limits written into the implementation')
add('There must be at least one response and at most one per prosecution finding. Validation rejects unknown or repeated targets, invalid positions, empty claims and missing/unknown evidence references. If a stage fails, its error reaches the caller; an incomplete combined debate is not returned as a success.')
add('The successful full flow is designed for five application-level model calls: two per independent report and one reply. The reply has no editing pass in this first version. SDK transport retries are separate. The same configured provider/model is used throughout.')
section('What colleagues need to test next')
add('<b>First, offline:</b> verify that the initial defense report does not see prosecution text; the later reply does; the evidence is unchanged; the original reports are preserved; and a successful run makes exactly five generator calls. Test failures in both initial reports and the reply, empty evidence, invalid targets and invalid citations.')
add('<b>Then, with the API:</b> check whether the reply addresses a specific argument and contributes a useful correction or qualification instead of repeating defense. Check support for every factual detail, justified agreement, and handling of conflicting documents. Compare initial defense versus initial defense plus reply.')
section('How to improve it after those checks')
add('Use observed failures to improve prompts and validation. Evaluate usefulness and extra latency/usage before adding another editing pass or more rounds. Define analyst-facing failure handling and decide when the service/UI should call run_debate. Multi-round debate and a final adjudicator are not part of this small implementation.')
add('Implementation references: contracts.py, prompts.py, llm.py, defense.py and orchestrator.py. Provider boundaries and local validation follow handbook concepts, but full conformance and behaviour are not yet verified. [S3, S5]', 'SmallText')
story.append(PageBreak())

title('4. Preprocessing: connected in code,')
add('not yet verified with real prepared data', 'Section')
add('The teammate\'s new pipeline lives under <b>aml/</b>. Mariana\'s integration work changed service.py to use that pipeline instead of the older alerts_model paths. This adapter is a first connection, not a completed end-to-end investigation.')
table(['Input','Current source / meaning'],[
('Prepared features','aml/data/processed/preprocessed/{train,val,test}_features.parquet: model inputs keyed by row_id.'),
('Prepared metadata','The matching *_meta.parquet files: timestamp, accounts, amount/payment format and daily-population controls.'),
('Saved scorer','aml/models/txn_scorer.joblib, loaded through aml/src/sample_modeling.py. Returns score, alert and available explanation text.'),
('Related history','Caller-supplied related_transactions. Automatic retrieval from the full transaction store/database is still missing.'),
('Agent evidence','Allowed transaction facts and model outputs; optional document excerpts. Evaluation labels and sampling controls are not evidence of suspicious behaviour.'),
], [123,384])
section('What the adapter already does')
add('It matches feature rows and metadata by row_id and rejects duplicate or mismatched rows. It scores the eligible day rather than just one transaction, because alert selection depends on the daily comparison population and weights. The eligible subset follows modelling: own_date_split matches the split, and at least one side is a seed account. These sampling fields are not sent as risk evidence.')
add('It obtains the selected row\'s explanation and creates the context for independent reviews. Small fake parquet files and a fake scorer were used to check this handoff; no real-data investigation was established by those tests.')
section('What is still needed')
add('<b>1. Obtain the six prepared files.</b> They are absent from the expected folder in this checkout as of 16 November. The committed test_scores.parquet file is an evaluation output, not a replacement for the full feature/metadata input.')
add('<b>2. Align the environment.</b> During the earlier model-load check, scikit-learn warned that a saved estimator used version 1.5.2 while the local environment used 1.9.1. Loading succeeded, but prediction equivalence was not established. Resolve this with the modelling teammate using the agreed locked environment.')
add('<b>3. Verify one eligible day.</b> Compare scores and alert flags against modelling, then inspect a selected context with simulated text generation. Confirm row IDs, units, currencies, timestamp/timezone conventions and missing values.')
add('<b>4. Connect account history.</b> Fetch relevant transactions with an explicit cutoff at the alert time; exclude future activity and labels. Preserve source IDs and communicate coverage so that unavailable history is not mistaken for no activity.')
add('<b>Complete when:</b> a real prepared alert has reproducible scoring, correctly sourced evidence and traceable history; the offline handoff passes, followed by a controlled live review. [S3, S6]', 'SmallText')
story.append(PageBreak())

title('5. Remaining work and priorities')
table(['Priority / owner','Task and completion evidence'],[
('1 - Agent colleagues','Test the new debate offline, then assess one live reply. Record failures, call counts and whether the reply adds supported analysis.'),
('1 - Data + agents','Obtain prepared files, resolve model/environment compatibility and verify daily scoring. Implement history retrieval with source IDs and time/coverage limits.'),
('2 - Mariana + reviewers','Expand beyond these development cases, including conflicting documents and unseen examples. Use labelled criteria, repeated runs and independent human assessment.'),
('2 - Agents + data team','Resolve citations against actual transaction records and exact document passages. Current ID membership checks alone do not verify the underlying source or meaning.'),
('3 - Team / supervisor','Confirm jurisdiction and approved regulatory sources, then retrieve relevant passages with title, version/date, URL and section. Regulation retrieval remains planned.'),
('3 - Backend + agents','Persist reviews, original evidence, reports/replies, model/prompt versions and analyst actions. Record actual model, usage, latency and failures; TXT outputs are not a complete audit/tracing system.'),
('3 - Backend + UI team','Connect alert queue, case file and case-specific chat. Present sources and uncertainty clearly; coordinate application and database ownership.'),
], [129,378])
section('Handbook alignment: what remains to demonstrate')
add('The code separates configuration, prompts, shared types, provider calls and orchestration. It rejects invalid output locally and permits fake generators. Full handbook compliance is not claimed: colleagues still need a clean-clone/locked-environment check, debate tests, review of schema constraints against Chapter 3, systematic tracing and broader labelled evaluations.')
add('The handbook asks for at least ten labelled real inputs and field-level scoring (printed p.11), manual evaluation in at least two sets and offline loop checks (p.24), and actual request model/usage recording on Claude (p.37). The five fictional scenarios, including one rejection-only case, do not meet that wider evaluation requirement. [S5]')
section('Sources and scope of this update')
add('<b>[S1]</b> Original AGENT_PROJECT_STATUS.pdf, 8 November, pp.1-4: baseline structure and outstanding components. This update supersedes that snapshot.<br/><b>[S2]</b> CapstoneProjectProposal.pdf, p.1: intended analyst workflow, regulatory citations, database/backend and UI scope; original team allocation.<br/><b>[S3]</b> Current agents/ source files, read on 16 November; no agent code or tests executed for this update.<br/><b>[S4]</b> output/output/cases_04_05/final_report.txt and its saved reports; earlier baseline record at output/output/final_iteration_report.txt.<br/><b>[S5]</b> Capstone_Handbook.pdf: printed pp.3-4 (PDF pp.4-5), p.11 (PDF p.12), p.24 (PDF p.25), pp.35-37 (PDF pp.36-38).<br/><b>[S6]</b> aml/README.md, aml/src/sample_modeling.py, agents/service.py and the integration checkpoint in agents/README.md.', 'SmallText')
add('<b>Decision-support boundary:</b> the project assists an analyst. A score, cited report or debate reply is not a legal finding or an automatic decision.', 'SmallText')

SimpleDocTemplate(str(output), pagesize=(595.28,841.89), leftMargin=44, rightMargin=44,
    topMargin=38, bottomMargin=49, title='CLEAR Agent Layer - Project status and Mariana\'s work',
    author='CLEAR project').build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(output)
assert len(reader.pages) == 6, f'Expected six pages, got {len(reader.pages)}'
doc = pdfium.PdfDocument(str(output))
for i, page in enumerate(doc):
    page.render(scale=1.25).to_pil().save(f'tmp/pdfs/agent_status_updated_{i+1}.png')
print(output.resolve())
