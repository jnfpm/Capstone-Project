"""Preserve the original four pages and append a dated progress update."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader, PdfWriter
import pypdfium2 as pdfium

source = Path('tmp/pdfs/AGENT_PROJECT_STATUS_before_20261116.pdf')
appendix = Path('tmp/pdfs/agent_status_addendum.pdf')
output = Path('output/pdf/AGENT_PROJECT_STATUS.pdf')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='HeadingA', fontName='Helvetica-Bold', fontSize=21, leading=26, textColor=colors.HexColor('#193650'), spaceAfter=14))
styles.add(ParagraphStyle(name='SectionA', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#087F83'), spaceBefore=9, spaceAfter=6))
styles.add(ParagraphStyle(name='BodyA', fontName='Helvetica', fontSize=10, leading=13.5, textColor=colors.HexColor('#193650'), spaceAfter=8))
styles.add(ParagraphStyle(name='NoteA', fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor('#52616D'), spaceAfter=8))
story=[]
def add(text, style='BodyA'):
    story.append(Paragraph(text, styles[style]))
def section(text):
    add(text,'SectionA')
def footer(c,d):
    c.setStrokeColor(colors.HexColor('#D2DDE5'))
    c.line(44,38,551,38)
    c.setFont('Helvetica',8)
    c.setFillColor(colors.HexColor('#52616D'))
    c.drawString(44,25,'CLEAR Agent Layer | Addendum - 16 November 2026')
    c.drawRightString(551,25,f'Addendum {d.page} / 3 | Document page {d.page+4}')

add('Addendum: Mariana\'s agent work','HeadingA')
add('Progress update | 16 November 2026','NoteA')
add('<b>The preceding four pages are the original document, preserved without changes.</b> They describe the project on 8 November. This addendum records later work; references there to 12 tests, the old model paths and no debate are historical, not the current status.')
section('1. What was already there, and what was developed')
add('The original agent skeleton already included prosecution, defense, orchestration, evidence construction and a prepared-data service. Mariana\'s subsequent agent workstream strengthened that foundation through reference cases, prompt improvements, live report evaluation, failure checks, a new preprocessing adapter and a first bounded debate. The preprocessing pipeline itself was delivered by a teammate.')
section('2. Reference cases and evidence')
add('<b>Case 01:</b> three incoming transfers of 5,000 USD and an outgoing transfer of 14,900 USD. Check totals, timing and account links without treating the 100 USD difference as a balance.<br/><b>Case 02:</b> three payments of 1,200 USD, 31 days apart. Assess regularity without inventing a business purpose.<br/><b>Case 03:</b> an alert timestamp and model signal, but no transactions. Explain insufficiency without inventing activity.<br/><b>Case 04:</b> one payment plus fictional agreement/invoice excerpts. Compare the payment with documented terms, without claiming verified authenticity or services.<br/><b>Case 05:</b> empty evidence. Reject the request before calling the model; this is an offline check, not a live-report case.')
section('3. What improved in the reports')
add('Instructions now require each finding to cite the facts supporting every material detail, including hypotheses and the final assessment. A payment-to-invoice comparison needs document and transaction references. Claims about several senders need the source-account references. A citation in an earlier finding does not automatically support the next one.')
add('Writing instructions reduce repeated transaction summaries, conclusions and missing-information requests. They distinguish documented purpose from verified legitimacy, contractual monthly fees from observed recurring payments, and missing verification from suspicious evidence. Each independent report uses a draft plus one editing pass, followed by validation.')
section('4. What was actually checked')
add('The 15 November session produced 28 final reports across 14 successful case executions. Case 04 was refined through successive versions; v6 and its unchanged confirmation were satisfactory in assistant review. Cases 01-03 were checked with v5, not live-rerun after the document-specific v6 addition. A connection failure was recorded separately as not graded.')
add('<b>32 offline tests passed before the debate implementation.</b> They covered evidence, output validation, orchestration failures, the service adapter and empty-evidence rejection. The case 04 test\'s list-versus-string error was also fixed. These results do not validate the new debate or prove real-model accuracy.')
add('Evidence: agents/reference_cases/, agents/tests/ and output/output/cases_04_05/final_report.txt. That folder contains seven current TXT files and ten archived intermediate files. Evaluations were conducted by the assistant that revised the prompts; independent human review remains necessary.','NoteA')
story.append(PageBreak())

add('Addendum: the first debate','HeadingA')
add('<b>Implemented on 16 November; not executed or tested.</b> Mariana requested implementation only. Colleagues will verify it. The earlier independent-report results must not be presented as debate results.')
section('5. How the small debate works')
add('<b>First:</b> prosecution produces its independent report.<br/><b>Second:</b> defense produces its own independent report, without seeing prosecution\'s report.<br/><b>Third:</b> both reports are validated. Defense then receives the original evidence and both reports, and writes a single reply to selected prosecution conclusions.<br/><b>Result:</b> the two original reports and the later defense reply are returned separately.')
add('The new entry point is <b>orchestrator.run_debate(context)</b>. The existing review_case(context), prepared-data service and experiment runner still produce independent reviews only. Debate is not started automatically by the older workflow.')
section('6. What each reply must contain')
add('<b>prosecution_finding:</b> the conclusion number being answered; 1 means the first prosecution finding.<br/><b>position:</b> agree, qualify (add a qualification), or disagree.<br/><b>claim:</b> the specific response and explanation.<br/><b>evidence_ids:</b> references to original case evidence, not to arguments as if they were facts.')
add('Agreement is valid. Defense should not manufacture disagreement or repeat its whole initial report. For example, it may accept a sequence of transfers while explaining why the sequence does not prove that particular incoming funds financed the outgoing payment. The explanation still needs the relevant original evidence.')
section('7. Boundaries implemented in the code')
add('There is one reply round, with at least one response and at most one per prosecution finding. Validation checks the role, target numbers, duplicate targets, permitted positions, non-empty claims and original evidence IDs. These checks do not establish semantic truth.')
add('There is no judge, second round or automatic guilt/innocence decision. The original reports are preserved. If a stage fails, the error reaches the caller rather than returning an incomplete debate as a success. The successful path is designed for five application-level model calls: two per initial report and one reply. SDK transport retries are separate; the reply has no editing pass in this first version.')
section('8. Files changed for this addition')
add('<b>contracts.py:</b> reply types, schema and validate_reply.<br/><b>prompts.py:</b> separate instructions and numbered prosecution arguments.<br/><b>llm.py:</b> one reply call, JSON parsing and validation.<br/><b>defense.py:</b> create_reply, separate from the independent create_report.<br/><b>orchestrator.py:</b> run_debate and SingleRoundDebateResult with prosecution, defense and defense_reply.')
section('9. What must happen before calling the debate validated')
add('Colleagues should first test independence, unchanged evidence, preservation of initial reports, the five-call successful path, invalid targets/citations and failures at each stage. Then assess live replies for useful argument-specific analysis, justified agreement and complete factual support. Compare the initial defense report with the extra reply to measure whether it adds value.')
add('These are planned verification tasks. No tests or live calls were run for the debate implementation or for this PDF update.','NoteA')
story.append(PageBreak())

add('Addendum: remaining work','HeadingA')
section('10. Finish the preprocessing integration')
add('<b>Already done:</b> service.py now uses aml/src/sample_modeling.py, aml/models/txn_scorer.joblib and aml/data/processed/preprocessed/. It aligns rows by row_id, checks duplicates/mismatches and scores the eligible daily population before reviewing a selected alert. Sampling controls and training labels are not passed as suspicious evidence. This adapter was checked with small fake parquet files and a fake scorer.')
add('<b>Still missing:</b> the six train/val/test feature and metadata parquet files are absent from the expected folder in this checkout as of 16 November. Obtain them and compare one eligible day\'s scores and flags with modelling. The committed test_scores.parquet is not a replacement for those inputs.')
add('During the earlier model-load check, scikit-learn reported saved-estimator version 1.5.2 versus local version 1.9.1. Loading succeeded, but score equivalence was not demonstrated. Align the environment with the modelling teammate before real-data validation.')
add('The service still relies on caller-supplied related_transactions. Add automatic account-history retrieval with stable source IDs, an explicit cutoff at the alert time and clear coverage limits. Confirm timestamp/timezone conventions, currencies, units and missing values. Then inspect a real context offline before a controlled live review.')
section('11. Evaluate and improve the debate')
add('After the first tests, fix observed failures in references, reasoning and repetition. Evaluate the extra latency and usage. Add another editing pass or further rounds only if the evaluation justifies them; neither is currently implemented. Decide how the service and UI should expose the reply and handle failures.')
section('12. Work from the original document that remains open')
add('<b>Source verification:</b> resolve evidence IDs to actual database records and exact document passages; ID membership alone is not source verification.<br/><b>Regulatory retrieval:</b> agree jurisdiction and approved sources, then retrieve passages with provenance, version/date and section references.<br/><b>Audit and tracing:</b> persist evidence, reports/replies, versions, analyst actions and actual model/usage/latency. Local TXT outputs are not a complete audit system.<br/><b>Application:</b> connect the backend, alert queue, case file and case-specific chat with the responsible teammates.')
section('13. Suggested next order')
add('<b>First, in parallel:</b> colleagues test the new debate; the data/agent workstream obtains compatible prepared inputs and verifies daily scoring.<br/><b>Next:</b> add time-bounded history retrieval and assess unseen cases, including contradictory document terms, with independent human review.<br/><b>Then:</b> improve debate from those results, complete source/regulation verification, and integrate persistence and the analyst interface.')
add('The handbook asks for at least ten labelled real inputs and field-level scoring (printed p.11 / PDF p.12), offline loop checks and at least two manual evaluation sets (p.24 / PDF p.25), and recording actual model and usage on Claude (p.37 / PDF p.38). Four fictional report cases plus one rejection scenario do not fulfil that wider evaluation requirement. Full handbook compliance is not claimed.','NoteA')
add('Sources: original status document (preceding pages); current agents/ source; aml/README.md; saved cases_04_05/final_report.txt; CapstoneProjectProposal.pdf p.1; Capstone_Handbook.pdf pages cited above. Preprocessing is a teammate responsibility; this addendum describes Mariana\'s agent work and the interfaces still to coordinate.','NoteA')
add('<b>Boundary:</b> reports and debate replies support an analyst; they are not legal findings or automatic decisions.','NoteA')

SimpleDocTemplate(str(appendix), pagesize=(595.28,841.89),leftMargin=44,rightMargin=44,
    topMargin=38,bottomMargin=49).build(story,onFirstPage=footer,onLaterPages=footer)
old, extra = PdfReader(source), PdfReader(appendix)
assert len(old.pages)==4 and len(extra.pages)==3
writer=PdfWriter()
writer.append(old)
writer.append(extra)
writer.add_metadata({'/Title':'CLEAR Agent Layer - Original status and Mariana progress addendum'})
with output.open('wb') as stream:
    writer.write(stream)
merged=PdfReader(output)
assert len(merged.pages)==7
old_render=pdfium.PdfDocument(str(source))
new_render=pdfium.PdfDocument(str(output))
for i in range(4):
    # Verify that copying pages did not change their content or appearance.
    assert merged.pages[i].extract_text()==old.pages[i].extract_text()
    before=old_render[i].render(scale=1).to_pil()
    after=new_render[i].render(scale=1).to_pil()
    assert before.size==after.size and before.tobytes()==after.tobytes()
for i in range(4,7):
    new_render[i].render(scale=1.25).to_pil().save(f'tmp/pdfs/agent_addendum_{i+1}.png')
print('7 pages. Original pages 1-4 match in text and rendered pixels.')

