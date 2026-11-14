from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader
import pypdfium2 as pdfium

out = Path('output/pdf/CLEAR_Weekly_Agent_Checklist.pdf')
out.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleC', fontName='Helvetica-Bold', fontSize=21, leading=25, spaceAfter=13, textColor=colors.HexColor('#18334B')))
styles.add(ParagraphStyle(name='HeadC', fontName='Helvetica-Bold', fontSize=12, leading=16, spaceBefore=10, spaceAfter=7, textColor=colors.HexColor('#007F79')))
styles.add(ParagraphStyle(name='BodyC', fontSize=10, leading=14, spaceAfter=8))
story = []
def add(text, style='BodyC'):
    story.append(Paragraph(text, styles[style]))
def task(code, title, action, done):
    story.append(KeepTogether([Paragraph(f'[ ] {code}. {title}', styles['HeadC']), Paragraph(action, styles['BodyC']), Paragraph('<b>Done when:</b> '+done, styles['BodyC'])]))
def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.drawString(44, 27, 'CLEAR | Updated 14 November 2026 | Working plan')
    canvas.drawRightString(551, 27, str(doc.page))

add('Agent checklist: where we are', 'TitleC')
add('Owner: Mariana | Prosecution, defense and orchestrator')
add('The assistant implements and explains each change simply. Code comments stay in English. Completed means verified at the stated scope, not that the full project is finished.')
add('1. Understand the flow - current baseline understood', 'HeadC')
add('[x] Evidence -> independent prosecution and defense reports -> validation.<br/>[x] Understand stop-on-error behaviour.<br/>[ ] Learn the new data handoff one step at a time.')
add('2. Reference cases - partly complete', 'HeadC')
add('[x] Three original cases and evaluation criteria.<br/>[x] Two drafts prepared and read by Mariana: documented payment (04) and empty evidence (05).<br/>[ ] Connect case 04 documents to the context.<br/>[ ] Verify case 05 rejection offline. Neither draft is in the live runner.')
add('3-4. Prosecution and defense - baseline evaluated', 'HeadC')
add('[x] Improve claim-to-evidence links and reduce repeated conclusions.<br/>[x] Preserve previous experimental outputs and final analysis.<br/>[ ] Evaluate unseen cases and obtain human assessment. Good results on three fictional cases do not establish wider reliability.')
add('5. Orchestrator - independent reviews checked', 'HeadC')
add('[x] Check prosecution failure, defense failure and invalid citations.<br/>[ ] Implement a bounded debate later, after the evidence handoff works.')
add('6. Preprocessing integration - initial adapter complete', 'HeadC')
add('[x] Bring in the new aml pipeline and preserve prompt edits.<br/>[x] Update scorer and file paths; match daily eligibility rules.<br/>[x] Check row alignment, duplicate IDs and label exclusion.<br/>[x] 29 offline tests passed in the integration session; no API calls.<br/>[x] Integration published to main and MarianaSFMartins (074c990).')
add('<b>Still missing:</b> the six prepared parquet files in aml/data/processed/preprocessed/. The saved model loads but reports a scikit-learn version mismatch (saved 1.5.2; local 1.9.1). Real scoring equivalence and automatic account-history retrieval remain unverified or unimplemented.')
add('7. Debate and final evaluation - pending', 'HeadC')
add('Keep the original seven-stage plan. Git work is complete for this integration; resume with the data handoff below.')
story.append(PageBreak())
add('Today: complete the data map', 'TitleC')
task('T1', 'Understand what reaches the agents', 'Read service.py and case_context.py with the assistant. Map row_id, timestamp, account pair, amount, payment format, model score and reasons to their source. Separate the selected transaction from its history.', 'Mariana can explain what is supplied, what is missing and why missing history does not mean no activity.')
task('T2', 'Locate the missing inputs', 'Obtain train/val/test_features.parquet and train/val/test_meta.parquet from the new preprocessing output. Check model/library versions with the teammate. Locate the full transaction-history source separately.', 'Paths, required columns and compatible model environment are identified. Do not silently use old alerts_model data.')
task('T3', 'Continue without waiting for real data', 'Implement explicit document-evidence support for case 04, keeping evaluation instructions out of prompts. Add the empty-context rejection check for case 05. Preserve any unsaved editor changes before editing shared files.', 'Focused offline checks pass: documents are available by ID; no evaluation text leaks; empty evidence causes zero model calls. Run the full suite after code changes.')
add('Following sessions: data first, then live review', 'HeadC')
add('[ ] With the real artifacts, reproduce one eligible day of scores and alert flags against modelling.<br/>[ ] Add account-history retrieval with a timestamp cutoff and clear coverage limits. Exclude labels and future transactions.<br/>[ ] Inspect the assembled evidence and run a simulated end-to-end review.<br/>[ ] Only then run one real API review and assess both reports against the evidence.<br/>[ ] Evaluate case 04 after its document integration is verified.')
add('Later: bounded debate and broader assessment', 'HeadC')
add('[ ] Add one defense reply addressing specific prosecution arguments.<br/>[ ] Keep original reports, citations and a clear one-round limit.<br/>[ ] Test failure paths and call counts offline.<br/>[ ] Expand to at least ten live-review cases and two evaluation series, following the existing plan; include unseen cases and human assessment.<br/>[ ] Record model/code versions, support for claims, failures and usage/latency when available. Case 05 is an offline rejection case, not a live report.')
add('<b>When to test:</b> reading files needs no tests. After implementation, run the affected tests, then the full suite. Do not regenerate old outputs just to resume work.<br/><font face="Courier" size="8">.\\.venv\\Scripts\\python.exe -B -m unittest discover -s agents/tests -v</font>')
SimpleDocTemplate(str(out), pagesize=(595.28,841.89), leftMargin=44, rightMargin=44, topMargin=35, bottomMargin=45, title='CLEAR - Current agent checklist').build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(out)
assert len(reader.pages) == 2, len(reader.pages)
doc = pdfium.PdfDocument(str(out))
for i, page in enumerate(doc):
    page.render(scale=1.2).to_pil().save(f'tmp/pdfs/current_checklist_{i+1}.png')
print(out.resolve())
