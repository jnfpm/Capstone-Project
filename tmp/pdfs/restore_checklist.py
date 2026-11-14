from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
s=getSampleStyleSheet()
for name,size,leading,color in [('TitleX',23,28,'#18334B'),('HeadX',14,18,'#007F79'),('BodyX',10,14,'#18334B'),('SmallX',8.5,12,'#52616D')]:
    s.add(ParagraphStyle(name=name,fontName='Arial-Bold' if name in ['TitleX','HeadX'] else 'Arial',fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=9))
story=[]
def p(t,style='BodyX'): return Paragraph(t,s[style])
def add(t,style='BodyX'): story.append(p(t,style))
def title(t,sub): add(sub,'SmallX');add(t,'TitleX')
def task(code,title,action,done,status='[ ]'):
    table=Table([[p(status,'SmallX'),p(f'<b>{code} - {title}</b><br/>{action}<br/><font color="#52616D"><b>Done when:</b> {done}</font>')]],colWidths=[30,469])
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    story.append(KeepTogether(table))
def footer(c,d):
    c.setFont('Arial',8);c.setFillColor(colors.HexColor('#52616D'));c.drawString(48,28,'CLEAR | Mariana | Checklist v1.1 | 11 November 2026');c.drawRightString(547,28,f'{d.page} / 4')

title('Weekly agent checklist','10-13 NOVEMBER 2026 | RECREATED FROM THE 10 NOVEMBER PLAN')
add('Same task IDs and sequence; progress updated on 11 November. The previous local PDF was no longer available. This edition is the working reference for this chat.','SmallX')
add('Goal and working agreement','HeadX')
add('Develop independent prosecution and defense reports, then strengthen the orchestrator. Use synthetic cases while preprocessing is updated. The assistant implements changes and explains them simply in Portuguese; code, comments and project documentation stay in English. Prefer small changes with a clear purpose.')
add('Current position: A4','HeadX')
add('<b>A1-A3 are complete.</b> The latest completed suite passed 17 tests. The first case produces 44 evidence items: 5 case facts, 36 transaction facts and 3 model outputs. Provider: google. Configured model: gemini-flash-lite-latest. The matching API key is not available to the current process. No live reports have been evaluated.')
task('A1','Review both role instructions','Read agents/prompts.py, prosecution.py and defense.py. Explain each role and identify general prompt improvements.','Roles documented and explanatory English comments added.','[x]')
task('A2','Fix JSON and clarify instructions','Use a valid role-specific JSON example. Separate facts from hypotheses, require supporting citations and explain insufficient evidence.','Implemented in prompts.py without changing the report fields or adding debate.','[x]')
task('A3','Check changes offline','Run context tests, LLM tests and the complete suite. Check valid JSON examples and unchanged evidence for both roles.','7 context tests, 4 LLM tests and 17 tests in the full suite passed.','[x]')
task('A4','Confirm configuration and inspect evidence','Inspect the first case and verify provider/model settings through agents/config.py. The user supplies the private key locally in .env; never paste it into chat.','Evidence is understood and a nonempty matching key is configured. A successful live call is still required to prove access.')
add('<b>Next action:</b> finish A4 by configuring the Google key locally, or explicitly choose another provider. Do not mark live access as verified until a call returns.','SmallX')

story.append(PageBreak());title('Evaluate before improving','WEDNESDAY / THURSDAY | BASELINE, DEFENSE AND PROSECUTION')
add('The dates are targets, not permission to skip checks. Carry unfinished work forward. Each real review through review_case makes two provider calls.','SmallX')
task('B1','Generate and save the baseline','After A4, run case 1 first using the real-review example in agents/reference_cases/README.md. If successful, run cases 2 and 3. Save the reports before further prompt changes.','Six reports saved with case, role, date, provider, configured model and code commit. Record the returned model when available. API failures are not graded as model failures.')
task('B2','Evaluate the three defense reports','Compare each report with evaluation.defense, forbidden and limitations. Inspect the actual cited facts, not just whether IDs exist.','Each criterion has Pass / Partial / Fail and an example from the report.')
task('B3','Make a targeted defense improvement','Use an observed failure to guide a small change to defense instructions in prompts.py. Edit defense.py only for a demonstrated code issue.','The reason for the change is documented. If no change is justified, record that instead.')
task('B4','Verify defense again','Run the offline suite after edits and regenerate affected defense reports. defense.create_report(context) can review one side with one call. Keep baseline and revised outputs.','The comparison shows whether the intended behavior improved and whether new errors appeared.')
task('C1','Evaluate the three prosecution reports','Check evaluation.prosecution, forbidden conclusions, amounts, timing, account relationships and citation support. Pay special attention to the insufficient-information case.','Each criterion is scored and remaining issues are explicit.')
task('C2','Make a targeted prosecution improvement','Adjust prosecution instructions for an observed failure. Do not force an accusation where the supplied data cannot support a pattern.','The improvement is tied to evidence from the baseline, or no change is needed and that decision is recorded.')
task('C3','Test and compare both roles','Run the complete suite after edits. Reevaluate affected prosecution cases and defense if shared instructions changed.','Comparison uses concrete report findings. Earlier outputs remain available.')
add('<b>Keep changes small:</b> one clear reason per change; no case-specific expected answers in prompts. API runs are manual evaluations, never part of offline tests.','SmallX')

story.append(PageBreak());title('Make coordination reliable','FRIDAY | ORCHESTRATOR AND HANDOFF')
task('D1','Document the current sequence','Read agents/orchestrator.py and tests/test_orchestrator.py. Trace context, prosecution, defense, validation and result.','You understand that both reviews are independent and errors currently propagate.')
task('D2','Choose failure behavior','Decide whether a failed agent stops the review or produces an explicitly incomplete result. Keeping the current stop-on-error policy is valid. Plan contract changes first if partial results are needed.','Incomplete work cannot appear as a complete review; the selected behavior is documented.')
task('D3','Test the chosen behavior','Use scripted responses or mocks in test_orchestrator.py for success, prosecution failure, defense failure and invalid reports. Assert observable outcomes.','Tests prove the selected policy; a deliberately broken behavior makes the relevant test fail. No API calls are required.')
task('E1','Run final checks','Run the offline suite. Once stable, generate and assess a final set of six reports on the same cases. Keep all runs and do not hide failed criteria.','Tests pass and each final report has a written assessment; remaining failures are listed.')
task('E2','Prepare a progress note','Summarize changes, tests, before/after results, model settings and limitations. Check with your colleague whether preprocessing changes affect the supplied context fields.','Your colleague can identify what works, what is incomplete and what integration requires.')
task('E3','Publish and choose the next priority','Review the diff and publish intended changes to MarianaSFMartins. Keep secrets and local environment files out of Git.','Tested changes are published and the next unfinished task is recorded.')
add('Scope and handbook alignment','HeadX')
add('The proposal describes evidence-based AML alert review and eventually debate, history gathering and regulatory retrieval. This week builds the independent-review foundation; those broader features remain future work.','SmallX')
add('Capstone_Handbook.pdf: physical PDF pp. 4-6 cover environment, config and keys; pp. 10-12 distinguish structured output from truth; pp. 21-25 separate offline code tests from repeated model evaluations. The handbook asks for at least ten labelled inputs and at least two evaluation sets. Our three cases are a starter set only.','SmallX')
add('Alignment still to address: the current code requests JSON and validates it manually; it does not yet enforce a full output schema through the API as taught in chapter 3. Record this gap and implement a small, tested change when scheduled. Do not claim full handbook compliance.','SmallX')

story.append(PageBreak());title('Commands and tracking','RUN FROM THE REPOSITORY ROOT')
add('Offline test commands','HeadX')
add('Use the existing project environment. Type each command on one line. The complete suite includes the two focused groups.','SmallX')
for label,cmd in [ ('T1 - Context and reference cases',r'.\.venv\Scripts\python.exe -B -m unittest discover -s agents\tests -p "test_case_context.py" -v'),('T2 - LLM response and prompt checks',r'.\.venv\Scripts\python.exe -B -m unittest discover -s agents\tests -p "test_llm.py" -v'),('T3 - Complete suite',r'.\.venv\Scripts\python.exe -B -m unittest discover -s agents\tests -v')]:
    add(label);add(cmd,'SmallX')
add('The handbook uses uv to manage the environment. For this unittest suite, the equivalent command is <b>uv run python -B -m unittest discover -s agents/tests -v</b>. Existing commands are retained here for continuity. No need to repeat tests simply because you read a file.','SmallX')
add('Review record','HeadX')
add('Suggested folder: output/agent_reviews/. Use distinct names such as baseline, defense_v1 and final. Record the case, role, timestamp, provider, model setting, code commit and report path.','SmallX')
rows=[[p('<b>Criterion</b>','SmallX'),p('<b>Result</b>','SmallX'),p('<b>Example / action</b>','SmallX')]]
for v in ['Role-specific criteria covered','Claims supported by cited facts','Amounts and relationships correct','Facts separated from hypotheses','No invented facts or forbidden conclusion']:
    rows.append([p(v,'SmallX'),p('Pass / Partial / Fail','SmallX'),p('')])
t=Table(rows,colWidths=[200,110,189],rowHeights=[25]+[34]*5)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F2F1')),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#CAD8DE')),('VALIGN',(0,0),(-1,-1),'TOP')]))
story.append(t);story.append(Spacer(1,12))
add('A report passes only when all relevant criteria pass. An invented fact, unsupported citation or prohibited conclusion prevents acceptance. Report provider/network errors separately as not graded.','SmallX')
add('How we follow the plan','HeadX')
add('The assistant implements and explains each step before moving on. Mark a task complete only after its checks. Use the next unfinished ID at the start of each session. For a new chat, attach this PDF and state the last completed task.','SmallX')
add('<b>Last completed:</b> A3 &nbsp; | &nbsp; <b>Current:</b> A4<br/><b>Pending:</b> private API key setup, then a first successful live call.<br/><b>Next after A4:</b> B1 - baseline reports.')
out=Path('output/pdf/CLEAR_Weekly_Agent_Checklist.pdf')
SimpleDocTemplate(str(out),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=40,bottomMargin=54,title='CLEAR - Weekly Agent Checklist',author='CLEAR project').build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(out);print('Pages:',len(r.pages));assert len(r.pages)==4
for i,pag in enumerate(r.pages,1):print(i,len(pag.extract_text()))
