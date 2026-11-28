"""Print a legally contextualized example debate without provider API calls."""  # Document this offline demonstration module.

import json  # Serialize the simulated responses in the agent response format.
from collections.abc import Sequence  # Describe the evidence and finding collections used below.

from agents.case_context import CaseContext  # Import the fictional case context type.
from agents.contracts import Evidence  # Import the shared evidence record type.
from agents.orchestrator import run_debate  # Import the debate flow to demonstrate.
from agents.reference_cases import build_reference_context  # Load a fictional reference case.
from agents.regulation_retriever import RegulationPassage, retrieve_regulations  # Search the approved local legal corpus.


def _find_passage(query: str, document_key: str, article_prefix: str) -> RegulationPassage:  # Retrieve a specific relevant legal passage.
    results = retrieve_regulations(query, max_results=10)  # Search only registered local legal sources.
    match = next(  # Select the requested document section from the ranked local results.
        (passage for passage in results.passages if passage.document_key == document_key and passage.article.replace("\xa0", " ").startswith(article_prefix)),  # Normalize non-breaking spaces in HTML headings.
        None,  # Return no match if the expected local passage is missing.
    )  # Finish selecting the legal passage.
    if match is None:  # Stop instead of silently running a legally incomplete demo.
        raise RuntimeError(f"Could not retrieve {document_key} {article_prefix}; check the local legislation corpus.")  # Name the missing legal source.
    return match  # Return the exact source passage with its metadata and stable ID.


def _legal_context(context: CaseContext) -> tuple[CaseContext, list[RegulationPassage], dict[str, str]]:  # Add date-aware legal evidence to the case.
    reporting = _find_passage(  # Retrieve the historical suspicious-reporting provision.
        "obliged entity knows suspects reasonable grounds suspect funds proceeds criminal activity FIU reporting",  # Search the reporting threshold language.
        "eu_2015_849_2021",  # Use the historical AML Directive consolidation in force for the scenario date.
        "Article 33",  # Select the suspicious-activity reporting provision.
    )  # Finish retrieving the historical reporting provision.
    definition = _find_passage(  # Retrieve the historical EU money-laundering definition.
        "money laundering conversion transfer property knowing derived criminal activity conceal disguise illicit origin",  # Search the elements in the definition.
        "eu_2015_849_2021",  # Use the historical AML Directive consolidation.
        "Article 1",  # Select its definition provision.
    )  # Finish retrieving the historical definition.
    future_definition = _find_passage(  # Retrieve the AMLR definition for future-framework comparison only.
        "money laundering means conduct set out in Article 3 Directive 2018/1673 knowledge intent purpose objective factual circumstances",  # Search the AMLR definition wording.
        "eu_2024_1624",  # Select the newly registered Anti-Money Laundering Regulation.
        "Article 2",  # Select the AMLR definition provision.
    )  # Finish retrieving the AMLR definition.
    future_reporting = _find_passage(  # Retrieve the AMLR reporting provision for future comparison only.
        "obliged entities report FIU knows suspects reasonable grounds suspect funds proceeds criminal activity regardless amount",  # Search the AMLR reporting-threshold wording.
        "eu_2024_1624",  # Select the AMLR source entry.
        "Article 69",  # Select the reporting-of-suspicions provision.
    )  # Finish retrieving the AMLR reporting provision.
    passages = [reporting, definition, future_definition, future_reporting]  # Keep the selected historical and future provisions together.
    applicability = Evidence(  # Add a clear case-specific temporal and jurisdiction limit.
        id="legal-scope-2022-reference-case",  # Give the legal-scope note a stable evidence ID.
        source="case_record",  # Mark this as case scope metadata, not statutory text.
        reference="legal applicability note for synthetic case 01",  # Identify the scope note in citations.
        description="The transactions in this fictional case are dated 2022-09-08. Use the 2021 consolidated Directive (EU) 2015/849 as the EU framework for that date, subject to the applicable national transposition. Regulation (EU) 2024/1624 generally applies from 2027-07-10, so its passages are future-framework comparison only and are not rules governing these 2022 transactions. No national statute or regulated-entity status is supplied.",  # State the date and limits needed to avoid retroactive or jurisdictional claims.
    )  # Finish the temporal scope note.
    legal_evidence = tuple(  # Convert each retrieved official passage to the common evidence type.
        Evidence(  # Build one citeable legal evidence record.
            id=passage.evidence_id,  # Preserve the stable source passage ID.
            source="regulation",  # Mark the evidence as a legal source.
            reference=f"{passage.title}; {passage.article}; version {passage.version}; applies: {passage.applies_from}; {passage.url}",  # Preserve title, section, version, applicability, and URL.
            description=passage.text,  # Provide the retrieved source wording without rewriting it.
        )  # Finish converting one passage.
        for passage in passages  # Include every selected legal provision.
    )  # Finish converting the legal passages.
    combined = CaseContext(  # Make the legal texts available before debate generation starts.
        alert_id=context.alert_id,  # Preserve the reference case identifier.
        evidence=(*context.evidence, applicability, *legal_evidence),  # Give both roles identical case facts and legal sources.
    )  # Finish the legally contextualized case.
    ids = {  # Store readable names for the legal citations used by the fixed example arguments.
        "historical_reporting": reporting.evidence_id,  # Identify the 2022 reporting rule passage.
        "historical_definition": definition.evidence_id,  # Identify the 2022 definition passage.
        "future_definition": future_definition.evidence_id,  # Identify the future AMLR definition passage.
        "future_reporting": future_reporting.evidence_id,  # Identify the future AMLR reporting passage.
        "scope": applicability.id,  # Identify the scenario's date and jurisdiction limits.
    }  # Finish the legal citation map.
    return combined, passages, ids  # Return the enriched case, source passages, and citation IDs.


def _transaction_evidence_ids(context: CaseContext) -> list[str]:  # Collect facts supporting the transaction-pattern example.
    incoming_prefixes = ("transaction-in-1", "transaction-in-2", "transaction-in-3")  # Name the three fictional incoming transfers.
    evidence_ids = [  # Start citations with each incoming transfer's source facts.
        item.id  # Select the stable evidence ID for a matching transaction field.
        for prefix in incoming_prefixes  # Include each incoming transfer.
        for field in ("timestamp", "amount-usd", "acct-from", "acct-to")  # Cite time, amount, sender, and receiver.
        for item in context.evidence  # Search the combined case evidence.
        if item.id.startswith(prefix) and item.id.endswith(field)  # Match the requested transaction and field.
    ]  # Finish collecting incoming transfer citations.
    evidence_ids.extend(  # Add the outgoing transfer facts needed for comparison.
        item.id  # Select the stable evidence ID for this outgoing field.
        for field in ("timestamp", "amount-usd", "acct-from", "acct-to")  # Cite time, amount, sender, and receiver.
        for item in context.evidence  # Search the combined case evidence.
        if item.id.startswith("transaction-out-1") and item.id.endswith(field)  # Match the outgoing transaction.
    )  # Finish collecting outgoing transfer citations.
    return evidence_ids  # Return only existing transaction evidence IDs.


def _simulated_generator(context: CaseContext, legal_ids: dict[str, str]):  # Create deterministic, legally contextualized sample responses.
    transaction_ids = _transaction_evidence_ids(context)  # Prepare the original transaction citations.
    prosecution_ids = [*transaction_ids, legal_ids["historical_reporting"], legal_ids["scope"]]  # Cite the historical reporting rule and temporal limit.
    defense_ids = [*transaction_ids, legal_ids["historical_definition"], legal_ids["scope"]]  # Cite the historical definition and temporal limit.

    def generate(system_prompt: str, user_prompt: str) -> str:  # Match the provider-call function used by the orchestrator.
        if "bounded AML debate round" in system_prompt:  # Detect the single later defense reply.
            reply_ids = [*transaction_ids, legal_ids["historical_reporting"], legal_ids["historical_definition"], legal_ids["future_reporting"], legal_ids["scope"]]  # Cite facts, applicable law, future law, and temporal scope.
            return json.dumps(  # Return one schema-valid simulated defense reply.
                {  # Start the reply object.
                    "agent": "defense",  # Mark the response as the defense's reply.
                    "replies": [  # Provide one response to the prosecution finding.
                        {  # Start the response item.
                            "prosecution_finding": 1,  # Respond to prosecution finding one.
                            "position": "qualify",  # Qualify the risk interpretation without denying observed facts.
                            "claim": "I agree that the sequence warrants checking whether the Article 33 reporting threshold is met. I qualify the legal inference: the records show timing and account links, but do not establish criminal proceeds, knowledge, or a purpose to conceal. The AMLR passages are a future comparison only because this case is dated 2022.",  # State a bounded reply tied to historical law and the case date.
                            "evidence_ids": reply_ids,  # Cite original transaction facts and source-backed legal passages.
                        }  # Finish the response item.
                    ],  # Finish the replies list.
                }  # Finish the reply object.
            )  # Serialize the simulated reply.
        agent = "prosecution" if "prosecution-side" in system_prompt else "defense"  # Identify the requested initial-report role.
        if agent == "prosecution":  # Select the prosecution's prewritten legal analysis.
            claim = "Under Article 33 of Directive (EU) 2015/849, an obliged entity reports when it knows, suspects, or has reasonable grounds to suspect that funds are criminal proceeds. Three USD 5,000 transfers reached bank-A:central at five-minute intervals, followed by a USD 14,900 transfer from that account five minutes later. This pattern could warrant checking whether that reporting threshold is met, but the pattern alone does not establish criminal proceeds."  # Apply the historical reporting standard cautiously to the observed sequence.
            evidence_ids = prosecution_ids  # Cite transaction facts, Article 33, and the temporal-scope note.
        else:  # Select the defense's prewritten legal analysis.
            claim = "Article 1 of Directive (EU) 2015/849 describes money-laundering conduct as involving property derived from criminal activity, knowledge of that origin, and specified conduct such as transfer for concealment. These records show a rapid sequence through bank-A:central, but do not establish criminal origin, knowledge, or a concealment purpose. The evidence can support further checking; it is not enough to conclude that money laundering occurred."  # Compare the observed facts with the historical definition without deciding guilt.
            evidence_ids = defense_ids  # Cite transaction facts, the applicable-era definition, and temporal scope.
        return json.dumps(  # Return a schema-valid simulated independent report.
            {  # Start the report object.
                "agent": agent,  # Preserve the role requested by the orchestrator.
                "findings": [  # Include one example finding for this role.
                    {  # Start the finding object.
                        "claim": claim,  # Add the role-specific prewritten argument.
                        "evidence_ids": evidence_ids,  # Cite source evidence that exists in the case context.
                    }  # Finish the finding object.
                ],  # Finish the findings list.
            }  # Finish the report object.
        )  # Serialize the simulated report.

    return generate  # Return the fake generator to run_debate.


def _print_findings(title: str, findings: Sequence, evidence_by_id: dict[str, Evidence]) -> None:  # Print claims with readable citations.
    print(f"\n=== {title} ===")  # Mark the start of this debate section.
    for number, finding in enumerate(findings, start=1):  # Visit each report finding.
        print(f"\n{number}. {finding.claim}")  # Show the complete example argument.
        print("   Cited sources:")  # Label the source details.
        for evidence_id in finding.evidence_ids:  # Resolve each cited ID to its case evidence.
            item = evidence_by_id[evidence_id]  # Get the evidence item validated by the orchestrator.
            print(f"   - {item.reference}: {item.description}")  # Show provenance and passage or transaction fact.


def main() -> None:  # Run and print the offline legally contextualized example.
    base_context = build_reference_context("01_concentrated_transfers")  # Load the fictional 2022 transaction case.
    context, passages, legal_ids = _legal_context(base_context)  # Retrieve relevant laws and attach them before debate.
    generator = _simulated_generator(context, legal_ids)  # Create prewritten responses without a provider API.
    result = run_debate(context, text_generator=generator)  # Run the real orchestrator on the enriched evidence context.
    evidence_by_id = {item.id: item for item in context.evidence}  # Index all original and legal evidence for display.
    print("SIMULATED LEGAL DEBATE — PREWRITTEN RESPONSES; NO MODEL API CALLS")  # Clearly label the output as illustrative.
    print("Case date: 2022-09-08; AMLR 2024/1624 is included as future law, not applied retroactively.")  # State the date boundary before the reports.
    print("Both initial reviewers and the later reply received the same transaction and legal evidence.")  # State how law entered the debate.
    print("Simulated text-generator calls: 5")  # Show the five mocked draft, revision, and reply calls.
    print("\n=== LEGAL PASSAGES SUPPLIED TO THE DEBATE ===")  # Show the source material the roles received.
    for passage in passages:  # Visit each retrieved statute passage.
        print(f"\n{passage.title} — {passage.article}")  # Show its official title and section.
        print(f"Version: {passage.version}; applicability: {passage.applies_from}")  # Show historical or future legal scope.
        print(passage.text)  # Print the exact locally retrieved passage text.
    print("\nCASE SCOPE: No national statute or regulated-entity status is supplied; do not make a criminal-law finding.")  # State the remaining jurisdictional limit.
    _print_findings("PROSECUTION", result.prosecution.findings, evidence_by_id)  # Print the prosecution's simulated argument and citations.
    _print_findings("INDEPENDENT DEFENSE", result.defense.findings, evidence_by_id)  # Print the independent defense argument and citations.
    print("\n=== DEFENSE REPLY ===")  # Mark the later response to prosecution.
    for reply in result.defense_reply.replies:  # Visit each reply item.
        print(f"\nResponse to prosecution finding {reply.prosecution_finding} ({reply.position})")  # Show the target and stance.
        print(reply.claim)  # Show the simulated legal response.
        print("Cited sources:")  # Label the response citations.
        for evidence_id in reply.evidence_ids:  # Resolve each reply citation against the case evidence.
            item = evidence_by_id[evidence_id]  # Get the validated source item.
            print(f"- {item.reference}: {item.description}")  # Show the exact cited passage or transaction fact.


if __name__ == "__main__":  # Avoid running the demo when another module imports it.
    main()  # Start the simulated debate when this module is executed.
