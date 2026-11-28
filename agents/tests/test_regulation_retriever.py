"""Check official AML source registration and local passage retrieval."""  # Describe the regulation-retrieval tests.

import unittest  # Use the standard-library test runner.

from agents.reference_cases.show_simulated_debate import _legal_context  # Check the demo's law-evidence setup.
from agents.reference_cases import build_reference_context  # Build the synthetic case without provider calls.
from agents.regulation_retriever import DOCUMENTS, retrieve_regulations  # Inspect registered sources and local retrieval.


class RegulationRetrieverTests(unittest.TestCase):  # Group checks for the local legal corpus.
    def test_AMLR_is_registered_with_official_source_and_application_dates(self):  # Verify the AMLR source metadata.
        source = next(document for document in DOCUMENTS if document["key"] == "eu_2024_1624")  # Select the AMLR entry.
        self.assertIn("32024R1624", source["url"])  # Confirm the citation points to EUR-Lex CELEX 32024R1624.
        self.assertIn("2027-07-10", source["applies_from"])  # Preserve its general application date.
        self.assertEqual(source["html"], "regulation_eu_2024_1624_amlr.html")  # Keep its local filename explicit.

    def test_search_retrieves_AMLR_article_69_from_the_local_corpus(self):  # Verify the AMLR reporting passage is searchable offline.
        result = retrieve_regulations(  # Search the local legislation corpus without network access.
            "obliged entity suspects reasonable grounds report FIU criminal activity proceeds funds",  # Query Article 69 reporting terms.
            max_results=10,  # Allow the AMLR result to appear alongside related acts.
        )  # Finish the local legal search.
        passage = next(  # Select the retrieved AMLR reporting provision.
            item for item in result.passages if item.document_key == "eu_2024_1624" and item.article.replace("\xa0", " ").startswith("Article 69")  # Match the source key and article heading.
        )  # Finish selecting the retrieved passage.
        self.assertIn("reasonable grounds to suspect", passage.text)  # Confirm the official passage contains the reporting threshold.
        self.assertIn("2027-07-10", passage.applies_from)  # Confirm future applicability remains attached to the passage.
        self.assertIn("32024R1624", passage.url)  # Confirm the passage cites the AMLR's official EUR-Lex URL.

    def test_simulated_case_receives_historical_law_and_future_AMLR_with_scope_note(self):  # Verify legal passages are attached before debate orchestration.
        base_context = build_reference_context("01_concentrated_transfers")  # Load the fictional 2022 case.
        context, passages, legal_ids = _legal_context(base_context)  # Attach retrieved law and its applicability metadata.
        regulation_ids = {item.id for item in context.evidence if item.source == "regulation"}  # Collect legal evidence IDs in the case.
        self.assertTrue(regulation_ids)  # Require legal passages to be present.
        self.assertIn(legal_ids["historical_reporting"], regulation_ids)  # Include the 2021 AML Directive reporting rule.
        self.assertIn(legal_ids["future_reporting"], regulation_ids)  # Include the AMLR reporting rule as future context.
        self.assertEqual(len(passages), 4)  # Supply historical definition/reporting and future AMLR definition/reporting passages.
        scope = next(item for item in context.evidence if item.id == legal_ids["scope"])  # Find the case's temporal-scope note.
        self.assertIn("not rules governing these 2022 transactions", scope.description)  # Prevent retroactive application of the AMLR.


if __name__ == "__main__":  # Support running this test module directly.
    unittest.main()  # Execute the regulation-retrieval checks.
