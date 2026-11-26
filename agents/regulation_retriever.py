"""Search the local corpus of official AML laws and supervisory guidance.

This is a transparent lexical retriever, not a legal reasoning system. It ranks
source paragraphs using BM25-style term weighting plus AML topic aliases, keeps
source metadata attached, and reports whether a source is applicable on the
case date or is only future/current reference material.
"""

from __future__ import annotations  # Allow modern annotations without runtime evaluation.

import argparse
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_LEGISLATIONS_DIR = Path(__file__).resolve().parent / "legislations"  # Use the local source corpus.
MAX_PASSAGE_CHARS = 900  # Keep a single cited excerpt short enough for case context.
DEFAULT_MAX_RESULTS = 5
BM25_K1 = 1.2
BM25_B = 0.75
TOPIC_MATCH_BONUS = 0.35
DOWNLOAD_USER_AGENT = "CLEAR AML capstone research tool/1.0"  # Identify this research download client.
DOCUMENTS = (  # Register official documents and the local text fallback for each source.
    {  # Describe the EU-wide Anti-Money Laundering Regulation.
        "key": "eu_2024_1624",  # Use a stable internal document key.
        "title": "Regulation (EU) 2024/1624 on the prevention of the use of the financial system for money laundering or terrorist financing",  # Preserve the official title.
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1624",  # Cite the official EUR-Lex HTML.
        "download_url": "http://publications.europa.eu/resource/celex/32024R1624?language=eng",  # Use the Publications Office CELLAR copy when EUR-Lex blocks automated downloads.
        "html": "regulation_eu_2024_1624_amlr.html",  # Store the official HTML under a clear AMLR filename.
        "text": "",  # Use the official HTML as the trusted source.
        "version": "Published in the Official Journal on 2024-06-19",  # Record the publication date.
        "applies_from": "Generally from 2027-07-10; Article 3(3)(n) and (o) entities from 2029-07-10",
        "consolidated_on": "2024-06-19",
        "general_application_from": "2027-07-10",
        "legal_form": "regulation",
    },  # Finish the AMLR source entry.
    {  # Describe the EU funds and crypto-assets transfer regulation.
        "key": "eu_2023_1113",  # Use a stable internal document key.
        "title": "Regulation (EU) 2023/1113 on information accompanying transfers of funds and certain crypto-assets",  # Preserve the official title.
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R1113",  # Cite the official EUR-Lex HTML.
        "html": "regulation_eu_2023_1113.html",  # Store the downloaded HTML under a predictable name.
        "text": "",  # This source has no trusted text fallback in the repository.
        "version": "2023-05-31",  # Record the act's adoption date.
        "applies_from": "2024-12-30",  # Record the general application date.
        "consolidated_on": "2023-06-09",
        "general_application_from": "2024-12-30",
        "legal_form": "regulation",
    },  # Finish the 2023 transfer regulation entry.
    {  # Describe the legacy EU AML directive provided by the user.
        "key": "eu_2015_849_2021",  # Identify the consolidation applicable to the synthetic case date.
        "title": "Directive (EU) 2015/849, consolidated text as at 2021-06-30",  # Preserve the historical version label.
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02015L0849-20210630",  # Cite the official EUR-Lex historical consolidation.
        "download_url": "http://publications.europa.eu/resource/celex/02015L0849-20210630?language=eng",  # Use the official CELLAR XHTML manifestation for reliable retrieval.
        "html": "directive_eu_2015_849_consolidated_2021_06_30.html",  # Store the historical text separately from the 2024 consolidation.
        "text": "",  # Use the official historical XHTML as the trusted source.
        "version": "Consolidated text dated 2021-06-30",  # Record the historical source version.
        "applies_from": "EU framework in force on the 2022-09-08 reference-case date; check national transposition",  # State scope without treating the directive as a criminal judgment.
        "consolidated_on": "2021-06-30",
        "legal_form": "directive",
    },  # Finish the historical directive source entry.
    {  # Describe the later consolidated version retained for current reference.
        "key": "eu_2015_849",  # Use a stable internal document key.
        "title": "Directive (EU) 2015/849 on the prevention of the use of the financial system for money laundering or terrorist financing",  # Preserve the official title.
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02015L0849-20241230",  # Cite the official consolidated EUR-Lex text.
        "html": "directive_eu_2015_849_consolidated_2024_12_30.html",  # Store the downloaded consolidated HTML.
        "text": "",  # The local PDF extraction is garbled and is intentionally not used.
        "version": "Consolidated text dated 2024-12-30",  # Identify the text version used by this corpus.
        "applies_from": "2015-06-25; repealed with effect from 2027-07-10",  # Preserve its temporal scope.
        "consolidated_on": "2024-12-30",
        "legal_form": "directive",
    },  # Finish the 2015 AML directive entry.
    {  # Describe the money-laundering criminal-law directive.
        "key": "eu_2018_1673",  # Use a stable internal document key.
        "title": "Directive (EU) 2018/1673 on combating money laundering by criminal law",  # Preserve the official title.
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32018L1673",  # Cite the official EUR-Lex HTML.
        "html": "",  # Use the readable text extracted from the supplied official PDF.
        "text": "directive_eu_2018_1673.txt",  # Use the text extracted from the local official PDF if needed.
        "version": "2018-10-23",  # Record the act's adoption date.
        "applies_from": "See national transposition measures",  # Avoid implying direct application dates for a directive.
        "consolidated_on": "2018-10-23",
        "legal_form": "directive",
    },  # Finish the 2018 criminal-law directive entry.
    {  # Describe the amended EBA risk-factor guidelines.
        "key": "eba_gl_2021_02_2023_03",  # Use a stable internal document key.
        "title": "EBA Guidelines on ML/TF risk factors, EBA/GL/2021/02, consolidated with EBA/GL/2023/03",  # Preserve the consolidated title.
        "url": "https://www.eba.europa.eu/sites/default/files/document_library/Publications/Guidelines/2023/EBA-GL-2023-03/1061654/Guidelines%20ML%20TF%20Risk%20Factors_consolidated.pdf.pdf",  # Cite the official EBA publication.
        "html": "",  # No EBA HTML version was supplied for this corpus.
        "text": "eba_gl_2021_02_consolidated_amended_2023_03.txt",  # Use the readable text extracted from the local EBA PDF.
        "version": "2021-02, amended and consolidated in 2023-03",  # Record the applicable guideline version.
        "applies_from": "See EBA guideline status and national supervisory adoption",  # Avoid claiming uniform direct effect.
        "consolidated_on": "2023-03-31",
        "legal_form": "guideline",
    },  # Finish the EBA guideline entry.
)  # Finish the source registry.
STOP_WORDS = {  # Ignore words too common to distinguish legal topics.
    "about", "after", "against", "from", "into", "that", "their", "there", "these",  # Common function words.
    "this", "those", "through", "under", "using", "with", "within", "without",  # More function words.
    "money", "laundering", "transaction", "transactions", "risk", "risks", "aml", "ml", "tf",  # Broad AML terms.
}  # Finish the search stop-word list.
TOPIC_TERMS = {
    "reporting": {"report", "reporting", "suspicion", "suspicious", "fiu", "authority", "disclosure", "tipping", "off"},
    "customer_due_diligence": {"customer", "client", "beneficial", "owner", "identity", "verification", "diligence", "identify", "verify"},
    "monitoring": {"monitor", "monitoring", "ongoing", "scrutiny", "pattern", "unusual", "alert", "transaction"},
    "risk_assessment": {"risk", "assessment", "factors", "geography", "country", "product", "channel", "delivery"},
    "record_keeping": {"record", "records", "retention", "retain", "documentation", "documents"},
    "transfers": {"transfer", "transfers", "payer", "payee", "originator", "beneficiary", "crypto", "asset", "traceability", "payment"},
    "criminal_law": {"criminal", "crime", "proceeds", "conceal", "conversion", "acquisition", "possession"},
}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "br", "tr"}  # Preserve paragraph boundaries from HTML.
PAGE_MARKER = re.compile(r"^--- PAGE (\d+) ---$")  # Read page markers from local text fallback files.
ARTICLE_MARKER = re.compile(r"^(?:Article|Guideline)\s+([0-9]+[A-Za-z]?(?:\.[0-9]+)*)\b", re.IGNORECASE)  # Find legal article or guideline headings.
PARAGRAPH_MARKER = re.compile(r"^(\d{1,2}(?:\.\d+)*\.?|\([a-z0-9]+\))\s+")  # Find paragraph labels where visible.
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9-]{2,}")  # Extract useful English search terms.


@dataclass(frozen=True)  # Keep each returned excerpt and its provenance immutable.
class RegulationPassage:  # Represent one source passage suitable for case evidence.
    evidence_id: str  # Stable identifier the agents can cite.
    document_key: str  # Internal key for the source document.
    title: str  # Official document title.
    url: str  # Official source URL.
    article: str  # Article, guideline, paragraph, or page reference when available.
    version: str  # Source version or adoption date.
    applies_from: str  # Temporal applicability note from the source registry.
    text: str  # Exact retrieved passage text, shortened only at the configured limit.
    score: int  # Relative retrieval score; this is not a confidence or applicability score.
    temporal_status: str = "applicability not assessed"  # Explain how the source relates to the case date.

    def as_evidence(self) -> dict[str, str]:  # Convert this passage to the agent evidence format.
        reference = f"{self.title}; {self.article}; version {self.version}; applies: {self.applies_from}; case-date status: {self.temporal_status}; {self.url}"  # Preserve provenance and temporal scope.
        return {  # Return a mapping accepted by the case-context builder.
            "id": self.evidence_id,  # Give the passage its stable evidence ID.
            "source": "regulation",  # Mark the source as a regulation or formal guideline.
            "reference": reference,  # Include title, section, version, date, and URL.
            "description": self.text,  # Include the retrieved source text verbatim.
        }  # Finish the agent-facing evidence mapping.


@dataclass(frozen=True)  # Keep the search output and its limitations together.
class RegulationSearchResult:  # Report selected passages and source coverage.
    passages: tuple[RegulationPassage, ...]  # Hold the ranked source passages.
    coverage_note: str  # Explain which local documents were available or missing.
    query_topics: tuple[str, ...] = ()  # Show which AML concepts informed the search.


class _HTMLTextExtractor(HTMLParser):  # Convert official HTML documents into paragraph text.
    def __init__(self) -> None:  # Initialize the parser's text buffers.
        super().__init__(convert_charrefs=True)  # Decode HTML entities such as non-breaking spaces.
        self.blocks: list[str] = []  # Store completed paragraph-like blocks.
        self.current: list[str] = []  # Store text for the block currently being parsed.
        self.skip_depth = 0  # Track script and style content that should not be indexed.

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # Handle opening HTML tags.
        if tag in {"script", "style", "noscript"}:  # Ignore executable and presentation content.
            self.skip_depth += 1  # Enter a region whose text is not legal source text.
        elif self.skip_depth == 0 and tag in BLOCK_TAGS:  # Keep source paragraphs separated.
            self._flush()  # Finish any text collected before this block.

    def handle_endtag(self, tag: str) -> None:  # Handle closing HTML tags.
        if tag in {"script", "style", "noscript"} and self.skip_depth > 0:  # Leave ignored regions safely.
            self.skip_depth -= 1  # Resume reading visible document text.
        elif self.skip_depth == 0 and tag in BLOCK_TAGS:  # Finish a visible text block.
            self._flush()  # Preserve each HTML paragraph as a searchable passage.

    def handle_data(self, data: str) -> None:  # Collect visible text nodes.
        if self.skip_depth == 0 and data.strip():  # Ignore blank nodes and skipped script/style data.
            self.current.append(data.strip())  # Keep the visible source wording.

    def _flush(self) -> None:  # Store the current block when a boundary is reached.
        value = " ".join(self.current).strip()  # Join adjacent inline elements in reading order.
        if value:  # Ignore empty formatting blocks.
            self.blocks.append(value)  # Add the completed source block.
        self.current.clear()  # Prepare the buffer for the next block.


def download_official_sources(  # Download official EUR-Lex HTML files into the local corpus.
    directory: Path = DEFAULT_LEGISLATIONS_DIR,  # Select the destination folder.
    *,  # Require the overwrite preference to be explicit.
    overwrite: bool = False,  # Preserve existing downloaded source files by default.
) -> tuple[Path, ...]:  # Return the files successfully saved.
    directory = Path(directory)  # Normalize the destination path.
    directory.mkdir(parents=True, exist_ok=True)  # Create the corpus folder when needed.
    downloaded: list[Path] = []  # Track saved official pages.
    for document in DOCUMENTS:  # Process each registered source in a predictable order.
        filename = document["html"]  # Read the local HTML filename.
        if not filename:  # Skip entries for which no HTML source is registered.
            continue  # Leave those entries to their supplied text fallback.
        destination = directory / filename  # Build the local destination path.
        if destination.is_file() and destination.stat().st_size > 0 and not overwrite:  # Reuse only a non-empty saved source by default.
            downloaded.append(destination)  # Report the existing source as available.
            continue  # Move to the next registered source.
        download_url = document.get("download_url", document["url"])  # Select the registered official retrieval endpoint.
        headers = {"User-Agent": DOWNLOAD_USER_AGENT}  # Identify the source download client.
        if "download_url" in document:  # Apply CELLAR's required format and language negotiation for this entry.
            headers.update({"Accept": "application/xhtml+xml", "Accept-Language": "eng"})  # Request the official English XHTML manifestation.
        request = Request(download_url, headers=headers)  # Build a request to the registered official endpoint.
        try:  # Fetch only the official URLs listed in the source registry.
            with urlopen(request, timeout=30) as response:  # Open the official page with a bounded timeout.
                content_type = response.headers.get("Content-Type", "").lower()  # Check the returned format.
                body = response.read()  # Read the source page once.
            preview = body[:5000].decode("utf-8", errors="replace").lower()  # Inspect the page for access challenges.
            if not body or ("html" not in content_type and b"<html" not in body[:1000].lower()):  # Reject empty responses and unexpected formats.
                raise ValueError(f"Official source did not return HTML: {download_url}")  # Do not cache empty responses or error pages.
            if "javascript is disabled" in preview or "not a robot" in preview:  # Reject anti-bot challenge pages.
                raise RuntimeError(f"The official source blocked the automatic download; save the official HTML manually: {document['url']}")  # Explain the manual fallback.
            destination.write_bytes(body)  # Save the official HTML for offline retrieval.
            downloaded.append(destination)  # Record the saved page.
        except (URLError, TimeoutError) as error:  # Report network errors without returning false success.
            raise RuntimeError(f"Could not download official source {document['url']}: {error}") from error  # Explain how to diagnose the failure.
    return tuple(downloaded)  # Return all saved or already-existing HTML paths.


def _load_document_blocks(document: Mapping[str, str], directory: Path) -> tuple[list[str], str]:  # Read local HTML or text for one source.
    html_name = document["html"]  # Get the preferred HTML filename.
    if html_name and (directory / html_name).is_file():  # Prefer the official HTML page when present.
        parser = _HTMLTextExtractor()  # Create an HTML parser for this document.
        parser.feed((directory / html_name).read_text(encoding="utf-8", errors="replace"))  # Parse the saved official page.
        parser.close()  # Flush any final text collected by the parser.
        parser._flush()  # Preserve text after the final block boundary.
        return parser.blocks, "HTML"  # Return source blocks and the source format.
    text_name = document["text"]  # Get the trusted local text fallback filename.
    if text_name and (directory / text_name).is_file():  # Use the fallback only when it exists.
        text = (directory / text_name).read_text(encoding="utf-8", errors="replace")  # Read the extracted source text.
        return [line.strip() for line in text.splitlines() if line.strip()], "TXT"  # Preserve text and page markers.
    return [], "missing"  # Report that this source is not locally searchable.


def _tokens(value: str) -> set[str]:  # Normalize query and passage text into search terms.
    return {token for token in TOKEN_PATTERN.findall(value.lower()) if token not in STOP_WORDS}  # Remove broad and common terms.


def _split_passages(blocks: list[str]) -> list[tuple[str, str, str]]:  # Attach article and page labels to text blocks.
    passages: list[tuple[str, str, str]] = []  # Store article label, page label, and passage text.
    current_article = "Preamble"  # Use a clear fallback until an article heading appears.
    current_page = ""  # HTML sources do not carry a printed page number.
    for block in blocks:  # Walk source text in its original order.
        page_match = PAGE_MARKER.match(block)  # Detect page markers from extracted PDF text.
        if page_match:  # Update the page label without treating it as legal text.
            current_page = f"PDF page {page_match.group(1)}"  # Preserve the source PDF page number.
            continue  # Continue to the next source block.
        article_match = ARTICLE_MARKER.match(block)  # Detect article or EBA guideline headings.
        if article_match:  # Update the section label for subsequent paragraphs.
            heading = article_match.group(0).rstrip(": ")  # Keep the heading text as printed.
            current_article = heading  # Attach the heading to following source passages.
            remainder = block[article_match.end():].strip(" :.-")  # Keep any legal text following the heading.
            if not remainder:  # Do not index a heading as if it were a legal rule.
                continue  # Continue to the next source block.
            block = remainder  # Keep heading and passage separate in the citation metadata.
        paragraph_match = PARAGRAPH_MARKER.match(block)  # Find numbered paragraph labels if present.
        paragraph = paragraph_match.group(1).rstrip(".") if paragraph_match else ""  # Preserve the paragraph number.
        label = current_article  # Start the citation with its article or guideline.
        if paragraph:  # Add a paragraph marker only when the source provides one.
            label = f"{label}, paragraph {paragraph}"  # Create a more exact section citation.
        if current_page:  # Preserve printed PDF page information for local text sources.
            label = f"{label}, {current_page}"  # Add the page label to the citation.
        passages.append((label, current_page, block))  # Keep the passage with its source location.
    return passages  # Return all labeled source passages.


def _topic_matches(text: str) -> set[str]:
    """Return AML concepts represented in text, including broad AML terms."""
    words = set(TOKEN_PATTERN.findall(text.lower()))
    return {
        topic for topic, aliases in TOPIC_TERMS.items()
        if words & aliases
    }


def _bm25_scores(
    documents: list[tuple[Mapping[str, str], str, str, str]],
    query_terms: set[str],
    query_topics: set[str],
) -> list[float]:
    """Score paragraphs with BM25 term weighting and a small topic-match bonus.

    The topic bonus helps bridge common wording differences (for example,
    "payment" and "payer"), while the score remains explainable and offline.
    It does not establish that a legal rule applies to a case.
    """
    token_counts = [Counter(_tokens(text)) for _, _, _, text in documents]
    average_length = sum(sum(counts.values()) for counts in token_counts) / max(len(token_counts), 1)
    document_frequency = Counter(
        term for counts in token_counts for term in counts.keys()
    )
    total_documents = len(token_counts)
    scores: list[float] = []
    for (_, _, _, text), counts in zip(documents, token_counts):
        length = sum(counts.values())
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (total_documents - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + BM25_K1 * (
                1 - BM25_B + BM25_B * length / max(average_length, 1)
            )
            score += inverse_frequency * frequency * (BM25_K1 + 1) / denominator
        passage_topics = _topic_matches(text)
        score += TOPIC_MATCH_BONUS * len(query_topics & passage_topics)
        scores.append(score)
    return scores


def _case_date_status(document: Mapping[str, str], case_date: date | None) -> str:
    """Describe source timing without deciding which law governs the case."""
    if case_date is None:
        return "case date not supplied"
    try:
        published = date.fromisoformat(document.get("consolidated_on", ""))
    except (ValueError, KeyError):
        return "check source version and applicability"
    if published > case_date:
        return "source version post-dates the case; reference context only"
    if document.get("legal_form") == "directive":
        return "directive version predates the case; check national transposition and law in force"
    if document.get("legal_form") == "guideline":
        return "guideline version predates the case; check supervisory adoption and scope"
    try:
        effective = date.fromisoformat(document["general_application_from"])
    except (ValueError, KeyError):
        return "check the regulation's application dates and scope"
    if effective > case_date:
        return "published by the case date, but general application begins later"
    return "published and general application date is on or before the case date; check jurisdiction and national implementation"


def retrieve_regulations(  # Search local legal texts and return the best matching excerpts.
    query: str,  # Provide case-related terms to search for.
    *,  # Require optional search settings to be keyword arguments.
    directory: Path = DEFAULT_LEGISLATIONS_DIR,  # Search the project's local corpus.
    max_results: int = DEFAULT_MAX_RESULTS,  # Bound the number of returned legal excerpts.
    case_date: date | None = None,  # Provide the alert date for source-scope annotations.
) -> RegulationSearchResult:  # Return ranked source passages and a coverage note.
    directory = Path(directory)  # Normalize the corpus path.
    if max_results < 1:  # Require a positive result limit.
        raise ValueError("max_results must be at least 1.")  # Explain the invalid search setting.
    query_terms = _tokens(query)  # Prepare query terms once for the corpus search.
    query_topics = _topic_matches(query)  # Identify legal concepts before broad terms are removed.
    candidates: list[tuple[Mapping[str, str], str, str, str]] = []  # Hold source, location, page and text.
    available: list[str] = []  # Track source documents searchable in this run.
    missing: list[str] = []  # Track source documents not yet available locally.
    for document in DOCUMENTS:  # Search every source registered for the project.
        blocks, source_format = _load_document_blocks(document, directory)  # Read the best available local form.
        if not blocks:  # Keep missing sources visible in the coverage report.
            missing.append(document["key"])  # Record this document as unavailable.
            continue  # Continue searching the available documents.
        available.append(f"{document['key']} ({source_format})")  # Record the searchable source and format.
        for article, page, text in _split_passages(blocks):  # Search passages while preserving citations.
            passage_terms = _tokens(text)  # Tokenize the source passage.
            if query_terms & passage_terms or query_topics & _topic_matches(text):
                candidates.append((document, article, page, text))
    scores = _bm25_scores(candidates, query_terms, query_topics)
    ranked = sorted(
        ((score, *candidate) for score, candidate in zip(scores, candidates) if score > 0),
        key=lambda row: (-row[0], row[1]["key"], row[2], row[3], row[4]),
    )  # Prefer useful paragraphs and keep tie-breaking deterministic.
    selected: list[RegulationPassage] = []  # Store final evidence passages.
    per_document: dict[str, int] = {}  # Prevent one long act from taking every result slot.
    for score, document, article, page, text in ranked:  # Visit candidates from best score downward.
        if len(selected) >= max_results:  # Enforce the global result limit.
            break  # Stop once enough evidence passages have been selected.
        if per_document.get(document["key"], 0) >= 2:  # Keep no more than two passages from one source.
            continue  # Allow other relevant documents a chance to appear.
        excerpt = text[:MAX_PASSAGE_CHARS].strip()  # Bound the passage while preserving its beginning.
        if len(text) > MAX_PASSAGE_CHARS:  # Mark an excerpt shortened for prompt size.
            excerpt = excerpt.rstrip() + " …"  # Show that the displayed source excerpt was shortened.
        identity = f"{document['key']}|{article}|{excerpt}"  # Build the stable evidence identity input.
        evidence_id = "reg-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]  # Create a stable evidence ID.
        selected.append(  # Add the selected legal excerpt with complete provenance.
            RegulationPassage(  # Construct a typed, source-backed passage.
                evidence_id=evidence_id,  # Set the stable citation ID.
                document_key=document["key"],  # Identify the source in the registry.
                title=document["title"],  # Preserve the official title.
                url=document["url"],  # Preserve the official source URL.
                article=article,  # Include the article or guideline citation.
                version=document["version"],  # Include the source version.
                applies_from=document["applies_from"],  # Include the temporal applicability note.
                text=excerpt,  # Include the exact source excerpt used by the agents.
                score=round(score * 100),  # Preserve an easy-to-read relative relevance score.
                temporal_status=_case_date_status(document, case_date),  # Keep legal timing visible to the reviewers.
            )  # Finish the passage record.
        )  # Add the passage to the result list.
        per_document[document["key"]] = per_document.get(document["key"], 0) + 1  # Count excerpts by source.
    if available:  # Summarize exactly which sources were searchable.
        note = f"Searchable legal sources: {', '.join(available)}."  # Report available local sources.
    else:  # Explain what must happen before any regulation search can succeed.
        note = "No legal source text is available locally; run the explicit official-source download command."  # Report the missing corpus.
    if missing:  # Add the missing source keys to the coverage note.
        note += f" Not locally available: {', '.join(missing)}."  # Prevent a partial corpus from looking complete.
    if not selected:  # Distinguish no lexical match from missing source files.
        note += " No passages matched the query terms."  # Report the empty search result accurately.
    return RegulationSearchResult(
        passages=tuple(selected),
        coverage_note=note,
        query_topics=tuple(sorted(query_topics)),
    )  # Return excerpts, topic coverage and source availability.


def main() -> None:  # Run the source-download command-line entry point.
    parser = argparse.ArgumentParser(description="Download CLEAR's registered official legal source pages.")  # Explain the command purpose.
    parser.add_argument("--download", action="store_true", help="Download official EUR-Lex HTML pages into agents/legislations.")  # Require explicit download intent.
    parser.add_argument("--overwrite", action="store_true", help="Replace existing downloaded HTML pages.")  # Make replacement explicit.
    args = parser.parse_args()  # Read the user's command-line options.
    if not args.download:  # Avoid network calls from a help or accidental invocation.
        parser.print_help()  # Show the safe usage instructions.
        return  # Exit without contacting external sources.
    for path in download_official_sources(overwrite=args.overwrite):  # Fetch only registered official pages.
        print(path)  # Report each source saved or already present.


if __name__ == "__main__":  # Allow this module to be run as a command.
    main()  # Execute the explicit downloader interface.
