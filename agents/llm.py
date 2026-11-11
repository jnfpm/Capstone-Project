"""Call the configured language model and validate its structured report."""

import json
from collections.abc import Callable, Sequence

from agents import config
from agents.contracts import AgentName, AgentReport, Evidence, Finding, validate_report
from agents.prompts import SYSTEM_PROMPTS, build_user_prompt

TextGenerator = Callable[[str, str], str]


def _generate_text(system_prompt: str, user_prompt: str) -> str:
    """Send prompts to the configured provider and return its text response."""
    if not config.API_KEY:
        raise RuntimeError(f"{config.KEY_NAME} is empty. Add it to the project .env file.")

    if config.PROVIDER == "google":
        from google import genai

        client = genai.Client(api_key=config.API_KEY)
        response = client.models.generate_content(
            model=config.MODEL,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
            },
        )
        if not response.text:
            raise RuntimeError("Google returned an empty response.")
        return response.text

    if config.PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=config.API_KEY)
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=config.ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # A token-limited response is incomplete even if its text happens to
        # parse as JSON. Reject it before parsing rather than repairing or
        # silently accepting a partial report. Retrying is a separate decision.
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Anthropic stopped at the response token limit "
                f"(ANTHROPIC_MAX_TOKENS={config.ANTHROPIC_MAX_TOKENS}). "
                "The report is incomplete and was not accepted. "
                "Increase the budget or reduce the requested output before retrying."
            )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text:
            raise RuntimeError("Anthropic returned an empty response.")
        return text

    raise ValueError(f"Unsupported provider: {config.PROVIDER!r}.")


def _parse_report(response_text: str, expected_agent: AgentName) -> AgentReport:
    """Convert provider JSON to the shared report type, rejecting bad shapes."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("The language model response was not valid JSON.") from error

    if not isinstance(payload, dict) or set(payload) != {"agent", "findings"}:
        raise ValueError("The language model response must contain agent and findings.")
    if payload["agent"] != expected_agent:
        raise ValueError(f"The language model response agent must be {expected_agent!r}.")
    if not isinstance(payload["findings"], list):
        raise ValueError("The language model findings must be a list.")

    findings = []
    for number, item in enumerate(payload["findings"], start=1):
        if not isinstance(item, dict) or set(item) != {"claim", "evidence_ids"}:
            raise ValueError(f"Finding {number} must contain claim and evidence_ids.")
        if not isinstance(item["claim"], str):
            raise ValueError(f"Finding {number} claim must be text.")
        evidence_ids = item["evidence_ids"]
        if not isinstance(evidence_ids, list) or not all(
            isinstance(evidence_id, str) for evidence_id in evidence_ids
        ):
            raise ValueError(f"Finding {number} evidence_ids must be a list of strings.")
        findings.append(Finding(claim=item["claim"], evidence_ids=evidence_ids))

    return AgentReport(agent=expected_agent, findings=findings)


def generate_report(
    agent: AgentName,
    evidence: Sequence[Evidence],
    *,
    text_generator: TextGenerator | None = None,
) -> AgentReport:
    """Generate a report, then check its shape and all evidence citations."""
    if agent not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported agent: {agent!r}.")

    evidence_json = json.dumps(
        [
            {
                "id": item.id,
                "source": item.source,
                "reference": item.reference,
                "description": item.description,
            }
            for item in evidence
        ],
        ensure_ascii=False,
    )
    user_prompt = build_user_prompt(agent, evidence_json)
    generate = text_generator or _generate_text
    response_text = generate(SYSTEM_PROMPTS[agent], user_prompt)
    report = _parse_report(response_text, agent)
    validate_report(report, evidence)
    return report
