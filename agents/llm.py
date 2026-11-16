"""Call the configured language model and validate its structured report."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict

from agents import config
from agents.contracts import (
    REPORT_JSON_SCHEMA,
    REPLY_JSON_SCHEMA,
    AgentName,
    AgentReport,
    Evidence,
    Finding,
    DefenseReply,
    ReplyFinding,
    validate_reply,
    validate_report,
)
from agents.prompts import (
    REVISION_SYSTEM_PROMPTS,
    SYSTEM_PROMPTS,
    DEFENSE_REPLY_SYSTEM_PROMPT,
    build_defense_reply_prompt,
    build_revision_prompt,
    build_user_prompt,
)

# Tests can supply a fake function with the same two text inputs as the API.
# This lets us test the flow without sending requests or spending API credit.
TextGenerator = Callable[[str, str], str]


def _generate_text(
    system_prompt: str, user_prompt: str, *, json_schema: dict | None = None
) -> str:
    """Send prompts to the configured provider and return its text response."""
    # Independent reports keep their existing schema. The debate reply supplies
    # its own schema because it must identify the arguments being answered.
    schema = REPORT_JSON_SCHEMA if json_schema is None else json_schema
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
        # Send the instructions, case text and required answer format together.
        # The schema controls the format, not whether the claims are correct.
        response = client.messages.create(
            model=config.MODEL,
            max_tokens=config.ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
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
    # The API returns text. Turn it into Python data before checking its fields.
    # Reject a response for the wrong role: defense must not become prosecution.
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
    """Draft, revise once against the same evidence, and validate both reports."""
    if agent not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported agent: {agent!r}.")

    # Step 1: turn the supplied facts into text that the model can read.
    # Both the first draft and its revision receive this same evidence text.
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
    # Step 2: use the fake function in tests, or the real API in a live review.
    generate = text_generator or _generate_text
    response_text = generate(SYSTEM_PROMPTS[agent], user_prompt)
    # Step 3: check the draft before asking the model to edit it.
    # A broken draft stops here; it is not silently accepted.
    report = _parse_report(response_text, agent)
    validate_report(report, evidence)

# Step 4: ask the same model to edit its own draft ONCE.
    # It checks wording and references using the original facts.
    # It never sees the other agent's report. This is not a debate.
    # Editing can still make mistakes, so the revised report needs checking too.
    revision_prompt = build_revision_prompt(
        agent, json.dumps(asdict(report), ensure_ascii=False), evidence_json
    )
    revised_text = generate(REVISION_SYSTEM_PROMPTS[agent], revision_prompt)
    # Step 5: check and return the revised version, not the original draft.
    # If this check fails, report the error instead of hiding it.
    revised_report = _parse_report(revised_text, agent)
    validate_report(revised_report, evidence)
    return revised_report


def generate_defense_reply(
    evidence: Sequence[Evidence],
    prosecution: AgentReport,
    defense: AgentReport,
    *,
    text_generator: TextGenerator | None = None,
) -> DefenseReply:
    """Make one reply call, with no editing call, retry loop or further round."""
    # Stop before spending a call if either initial report is invalid.
    if prosecution.agent != "prosecution" or defense.agent != "defense":
        raise ValueError("debate requires prosecution and defense reports in that order.")
    validate_report(prosecution, evidence)
    validate_report(defense, evidence)
    evidence_json = json.dumps([asdict(item) for item in evidence], ensure_ascii=False)
    prompt = build_defense_reply_prompt(prosecution, defense, evidence_json)
    # Keep the two-argument fake-generator interface used by offline tests.
    if text_generator is not None:
        response = text_generator(DEFENSE_REPLY_SYSTEM_PROMPT, prompt)
    else:
        response = _generate_text(
            DEFENSE_REPLY_SYSTEM_PROMPT, prompt, json_schema=REPLY_JSON_SCHEMA
        )
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("The defense reply was not valid JSON.") from error
    if not isinstance(payload, dict) or set(payload) != {"agent", "replies"}:
        raise ValueError("The defense reply must contain agent and replies.")
    if payload["agent"] != "defense" or not isinstance(payload["replies"], list):
        raise ValueError("The defense reply requires agent='defense' and a replies list.")
    replies = []
    for item in payload["replies"]:
        if not isinstance(item, dict) or set(item) != {
            "prosecution_finding", "position", "claim", "evidence_ids"
        }:
            raise ValueError("Each reply needs prosecution_finding, position, claim and evidence_ids.")
        if not isinstance(item["claim"], str):
            raise ValueError("A reply claim must be text.")
        if not isinstance(item["evidence_ids"], list) or not all(
            isinstance(value, str) for value in item["evidence_ids"]
        ):
            raise ValueError("Reply evidence_ids must be a list of strings.")
        replies.append(ReplyFinding(**item))
    reply = DefenseReply(agent="defense", replies=replies)
    # This validates structure and references. Human review still checks meaning.
    validate_reply(reply, prosecution, evidence)
    return reply
