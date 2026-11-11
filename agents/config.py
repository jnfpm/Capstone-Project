"""Central configuration for the agent.

Set PROVIDER to ``google`` or ``anthropic`` in the project-root .env file.
MODEL uses the provider-specific default below and can be overridden there.
ANTHROPIC_MAX_TOKENS sets the Anthropic response budget (positive integer, default 8192).
Store provider credentials in .env; never commit that file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROVIDER = os.getenv("PROVIDER", "google").strip().lower()
if PROVIDER not in ("google", "anthropic"):
    raise ValueError(f"PROVIDER must be 'google' or 'anthropic', not {PROVIDER!r}.")

KEY_NAME = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}[PROVIDER]
API_KEY = os.getenv(KEY_NAME, "")

MODELS = {
    "google": "gemini-flash-lite-latest",
    "anthropic": "claude-haiku-5-5",
}
MODEL = os.getenv("MODEL", MODELS[PROVIDER]).strip()

# Long evidence IDs also consume response tokens. The first live review was cut
# off at 1200; a later two-stage evaluation also hit 4096. Allow more room
# without hard-coding the budget in the API call or requesting longer reports.
# This is an upper limit, not a guarantee that every report will fit.
try:
    ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))
except ValueError as error:
    raise ValueError("ANTHROPIC_MAX_TOKENS must be a positive integer.") from error
if ANTHROPIC_MAX_TOKENS <= 0:
    raise ValueError("ANTHROPIC_MAX_TOKENS must be a positive integer.")
