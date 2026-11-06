"""Central configuration for the agent.

Set PROVIDER to ``google`` or ``anthropic`` in the project-root .env file.
MODEL defaults to the provider's recommended alias and can be overridden there.
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
