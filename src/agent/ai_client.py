"""Thin wrapper around OpenAI and Anthropic so the provider is swappable via env vars.

Set AI_PROVIDER=openai (default) or anthropic.
Set AI_MODEL to override the default model for the chosen provider.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_openai_client = None
_anthropic_client = None

_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}


def _provider() -> str:
    return os.getenv("AI_PROVIDER", "openai").lower()


def _model() -> str:
    return os.getenv("AI_MODEL", _DEFAULTS.get(_provider(), "gpt-4o-mini"))


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def chat_json(system: str, user: str) -> dict | list:
    """Call the AI and return a parsed JSON object or array."""
    if _provider() == "anthropic":
        return _anthropic_json(system, user)
    return _openai_json(system, user)


def chat_text(system: str, user: str) -> str:
    """Call the AI and return a plain-text response."""
    if _provider() == "anthropic":
        return _anthropic_text(system, user)
    return _openai_text(system, user)


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _openai_json(system: str, user: str) -> dict | list:
    response = _get_openai().chat.completions.create(
        model=_model(),
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def _openai_text(system: str, user: str) -> str:
    response = _get_openai().chat.completions.create(
        model=_model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _anthropic_json(system: str, user: str) -> dict | list:
    response = _get_anthropic().messages.create(
        model=_model(),
        max_tokens=4096,
        system=system + "\n\nReturn ONLY valid JSON — no markdown fences, no commentary.",
        messages=[{"role": "user", "content": user}],
        temperature=0.1,
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _anthropic_text(system: str, user: str) -> str:
    response = _get_anthropic().messages.create(
        model=_model(),
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.3,
    )
    return response.content[0].text.strip()
