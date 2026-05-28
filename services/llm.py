"""Anthropic Claude API wrapper for chat and dashboard generation."""

from __future__ import annotations

import os
import re

from anthropic import AsyncAnthropic

DEFAULT_CHAT_MODEL = "claude-sonnet-4-5"
DEFAULT_DASHBOARD_MODEL = "claude-sonnet-4-6"


class AnthropicClient:
    """Async wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file before calling the LLM."
            )
        self.client = AsyncAnthropic(api_key=key)

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str = DEFAULT_CHAT_MODEL,
        max_tokens: int = 1500,
    ) -> str:
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            raise RuntimeError(f"Anthropic chat request failed: {exc}") from exc

        return _extract_text(response)

    async def generate_dashboard(
        self,
        system_prompt: str,
        user_message: str,
        model: str = DEFAULT_DASHBOARD_MODEL,
        max_tokens: int = 8000,
    ) -> str:
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic dashboard generation failed: {exc}"
            ) from exc

        return _strip_markdown_fences(_extract_text(response))


def _extract_text(response) -> str:
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    if not parts:
        raise RuntimeError("Anthropic returned an empty response")
    return "\n".join(parts).strip()


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:html)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()
