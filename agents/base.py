from __future__ import annotations

import json
import logging
from typing import TypeVar

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentRunError(RuntimeError):
    """Raised when an agent fails to produce output_model-valid JSON after all retries."""


async def run_structured_agent(
    system_prompt: str,
    user_prompt: str,
    output_model: type[T],
    *,
    mcp_servers: dict | None = None,
    allowed_tools: list[str] | None = None,
    max_retries: int = 2,
) -> T:
    """Runs one Claude Agent SDK query and returns a validated `output_model` instance.

    Deliberately generic — no knowledge of any specific agent's domain models — so every
    future agent (Researcher, Technical Analyst, Quant, Risk, Trader) reuses this exact
    function. `mcp_servers`/`allowed_tools` are already threaded through even though the
    Scanner (Phase 1) doesn't use custom tools, so later agents that do need `@tool`-
    registered functions don't require any change here.

    On invalid/unparseable output, retries up to `max_retries` times, feeding the
    validation error back into the prompt so Claude can self-correct.
    """
    attempt_prompt = user_prompt
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            output_format={"type": "json_schema", "schema": output_model.model_json_schema()},
            mcp_servers=mcp_servers or {},
            allowed_tools=allowed_tools or [],
        )

        raw_text = ""
        async for message in query(prompt=attempt_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        raw_text += block.text

        try:
            parsed = json.loads(raw_text)
            return output_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Agent output invalid (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc
            )
            attempt_prompt = (
                f"{user_prompt}\n\nYour previous response was invalid: {exc}\n"
                "Return ONLY valid JSON that matches the required schema, with no other text."
            )

    raise AgentRunError(f"Agent failed after {max_retries + 1} attempts: {last_error}") from last_error
