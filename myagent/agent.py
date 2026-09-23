"""The Pydantic AI agent exposed by the server."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from pydantic_ai import Agent

DEFAULT_MODEL = os.getenv("MYAGENT_MODEL", "openai:gpt-4o-mini")
MODEL_ID = os.getenv("MYAGENT_MODEL_ID", "myagent")

agent = Agent(
    DEFAULT_MODEL,
    name=MODEL_ID,
    instructions=(
        "You are myagent, a concise and helpful assistant. "
        "Use the available tools when they give you a better answer than guessing."
    ),
    defer_model_check=True,
)


@agent.tool_plain
def current_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
