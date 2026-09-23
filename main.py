"""A Pydantic AI agent behind an OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

MODEL_ID = os.getenv("MYAGENT_MODEL", "gpt-4o-mini")

model = OpenAIChatModel(
    MODEL_ID,
    provider=OpenAIProvider(
        base_url=os.getenv("MYAGENT_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("MYAGENT_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
    ),
)
agent = Agent(model, instructions="You are myagent, a concise and helpful assistant.")


@agent.tool_plain
def current_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


app = FastAPI(title="myagent")


def to_agent_input(messages: list[ChatMessage]) -> tuple[str, list[ModelMessage], str | None]:
    """Split OpenAI-style messages into a prompt, prior history and extra instructions."""
    instructions = [m.content for m in messages if m.role in ("system", "developer") and m.content]
    conversation = [m for m in messages if m.role in ("user", "assistant")]
    if not conversation or conversation[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must have role 'user'.")

    history: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content=m.content)])
        if m.role == "user"
        else ModelResponse([TextPart(content=m.content)])
        for m in conversation[:-1]
    ]
    return conversation[-1].content, history, "\n\n".join(instructions) or None


async def stream_chunks(
    model_name: str, prompt: str, history: list[ModelMessage], instructions: str | None
) -> AsyncIterator[str]:
    completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    async with agent.run_stream(prompt, message_history=history, instructions=instructions) as run:
        async for text in run.stream_text(delta=True):
            yield chunk({"content": text})
    yield chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    prompt, history, instructions = to_agent_input(body.messages)

    if body.stream:
        return StreamingResponse(
            stream_chunks(body.model, prompt, history, instructions),
            media_type="text/event-stream",
        )

    result = await agent.run(prompt, message_history=history, instructions=instructions)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "myagent"}]}


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
