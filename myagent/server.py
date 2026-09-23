"""OpenAI-compatible HTTP server in front of the Pydantic AI agent."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from myagent.agent import MODEL_ID, agent
from myagent.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    Model,
    ModelList,
    Usage,
)

app = FastAPI(title="myagent", description="Pydantic AI agent with an OpenAI-compatible API")


def to_agent_input(
    messages: Sequence[ChatMessage],
) -> tuple[str, list[ModelMessage], str | None]:
    """Split OpenAI-style messages into a prompt, prior history and extra instructions."""
    instructions = [m.content for m in messages if m.role in ("system", "developer") and m.content]
    conversation = [m for m in messages if m.role in ("user", "assistant")]

    if not conversation or conversation[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must have role 'user'.")

    prompt = conversation[-1].content or ""
    history: list[ModelMessage] = []
    for message in conversation[:-1]:
        if message.role == "user":
            history.append(ModelRequest([UserPromptPart(content=message.content or "")]))
        else:
            history.append(ModelResponse([TextPart(content=message.content or "")]))

    return prompt, history, "\n\n".join(instructions) or None


async def stream_completion(
    request: ChatCompletionRequest,
    prompt: str,
    history: list[ModelMessage],
    instructions: str | None,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: ChoiceDelta, finish_reason: str | None = None) -> str:
        payload = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChunkChoice(delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    yield chunk(ChoiceDelta(role="assistant", content=""))
    async with agent.run_stream(
        prompt, message_history=history, instructions=instructions
    ) as result:
        async for text in result.stream_text(delta=True):
            yield chunk(ChoiceDelta(content=text))
    yield chunk(ChoiceDelta(), finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[Model(id=MODEL_ID)])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt, history, instructions = to_agent_input(request.messages)

    if request.stream:
        return StreamingResponse(
            stream_completion(request, prompt, history, instructions),
            media_type="text/event-stream",
        )

    try:
        result = await agent.run(prompt, message_history=history, instructions=instructions)
    except ModelHTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (UnexpectedModelBehavior, UserError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    usage = result.usage
    return ChatCompletion(
        model=request.model,
        choices=[Choice(message=ChatMessage(role="assistant", content=result.output))],
        usage=Usage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        ),
    )
