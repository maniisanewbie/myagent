from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from main import agent, app


@pytest.fixture
def client():
    with agent.override(model=TestModel(custom_output_text="hello from myagent")):
        yield TestClient(app)


def test_list_models(client: TestClient):
    body = client.get("/v1/models").json()
    assert body["data"][0]["object"] == "model"


def test_chat_completion(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "myagent",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "what can you do?"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from myagent"
    assert body["usage"]["total_tokens"] > 0


def test_chat_completion_streaming(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "myagent",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks) == (
        "hello from myagent"
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_last_message_must_be_user(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "myagent", "messages": [{"role": "assistant", "content": "hello"}]},
    )
    assert response.status_code == 400
