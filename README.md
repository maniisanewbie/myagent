# myagent

just learning agentic frameworks

A minimal [Pydantic AI](https://ai.pydantic.dev) agent served behind an OpenAI-compatible
REST API, so any ChatGPT API client (the `openai` SDK, `curl`, LangChain, Open WebUI, ...)
can talk to it by pointing its base URL at this server.

## Endpoints

| Method | Path                   | Notes                                             |
| ------ | ---------------------- | ------------------------------------------------- |
| POST   | `/v1/chat/completions` | Chat completions, with `"stream": true` SSE support |
| GET    | `/v1/models`           | Lists the single agent model                        |
| GET    | `/health`              | Liveness probe                                      |

## Run it

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...          # credentials for the underlying model
export MYAGENT_MODEL=openai:gpt-4o-mini   # any Pydantic AI model string
python -m myagent                      # serves on :8000 (HOST/PORT to override)
```

## Call it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "myagent", "messages": [{"role": "user", "content": "what time is it?"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
print(client.chat.completions.create(
    model="myagent",
    messages=[{"role": "user", "content": "what time is it?"}],
).choices[0].message.content)
```

`system`/`developer` messages become agent instructions, earlier `user`/`assistant`
messages become the run's message history, and the trailing `user` message is the prompt.

## Agent

The agent lives in `myagent/agent.py` and ships with one `current_time` tool; add more
with `@agent.tool_plain`. The HTTP layer is in `myagent/server.py`.

## Tests

```bash
pytest
```
