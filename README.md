# myagent

just learning agentic frameworks

A single-file [Pydantic AI](https://ai.pydantic.dev) agent served behind an OpenAI-compatible
REST API, so any ChatGPT API client (the `openai` SDK, `curl`, LangChain, Open WebUI, ...)
can talk to it by pointing its base URL at this server. Everything lives in `main.py`.

## Endpoints

| Method | Path                   | Notes                                               |
| ------ | ---------------------- | --------------------------------------------------- |
| POST   | `/v1/chat/completions` | Chat completions, with `"stream": true` SSE support |
| GET    | `/v1/models`           | Lists the single agent model                        |

## Run it

```bash
pip install -e ".[dev]"
export MYAGENT_API_KEY=...                         # or OPENAI_API_KEY
export MYAGENT_MODEL=gpt-4o-mini                   # any model the host serves
export MYAGENT_BASE_URL=https://api.openai.com/v1  # e.g. https://api.together.ai/v1, Ollama, vLLM
python main.py                                     # serves on :8000 (HOST/PORT to override)
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
completion = client.chat.completions.create(
    model="myagent",
    messages=[{"role": "user", "content": "what time is it?"}],
)
print(completion.choices[0].message.content)
```

`system`/`developer` messages become agent instructions, earlier `user`/`assistant`
messages become the run's message history, and the trailing `user` message is the prompt.
The agent ships with one `current_time` tool; add more with `@agent.tool_plain`.

## Tests

```bash
pytest
```
