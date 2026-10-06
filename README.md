# langgraph-router

Exploration of an embedding-based agent router on LangGraph. It is a spike, not a product: specialist "agents" are stand-ins (a generic chat model plus a canned system prompt). The routing catalog is a handful of example questions per department.

## What it does

1. Embed the latest user message.
2. Drop agents the user is not allowed to see (role pre-filter).
3. Score remaining agents by cosine similarity against their example questions.
4. Call every agent above `ROUTER_SCORE_THRESHOLD` (up to 3).
5. If several specialists reply, a default agent merges them. If none clear the threshold, only the default agent answers.

The only entrypoint is a local CLI. FastAPI is listed as a dependency but unused.

## Run locally

Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env   # then set API_KEY (and BASE_URL if you are not on api.openai.com)
uv sync
docker compose up -d   # Postgres (checkpointer) and MLflow
uv run python router.py
```

Startup hits the embedding API once for every canned example in `agent_registry.py`. That is expected; it is not a local-only load.

Conversation state is checkpointed in Postgres. On `quit` / `exit` the CLI prints the latest checkpoint stored for the thread.

MLflow is optional. To enable tracing, set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in `.env`.

## Config

| Variable | Purpose |
| --- | --- |
| `API_KEY` | Required. |
| `BASE_URL` | OpenAI-compatible endpoint. |
| `LLM_MODEL` | Chat model. Default / example: `gpt-5.6-luna`. |
| `EMBEDDING_MODEL` | Embedding model. |
| `ROUTER_SCORE_THRESHOLD` | Cosine cutoff to select a specialist (default `0.7`). |
| `POSTGRES_URI` | Checkpointer connection string (defaults to the compose service). |
| `MLFLOW_TRACKING_URI` | If unset, MLflow is not used. |

## Example queries

Short or one-word prompts usually miss the threshold and fall through to `default`. Paraphrases of the canned examples route:

- `Summarize the results of the employee engagement survey.` → `hr`
- `Suggest topics for next month's content calendar.` → `marketing`
- `Write a press release for our new product launch.` → `communication`

Lower `ROUTER_SCORE_THRESHOLD` if you want looser matching.

## Stand-ins (not the intended design)

- **Specialists** in `agent_registry.py`: `AgentSpec.call` is one `llm.invoke`, not a real department agent or tool graph.
- **Example questions**: they *are* the routing catalog, not eval fixtures.
- **Roles**: the CLI grants every role (`USER_ROLES` in `router.py`), so the auth filter does not hide anyone in the demo.
- **Memory**: Postgres checkpointer with a random `thread_id` per CLI run, so every run starts a new conversation.
- **Default agent**: fallback when nothing matches, and synthesizer when several specialists reply.

