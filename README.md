# llama_herd

Local LLM world-model drift research workflow for a Jetson Orin Nano Super Dev Kit.

This is a terminal-only experiment runner, not a search-quality benchmark. It studies how one initially blank first-person local LLM profile changes over repeated web-search episodes: topic drift, source preference, repetition, self-rule formation, hallucination, and stability of research behaviour.

The project assumes llama.cpp server and SearXNG are already running locally. It can optionally restart llama.cpp after transient failures if restart commands are configured in `.env`.

## Local Assumptions

- llama.cpp OpenAI-compatible API: `http://127.0.0.1:10000/v1`
- SearXNG search endpoint: `http://127.0.0.1:8888/search`
- Search episodes rely on SearXNG results only by default; no direct Wikipedia API is required.
- One local small instruct model served through llama.cpp
- Reasoning treated as off; prompts require short structured JSON and never request chain-of-thought
- Runs are sequential for Jetson memory friendliness

## Setup

```bash
python3 -m venv --prompt llama_herd .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Check the local model server:

```bash
curl http://127.0.0.1:10000/v1/models
```

Run the infrastructure check:

```bash
python scripts/check_infrastructure.py
```

## Initialize

```bash
python scripts/init_experiment.py
```

Use `--reset` to clear experiment data and recreate the blank profile:

```bash
python scripts/init_experiment.py --reset
```

## Run

Run one search episode:

```bash
python scripts/run_episode.py --agent agent_01
```

Run the default one-search test for the profile:

```bash
python scripts/run_batch.py --agent agent_01
```

Run a longer drift check later by setting the episode count explicitly:

```bash
python scripts/run_batch.py --agent agent_01 --episodes 10
```

Run text mode:

```bash
python scripts/run_text_mode.py
```

Analyze drift:

```bash
python scripts/analyze_drift.py
```

## Data Outputs

- Profile JSON: `data/profiles/agent_01.json`
- Full episode JSONL: `data/logs/episodes.jsonl`
- Per-agent JSONL: `data/logs/agent_01.jsonl`
- Source text and snippets: `data/sources/`
- Raw search cache: `data/sources/search_cache/`
- SQLite database: `data/db/world_model_lab.sqlite`

Everything written to SQLite should also be recoverable from JSONL. Each episode stores raw model request/response records plus prompt metadata, including prompt version and prompt file hash, so prompt changes can be monitored across runs.

## Failure Behaviour

LLM requests are retried with backoff. If `LLM_RESTART_COMMAND` is configured, the runner also restarts llama.cpp once and retries before giving up. Search failures, empty SearXNG results, source fetch failures, malformed model JSON, and weak snippets are logged as valid experimental data instead of being hidden.

No paid APIs, dashboards, Streamlit, LangChain, LlamaIndex, Celery, FastAPI, or heavy ML/NLP dependencies are used.
