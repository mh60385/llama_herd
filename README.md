# llama_herd

Local LLM world-model drift research workflow for a Jetson Orin Nano Super Dev Kit.

This is a terminal-only experiment runner, not a search-quality benchmark. It studies how one initially blank first-person local LLM profile changes over repeated web-search episodes: topic drift, source preference, repetition, self-rule formation, hallucination, and stability of research behaviour.

The project assumes llama.cpp server and SearXNG are already running locally. It can optionally restart llama.cpp after transient failures if restart commands are configured in `.env`.

## Local Assumptions

- llama.cpp OpenAI-compatible API: `http://127.0.0.1:10000/v1`
- SearXNG search endpoint: `http://127.0.0.1:8888/search`
- Initial profile seeding defaults to a broad deterministic public-world profile.
- Search episodes rely on SearXNG results only by default.
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

Initialization stores the profile seed and generated initial profile in `data/profiles/<agent_id>.json`. The same seed selects the same starting interests.

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

Run the full configured study sequentially:

```bash
python scripts/init_experiment.py --reset
python scripts/run_batch.py --all-agents
```

With the default config this runs 5 agents for 100 episodes each.

Resume a partially completed configured study:

```bash
python scripts/run_batch.py --all-agents --resume
```

Write a readable latest-status file:

```bash
python scripts/write_latest_status.py
```

Open:

```text
data/logs/latest_status.md
```

Write research metrics for a run:

```bash
python scripts/write_research_metrics.py
```

```text
data/metrics/research_metrics.md
data/metrics/research_metrics.json
```

Screen candidate models before the full study:

```bash
python scripts/model_screen.py \
  --models qwen2.5-1.5b-q4 llama-3.2-1b-q4
```

This writes `data/model_screen/summary.md` with a shortlist based on load, JSON, seed, topic, and prompt behavior.
The model-screening rationale and paper-oriented summary are documented in
`PIPELINE_SUMMARY.md`.

Run a short model comparison for shortlisted models:

```bash
python scripts/model_comparison.py \
  --models qwen2.5-1.5b-q4 llama-3.2-1b-q4 \
  --agents 3 \
  --episodes 10
```

This temporarily moves the current `data/` directory aside, runs isolated mini-studies per model,
restores the original `data/`, and writes `data/model_comparison/summary.md`.

Analyze drift:

```bash
python scripts/analyze_drift.py
```

Compute article-grade metrics offline after a run:

```bash
python -m llama_herd.metrics \
  --episodes data/logs/episodes.jsonl \
  --profiles data/profiles \
  --out data/metrics \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu
```

## Data Outputs

- Profile JSON: `data/profiles/agent_01.json`
- Full episode JSONL: `data/logs/episodes.jsonl`
- Per-agent JSONL: `data/logs/agent_01.jsonl`
- Source text and snippets: `data/sources/`
- Raw search cache: `data/sources/search_cache/`
- Offline metrics: `data/metrics/`

Each episode stores raw model request/response records plus prompt metadata, including prompt version and prompt file hash, so prompt changes can be monitored across runs.

## Failure Behaviour

LLM requests are retried with backoff. If `LLM_RESTART_COMMAND` is configured, the runner also restarts llama.cpp once and retries before giving up. Search failures, empty SearXNG results, source fetch failures, malformed model JSON, and weak snippets are logged as valid experimental data instead of being hidden.

No paid APIs, dashboards, Streamlit, LangChain, LlamaIndex, Celery, or FastAPI are used. Sentence Transformers is only used in the offline metrics step, not during live episode generation.
