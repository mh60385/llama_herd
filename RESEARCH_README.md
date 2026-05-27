# llama_herd Research README

## Study Purpose

`llama_herd` is a local LLM drift experiment for repeated web-search episodes.
It is not a search-quality benchmark and not a general model-intelligence
benchmark. The study asks whether a small local instruct model can sustain
structured, auditable research behaviour over time.

The main behaviours under observation are:

- topic drift
- repeated or looping interests
- source preference and source repetition
- uncertainty handling
- prompt-induced attractors
- hallucinated or over-stabilized self-beliefs
- whether weak observations become stable interests only after recurrence

## Primary Model

The primary model-prompt pair is:

```text
qwen2.5-1.5b-q4
```

This model was selected after a fixed model-screening battery. The screen tested
structured JSON compliance, seeded search-query behaviour, resistance to
tentative-interest leakage, resistance to recent-query echoing, and loose-topic
generation.

The screen evaluates suitability for this pipeline, not general model quality.
Full screening notes are in `PIPELINE_SUMMARY.md`.

## Agent Seeding

Agents are initialized from fixed seeds in `configs/agents.yaml`.

During initialization, each seed is hashed and used to deterministically sample
3 broad public-world starting interests. These become the agent's initial stable
interests.

This means starting topics are:

- reproducible
- broad
- not generated from live search
- not produced by the model during initialization

After initialization, topic development comes from repeated search episodes.
New interests are promoted only after recurrence checks.

## Episode Flow

Each episode:

1. Loads the agent profile.
2. Prompts the model for one search query from stable interests.
3. Searches through SearXNG.
4. Asks the model to select one source.
5. Fetches and summarizes the source.
6. Writes diary and reflection JSON.
7. Extracts weak observations.
8. Applies deterministic recurrence rules.
9. Saves episode logs and raw model outputs.

One source or one episode cannot directly become stable profile state.

## Production Run

Use the repository virtual environment:

```bash
./.venv/bin/python scripts/init_experiment.py --reset
./.venv/bin/python scripts/run_batch.py --all-agents --resume
```

Default run size:

```text
5 agents x 100 episodes = 500 episodes
```

## Safety And Recovery

Long runs are guarded by `SystemMonitor`.

Before every episode it checks:

- available RAM
- swap use
- `llama-server` Docker memory
- Jetson thermal zones
- max temperature

If memory is low or trending down, the runner can restart `llama-server`.
If temperature is high, it pauses for cooldown. If temperature or swap crosses
the stop threshold, the run stops instead of thrashing.

LLM requests also retry with backoff. If configured in `.env`, the client
restarts the Docker server after repeated request failures and retries once.

## Outputs

Important run outputs:

```text
data/profiles/
data/logs/episodes.jsonl
data/logs/<agent_id>.jsonl
data/logs/system_monitor.jsonl
data/sources/
data/metrics/
```

Generated data outputs are ignored by Git by default.

## Metrics

Lightweight run metrics:

```bash
./.venv/bin/python scripts/write_research_metrics.py
```

Drift summary:

```bash
./.venv/bin/python scripts/analyze_drift.py
```

Embedding-heavy article metrics should be computed offline after the run so
they do not compete with llama.cpp for memory.

## Paper Framing

Recommended wording:

```text
We selected the most suitable local model-prompt pair for the experimental
pipeline, rather than the best model in general.
```

The experiment measures structured behaviour and profile evolution under a
controlled local pipeline. It should be reported as a longitudinal drift and
auditing study, not as an external search benchmark.
