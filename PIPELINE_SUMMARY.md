# llama_herd Pipeline Summary

This project runs local LLM “research episodes” to study how an agent profile changes over time. The goal is to observe model self-governance, topic drift, source preference, repetition, and prompt-induced attractors while keeping the process auditable.

## Current Episode Flow

1. Load the agent profile from `data/profiles/<agent_id>.json`.
2. Build a search-query prompt from:
   - seeded initial profile
   - stable interests only
   - recent memory summary
3. Ask the local LLM for one web search query.
4. Search through SearXNG.
5. Ask the model to select one source.
6. Fetch/read the source when possible.
7. Ask the model to summarize the source.
8. Ask the model for a diary entry and reflection.
9. Extract weak observations from the episode.
10. Apply deterministic recurrence rules before anything becomes stable profile state.
11. Save JSONL and SQLite episode records.

## Profile State Model

The profile separates weak observations from stable identity-like state:

- `initial_profile`: starting profile generated from a seed source.
- `observations`: what the agent saw in one episode.
- `tentative_interests`: weak repeated signals.
- `stable_interests`: only promoted after recurrence checks.
- `recent_queries`: logged for auditing and anti-anchoring, but no longer used directly in search-query prompt conditioning.

One search result must not directly become a stable interest, preferred source, or self-rule.

## Recurrence Rule

Candidate interests are promoted only when:

- they appear in 3+ episodes,
- they appear across 2+ source domains,
- the relevant episode outputs passed JSON validation,
- they were not caused by near-identical repeated queries.

Malformed `source_summary` or profile/observation JSON blocks stable profile promotion. The raw observation can still be logged if safe.

## Prompt Leakage Fix

An issue was found where tentative interests and recent queries leaked into the next search query. Example: a weak “ethical AI” tentative topic kept appearing in later searches.

Fix:

- `search_query_prompt()` now excludes `tentative_interests`.
- `search_query_prompt()` now excludes `recent_queries`.
- Tentative interests remain visible to source selection, diary/reflection, and observation extraction.

Five-agent bait smoke after the fix:

```text
valid_json_count: 5/5
seed_following_count: 5/5
tentative_leak_count: 0
recent_query_echo_count: 0
passed: true
```

## System Guardrails

Long batch runs are intercepted before every episode by a system monitor.

It logs to:

```text
data/logs/system_monitor.jsonl
```

It checks:

- available RAM,
- swap used,
- `llama-server` Docker memory,
- Jetson thermal zones,
- max temperature,
- episode index.

Current Jetson Orin Nano Super Dev Kit defaults:

```yaml
memory_warn_available_mb: 1200
memory_restart_available_mb: 800
swap_warn_used_mb: 3200
swap_stop_used_mb: 3700
temperature_warn_c: 75.0
temperature_pause_c: 78.0
temperature_stop_c: 83.0
thermal_cooldown_seconds: 90.0
llm_restart_every_episodes: 0
llm_restart_if_memory_trending_down: true
llm_memory_trend_window: 4
llm_min_restart_gap_episodes: 3
```

If RAM is critically low, the runner restarts `llama-server`. It no longer restarts on every mild warning, because constant restarts made the Jetson run noisier without improving the experiment. If swap gets close to exhaustion, the run stops instead of thrashing. If temperature is too high, it pauses or stops safely.

## Model Smoke Results

Models were downloaded into:

```text
/home/deadbod/jetson-admin/local-llm/models
```

Prompt-adherence smoke used:

- context: `2048`
- max output: `192`
- strict JSON prompts
- search-query prompt
- source-summary prompt
- observation-extraction prompt

Results:

```text
qwen2.5-0.5b-q4       1/4   ~764 MiB   too weak
llama-3.2-1b-q4       3/4   ~1.89 GiB  almost usable
gemma-3-1b-q4         4/4   ~2.03 GiB  best small alternate
qwen2.5-1.5b-q4       4/4   ~2.29 GiB  current baseline
granite-3.3-2b-q4     4/4   ~2.39 GiB  good comparison model
llama-3.2-3b-iq4-xs   4/4   ~3.19 GiB  heavier quality comparison
ministral-3b-q4       0/4   ~2.58 GiB  not suitable
```

Current default model:

```text
bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
```

## Seeding Findings

Several seeding approaches were tested:

1. Hand-authored interest pool:
   - reproducible,
   - but too artificial and repetitive.

2. Pure model-generated seed profile:
   - technically model-generated,
   - but Qwen collapsed into repeated AI/tech/sustainability/generic themes.

3. Loose public-web topic prompt:
   - removed AI/tech collapse,
   - but still repeated topic bundles like origami/bicycle/quantum/rainforest.

4. Random Wikipedia article seeding:
   - best result so far.
   - uses random public-web material as seed evidence.
   - model abstracts interests from article titles/extracts.

Wikipedia seed smoke:

```text
total_agents: 5
valid_json_count: 4/5
unique_interest_count: 15
rows_with_ai/tech/governance_cluster: 0
```

Example generated interests:

```text
documentary filmmaking
hair straightening vs. natural hair styles
self-expression and cultural identity
Mercosur economic bloc
Freestyle skiing World Cup
PAD community development
Radio Stations
Music
Pop Culture
Horror films
Telugu cinema
Hollywood films
```

## Recommended Seeding Direction

Use Wikipedia-seeded profiles. This is now the default initialization path:

```text
seed -> deterministic Wikipedia page summaries -> model abstracts broad interests -> validate -> store source pages
```

Current implementation:

1. Select 3 deterministic Wikipedia article summaries from the seed.
2. Reject bad pages:
   - disambiguation,
   - list/index pages,
   - very short extracts,
   - surname/given-name pages.
3. Ask the LLM to generate an initial profile from the pages.
4. Validate:
   - valid JSON,
   - 3 interests,
   - interests are not exact article titles,
   - no biography/identity claims,
   - no strict self-rules,
   - retry once if invalid.
5. Store:
   - `profile_seed`,
   - `wiki_seed_pages`,
   - generated `initial_profile`,
   - raw seed-generation response.

This keeps initial themes model-generated while avoiding the model’s default AI/tech attractor.

## Offline Metrics

After a run, compute metrics offline so Sentence Transformers does not compete with llama.cpp for memory:

```bash
python -m llama_herd.metrics \
  --episodes data/logs/episodes.jsonl \
  --profiles data/profiles \
  --out data/metrics \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu
```

Metrics outputs:

```text
data/metrics/initial_seed_spread.csv
data/metrics/query_drift.csv
data/metrics/source_path_dependence.csv
data/metrics/memory_promotion_counts.csv
data/metrics/agent_divergence.csv
data/metrics/metrics_summary.json
```

## Key Logs

Useful logs:

```text
data/logs/episodes.jsonl
data/logs/agent_01.jsonl
data/logs/system_monitor.jsonl
data/logs/model_smoke_results.json
data/logs/seed_smoke_5_agents.jsonl
data/logs/loose_seed_topic_smoke.jsonl
data/logs/wiki_seed_smoke.jsonl
```

## Current Interpretation

Prompt leakage is a bug and has been fixed.

Model tendency to cluster around AI/ethics/governance under “research agent” framing is experimental signal, but it is not desirable for initial seeding if all agents start in the same attractor basin.

Wikipedia seeding appears to be the best compromise:

- not hand-authored,
- diverse,
- auditable,
- public-web grounded,
- still model-interpreted.
