# AGENTS

Repository-level instructions for coding agents working in this project.

## Goals

- Use the simplest solution that satisfies the request.
- Avoid overengineering.
- Keep responses and plans concise to reduce token usage.
- Prefer direct edits over broad refactors.

## Simplicity Rules

- Implement only what is explicitly requested.
- Choose the smallest viable change set.
- Do not add new abstractions, layers, or dependencies unless required.
- Do not add optional features, speculative improvements, or extra architecture.
- Stop once acceptance criteria are met.

## Challenge Threshold

- Do not challenge the user's desired implementation unless one of these is true:
  - An obvious, widely accepted best practice is missing.
  - There is a clear risk of future failures, regressions, data loss, or security issues.
- If challenging, keep it brief and propose the smallest safe alternative.

## Token Discipline

- Keep analysis brief and action-oriented.
- Use short progress updates.
- In final responses, report only what changed, where, and how to validate.
- Avoid long background explanations unless explicitly requested.

## Python Environment (Mandatory)

- Always use the repository's `.venv` for Python commands and tooling.
- Never use system Python if `.venv` exists.
- Preferred command style:
  - `./.venv/bin/python ...`
  - `./.venv/bin/pip ...`
- If activation is needed in a shell session, use:
  - `source .venv/bin/activate`

## Verification

- Run only the minimal checks needed for the specific change.
- Prefer targeted checks over full-suite runs unless explicitly requested.
