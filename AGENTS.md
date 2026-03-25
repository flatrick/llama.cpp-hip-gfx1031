# Project Agent Rules

This repository is worked on through OpenCode with a local llama.cpp server.

## General behavior

- Prefer small, targeted changes over broad rewrites.
- Read nearby files before editing.
- Preserve existing style, naming, and structure unless explicitly asked to change them.
- State the plan before making multi-file changes.
- When uncertain, ask for inspection or propose alternatives instead of inventing APIs.

## Editing rules

- Do not add new dependencies unless necessary.
- Do not rename files or public symbols unless required for the task.
- Prefer patching existing code over replacing entire files.
- Keep comments concise and only add them when they clarify intent.

## Validation rules

- After making code changes, propose the exact command to validate them.
- Prefer the smallest relevant validation first.
- If tests fail, summarize the failure before proposing a fix.

## Safety rules

- Treat shell commands as sensitive.
- Do not run destructive commands unless explicitly approved.
- Never run `git commit`, `git push`, or delete files without approval.
