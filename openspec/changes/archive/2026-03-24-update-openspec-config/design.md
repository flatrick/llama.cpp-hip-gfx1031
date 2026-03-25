## Context

`openspec/config.yaml` is the AI context file read by OpenSpec when generating artifacts.
Its `context` block is injected verbatim into every artifact prompt; its `rules` block
constrains artifact shape per type. Currently it has one inaccurate constraint, is written
as human narrative rather than AI-optimised context, and is missing several sections.

## Goals / Non-Goals

**Goals:**
- Correct the HSA_OVERRIDE inaccuracy
- Establish the repo's purpose for AI readers (gfx1031 survival guide, not personal dotfiles)
- Add structured sections: current launcher status, breakage taxonomy, decision logic
- Add missing rule types: `specs`, `design`
- Date-stamp volatile constraints so stale info is visible

**Non-Goals:**
- Changing any launcher scripts or Dockerfile
- Updating the README (separate change: `improve-readme`)
- Supporting GPUs other than gfx1031

## Decisions

### Decision 1: AI-optimised prose over human-readable narrative

The `context` block is read by an AI, not a human. Human-friendly prose wastes tokens
and buries the signal. Replacing paragraph narrative with structured, dense sections
(tables, bullet conditionals) makes the context more useful per token.

### Decision 2: Date-stamp volatile constraints inline

Version-sensitive facts (build regressions, Tensile library gaps) get a `[YYYY-MM]` prefix
rather than a separate "last updated" field. This way stale entries are visible at the line
level without requiring a separate staleness audit.

### Decision 3: Add a "current state" launcher status table

An AI proposing changes needs to know which launchers are broken today, or it will generate
tasks for broken paths. A compact table (script → status → reason) is the most efficient
representation.

### Decision 4: Add conditional decision logic

Flat facts ("rocBLAS missing gfx1031") are less useful to an AI than conditional rules
("if model has f16 tensors → native broken → recommend docker-rocm"). Adding an explicit
decision logic section teaches the AI how to reason about the constraints, not just what they are.

## Risks / Trade-offs

- **Stale date-stamps** → Someone reads `[2026-03]` months later and doesn't know if it's
  still true. Mitigation: the date makes the staleness explicit and prompts verification,
  which is better than undated facts that feel always-current.
- **Over-constraining AI suggestions** → Too much context can anchor an AI to known solutions
  and prevent it suggesting better alternatives. Mitigation: keep decision logic as "prefer X"
  not "only ever X".
