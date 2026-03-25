## Why

The `openspec/config.yaml` contains inaccurate constraints (incorrect HSA_OVERRIDE guidance),
lacks the repo's actual purpose as a gfx1031-on-Arch survival guide, and is missing the
context an AI needs to make good suggestions — specifically the breakage taxonomy, current
working state per launcher, and the gfx1031/gfx1030 relationship.

## What Changes

- Correct the HSA_OVERRIDE constraint (currently states it must be set for UD-Q4_K_XL native — incorrect; UD native is broken regardless, Docker is the fix)
- Add repo purpose statement: living compatibility guide for gfx1031 on Arch/EndeavourOS
- Add the gfx1031 ≈ gfx1030 background (why workarounds exist)
- Add current launcher status table (what works today vs what is broken)
- Date-stamp version-sensitive constraints so they don't silently mislead after upgrades
- Add breakage taxonomy (what categories of things break and why)
- Add `specs` and `design` rules sections (currently missing)
- Add decision logic for AI: conditional rules like "if model has f16 tensors → native won't work"
- Tighten prose to be AI-context-optimised (dense, precise) rather than README-style narrative

## Capabilities

### New Capabilities

- `openspec-config`: Accurate, AI-optimised context and rules for the openspec/config.yaml

### Modified Capabilities

<!-- None — this is the first meaningful version of the config -->

## Impact

- `openspec/config.yaml`: rewritten
- All future OpenSpec changes will benefit from the corrected context
