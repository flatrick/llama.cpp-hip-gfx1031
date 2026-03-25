## 1. Rewrite openspec/config.yaml

- [x] 1.1 Add repo purpose statement (gfx1031-on-Arch survival guide, not personal dotfiles)
- [x] 1.2 Add gfx1031/gfx1030 relationship section with date-stamped rocBLAS Tensile gap note
- [x] 1.3 Fix HSA_OVERRIDE constraint (remove incorrect "must be set for UD-Q4_K_XL on native")
- [x] 1.4 Add current launcher status table (Q4_K_M native ✅, UD native ❌, docker-llamacpp ❌, docker-rocm 🔄)
- [x] 1.5 Add breakage taxonomy (kernel updates, ROCm packages, llama.cpp upstream, rocBLAS, HF model changes)
- [x] 1.6 Add AI decision logic section (if f16 tensors → docker-rocm; VRAM budget formula; etc.)
- [x] 1.7 Date-stamp all version-sensitive constraints with [YYYY-MM] prefix
- [x] 1.8 Add `specs` rules (infra repo — prefer scenario-based verification over unit test language)
- [x] 1.9 Add `design` rules (only create design.md for multi-script changes or new Docker patterns)

## 2. Verify

- [x] 2.1 Run `openspec instructions proposal --change <any-new-change> --json` and confirm the context block reflects the updated content
