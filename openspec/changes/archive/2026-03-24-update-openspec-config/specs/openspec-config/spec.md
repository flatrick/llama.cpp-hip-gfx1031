## ADDED Requirements

### Requirement: Config states repo purpose
The config SHALL identify the repo as a gfx1031-on-Arch/EndeavourOS compatibility guide,
not generic personal dotfiles, so AI readers understand the target audience and scope.

#### Scenario: AI receives purpose context
- **WHEN** an AI reads the config context
- **THEN** it SHALL know the repo targets gfx1031 (RX 6700 XT) specifically on Arch/EOS
- **AND** it SHALL know the setup breaks frequently due to rolling-release package churn

### Requirement: Config explains the gfx1031/gfx1030 relationship
The config SHALL document why gfx1031-specific workarounds exist, so AI readers understand
the root cause rather than just the symptoms.

#### Scenario: AI understands the GPU relationship
- **WHEN** an AI reads the config context
- **THEN** it SHALL know gfx1031 and gfx1030 share the same ISA but ROCm treats them differently
- **AND** it SHALL know rocBLAS Tensile is missing gfx1031 as of ROCm 7.2 (date-stamped)
- **AND** it SHALL know HSA_OVERRIDE=10.3.0 was the historical workaround but is no longer used

### Requirement: Config contains accurate HSA_OVERRIDE guidance
The config SHALL correctly state that HSA_OVERRIDE_GFX_VERSION is NOT set in either
native launcher script, and that UD-Q4_K_XL native is broken regardless of the override.

#### Scenario: AI advises on HSA_OVERRIDE
- **WHEN** an AI is asked to troubleshoot UD-Q4_K_XL on native
- **THEN** it SHALL NOT suggest adding HSA_OVERRIDE as a fix
- **AND** it SHALL recommend the docker-rocm launcher instead

### Requirement: Config includes current launcher status
The config SHALL contain a structured status table showing which launchers work, which are
broken, and why, so AI readers do not propose changes for non-functional paths.

#### Scenario: AI avoids broken launchers
- **WHEN** an AI proposes a change affecting launcher scripts
- **THEN** it SHALL only propose work against launchers marked as working or in-progress
- **AND** it SHALL note breakage reasons when referencing broken launchers

### Requirement: Config includes breakage taxonomy
The config SHALL enumerate the categories of things that break this setup (kernel updates,
ROCm package updates, llama.cpp upstream changes, etc.) so AI readers understand the
fragility model.

#### Scenario: AI understands fragility
- **WHEN** an AI reads the config context
- **THEN** it SHALL know that any of: kernel update, ROCm update, llama.cpp update,
  or HuggingFace repo change can break the setup independently

### Requirement: Config includes AI decision logic
The config SHALL provide conditional rules (if X → then Y) that allow an AI to reason
about hardware/model compatibility, not just recite flat facts.

#### Scenario: AI applies decision logic for model selection
- **WHEN** an AI is asked about running a new model
- **THEN** it SHALL check tensor types first
- **AND** if f16 tensors are present it SHALL recommend docker-rocm over native

### Requirement: Config has rules for specs and design artifact types
The config rules section SHALL include entries for `specs` and `design` in addition to
the existing `proposal` and `tasks` entries.

#### Scenario: AI follows design rules
- **WHEN** an AI creates a design artifact for this repo
- **THEN** it SHALL apply the design rules from config
