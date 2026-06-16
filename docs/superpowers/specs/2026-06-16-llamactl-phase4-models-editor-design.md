# llamactl Phase 4 — Models Settings Editor — Design

**Date:** 2026-06-16
**Status:** Approved (design); ready for implementation planning
**Basis:** `2026-06-10-llamactl-tui-design.md` (overall design), Phase 4 row of
its phasing table; Phases 1–3 as shipped (core, Serve tab, Builds tab).

## Scope

Phase 4 adds the **Models tab**: a settings editor for the per-model TOML
configs in `configs/models/*.toml`. It covers the **full** design-doc scope:

- Edit / add / delete arbitrary keys across `[settings]`, `[backends.*]`,
  `[presets.*]`, plus the top-level `name` / `hf`.
- Comment- and formatting-preserving save (via `tomlkit`).
- A **Resolved view** toggle showing the merged settings for a chosen
  backend + preset.
- Create a new model (blank or by duplicating an existing one), delete a model,
  and import a legacy `models/*.json` into a new `.toml`.
- Live reload so saved changes appear in the Serve tab's model picker without
  restarting the app.

## Decisions (from brainstorm 2026-06-16)

| Decision | Choice |
|----------|--------|
| Scope | **Full** design scope (editor + create/duplicate/delete model + in-UI JSON import). |
| Value typing | **TOML-literal parse with string fallback**: parse `key = <input>`; if it doesn't parse, treat input as a string. |
| Edit/save architecture | **A — new `tomlkit`-document edit layer** (`core/config_edit.py`); the live `TOMLDocument` is the editor's working state, so comments survive. The `tomllib`-based `ModelConfig` read model is untouched. |
| Save trigger | **Explicit Save** with a dirty indicator (no per-keystroke writes). |
| Deferred fixes folded in | `resolve_settings` returns a **deep copy** (kills the shallow-copy aliasing footgun); the `flash_attn` str/bool inconsistency dissolves because the editor preserves each value's stored TOML type. |

## Architecture

New `llamactl/core/config_edit.py` (no Textual imports) and
`llamactl/ui/screens/models.py` (`ModelsScreen`). Obeys the established
`ui → core` one-way dependency.

The editor's working state for the selected model is a live
`tomlkit.TOMLDocument` loaded from the raw file. All edits mutate it in place,
so comments and formatting are preserved on save. The existing
`tomllib`-based `load_model` / `ModelConfig` (used by lifecycle, the Serve tab,
the Builds in-use check) is **not** changed in behavior — only refactored
slightly (below).

```
llamactl/
  core/
    config.py         # (existing) read model; small refactor — see below
    config_edit.py    # NEW: tomlkit load/edit/save, value typing, new/duplicate
  ui/screens/
    models.py         # NEW: ModelsScreen (list + tree + resolved view + actions)
```

### `core/config_edit.py` surface

- `load_doc(path: Path) -> TOMLDocument` — `tomlkit.parse(path.read_text())`.
- `parse_value(raw: str) -> Any` — the TOML-literal rule: attempt
  `tomlkit.parse(f"_x_ = {raw}")["_x_"]`; on a parse error return `raw`
  (a plain string). Never raises.
- `set_value(doc, section, key, raw)` — navigate/create `section`, set
  `key = parse_value(raw)`.
- `delete_key(doc, section, key)` — remove a key from a section.
- `ensure_section(doc, section)` — create an empty table for a new
  `[backends.*]` / `[presets.<name>]` if absent.
- `new_model_doc(name, hf) -> TOMLDocument` — minimal doc: `name`, `hf`,
  empty `[settings]`.
- `duplicate_doc(src: TOMLDocument) -> TOMLDocument` — re-parse a dump of `src`
  to get an independent document with comments intact.
- `save_doc(path, doc)` — atomic write (temp file + `replace`, mirroring
  `registry.save_registry`).
- `model_id_from_name(name: str) -> str` — slugify a name into a filename stem.

`section` values: `""` (top-level `name` / `hf`), `"settings"`,
`"backends.rocm"`, `"backends.vulkan"`, or `"presets.<name>"`.

### `core/config.py` refactor

Extract `_build_model_config(data: dict, path: Path) -> ModelConfig` from
`load_model` (the validation + construction body), so both `load_model` and the
Resolved view can build a `ModelConfig`. The Resolved view calls it on
`doc.unwrap()` (tomlkit → plain Python types). `resolve_settings` is changed to
return a **deep copy** of the merged dict.

## Value typing & editing

Tree leaves render as `key = value`. Editing a leaf opens an `Input` seeded with
the current value's text; on commit, `parse_value` types it:

| Input | Stored as |
|-------|-----------|
| `true` / `false` | bool |
| `16384` | int |
| `0.7` | float |
| `["a", "b"]` | list |
| `"on"` (quoted) | string `on` |
| `on` / `auto` / `q8_0` (bare, unparseable) | string |

Quote to force a string. This matches how values are stored and makes the
mapper's bool-vs-string distinction (`jinja = true` → bare `--jinja`;
`flash_attn = "on"` → `--flash-attn on`) correct by construction. Existing
values display with their stored type, so the previously-deferred assumption
that on/off settings are uniformly boolean is no longer made anywhere.

Validation: when adding a key, its name must be a valid TOML bare key
(otherwise the add is rejected in-UI with a message); a duplicate key within a
table is rejected. There is no dedicated rename operation — renaming a key is
delete-then-add, so it goes through the same add validation. `parse_value`
itself never errors.

## Models screen UI

- **Left:** model list (display names). Files that fail to parse are shown
  disabled, reusing `load_all`'s `(configs, errors)` return.
- **Right:** a `Tree` of the selected model — top-level (`name`, `hf`),
  `[settings]`, `[backends.rocm]`, `[backends.vulkan]`, and each `[presets.*]`.
- **Resolved view toggle:** swaps the editable tree for a read-only merged view
  for a chosen backend + preset (`Select` widgets), computed via
  `resolve_settings(_build_model_config(doc.unwrap(), path), preset, backend, {})`.
- **Actions:** **Save** (dirty `*` indicator; disabled when clean), **New**,
  **Duplicate**, **Delete**, **Import JSON**.

Explicit save only — edits accumulate on the in-memory doc and are written on
Save, so the file (and its comments) are not rewritten on every keystroke, and
**Discard** can revert by reloading the doc from disk.

## Create / duplicate / delete / import

- **New:** prompt `name` + `hf` → `id = model_id_from_name(name)`; reject if
  `configs/models/<id>.toml` already exists; write `new_model_doc`; reload;
  select it.
- **Duplicate:** copy the selected model's doc (comments preserved) under a new
  name/id (same collision check).
- **Delete:** confirmed → remove the file → reload → clear selection.
- **Import JSON:** a file-path `Input` → reuse `migrate.convert_model` on the
  chosen JSON → save the resulting document as a new `.toml` (collision-checked)
  → reload. Reuses the existing migration code rather than duplicating it.

## Save semantics, reload, error handling

- **Save** → `save_doc` (atomic) → reload `app._models` / `_model_errors` via
  `load_all` → refresh the **Serve** tab's model picker and the Models list →
  clear the dirty flag.
- Switching models, or creating/deleting, with unsaved edits prompts
  discard/cancel.
- Errors surface in-UI: invalid key name rejected with a message; file
  write/parse failures shown; a model file that won't parse stays disabled in
  the list (existing `load_all` behavior). `parse_value` never raises.
- No coupling to a running server — a note states that edits apply on the next
  launch (the Serve tab reads resolved settings at launch time).

## Testing

Core (pytest, pure / fake — no real TUI):

- `parse_value`: table covering bool, int, float, list, quoted string, and bare
  (unparseable) string.
- `set_value` / `delete_key`: **comment-preservation round-trip** — load a doc
  containing a comment, edit one value, dump, and assert the comment survives
  and the value changed.
- `new_model_doc` / `duplicate_doc`: correct structure; duplicate is independent
  of the source and keeps comments.
- `save_doc`: atomic write; round-trips equal.
- `model_id_from_name`: slug rules.
- `_build_model_config` refactor: existing `load_model` and run.py-parity tests
  stay green; `resolved` built from a doc equals `resolve_settings`.
- `resolve_settings` deep copy: mutating the returned dict (including a
  list-valued setting) does not alter the `ModelConfig`.

UI (Textual `Pilot` smoke tests): Models tab loads; selecting a model populates
the tree; editing a value sets the dirty flag; Save writes the file, clears
dirty, and reloads; the Resolved view toggle renders; new / duplicate / delete /
import flows complete.

## Out of scope (Phase 4)

- The opt-in `llama-server --help` flag validation (tied to a built artifact;
  belongs with the Builds/launch flow, not the editor).
- Editing the global config (`configs/llamactl.toml`) — this tab edits per-model
  files only.
- Changing `run.py`, `models/*.json`, `build.*.sh`, or the Dockerfiles.
- The Test/OOM tab (Phase 5).

## Relationship to OpenSpec

Per the 2026-06-06 spec-driven retrofit design, implementation should land
through an OpenSpec change extending the `llamactl` capability spec with
scenario-based WHEN/THEN requirements for value typing, comment-preserving
edits, the resolved view, model create/duplicate/delete/import, and the
save/reload flow — with this document as the design basis.
