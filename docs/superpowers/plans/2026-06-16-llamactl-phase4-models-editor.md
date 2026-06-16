# llamactl Phase 4 — Models Settings Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Models tab that edits per-model `configs/models/*.toml` with comment-preserving saves, a resolved-settings view, and create/duplicate/delete/import — backed by a testable `tomlkit` edit layer.

**Architecture:** A new `llamactl/core/config_edit.py` (no Textual) owns `tomlkit`-document load/edit/save plus TOML-literal value typing; the live `TOMLDocument` is the editor's working state so comments survive. `config.py` gets a tiny refactor (`_build_model_config` extracted; `resolve_settings` deep-copies). A new `llamactl/ui/screens/models.py` (`ModelsScreen`) drives it: model list + editable tree + section/key/value edit bar + resolved-view toggle + action buttons.

**Tech Stack:** Python 3.14, Textual 8.2.7, tomlkit/tomllib, pytest + pytest-asyncio.

**Design basis:** `docs/superpowers/specs/2026-06-16-llamactl-phase4-models-editor-design.md`

## File Structure

**Create:**
- `llamactl/core/config_edit.py` — `load_doc`, `parse_value`, `value_to_literal`, `is_valid_key`, `set_value`, `delete_key`, `ensure_section`, `new_model_doc`, `duplicate_doc`, `save_doc`, `model_id_from_name`.
- `llamactl/ui/screens/models.py` — `ModelsScreen`.
- `tests/llamactl/test_config_edit.py` — core unit tests.
- `tests/llamactl/test_models_ui.py` — Textual Pilot smoke tests.

**Modify:**
- `llamactl/core/config.py` — extract `_build_model_config`; `resolve_settings` returns a deep copy.
- `llamactl/ui/app.py` — wire `ModelsScreen` into the Models tab.
- `llamactl/ui/screens/serve.py` — add `_LaunchForm.refresh_model_options()` for the post-save picker refresh.

`configs/models/*.toml` are the edited files; nothing in `state/` is involved.

---

## Task 1: config.py refactor — extract `_build_model_config`, deep-copy `resolve_settings`

**Files:**
- Modify: `llamactl/core/config.py`
- Test: `tests/llamactl/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_config.py — append
import copy as _copy

from llamactl.core.config import _build_model_config, ModelConfig, resolve_settings


def test_build_model_config_from_plain_dict(tmp_path):
    data = {"hf": "org/m:f", "name": "M", "settings": {"ctx_size": 4096}}
    mc = _build_model_config(data, tmp_path / "m.toml")
    assert isinstance(mc, ModelConfig)
    assert mc.id == "m"
    assert mc.hf == "org/m:f"
    assert mc.settings == {"ctx_size": 4096}


def test_resolve_settings_returns_deep_copy():
    mc = ModelConfig(
        id="m", name="M", hf="org/m:f",
        settings={"stop": ["</s>"], "ctx_size": 4096},
        backends={}, presets={}, images={}, path=__import__("pathlib").Path("/x/m.toml"),
    )
    resolved = resolve_settings(mc, None, "rocm", {})
    resolved["stop"].append("INJECTED")
    resolved["ctx_size"] = 999
    # Mutating the resolved dict must not touch the model's own settings
    assert mc.settings["stop"] == ["</s>"]
    assert mc.settings["ctx_size"] == 4096
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_config.py -k "build_model_config or deep_copy" -v`
Expected: FAIL — `ImportError: cannot import name '_build_model_config'`.

- [ ] **Step 3: Refactor `config.py`**

Add `import copy` near the top (after `import tomllib`). Replace the body of `load_model` (currently lines ~70-103) so the construction lives in a new helper:

```python
def load_model(path: Path) -> ModelConfig:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return _build_model_config(data, path)


def _build_model_config(data: dict[str, Any], path: Path) -> ModelConfig:
    if "hf" not in data:
        raise ConfigError(f"{path}: missing required key 'hf'")
    hf = _require_str(path, "hf", data["hf"])
    name = _require_str(path, "name", data.get("name", path.stem))
    settings = dict(_require_table(path, "settings", data.get("settings", {})))
    backends = {
        k: dict(_require_table(path, f"backends.{k}", v))
        for k, v in _require_table(path, "backends", data.get("backends", {})).items()
    }
    presets = {
        k: dict(_require_table(path, f"presets.{k}", v))
        for k, v in _require_table(path, "presets", data.get("presets", {})).items()
    }
    images = dict(_require_table(path, "images", data.get("images", {})))
    return ModelConfig(
        id=path.stem, name=name, hf=hf, settings=settings,
        backends=backends, presets=presets, images=images, path=path,
    )
```

Change the final line of `resolve_settings` from `return settings` to:
```python
    return copy.deepcopy(settings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/llamactl/test_config.py tests/llamactl/test_parity.py -v`
Expected: PASS — the new tests plus all pre-existing config/parity tests (the refactor is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config.py tests/llamactl/test_config.py
git commit -m "refactor: extract _build_model_config; resolve_settings returns deep copy"
```

---

## Task 2: Value typing — `parse_value`, `value_to_literal`, `is_valid_key`

**Files:**
- Create: `llamactl/core/config_edit.py`
- Test: `tests/llamactl/test_config_edit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_config_edit.py
from __future__ import annotations

from pathlib import Path

import pytest

from llamactl.core.config_edit import is_valid_key, parse_value, value_to_literal


def test_parse_value_types():
    assert parse_value("true") is True
    assert parse_value("false") is False
    assert parse_value("16384") == 16384 and isinstance(parse_value("16384"), int)
    assert parse_value("0.7") == 0.7 and isinstance(parse_value("0.7"), float)
    assert parse_value('["a", "b"]') == ["a", "b"]
    assert parse_value('"on"') == "on" and isinstance(parse_value('"on"'), str)


def test_parse_value_bare_word_is_string():
    assert parse_value("on") == "on"
    assert parse_value("q8_0") == "q8_0"
    assert parse_value("auto") == "auto"


def test_value_to_literal_round_trips_through_parse_value():
    for raw_value in (True, False, 16384, 0.7, "on", ["a", "b"]):
        literal = value_to_literal(raw_value)
        assert parse_value(literal) == raw_value
    # bool stays bool (not coerced to int)
    assert isinstance(parse_value(value_to_literal(True)), bool)
    # string renders quoted so it parses back as a string
    assert value_to_literal("on") == '"on"'


def test_is_valid_key():
    assert is_valid_key("ctx_size")
    assert is_valid_key("cache-type-k")
    assert not is_valid_key("bad key")     # space
    assert not is_valid_key("has.dot")     # dot not allowed in a bare segment
    assert not is_valid_key("")            # empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_config_edit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llamactl.core.config_edit'`.

- [ ] **Step 3: Create `config_edit.py` with the typing helpers**

```python
# llamactl/core/config_edit.py
"""Comment-preserving edit/save layer for per-model TOML configs.

Operates on a live tomlkit.TOMLDocument so comments and formatting survive
edits. core never imports UI. Value typing follows the TOML-literal rule:
parse `key = <input>`; if it does not parse, the input is a plain string.
"""

from __future__ import annotations

import re
from typing import Any

import tomlkit
from tomlkit import TOMLDocument

_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_value(raw: str) -> Any:
    """TOML-literal typing with string fallback. Never raises.

    `true`/`false`->bool, `16384`->int, `0.7`->float, `["a","b"]`->list,
    `"on"`->str. A bare word that is not valid TOML (e.g. `q8_0`, `on`, `auto`)
    becomes a plain string.
    """
    try:
        return tomlkit.parse(f"_x_ = {raw}").unwrap()["_x_"]
    except Exception:
        return raw


def value_to_literal(value: Any) -> str:
    """Render a Python value as the TOML value literal that parse_value inverts.

    bool -> true/false, str -> quoted, int/float -> bare, list -> inline array.
    """
    doc = tomlkit.document()
    doc["_x_"] = value
    return tomlkit.dumps(doc).split("=", 1)[1].strip()


def is_valid_key(key: str) -> bool:
    """True if `key` is a valid single TOML bare key (A-Za-z0-9_-, non-empty)."""
    return bool(_BARE_KEY_RE.match(key))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_config_edit.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config_edit.py tests/llamactl/test_config_edit.py
git commit -m "feat: TOML-literal value typing for the model editor"
```

---

## Task 3: Document load/save with comment preservation

**Files:**
- Modify: `llamactl/core/config_edit.py`
- Test: `tests/llamactl/test_config_edit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_config_edit.py — append
from llamactl.core.config_edit import load_doc, save_doc


def test_load_and_save_round_trip_preserves_comments(tmp_path):
    src = tmp_path / "m.toml"
    src.write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\n'
        '# tuning comment\nctx_size = 4096\n',
        encoding="utf-8",
    )
    doc = load_doc(src)
    out = tmp_path / "out.toml"
    save_doc(out, doc)
    text = out.read_text(encoding="utf-8")
    assert "# tuning comment" in text
    assert 'ctx_size = 4096' in text


def test_save_doc_is_atomic_leaves_no_tmp(tmp_path):
    doc = load_doc_from_text('hf = "x"\n', tmp_path)
    target = tmp_path / "m.toml"
    save_doc(target, doc)
    assert target.exists()
    assert not (tmp_path / "m.toml.tmp").exists()


def load_doc_from_text(text: str, tmp_path) -> object:
    p = tmp_path / "seed.toml"
    p.write_text(text, encoding="utf-8")
    return load_doc(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_config_edit.py -k "round_trip or atomic" -v`
Expected: FAIL — `ImportError: cannot import name 'load_doc'`.

- [ ] **Step 3: Add load/save to `config_edit.py`**

Add `from pathlib import Path` to the imports, then append:

```python
def load_doc(path: Path) -> TOMLDocument:
    return tomlkit.parse(path.read_text(encoding="utf-8"))


def save_doc(path: Path, doc: TOMLDocument) -> None:
    """Atomic write (temp file + replace), mirroring registry.save_registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    tmp.replace(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_config_edit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config_edit.py tests/llamactl/test_config_edit.py
git commit -m "feat: load_doc/save_doc with atomic comment-preserving write"
```

---

## Task 4: Edit operations — `set_value`, `delete_key`, `ensure_section`

**Files:**
- Modify: `llamactl/core/config_edit.py`
- Test: `tests/llamactl/test_config_edit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_config_edit.py — append
from llamactl.core.config_edit import delete_key, ensure_section, set_value


def _doc(text: str):
    return tomlkit.parse(text)


def test_set_value_updates_existing_and_preserves_comment():
    doc = _doc('name = "M"\nhf = "x"\n\n[settings]\n# keep me\nctx_size = 4096\n')
    set_value(doc, "settings", "ctx_size", "8192")
    text = tomlkit.dumps(doc)
    assert "# keep me" in text
    assert doc["settings"]["ctx_size"] == 8192
    assert isinstance(doc["settings"]["ctx_size"], int)


def test_set_value_creates_nested_section():
    doc = _doc('hf = "x"\n')
    set_value(doc, "backends.rocm", "cache_type_k", "q8_0")
    assert doc["backends"]["rocm"]["cache_type_k"] == "q8_0"


def test_set_value_top_level():
    doc = _doc('hf = "x"\n')
    set_value(doc, "", "name", '"New Name"')
    assert doc["name"] == "New Name"


def test_set_value_bool_stays_bare_flag():
    doc = _doc('hf = "x"\n[settings]\n')
    set_value(doc, "settings", "jinja", "true")
    assert doc["settings"]["jinja"] is True


def test_delete_key_removes_only_that_key():
    doc = _doc('hf = "x"\n[settings]\na = 1\nb = 2\n')
    delete_key(doc, "settings", "a")
    assert "a" not in doc["settings"]
    assert doc["settings"]["b"] == 2


def test_ensure_section_creates_empty_preset_table():
    doc = _doc('hf = "x"\n')
    ensure_section(doc, "presets.thinking")
    assert "thinking" in doc["presets"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_config_edit.py -k "set_value or delete_key or ensure_section" -v`
Expected: FAIL — `ImportError: cannot import name 'set_value'`.

- [ ] **Step 3: Add edit ops to `config_edit.py`**

```python
def _section_table(doc: TOMLDocument, section: str, create: bool):
    """Return the table for `section` ("" = doc root). create=True makes any
    missing intermediate tables; create=False returns None if absent."""
    if section == "":
        return doc
    node: Any = doc
    for part in section.split("."):
        if part not in node:
            if not create:
                return None
            node[part] = tomlkit.table()
        node = node[part]
    return node


def ensure_section(doc: TOMLDocument, section: str) -> None:
    """Create an empty table for `section` (e.g. 'presets.thinking') if absent."""
    _section_table(doc, section, create=True)


def set_value(doc: TOMLDocument, section: str, key: str, raw: str) -> None:
    """Set section.key = parse_value(raw), creating the section if needed."""
    table = _section_table(doc, section, create=True)
    table[key] = parse_value(raw)


def delete_key(doc: TOMLDocument, section: str, key: str) -> None:
    """Remove section.key if both exist; a no-op otherwise."""
    table = _section_table(doc, section, create=False)
    if table is not None and key in table:
        del table[key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_config_edit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config_edit.py tests/llamactl/test_config_edit.py
git commit -m "feat: set_value/delete_key/ensure_section edit ops"
```

---

## Task 5: New / duplicate doc + `model_id_from_name`

**Files:**
- Modify: `llamactl/core/config_edit.py`
- Test: `tests/llamactl/test_config_edit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_config_edit.py — append
from llamactl.core.config_edit import duplicate_doc, model_id_from_name, new_model_doc


def test_new_model_doc_minimal_shape():
    doc = new_model_doc("My Model", "org/m:f")
    assert doc["name"] == "My Model"
    assert doc["hf"] == "org/m:f"
    assert "settings" in doc and len(doc["settings"]) == 0


def test_duplicate_doc_is_independent_and_keeps_comments():
    src = tomlkit.parse('name = "M"\nhf = "x"\n[settings]\n# c\na = 1\n')
    dup = duplicate_doc(src)
    dup["settings"]["a"] = 999
    assert src["settings"]["a"] == 1           # source untouched
    assert "# c" in tomlkit.dumps(dup)         # comment carried over


def test_model_id_from_name_slug():
    assert model_id_from_name("Qwen3 8B Instruct") == "Qwen3-8B-Instruct"
    assert model_id_from_name("a/b:c") == "a-b-c"
    assert model_id_from_name("  ") == "model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_config_edit.py -k "new_model or duplicate or model_id" -v`
Expected: FAIL — `ImportError: cannot import name 'new_model_doc'`.

- [ ] **Step 3: Add to `config_edit.py`**

```python
def new_model_doc(name: str, hf: str) -> TOMLDocument:
    doc = tomlkit.document()
    doc["name"] = name
    doc["hf"] = hf
    doc["settings"] = tomlkit.table()
    return doc


def duplicate_doc(src: TOMLDocument) -> TOMLDocument:
    """Independent copy with comments intact (re-parse a dump of src)."""
    return tomlkit.parse(tomlkit.dumps(src))


def model_id_from_name(name: str) -> str:
    """Slugify a display name into a filename stem (A-Za-z0-9._- ; '-' joins)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return slug or "model"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_config_edit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/core/config_edit.py tests/llamactl/test_config_edit.py
git commit -m "feat: new_model_doc/duplicate_doc/model_id_from_name"
```

---

## Task 6: ModelsScreen — list, editable tree, edit bar, Save; app wiring

**Files:**
- Create: `llamactl/ui/screens/models.py`
- Modify: `llamactl/ui/app.py` (import + Models TabPane body)
- Test: `tests/llamactl/test_models_ui.py`

This is the core editor screen. The interaction model avoids fragile inline-tree
editing: selecting a tree leaf fills a section/key/value edit bar; **Set** writes
`section.key = value` (create or update), **Delete key** removes it, **Save**
writes the file. Editing marks the screen dirty.

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_models_ui.py
from __future__ import annotations

from pathlib import Path

import pytest


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\nctx_size = 4096\n'
    )
    (tmp_path / "state").mkdir()
    return tmp_path


@pytest.mark.asyncio
async def test_models_tab_lists_and_loads_model(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from textual.widgets import Button, Input, Tree

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        assert app.query_one("#model-list") is not None
        assert app.query_one("#editor-tree", Tree) is not None
        assert app.query_one("#btn-save", Button) is not None
        screen = app.query_one("llamactl.ui.screens.models:ModelsScreen".split(":")[-1])  # noqa
        # Select the only model and confirm the tree populated with its key
        from llamactl.ui.screens.models import ModelsScreen
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        tree = app.query_one("#editor-tree", Tree)
        labels = _tree_leaf_labels(tree)
        assert any("ctx_size = 4096" in lbl for lbl in labels)


@pytest.mark.asyncio
async def test_edit_value_sets_dirty_and_save_writes(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        # Edit settings.ctx_size -> 8192 via the edit bar API
        ms.apply_set("settings", "ctx_size", "8192")
        assert ms.is_dirty
        ms.action_save()
        await pilot.pause()
        assert not ms.is_dirty
        text = (tmp_path / "configs" / "models" / "m.toml").read_text()
        assert "ctx_size = 8192" in text


def _tree_leaf_labels(tree) -> list[str]:
    out: list[str] = []

    def walk(node):
        out.append(str(node.label))
        for child in node.children:
            walk(child)

    walk(tree.root)
    return out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_models_ui.py -v`
Expected: FAIL — Models tab is still a placeholder `Static`; `#model-list` not found.

- [ ] **Step 3: Create `llamactl/ui/screens/models.py`**

```python
"""Models tab — per-model TOML editor with comment-preserving saves."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Input, Label, OptionList, Select, Static, Tree
from textual.widgets.option_list import Option

from llamactl.core import config_edit as ce

if TYPE_CHECKING:
    from llamactl.ui.app import LlamaCtlApp


class ModelsScreen(Widget):
    DEFAULT_CSS = """
    ModelsScreen { height: 1fr; layout: horizontal; }
    ModelsScreen #model-list { width: 32; border: solid $panel; }
    ModelsScreen #editor-pane { width: 1fr; layout: vertical; }
    ModelsScreen #editor-tree { height: 1fr; border: solid $panel; }
    ModelsScreen #edit-bar { height: auto; layout: horizontal; }
    ModelsScreen #edit-bar Input, ModelsScreen #edit-bar Select { width: 1fr; margin-right: 1; }
    ModelsScreen .button-row { height: auto; layout: horizontal; }
    ModelsScreen .button-row Button { margin-right: 1; }
    ModelsScreen #editor-title { height: auto; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._doc = None          # tomlkit document of the selected model
        self._model_id: str | None = None
        self.is_dirty = False

    # ── compose ──────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield OptionList(id="model-list")
        with Widget(id="editor-pane"):
            yield Static("(no model selected)", id="editor-title")
            yield Tree("model", id="editor-tree")
            with Widget(id="edit-bar"):
                yield Select(options=[], id="section-select", allow_blank=True)
                yield Input(placeholder="key", id="key-input")
                yield Input(placeholder="value (TOML literal)", id="value-input")
                yield Button("Set", id="btn-set")
                yield Button("Delete key", id="btn-del-key", variant="error")
            with Widget(classes="button-row"):
                yield Button("Save", id="btn-save", variant="success", disabled=True)
                yield Button("New", id="btn-new")
                yield Button("Duplicate", id="btn-dup")
                yield Button("Delete model", id="btn-del-model", variant="error")

    def on_mount(self) -> None:
        self._reload_model_list()

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def _models_dir(self) -> Path:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        return app._config_dir / "models"

    def _reload_model_list(self) -> None:
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        ol = self.query_one("#model-list", OptionList)
        ol.clear_options()
        for m in app._models:
            ol.add_option(Option(m.name, id=m.id))
        for path, _err in app._model_errors.items():
            ol.add_option(Option(f"[red]{path.stem} (parse error)[/red]", id=path.stem))

    def _set_dirty(self, dirty: bool) -> None:
        self.is_dirty = dirty
        title = self.query_one("#editor-title", Static)
        mark = " *" if dirty else ""
        title.update(f"{self._model_id or '(no model selected)'}{mark}")
        self.query_one("#btn-save", Button).disabled = not dirty

    def _section_options(self) -> list[tuple[str, str]]:
        opts = [("(top-level)", ""), ("settings", "settings"),
                ("backends.rocm", "backends.rocm"), ("backends.vulkan", "backends.vulkan")]
        if self._doc is not None and "presets" in self._doc:
            for pk in self._doc["presets"]:
                opts.append((f"presets.{pk}", f"presets.{pk}"))
        return opts

    def _rebuild_tree(self) -> None:
        tree = self.query_one("#editor-tree", Tree)
        tree.clear()
        if self._doc is None:
            return
        root = tree.root
        root.expand()

        def add_table(parent, section: str, table) -> None:
            for k, v in table.items():
                if hasattr(v, "items") and section in ("backends", "presets"):
                    continue  # nested tables handled explicitly below
                parent.add_leaf(f"{k} = {ce.value_to_literal(v)}", data=(section, k))

        top = root.add("(top-level)", expand=True)
        for k in ("name", "hf"):
            if k in self._doc:
                top.add_leaf(f"{k} = {ce.value_to_literal(self._doc[k])}", data=("", k))
        if "settings" in self._doc:
            node = root.add("settings", expand=True)
            for k, v in self._doc["settings"].items():
                node.add_leaf(f"{k} = {ce.value_to_literal(v)}", data=("settings", k))
        if "backends" in self._doc:
            for bk, bt in self._doc["backends"].items():
                node = root.add(f"backends.{bk}", expand=True)
                for k, v in bt.items():
                    node.add_leaf(f"{k} = {ce.value_to_literal(v)}", data=(f"backends.{bk}", k))
        if "presets" in self._doc:
            for pk, pt in self._doc["presets"].items():
                node = root.add(f"presets.{pk}", expand=True)
                for k, v in pt.items():
                    node.add_leaf(f"{k} = {ce.value_to_literal(v)}", data=(f"presets.{pk}", k))
        self.query_one("#section-select", Select).set_options(self._section_options())

    # ── public API (also drives tests) ───────────────────────────────────────
    def select_model(self, model_id: str) -> None:
        path = self._models_dir / f"{model_id}.toml"
        try:
            self._doc = ce.load_doc(path)
        except Exception as exc:
            self.notify(f"Cannot open {model_id}: {exc}", severity="error")
            return
        self._model_id = model_id
        self._rebuild_tree()
        self._set_dirty(False)

    def apply_set(self, section: str, key: str, raw: str) -> None:
        if self._doc is None:
            return
        if not ce.is_valid_key(key):
            self.notify(f"Invalid key name: {key!r}", severity="warning")
            return
        ce.set_value(self._doc, section, key, raw)
        self._rebuild_tree()
        self._set_dirty(True)

    def apply_delete(self, section: str, key: str) -> None:
        if self._doc is None:
            return
        ce.delete_key(self._doc, section, key)
        self._rebuild_tree()
        self._set_dirty(True)

    def action_save(self) -> None:
        if self._doc is None or self._model_id is None:
            return
        path = self._models_dir / f"{self._model_id}.toml"
        try:
            ce.save_doc(path, self._doc)
        except Exception as exc:
            self.notify(f"Save failed: {exc}", severity="error")
            return
        self._reload_app_models()
        self._set_dirty(False)
        self.notify("Saved.")

    def _reload_app_models(self) -> None:
        from llamactl.core.config import load_all
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        app._models, app._model_errors = load_all(self._models_dir)
        self._reload_model_list()

    # ── events ───────────────────────────────────────────────────────────────
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-list" and event.option.id:
            self.select_model(event.option.id)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if not data:
            return
        section, key = data
        self.query_one("#section-select", Select).value = section
        self.query_one("#key-input", Input).value = key
        if self._doc is not None:
            table = ce._section_table(self._doc, section, create=False)
            if table is not None and key in table:
                self.query_one("#value-input", Input).value = ce.value_to_literal(table[key])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-set":
            section = str(self.query_one("#section-select", Select).value or "")
            key = self.query_one("#key-input", Input).value.strip()
            raw = self.query_one("#value-input", Input).value.strip()
            if key:
                self.apply_set(section, key, raw)
        elif bid == "btn-del-key":
            section = str(self.query_one("#section-select", Select).value or "")
            key = self.query_one("#key-input", Input).value.strip()
            if key:
                self.apply_delete(section, key)
        elif bid == "btn-save":
            self.action_save()
```

(`btn-new`, `btn-dup`, `btn-del-model`, Import, and the resolved view are added
in later tasks; their handlers are appended to `on_button_pressed` then.)

- [ ] **Step 4: Wire into `app.py`**

Add the import next to the others (around line 12):
```python
from llamactl.ui.screens.models import ModelsScreen
```
Replace the Models TabPane body (around line 57-58):
```python
            with TabPane("Models", id="models"):
                yield ModelsScreen()
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/llamactl/test_models_ui.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add llamactl/ui/screens/models.py llamactl/ui/app.py tests/llamactl/test_models_ui.py
git commit -m "feat: Models tab editor — list, tree, edit bar, save"
```

---

## Task 7: Resolved-view toggle

**Files:**
- Modify: `llamactl/ui/screens/models.py`
- Test: `tests/llamactl/test_models_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_models_ui.py — append
@pytest.mark.asyncio
async def test_resolved_view_renders_merged_settings(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen
    from textual.widgets import Static

    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "llamactl.toml").write_text(
        'default_backend = "rocm"\ndefault_mode = "container"\nport = 8080\n'
        'vram_budget_gb = 11.0\nname_prefix = "llamactl"\n'
    )
    (tmp_path / "configs" / "models" / "m.toml").write_text(
        'name = "M"\nhf = "org/m:f"\n\n[settings]\nctx_size = 4096\n'
        '\n[backends.rocm]\ncache_type_k = "f16"\n'
    )
    (tmp_path / "state").mkdir()

    app = LlamaCtlApp(repo_root=tmp_path)
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.select_model("m")
        await pilot.pause()
        text = ms.resolved_text("rocm", None)
        assert "ctx_size" in text and "4096" in text
        assert "cache_type_k" in text and "f16" in text   # backend override merged in
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_models_ui.py -k resolved -v`
Expected: FAIL — `AttributeError: 'ModelsScreen' object has no attribute 'resolved_text'`.

- [ ] **Step 3: Add the resolved view**

Add imports at the top of `models.py`:
```python
from llamactl.core.config import _build_model_config, resolve_settings
```
Add a `resolved_text` method and a toggle. Append these methods to `ModelsScreen`:
```python
    def resolved_text(self, backend: str, preset: str | None) -> str:
        if self._doc is None or self._model_id is None:
            return "(no model selected)"
        path = self._models_dir / f"{self._model_id}.toml"
        try:
            mc = _build_model_config(self._doc.unwrap(), path)
            resolved = resolve_settings(mc, preset, backend, {})
        except Exception as exc:
            return f"[red]{exc}[/red]"
        lines = [f"{k} = {ce.value_to_literal(v)}" for k, v in sorted(resolved.items())]
        return "\n".join(lines) or "(empty)"
```
Add a resolved-view `Static` and a toggle button to `compose`. In the
`button-row`, after the "Delete model" button, add:
```python
                yield Button("Resolved view", id="btn-resolved")
```
And after the `Tree` in `editor-pane`, add a hidden resolved pane:
```python
            yield Static("", id="resolved-view")
```
Add to the `DEFAULT_CSS`:
```css
    ModelsScreen #resolved-view { height: 1fr; border: solid $panel; display: none; }
    ModelsScreen.-resolved #editor-tree { display: none; }
    ModelsScreen.-resolved #resolved-view { display: block; }
    """
```
(Append those three rules just before the closing `"""` of `DEFAULT_CSS`.)
Add a handler branch in `on_button_pressed`:
```python
        elif bid == "btn-resolved":
            showing = self.has_class("-resolved")
            if not showing:
                app: LlamaCtlApp = self.app  # type: ignore[assignment]
                backend = app._global_cfg.default_backend
                self.query_one("#resolved-view", Static).update(self.resolved_text(backend, None))
            self.toggle_class("-resolved")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_models_ui.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/models.py tests/llamactl/test_models_ui.py
git commit -m "feat: Models tab resolved-settings view"
```

---

## Task 8: New / Duplicate / Delete model

**Files:**
- Modify: `llamactl/ui/screens/models.py`
- Test: `tests/llamactl/test_models_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_models_ui.py — append
@pytest.mark.asyncio
async def test_create_duplicate_delete_model(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        models_dir = tmp_path / "configs" / "models"

        # Create
        ms.create_model("New One", "org/new:f")
        await pilot.pause()
        assert (models_dir / "New-One.toml").exists()
        assert any(m.id == "New-One" for m in app._models)

        # Create collision is rejected (no overwrite)
        ms.create_model("New One", "org/other:f")
        assert (models_dir / "New-One.toml").read_text().count("org/new:f") == 1

        # Duplicate the original m
        ms.select_model("m")
        ms.duplicate_model("m copy")
        await pilot.pause()
        assert (models_dir / "m-copy.toml").exists()

        # Delete
        ms.delete_model("New-One")
        await pilot.pause()
        assert not (models_dir / "New-One.toml").exists()
        assert not any(m.id == "New-One" for m in app._models)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_models_ui.py -k create_duplicate_delete -v`
Expected: FAIL — `AttributeError: ... has no attribute 'create_model'`.

- [ ] **Step 3: Add the model-level operations**

Append to `ModelsScreen`:
```python
    def create_model(self, name: str, hf: str) -> None:
        model_id = ce.model_id_from_name(name)
        path = self._models_dir / f"{model_id}.toml"
        if path.exists():
            self.notify(f"Model '{model_id}' already exists.", severity="warning")
            return
        ce.save_doc(path, ce.new_model_doc(name, hf))
        self._reload_app_models()
        self.select_model(model_id)

    def duplicate_model(self, new_name: str) -> None:
        if self._doc is None:
            self.notify("Select a model to duplicate first.", severity="warning")
            return
        model_id = ce.model_id_from_name(new_name)
        path = self._models_dir / f"{model_id}.toml"
        if path.exists():
            self.notify(f"Model '{model_id}' already exists.", severity="warning")
            return
        dup = ce.duplicate_doc(self._doc)
        dup["name"] = new_name
        ce.save_doc(path, dup)
        self._reload_app_models()
        self.select_model(model_id)

    def delete_model(self, model_id: str) -> None:
        path = self._models_dir / f"{model_id}.toml"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        if self._model_id == model_id:
            self._model_id = None
            self._doc = None
            self._rebuild_tree()
            self._set_dirty(False)
        self._reload_app_models()
```

For the buttons, add three inputs and wire the handlers. Add to `compose`, in a
new row before the existing `button-row`:
```python
            with Widget(classes="button-row"):
                yield Input(placeholder="new/dup name", id="new-name")
                yield Input(placeholder="hf (for New)", id="new-hf")
```
Add handler branches to `on_button_pressed`:
```python
        elif bid == "btn-new":
            name = self.query_one("#new-name", Input).value.strip()
            hf = self.query_one("#new-hf", Input).value.strip()
            if name and hf:
                self.create_model(name, hf)
        elif bid == "btn-dup":
            name = self.query_one("#new-name", Input).value.strip()
            if name:
                self.duplicate_model(name)
        elif bid == "btn-del-model":
            if self._model_id:
                self.delete_model(self._model_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_models_ui.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/models.py tests/llamactl/test_models_ui.py
git commit -m "feat: Models tab create/duplicate/delete model"
```

---

## Task 9: Import a legacy JSON model

**Files:**
- Modify: `llamactl/ui/screens/models.py`
- Test: `tests/llamactl/test_models_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_models_ui.py — append
@pytest.mark.asyncio
async def test_import_json_creates_toml(tmp_path: Path) -> None:
    import json
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    src = tmp_path / "legacy.json"
    src.write_text(json.dumps({"hf": "org/legacy:f", "name": "Legacy",
                               "defaults": {"ctx_size": 2048}}))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.import_json(src)
        await pilot.pause()
        out = tmp_path / "configs" / "models" / "legacy.toml"
        assert out.exists()
        assert 'hf = "org/legacy:f"' in out.read_text()
        assert any(m.id == "legacy" for m in app._models)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_models_ui.py -k import_json -v`
Expected: FAIL — `AttributeError: ... has no attribute 'import_json'`.

- [ ] **Step 3: Add the import operation**

Append to `ModelsScreen` (reuses the existing migration converter):
```python
    def import_json(self, json_path: Path) -> None:
        import json as _json
        from llamactl.core.migrate import convert_model
        model_id = json_path.stem
        dest = self._models_dir / f"{model_id}.toml"
        if dest.exists():
            self.notify(f"Model '{model_id}' already exists.", severity="warning")
            return
        try:
            raw = _json.loads(json_path.read_text(encoding="utf-8"))
            doc, warnings = convert_model(raw, model_id)
        except Exception as exc:
            self.notify(f"Import failed: {exc}", severity="error")
            return
        ce.save_doc(dest, doc)
        for w in warnings:
            self.notify(w, severity="warning")
        self._reload_app_models()
        self.select_model(model_id)
```
Add an import row to `compose` (after the new/dup row):
```python
            with Widget(classes="button-row"):
                yield Input(placeholder="path/to/legacy.json", id="import-path")
                yield Button("Import JSON", id="btn-import")
```
Add the handler branch to `on_button_pressed`:
```python
        elif bid == "btn-import":
            raw_path = self.query_one("#import-path", Input).value.strip()
            if raw_path:
                self.import_json(Path(raw_path).expanduser())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_models_ui.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/models.py tests/llamactl/test_models_ui.py
git commit -m "feat: Models tab JSON import via convert_model"
```

---

## Task 10: Refresh the Serve model picker after save

**Files:**
- Modify: `llamactl/ui/screens/serve.py` (add `_LaunchForm.refresh_model_options`)
- Modify: `llamactl/ui/screens/models.py` (call it after reload)
- Test: `tests/llamactl/test_models_ui.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llamactl/test_models_ui.py — append
@pytest.mark.asyncio
async def test_new_model_appears_in_serve_picker(tmp_path: Path) -> None:
    from llamactl.ui.app import LlamaCtlApp
    from llamactl.ui.screens.models import ModelsScreen
    from llamactl.ui.screens.serve import _LaunchForm
    from textual.widgets import Select

    app = LlamaCtlApp(repo_root=_repo(tmp_path))
    async with app.run_test(headless=True) as pilot:
        app.query_one("TabbedContent").active = "models"
        await pilot.pause()
        ms = app.query_one(ModelsScreen)
        ms.create_model("Fresh Model", "org/fresh:f")
        await pilot.pause()
        form = app.query_one(_LaunchForm)
        values = [v for _label, v in form.query_one("#model-select", Select)._options]
        assert "Fresh-Model" in values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llamactl/test_models_ui.py -k serve_picker -v`
Expected: FAIL — the new model id is not in the Serve model select options.

- [ ] **Step 3: Add `refresh_model_options` to `_LaunchForm` and call it**

In `llamactl/ui/screens/serve.py`, append a method to `_LaunchForm` (next to `refresh_artifact_options`):
```python
    def refresh_model_options(self) -> None:
        """Repopulate the model select from app._models (after a Models-tab save)."""
        app = self.app
        self._models = getattr(app, "_models", [])
        self._model_map = {m.id: m for m in self._models}
        try:
            sel = self.query_one("#model-select", Select)
        except NoMatches:
            return
        sel.set_options([(m.name, m.id) for m in self._models])
```

In `llamactl/ui/screens/models.py`, extend `_reload_app_models` to also refresh
the Serve picker:
```python
    def _reload_app_models(self) -> None:
        from llamactl.core.config import load_all
        from llamactl.ui.screens.serve import _LaunchForm
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        app._models, app._model_errors = load_all(self._models_dir)
        self._reload_model_list()
        try:
            self.app.query_one(_LaunchForm).refresh_model_options()
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llamactl/test_models_ui.py tests/llamactl/test_serve_ui.py -v`
Expected: PASS — including the existing serve tests (the new method doesn't change existing behavior).

- [ ] **Step 5: Commit**

```bash
git add llamactl/ui/screens/serve.py llamactl/ui/screens/models.py tests/llamactl/test_models_ui.py
git commit -m "feat: refresh Serve model picker after a Models-tab save"
```

---

## Task 11: Full suite + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the whole llamactl suite**

Run: `pytest tests/llamactl/ -v`
Expected: all tests pass (Phase 1-4 — `test_config_edit.py`, `test_models_ui.py`, and all prior).

- [ ] **Step 2: Import-cleanliness check**

Run: `python -c "import llamactl.core.config_edit, llamactl.ui.screens.models, llamactl.ui.app"`
Expected: no output, exit 0.

- [ ] **Step 3: Manual smoke (optional but recommended)**

Run: `python -m llamactl`
Switch to the **Models** tab, select a model, confirm the tree shows its
`[settings]`/`[backends.*]`/`[presets.*]`, edit a value via the edit bar + **Set**,
**Save**, and re-open the file to confirm the change and that comments survived.
Try **Resolved view**, **New**, **Duplicate**, **Delete**, and **Import JSON**.
Press `q` to quit.

- [ ] **Step 4: Commit any fixups**

```bash
git add -A
git commit -m "chore: Phase 4 Models editor fixups"
```

---

## Self-Review

**Spec coverage:**
- Edit/add/delete keys + name/hf → Tasks 4, 6 (`set_value`/`delete_key`, edit bar).
- TOML-literal value typing (flash_attn/jinja correctness) → Task 2.
- Comment-preserving save → Tasks 3, 6.
- Resolved-view toggle → Task 7.
- Create / duplicate / delete model → Task 8.
- JSON import (reusing `convert_model`) → Task 9.
- Live reload into Serve picker → Tasks 6 (`_reload_app_models`), 10.
- `_build_model_config` extraction + `resolve_settings` deep copy (deferred fixes) → Task 1.
- Key-name validation → Task 6 (`apply_set` uses `is_valid_key`); collision checks → Tasks 8, 9.
- Testing surface (parse_value table, comment round-trip, atomic save, slug, deep-copy, resolved-equals, UI smoke) → Tasks 1-10.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Later UI tasks append handlers/methods to the `ModelsScreen` defined in Task 6, with the exact code and slot shown.

**Type consistency:** `config_edit` names (`parse_value`, `value_to_literal`, `is_valid_key`, `load_doc`, `save_doc`, `set_value`, `delete_key`, `ensure_section`, `new_model_doc`, `duplicate_doc`, `model_id_from_name`) are used consistently across tasks and tests. `ModelsScreen` public methods (`select_model`, `apply_set`, `apply_delete`, `action_save`, `resolved_text`, `create_model`, `duplicate_model`, `delete_model`, `import_json`, `_reload_app_models`) are referenced consistently. `section` strings (`""`, `"settings"`, `"backends.rocm"`, `"backends.vulkan"`, `"presets.<name>"`) match between core and UI. `_section_table` is reused by the UI for value-input seeding.

**Executor notes:**
- Textual `OptionList.add_option(Option(label, id=...))` and `Tree.root.add(..., expand=True)` / `add_leaf(label, data=...)` are the 8.2.7 APIs; if a signature differs on the installed version, adapt (e.g. `Tree` node `data=` is supported in 8.2.7). The tests read `select._options` and walk `tree.root` directly, matching the Phase 3 test style.
- `doc.unwrap()` returns plain Python types for `_build_model_config`; if a tomlkit version lacks `.unwrap()` on the document, use `dict(doc)` recursively — but 8.2.x ships `unwrap()`.
