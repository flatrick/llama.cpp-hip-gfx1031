"""Models tab — per-model TOML editor with comment-preserving saves."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Tree
from textual.widgets.option_list import Option
from textual.widgets import OptionList

from llamactl.core import config_edit as ce
from llamactl.core.config import _build_model_config, resolve_settings

if TYPE_CHECKING:
    from llamactl.ui.app import LlamaCtlApp


class ModelsScreen(Widget):
    BINDINGS = [Binding("ctrl+s", "save", "Save")]

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
    ModelsScreen #resolved-view { height: 1fr; border: solid $panel; display: none; }
    ModelsScreen.-resolved #editor-tree { display: none; }
    ModelsScreen.-resolved #resolved-view { display: block; }
    ModelsScreen #resolved-bar { height: auto; layout: horizontal; display: none; }
    ModelsScreen #resolved-bar Select { width: 1fr; margin-right: 1; }
    ModelsScreen.-resolved #resolved-bar { display: block; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._doc = None
        self._model_id: str | None = None
        self.is_dirty = False

    def compose(self) -> ComposeResult:
        yield OptionList(id="model-list")
        with Widget(id="editor-pane"):
            yield Static("(no model selected)", id="editor-title")
            yield Tree("model", id="editor-tree")
            with Widget(id="resolved-bar"):
                yield Select(options=[("ROCm", "rocm"), ("Vulkan", "vulkan")],
                             value="rocm", id="rv-backend")
                yield Select(options=[], id="rv-preset", allow_blank=True)
            yield Static("", id="resolved-view")
            with Widget(id="edit-bar"):
                yield Select(options=[], id="section-select", allow_blank=True)
                yield Input(placeholder="key", id="key-input")
                yield Input(placeholder="value (TOML literal)", id="value-input")
                yield Button("Set", id="btn-set")
                yield Button("Delete key", id="btn-del-key", variant="error")
            with Widget(classes="button-row"):
                yield Input(placeholder="new/dup name", id="new-name")
                yield Input(placeholder="hf (for New)", id="new-hf")
            with Widget(classes="button-row"):
                yield Input(placeholder="path/to/legacy.json", id="import-path")
                yield Button("Import JSON", id="btn-import")
            with Widget(classes="button-row"):
                yield Button("Save", id="btn-save", variant="success", disabled=True)
                yield Button("New", id="btn-new")
                yield Button("Duplicate", id="btn-dup")
                yield Button("Delete model", id="btn-del-model", variant="error")
                yield Button("Resolved view", id="btn-resolved")

    def on_mount(self) -> None:
        self._reload_model_list()

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
        for path in app._model_errors:
            ol.add_option(Option(f"[red]{path.stem} (parse error)[/red]", id=path.stem))

    def _update_dirty(self, dirty: bool) -> None:
        self.is_dirty = dirty
        title = self.query_one("#editor-title", Static)
        mark = " *" if dirty else ""
        title.update(f"{self._model_id or '(no model selected)'}{mark}")
        self.query_one("#btn-save", Button).disabled = not dirty

    def _selected_section(self) -> str:
        """Section string for the dropdown; blank (Select.NULL/None) -> "" (top-level)."""
        value = self.query_one("#section-select", Select).value
        if value is Select.NULL or value is None:
            return ""
        return str(value)

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

    def select_model(self, model_id: str) -> None:
        path = self._models_dir / f"{model_id}.toml"
        try:
            self._doc = ce.load_doc(path)
        except Exception as exc:
            self.notify(f"Cannot open {model_id}: {exc}", severity="error")
            return
        self._model_id = model_id
        self._rebuild_tree()
        self._update_dirty(False)

    def apply_set(self, section: str, key: str, raw: str) -> None:
        if self._doc is None:
            return
        if not ce.is_valid_key(key):
            self.notify(f"Invalid key name: {key!r}", severity="warning")
            return
        ce.set_value(self._doc, section, key, raw)
        self._rebuild_tree()
        self._update_dirty(True)

    def apply_delete(self, section: str, key: str) -> None:
        if self._doc is None:
            return
        ce.delete_key(self._doc, section, key)
        self._rebuild_tree()
        self._update_dirty(True)

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
        self._update_dirty(False)
        self.notify("Saved.")

    def _reload_app_models(self) -> None:
        from llamactl.core.config import load_all
        app: LlamaCtlApp = self.app  # type: ignore[assignment]
        app._models, app._model_errors = load_all(self._models_dir)
        self._reload_model_list()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "model-list" and event.option.id:
            if self.is_dirty:
                self.notify("Discarded unsaved changes.", severity="warning")
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
            section = self._selected_section()
            key = self.query_one("#key-input", Input).value.strip()
            raw = self.query_one("#value-input", Input).value.strip()
            if key:
                self.apply_set(section, key, raw)
        elif bid == "btn-del-key":
            section = self._selected_section()
            key = self.query_one("#key-input", Input).value.strip()
            if key:
                self.apply_delete(section, key)
        elif bid == "btn-save":
            self.action_save()
        elif bid == "btn-new":
            name = self.query_one("#new-name", Input).value.strip()
            hf = self.query_one("#new-hf", Input).value.strip()
            if not (name and hf):
                self.notify("New model needs both a name and an hf repo.", severity="warning")
            else:
                if self.is_dirty:
                    self.notify("Discarded unsaved changes.", severity="warning")
                self.create_model(name, hf)
        elif bid == "btn-dup":
            name = self.query_one("#new-name", Input).value.strip()
            if not name:
                self.notify("Duplicate needs a new name.", severity="warning")
            else:
                if self.is_dirty:
                    self.notify("Discarded unsaved changes.", severity="warning")
                self.duplicate_model(name)
        elif bid == "btn-del-model":
            if not self._model_id:
                self.notify("Select a model to delete first.", severity="warning")
            else:
                if self.is_dirty:
                    self.notify("Discarded unsaved changes.", severity="warning")
                self.delete_model(self._model_id)
        elif bid == "btn-import":
            raw_path = self.query_one("#import-path", Input).value.strip()
            if raw_path:
                self.import_json(Path(raw_path).expanduser())
            else:
                self.notify("Enter a path to a legacy .json file.", severity="warning")
        elif bid == "btn-resolved":
            if not self.has_class("-resolved"):
                preset_opts = []
                if self._doc is not None and "presets" in self._doc:
                    preset_opts = [(pk, pk) for pk in self._doc["presets"]]
                self.query_one("#rv-preset", Select).set_options(preset_opts)
                self.render_resolved()
            self.toggle_class("-resolved")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("rv-backend", "rv-preset") and self.has_class("-resolved"):
            self.render_resolved()

    def create_model(self, name: str, hf: str) -> None:
        model_id = ce.model_id_from_name(name)
        path = self._models_dir / f"{model_id}.toml"
        if path.exists():
            self.notify(f"Model id '{model_id}' already exists (from name '{name}').", severity="warning")
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
            self.notify(f"Model id '{model_id}' already exists (from name '{new_name}').", severity="warning")
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
            self._update_dirty(False)
        self._reload_app_models()

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

    def render_resolved(self) -> None:
        backend_val = self.query_one("#rv-backend", Select).value
        backend = "rocm" if backend_val is Select.NULL or backend_val is None else str(backend_val)
        preset_val = self.query_one("#rv-preset", Select).value
        preset = None if preset_val is Select.NULL or preset_val is None else str(preset_val)
        self.query_one("#resolved-view", Static).update(self.resolved_text(backend, preset))

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
