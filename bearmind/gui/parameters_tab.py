"""Edit one configuration layer and preview the stages it invalidates."""
import math
from collections import Counter

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                               QVBoxLayout, QWidget)

from bearmind import bad_frames, config, registry, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import State


SECTIONS = ("scan", "crop", "mc", "thresholds", "cnmf", "gate", "autoinspect", "estimates", "inspection",
            "preview", "checks", "pool", "memory", "runner", "monitor", "Frame repair")
FRAME_KEYS = ("bad_frames", "bad_frames_action", "bad_frames_max_interpolate_s", "bad_frames_auto",
              "bad_frames_trim_tail_from")


def _fields(data, section):
    if section == "Frame repair":
        return {k: data[k] for k in FRAME_KEYS if k in data}
    return data.get(section) or {}


def _replacement(data, section, patch):
    if section == "Frame repair":
        unknown = set(patch) - set(FRAME_KEYS)
        if unknown:
            raise ValueError(f"Unknown frame repair key: {', '.join(sorted(unknown))}")
        return {**{k: v for k, v in data.items() if k not in FRAME_KEYS}, **patch}
    updated = dict(data)
    if patch:
        updated[section] = patch
    else:
        updated.pop(section, None)
    return updated


def _validate_keys(patch, allowed, where=""):
    for key, value in patch.items():
        name = f"{where}.{key}" if where else str(key)
        if key not in allowed:
            raise ValueError(f"Unknown parameter: {name}")
        existing = allowed[key]
        if isinstance(existing, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping.")
            _validate_keys(value, existing, name)
        elif isinstance(existing, bool) and type(value) is not bool:
            raise ValueError(f"{name} must be true or false.")
        elif type(existing) in (int, float) and (
                type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number.")
        elif isinstance(existing, list) and not isinstance(value, (list, tuple, type(None))):
            raise ValueError(f"{name} must be a list.")


class ParametersTab(QWidget):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.paths = SeriesPaths(root)
        self.original = {}
        self.path = None
        self.scope = QComboBox()
        self.scope.addItems(["Series", "Mouse", "Session"])
        self.target = QComboBox()
        self.section = QComboBox()
        self.section.addItems(SECTIONS)
        self.override = QPlainTextEdit()
        self.resolved = QPlainTextEdit()
        self.resolved.setReadOnly(True)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        reload = QPushButton("Reload parameters")
        self.save = QPushButton("Review and save")
        reload.clicked.connect(self._reload)
        self.save.clicked.connect(self._preview)
        layout = QVBoxLayout(self)
        selectors = QHBoxLayout()
        for title, combo in (("Scope", self.scope), ("Target", self.target), ("Section", self.section)):
            selectors.addWidget(QLabel(title))
            selectors.addWidget(combo)
        selectors.addStretch(1)
        layout.addLayout(selectors)
        layout.addWidget(self.details)
        editors = QHBoxLayout()
        for name, widget in (("Overrides at the selected layer", self.override),
                             ("Resolved values and their layers", self.resolved)):
            pane = QVBoxLayout()
            pane.addWidget(QLabel(name))
            pane.addWidget(widget, 1)
            editors.addLayout(pane, 1)
        layout.addLayout(editors, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(reload)
        buttons.addWidget(self.save)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.scope.currentTextChanged.connect(self._targets)
        self.target.currentTextChanged.connect(self._reload)
        self.section.currentTextChanged.connect(self._reload)
        self._targets()

    def refresh_targets(self):
        self._targets()

    def _targets(self):
        records = registry.all_records(self.paths)
        scope = self.scope.currentText()
        choices = (["All sessions"] if scope == "Series" else
                   sorted({r.mouse for r in records}) if scope == "Mouse" else [r.sid for r in records])
        old = self.target.currentText()
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItems(choices)
        if old in choices:
            self.target.setCurrentText(old)
        self.target.blockSignals(False)
        self._reload()

    def _selection(self):
        scope, target = self.scope.currentText(), self.target.currentText()
        if scope == "Series":
            return self.paths.series_config, None, None
        if not target:
            return None, None, None
        if scope == "Mouse":
            return self.paths.mouse_config(target), target, None
        rec = registry.load_record(self.paths, target)
        return self.paths.session_config(target), rec.mouse, target

    def _reload(self):
        self.path = None
        self.save.setEnabled(False)
        try:
            path, mouse, sid = self._selection()
            if path is None:
                self.details.setText("Scan the series to list sessions and mice.")
                self.override.clear()
                self.resolved.clear()
                return
            section = self.section.currentText()
            original = config.read_yaml(path)
            effective = config.resolve(self.paths, sid=sid, mouse=mouse)
            layers = [("Defaults", config.load_defaults()), ("Series", config.read_yaml(self.paths.series_config))]
            if mouse:
                layers.append(("Mouse", config.read_yaml(self.paths.mouse_config(mouse))))
            if sid:
                layers.append(("Session", original))
            blocks = []
            for label, data in [("Resolved", effective), *reversed(layers)]:
                fields = _fields(data, section)
                if fields:
                    blocks.append(f"# {label}\n" + yaml.safe_dump(fields, sort_keys=False))
            self.resolved.setPlainText("\n".join(blocks))
            self.override.setPlainText(yaml.safe_dump(_fields(original, section), sort_keys=False))
            self.details.setText(f"{path}\nEmpty mapping removes the selected layer’s overrides for this section.")
            self.path, self.original = path, original
            self.save.setEnabled(True)
        except (OSError, ValueError) as exc:
            self.details.setText(str(exc))

    def _preview(self):
        section = self.section.currentText()
        try:
            patch = yaml.safe_load(self.override.toPlainText())
            if patch is None:
                patch = {}
            if not isinstance(patch, dict) or not all(isinstance(k, str) for k in patch):
                raise ValueError("Enter a YAML mapping of parameter names to values.")
            if section != "Frame repair":
                allowed = config.deep_merge(_fields(config.load_defaults(), section), _fields(self.original, section))
                if section == "cnmf":
                    allowed.update(gSig=4, min_corr=.85, min_pnr=8.)
                if section == "mc" and patch.get("pw_rigid"):
                    raise ValueError("This pipeline supports rigid motion correction. Set pw_rigid to false.")
                _validate_keys(patch, allowed)
            updated = _replacement(self.original, section, patch)
            if updated == self.original:
                self.details.setText("No parameter changes.")
                return
            _, mouse, sid = self._selection()
            paths = self.paths
            records = [r for r in registry.all_records(paths) if (sid is None or r.sid == sid)
                       and (mouse is None or r.mouse == mouse)]
            proposed = runner.StageView(paths, resolve=lambda r: config.resolve(
                paths, sid=r.sid, mouse=r.mouse, overrides={self.path: updated}))
            prior = runner.StageView(paths)
            changes = Counter()
            if not records and section == "Frame repair":
                bad_frames.settings(config.resolve(paths, overrides={self.path: updated}), 2**31 - 1)
            for rec in records:
                if (problem := proposed.problem(rec)) and problem != prior.problem(rec):
                    raise ValueError(f"{rec.sid}: {problem}")
                for stage, entry in rec.stages.items():
                    if stage in registry.ALL_STAGES and entry.state is State.DONE and (
                            proposed.hash(rec, stage) != prior.hash(rec, stage)):
                        changes[stage] += 1
            affected = "\n".join(f"{stage}: {count} finished sessions" for stage, count in sorted(changes.items()))
            text = f"Apply the edited parameters in {self.path}?\n\n"
            text += f"Completed stages to review in Monitor:\n{affected}" if changes else "No finished stage becomes out of date."
            if QMessageBox.question(self, "Save parameters", text) != QMessageBox.StandardButton.Yes:
                return
            config.replace_yaml(self.path, updated, self.original)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "Parameters", str(exc))
            return
        self._reload()
