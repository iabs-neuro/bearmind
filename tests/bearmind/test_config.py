import os
import tempfile
from pathlib import Path

import yaml

from bearmind.config import (load_defaults, machine_config, params_hash,
                             resolve, update_yaml)
from bearmind.paths import SeriesPaths


def test_merge_order(tmp_path):
    p = SeriesPaths(tmp_path)
    update_yaml(p.series_config, {"cnmf": {"merge_thr": 0.7}})
    update_yaml(p.mouse_config("NOF_H01"), {"crop": {"box": [400, 420]}})
    update_yaml(p.session_config("NOF_H01_1D"), {"cnmf": {"gSig": 5, "min_corr": 0.85, "min_pnr": 8}})
    cfg = resolve(p, sid="NOF_H01_1D")
    assert cfg["cnmf"]["merge_thr"] == 0.7 and cfg["cnmf"]["gSig"] == 5
    assert cfg["crop"]["box"] == [400, 420]
    assert cfg["autoinspect"]["pxlthr_distance_boundary"] == 15
    assert cfg["cnmf"]["tsub"] == 1 and "fr" not in cfg["cnmf"]
    assert "machine" in cfg


def test_hash_stable_and_sensitive():
    assert params_hash({"a": 1, "b": [1, 2]}) == params_hash({"b": [1, 2], "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_defaults_int_keys_and_values():
    d = load_defaults()
    assert d["memory"]["patch_gb_per_mframe"] == {1: 245, 2: 163}
    assert all(isinstance(k, int) for k in d["memory"]["patch_gb_per_mframe"])
    assert d["mc"]["max_shifts"] == [35, 35] and "write_mp4" not in d["mc"]
    assert d["autoinspect"]["ml_threshold"] == 0.72 and d["gate"]["enabled"] is True
    assert d["checks"]["mc_max_shift_px"] == {"warn": 15, "error": 30}


def test_lists_replace_and_missing_files_empty(tmp_path):
    p = SeriesPaths(tmp_path)
    update_yaml(p.series_config, {"mc": {"max_shifts": [10]}})
    cfg = resolve(p)
    assert cfg["mc"]["max_shifts"] == [10]
    assert cfg["mc"]["strides"] == [48, 48]


def test_resolve_derives_mouse_from_sid(tmp_path):
    p = SeriesPaths(tmp_path)
    update_yaml(p.mouse_config("NOF_H01"), {"crop": {"margin_px": 3}})
    assert resolve(p, sid="NOF_H01_2D")["crop"]["margin_px"] == 3
    assert resolve(p, mouse="NOF_H01")["crop"]["margin_px"] == 3
    assert resolve(p)["crop"]["margin_px"] == 10


def test_update_yaml_merges_atomically(tmp_path):
    f = tmp_path / "a" / "b.yaml"
    update_yaml(f, {"x": {"a": 1}, "l": [1]})
    update_yaml(f, {"x": {"b": 2}, "l": [2, 3]})
    assert yaml.safe_load(f.read_text()) == {"x": {"a": 1, "b": 2}, "l": [2, 3]}
    assert not f.with_name(f.name + ".tmp").exists()


def test_machine_config(tmp_path):
    m = machine_config()
    home = Path(os.environ["BEARMIND_HOME"])
    assert (home / "machine.yaml").exists()
    assert m["temp_dir"] == str(Path(tempfile.gettempdir()) / "bearmind")
    assert m["cnmf_ram_budget_gb"] == 0.8 * m["ram_gb"]
    assert m["n_cpus"] >= 1
    update_yaml(home / "machine.yaml", {"n_cpus": 3})
    assert machine_config()["n_cpus"] == 3
