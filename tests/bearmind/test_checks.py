import pytest

from bearmind.checks import evaluate

CFG = {"checks": {
    "mc_max_shift_px": {"warn": 15, "error": 30},
    "mc_template_corr": {"warn": 0.6},
    "cnmf_n_components_vs_median": {"warn": 2.0, "error": 3.0},
    "cnmf_missed_cells": {"warn": 20},
    "auto_deleted_frac": {"warn": 0.8, "error": 0.95},
    "export_n_neurons": {"warn": 30, "error": 5},
    "export_median_snr": {"warn": 3.0},
    "min_sessions_for_relative": 3,
}}


def by_name(checks):
    return {c.name: c for c in checks}


def test_shift_warning_and_error():
    c = by_name(evaluate("mc", {"max_shift_px": 18.0}, [], CFG))["mc_max_shift_px"]
    assert c.level == "warning" and c.stage == "mc" and c.value == 18.0
    assert c.message == "max shift 18.0 px (warn > 15, error > 30)"
    c = by_name(evaluate("mc", {"max_shift_px": 31.0}, [], CFG))["mc_max_shift_px"]
    assert c.level == "error"


def test_shift_ok_and_low_corr_warning():
    res = by_name(evaluate("mc", {"max_shift_px": 3.0, "mean_template_corr": 0.5}, [], CFG))
    assert res["mc_max_shift_px"].level == "ok"
    assert res["mc_template_corr"].level == "warning"


def test_components_relative_error():
    series = [{"n_components": n} for n in (600, 612, 700, 2950)]
    c = by_name(evaluate("cnmf", {"n_components": 2950}, series, CFG))["cnmf_n_components_vs_median"]
    assert c.level == "error"
    assert "4.5x" in c.message and "(656)" in c.message


def test_components_message_example():
    series = [{"n_components": 600}, {"n_components": 612}, {"n_components": 2950}]
    c = by_name(evaluate("cnmf", {"n_components": 2950}, series, CFG))["cnmf_n_components_vs_median"]
    assert c.message.startswith("2950 components, 4.8x series median (612)")
    assert c.level == "error"


def test_no_relative_check_with_two_sessions():
    series = [{"n_components": 600}, {"n_components": 2950}]
    res = by_name(evaluate("cnmf", {"n_components": 2950, "missed_cells": 5}, series, CFG))
    assert "cnmf_n_components_vs_median" not in res
    assert res["cnmf_missed_cells"].level == "ok"


def test_few_neurons_error_and_warning():
    c = by_name(evaluate("export", {"n_neurons": 4}, [], CFG))["export_n_neurons"]
    assert c.level == "error"
    assert c.message == "4 neurons (warn < 30, error < 5)"
    assert by_name(evaluate("export", {"n_neurons": 12}, [], CFG))["export_n_neurons"].level == "warning"


def test_missing_or_none_metric_gives_no_check():
    assert evaluate("export", {"n_neurons": None}, [], CFG) == []
    assert evaluate("autoinspect", {}, [], CFG) == []


def test_stage_without_checks():
    for s in ("crop", "thresholds", "gate"):
        assert evaluate(s, {"x": 1}, [], CFG) == []


def test_autoinspect_deleted_frac():
    c = by_name(evaluate("autoinspect", {"deleted_frac": 0.9}, [], CFG))["auto_deleted_frac"]
    assert c.level == "warning"
