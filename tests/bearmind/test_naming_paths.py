from bearmind.naming import session_id, mouse_id
from bearmind.paths import SeriesPaths


def test_session_id_and_mouse():
    assert session_id("x/NOF_H01_1D_0.avi") == "NOF_H01_1D"
    assert session_id("HOS_D04_1D_2T") == "HOS_D04_1D_2T"
    assert session_id("readme.txt") is None
    assert mouse_id("LNOF_J01_1D") == "LNOF_J01"


def test_paths_layout(tmp_path):
    p = SeriesPaths(tmp_path / "серия 1")
    assert p.mc_video("NOF_H01_1D").relative_to(p.root).as_posix() == "3_MC/NOF_H01_1D_MC.avi"
    assert p.export_filters("NOF_H01_1D").relative_to(p.root).as_posix() == "6_Export/filters/NOF_H01_1D_filters.mat"
    assert p.preview("NOF_H01_1D", "mc_dff").name == "NOF_H01_1D_mc_dff.mp4"
    assert p.rel(p.estimates("NOF_H01_1D")) == "4_Estimates/NOF_H01_1D_estimates.pickle"


def test_threshold_choices_path(tmp_path):
    assert SeriesPaths(tmp_path).threshold_choices_csv == tmp_path / "_bearmind" / "threshold_choices.csv"


def test_inspection_paths(tmp_path):
    p = SeriesPaths(tmp_path)
    assert p.rel(p.inspection_lock) == "_bearmind/inspect.lock"
    assert p.rel(p.inspection_progress_dir("NOF_H01_1D")) == "5_Inspection/NOF_H01_1D/progress"
    assert p.rel(p.inspection_cache("NOF_H01_1D")) == "5_Inspection/NOF_H01_1D/NOF_H01_1D_inspect_cache.npz"
    assert p.inspection_cache("NOF_H01_1D").parent == p.inspection_dir("NOF_H01_1D") == \
        p.manual_result("NOF_H01_1D").parent
