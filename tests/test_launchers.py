"""The two student launchers find the bearmind env the same way: five env folders, then `conda run`.

run_bearmind.bat runs a command in a console; bearmind_gui.bat starts the window with pythonw (no console). Both look
in the per-user and all-users Anaconda and Miniconda folders and in %USERPROFILE%\\.conda\\envs (where conda puts a
user's envs when Anaconda is installed for all users). When none holds the env, both let conda find it.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_BAT, GUI_BAT = REPO / "run_bearmind.bat", REPO / "bearmind_gui.bat"
FAILED = '[FAILED] conda env "bearmind" not found.'

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launchers")


def _text(bat: Path) -> str:
    return bat.read_text(encoding="ascii")


def _env_folders(bat: Path) -> list:
    m = re.search(r"^for %%P in \((.*)\) do if not defined PY", _text(bat), re.M)
    assert m, f"{bat.name}: no loop over env folders"
    return [re.sub(r"\\pythonw?\.exe$", "", p, flags=re.I) for p in re.findall(r'"([^"]+)"', m[1])]


def test_launchers_search_the_same_env_folders():
    run, gui = _env_folders(RUN_BAT), _env_folders(GUI_BAT)
    assert run == gui
    assert r"%USERPROFILE%\.conda\envs\bearmind" in run
    assert len(run) == 5


def test_launchers_fall_back_to_conda_run():
    conda_lines = []
    for bat in (RUN_BAT, GUI_BAT):
        text = _text(bat)
        assert "if not defined PY goto conda_run" in text and "\n:conda_run" in text, bat.name
        assert re.search(r'^call "%CONDA_CMD%" run --no-capture-output -n bearmind python -m bearmind (gui )?%\*$',
                         text, re.M), bat.name
        assert FAILED in text, bat.name
        # `conda run` reads CONDA_BAT as the path of conda.bat: a launcher variable of that name breaks it
        assert not re.search(r"^\s*set \"?CONDA_BAT", text, re.M | re.I), bat.name
        conda_lines.append([line for line in text.splitlines() if line.lstrip().startswith(('set "CONDA_CMD=',
                                                                                         "if not defined CONDA_CMD"))])
    assert conda_lines[0] == conda_lines[1] and conda_lines[0]


def _fake_home(tmp_path: Path):
    """An environment whose user folder and ProgramData hold no conda: the launchers see no env and no conda."""
    home, programdata = tmp_path / "home", tmp_path / "programdata"
    home.mkdir()
    programdata.mkdir()
    env = dict(os.environ)  # on Windows the keys are upper case
    env.update(USERPROFILE=str(home), PROGRAMDATA=str(programdata))
    env.pop("CONDA_EXE", None)
    return home, env


def _bat(bat: Path, env: dict, cwd: Path, *args):
    return subprocess.run(["cmd", "/c", str(bat), *args], env=env, cwd=cwd, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=180)


def test_run_launcher_finds_the_env_in_the_user_conda_folder(tmp_path):
    home, env = _fake_home(tmp_path)
    miss = _bat(RUN_BAT, env, tmp_path, "--help")
    assert miss.returncode == 1 and FAILED in miss.stdout  # the fake folders hide this machine's env
    link = home / ".conda" / "envs" / "bearmind"
    link.parent.mkdir(parents=True)
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), sys.prefix], capture_output=True, text=True)
    if made.returncode:
        pytest.skip(f"cannot create a junction here: {made.stdout} {made.stderr}")
    try:
        found = _bat(RUN_BAT, env, tmp_path, "--help")
    finally:
        os.rmdir(link)  # removes the junction only, never the env it points to
    assert found.returncode == 0 and "usage: run_bearmind [-h] command" in found.stdout, found.stdout + found.stderr


@pytest.mark.parametrize("bat,usage", [(RUN_BAT, "usage: run_bearmind [-h] command"),
                                       (GUI_BAT, "usage: run_bearmind gui")])
def test_launcher_falls_back_to_conda_run(tmp_path, bat, usage):
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    prefix = Path(sys.prefix)
    if not conda or prefix.name != "bearmind" or not (prefix.parent.parent / "condabin").is_dir():
        pytest.skip("needs the bearmind env in the envs folder of a conda install, and conda")
    _, env = _fake_home(tmp_path)
    env["CONDA_EXE"] = conda
    out = _bat(bat, env, tmp_path, "--help")
    assert out.returncode == 0 and usage in out.stdout, out.stdout + out.stderr
