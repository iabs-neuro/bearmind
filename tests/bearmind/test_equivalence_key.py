"""Fast checks of the equivalence cache key and the pool watchdog (equivalence_runs.py); no data needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import equivalence_runs as er  # noqa: E402


def _repo(root):
    for rel in ("bearmind/__init__.py", "bearmind/stages/cnmf.py", "bm_qc.py", "bm_batch_routines.py",
                "tests/bearmind/equivalence_runs.py", "tests/bearmind/test_other.py", "docs/notes.md",
                "bearmind/__pycache__/cnmf.cpython-312.pyc"):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# {rel}\n", encoding="utf-8")
    return root


def test_code_sig_follows_every_package_and_root_module(tmp_path):
    root = _repo(tmp_path)
    base = er.code_sig(root)
    for rel in ("bm_qc.py", "bearmind/stages/cnmf.py", "bearmind/__init__.py", "bm_batch_routines.py",
                "tests/bearmind/equivalence_runs.py"):
        f = root / rel
        text = f.read_text(encoding="utf-8")
        f.write_text(text + "x = 1\n", encoding="utf-8")
        assert er.code_sig(root) != base, rel
        f.write_text(text, encoding="utf-8")
        assert er.code_sig(root) == base, rel
    (root / "bearmind" / "new_module.py").write_text("", encoding="utf-8")
    assert er.code_sig(root) != base


def test_code_sig_ignores_files_that_never_run(tmp_path):
    root = _repo(tmp_path)
    base = er.code_sig(root)
    for rel in ("tests/bearmind/test_other.py", "docs/notes.md", "bearmind/__pycache__/cnmf.cpython-312.pyc"):
        (root / rel).write_text("changed\n", encoding="utf-8")
    assert er.code_sig(root) == base


def test_code_files_of_the_repo():
    names = {f.relative_to(er.REPO).as_posix() for f in er.code_files()}
    assert {"bm_qc.py", "bm_batch_routines.py", "auto_inspector.py", "bearmind/stages/cnmf.py",
            "bearmind/registry.py", "bearmind/config.py", "tests/bearmind/equivalence_runs.py"} <= names
    assert "tests/bearmind/test_equivalence.py" not in names
    assert er.code_sig() == er.code_sig()


def test_pool_watch_flags_a_replaced_worker_only():
    w = er.PoolWatch()
    assert not w.update({1})
    assert not w.update({1, 2, 3})  # pool start spread over two samples
    assert not w.update({1, 2})  # 3 died; no replacement yet
    assert w.update({1, 2, 4}) == {3}  # replacement in a later sample

    w = er.PoolWatch()
    w.update({1, 2, 3})
    assert w.update({1, 2, 4}) == {3}  # death and replacement within one sample

    w = er.PoolWatch()
    w.update({1, 2, 3})
    assert not w.update({1})  # terminate: workers go, none join
    assert not w.update(set())
    assert not w.update({7, 8})  # a later pool is a new pool
    assert not w.update({7, 8})
