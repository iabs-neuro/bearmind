"""Regression: OpenBLAS 0.3.34 segfaults on Zen 4 in float64 matmul (inner dim ~239)."""
import os
import subprocess
import sys

_MATMUL = (
    "import bearmind, numpy as np\n"
    "rng = np.random.default_rng(0)\n"
    "a = rng.random((2000, 239)); b = rng.random((239, 600))\n"
    "c = a.dot(b)\n"
    "ref = np.einsum('ij,jk->ik', a[:50], b)\n"
    "assert np.allclose(c[:50], ref)\n"
    "print('ok')\n"
)


def _run(code, **env_over):
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_CORETYPE"}
    env.update(env_over)
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=300)


def test_matmul_after_import_bearmind_does_not_crash():
    r = _run(_MATMUL)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_import_keeps_preset_coretype():
    r = _run("import bearmind, os; print(os.environ['OPENBLAS_CORETYPE'])",
             OPENBLAS_CORETYPE="Nehalem")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "Nehalem"
