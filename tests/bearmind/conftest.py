import pytest


@pytest.fixture(autouse=True)
def _isolated_bearmind_home(tmp_path_factory, monkeypatch):
    """Keep machine.yaml out of the real home directory in every test."""
    monkeypatch.setenv("BEARMIND_HOME", str(tmp_path_factory.mktemp("bearmind_home")))
