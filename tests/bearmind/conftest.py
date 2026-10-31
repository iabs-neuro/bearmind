import pytest


@pytest.fixture(autouse=True)
def _isolated_bearmind_home(tmp_path_factory, monkeypatch):
    """Keep machine.yaml out of the real home directory in every test."""
    monkeypatch.setenv("BEARMIND_HOME", str(tmp_path_factory.mktemp("bearmind_home")))


@pytest.fixture
def make_avi():
    """Factory writing a grey uint8 (T, h, w) array to an AVI via PyAV."""
    from fractions import Fraction

    import av

    def _make(path, frames, codec="ffv1", fps=20):
        with av.open(str(path), "w", format="avi") as c:
            s = c.add_stream(codec, rate=Fraction(fps))
            s.height, s.width = frames.shape[1:]
            s.pix_fmt = "gray" if codec == "ffv1" else "yuvj420p"
            for f in frames:
                fr = av.VideoFrame.from_ndarray(f, format="gray")
                for p in s.encode(fr.reformat(format=s.pix_fmt)):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        return path

    return _make
