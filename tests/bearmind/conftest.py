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


@pytest.fixture
def make_series(make_avi):
    """Factory building a tiny series: bonsai (nested, TS in 0_TimeStamps), legacy or legacy_split."""
    import json

    import numpy as np

    def _ts_bonsai(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{i * 500000}\n" for i in range(rows)), encoding="utf-8")

    def _ts_legacy(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Frame Number,Time Stamp (ms),Buffer Index\n"
                        + "".join(f"{i},{i * 50},0\n" for i in range(rows)), encoding="utf-8")

    def _chunks(folder, names, n_frames):
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for nm in names:
            make_avi(folder / nm, rng.integers(0, 255, (n_frames, 16, 16), dtype=np.uint8))

    def _make(root, layout="bonsai", sid=None, n_chunks=3, chunk_frames=10, ts_rows=None,
              with_ts=True, with_meta=True, with_ts_in_session=False):
        raw = root / "1_Raw"
        rows = n_chunks * chunk_frames if ts_rows is None else ts_rows
        if layout == "bonsai":
            sid = sid or "NOF_H01_1D"
            folder = raw / "HM_NOF_1D" / sid
            _chunks(folder, [f"{sid}_{i}.avi" for i in range(n_chunks)], chunk_frames)
            if with_ts:
                where = folder if with_ts_in_session else root / "0_TimeStamps"
                _ts_bonsai(where / f"{sid}_Mini_TS.csv", rows)
        else:
            sid = sid or "LNOF_J01_1D"
            starts = ["10_00_00"] if layout == "legacy" else ["10_00_00", "11_00_00"]
            for k, hms in enumerate(starts):
                mini = raw / sid / "2024_10_15" / hms / "Miniscope"
                _chunks(mini, [f"{i}.avi" for i in range(n_chunks)], chunk_frames)
                if with_ts:
                    _ts_legacy(mini / "timeStamps.csv", rows)
                (mini / "metaData.json").write_text(json.dumps({"frameRate": 20}), encoding="utf-8")
                if with_meta:
                    (mini.parent / "metaData.json").write_text(json.dumps(
                        {"recordingStartTime": {"msecSinceEpoch": 1728989165178 + k * 3600_000}}),
                        encoding="utf-8")
        return root

    return _make
