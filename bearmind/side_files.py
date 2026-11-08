"""The side files of a manual-inspection result: traces.csv and the filter images, as the notebook's SaveResults
writes them, for the components the result keeps.

The inspection window (bm_examinator) writes them in a background thread; the inspection server writes them in a
child process, `python -m bearmind.side_files RESULT FOLDER BASE [SIGMA]`, which reads the result pickle the
server has just written. The side files of a session of 1880 components and 64 300 frames take about 90 s of
mostly Python work: in a thread of the server they would hold its GIL that long and slow the pages. This module
imports no CaImAn, no Bokeh and no torch, so the child starts in a second or two.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tfl
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

from bearmind import fileio

log = logging.getLogger("bm_examinator")  # the window's logger: the server's series log keeps it


def kept_for_side_files(estimates) -> np.ndarray:
    """The components the side files hold, in idx_components order: idx_components without those that
    autoinspection marked deleted (metrics_df['delete'] == 1), which it leaves in idx_components for the window
    to show. metrics_df names components by its 'component_idx' column, or by row number without one.
    metrics_df is in the file's numbering and is stale after a Merge in the window (capcan mode); no caller
    reaches that case, since the inspection server refuses inputs with metrics_df."""
    idx = np.asarray(estimates.idx_components, dtype=int)
    df = getattr(estimates, 'metrics_df', None)
    if df is None or 'delete' not in df.columns:
        return idx
    comps = df['component_idx'].to_numpy() if 'component_idx' in df.columns else np.arange(len(df))
    deleted = set(int(c) for c in comps[(df['delete'] == 1).to_numpy()])
    return np.array([c for c in idx if int(c) not in deleted], dtype=int)


def _write_whole(path, write):
    """write(tmp) beside path, then rename over it: a reader never finds half a file, and a failed write
    leaves the earlier file. The rename waits for a reader that holds the earlier file (fileio.replace)."""
    # Whole against a crash of this process, not against a power loss: no fsync, which for a result of
    # hundreds of MB would break the 0.5 s target of Save results.
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    try:
        write(tmp)
        fileio.replace(tmp, path)
    except BaseException:
        try:
            fileio._unlink(tmp)
        except OSError:
            # the write's own error is the one to raise
            log.warning('could not remove %s', tmp, exc_info=True)
        raise


def _filter_images(estimates, kept, sigma):
    """SaveResults' filter image of each kept component: the footprint, smoothed, scaled to uint8."""
    A = estimates.A.tocsc()[:, kept]
    shape = estimates.imax.shape[::-1]  # as SaveResults: a footprint's pixels read in C order, i.e. transposed
    ims = []
    for i in range(len(kept)):
        im = A[:, i].toarray().reshape(shape)
        if sigma:  # gaussian smoothing of neural contours, omitted if sigma=0
            im = gaussian_filter(im, sigma=sigma)
        ims.append((im * 255 / np.max(im)).astype(np.uint8))
    return ims


def _write_filters(filters, ims, written):
    """filter_NNN.tif of each image in the folder filters, then the removal of filter files of an earlier,
    larger save. written: the names written, filled as it goes."""
    filters.mkdir(exist_ok=True)
    for i, im in enumerate(ims):
        name = f'filter_{i + 1:03d}.tif'
        tfl.imwrite(filters / name, im)
        written.add(name)
    for p in filters.glob('filter_*.tif'):
        if p.name not in written:
            fileio._unlink(p)


def _first_line(e):
    return ''.join(traceback.format_exception_only(type(e), e)).strip().splitlines()[0]


def write_side_files(estimates, folder, base, sigma=3) -> None:
    """SaveResults' files for the components of kept_for_side_files, in folder (made if missing):
    <base>traces.csv (time_s and one column per component), <base>filters/filter_NNN.tif (footprints smoothed
    with sigma) and <base>filters_session.mat (the same images as one array 'A').

    The folder holds one save: filter files left by an earlier save of more components are removed. A failure
    of one of the three does not stop the others; after all were tried, one RuntimeError names what failed and
    the files of an earlier save left in its place."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    kept = kept_for_side_files(estimates)
    traces = folder / f'{base}traces.csv'
    filters = folder / f'{base}filters'
    mat = folder / f'{base}filters_session.mat'
    failed = []  # (path, exception)

    def attempt(path, write):
        try:
            write()
        except Exception as e:
            failed.append((path, e))

    def write_traces():
        stamped = np.concatenate(([estimates.time], estimates.C[kept]), axis=0)
        _write_whole(traces, lambda p: pd.DataFrame(stamped.T).to_csv(
            p, index=False, header=['time_s', *np.arange(len(kept))]))

    attempt(traces, write_traces)
    written = set()
    try:
        ims = _filter_images(estimates, kept, sigma)
    except Exception as e:
        failed += [(filters, e), (mat, e)]
    else:
        attempt(filters, lambda: _write_filters(filters, ims, written))
        attempt(mat, lambda: _write_whole(mat, lambda p: savemat(p, {"A": np.array(ims)}, appendmat=False)))
    if not failed:
        return

    left = []  # files of an earlier save where this one failed
    for path, _ in failed:
        if path == filters:
            old = [p for p in filters.glob('filter_*.tif') if p.name not in written] if filters.is_dir() else []
            if old:
                left.append(f'{len(old)} files in {filters.name}')
        elif path.is_file():
            left.append(path.name)
    what = '; '.join(f'{path.name}: {_first_line(e)}' for path, e in failed)
    earlier = f'left from an earlier save: {", ".join(left)}' if left else 'no files of an earlier save are left'
    raise RuntimeError(f'side files in {folder} not all written ({what}); {earlier}') from failed[0][1]


# --- the child process ---

class _Loaded:
    """Stands in for a CaImAn class while a result is read in the child: the side files need only its attributes
    (numpy, scipy and pandas objects), and importing CaImAn would take the child several seconds."""

    def __init__(self, *args, **kwargs):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "caiman" or module.startswith("caiman."):
            return _Loaded
        return super().find_class(module, name)


def load_result(path):
    """The estimates pickled at path, with CaImAn's classes as plain attribute holders; read with CaImAn itself
    when that fails (a CaImAn object that is not a plain holder)."""
    with open(path, "rb") as fh:
        try:
            return _Unpickler(fh).load()
        except Exception:
            log.info("%s: read again with CaImAn", path, exc_info=True)
            fh.seek(0)
            return pickle.load(fh)


def _file_name(e):
    f = getattr(e, "filename2", None) or getattr(e, "filename", None)
    return Path(os.fsdecode(f)).name if isinstance(f, (str, bytes, os.PathLike)) else None


def main(argv=None) -> int:
    """The child: RESULT FOLDER BASE [SIGMA]. On a failure, one JSON line on stdout names the error's type and
    file (the parent's page shows them; a Windows error text is localized) and stderr has the traceback."""
    argv = sys.argv[1:] if argv is None else argv
    source, folder, base = argv[0], argv[1], argv[2]
    sigma = float(argv[3]) if len(argv) > 3 else 3
    try:
        write_side_files(load_result(source), folder, base, sigma)
    except Exception as e:
        cause = e.__cause__ if e.__cause__ is not None else e
        print(json.dumps({"type": type(cause).__name__, "file": _file_name(cause) or _file_name(e),
                          "error": _first_line(e)}), flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
