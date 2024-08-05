import numpy as np
import pandas as pd

ALLOWED_FPS = [10, 20, 30, 40, 50]
MAX_UNIQUE_VALS = 0.05  # if more than X% vals are unique in constant/categorical data, this is strange


def find_time_column(df, verbose=True):
    cols = df.columns.values
    try:
        time_col = cols[np.array(['time' in str(c.lower()) for c in cols])][0]
        if verbose:
            print(f'Assuming column "{time_col}" with index {list(cols).index(time_col)} as a timeline')

    except IndexError:
        if verbose:
            print('No specified time column found, searching for a timeline without name...')
        candidates = []
        counters = []
        for i, col in enumerate(cols):
            data = df[col].values
            try:
                if np.all(np.diff(data)>0):
                    candidates.append(i)
                    if set(np.diff(data)) == {1}:
                        counters.append(i)
            except:
                pass

        if len(candidates) == 0:
            if verbose:
                print('No column in the data looks like a timeline, assuming data has no timestamps')
            time_col = None

        else:
            if verbose:
                print(f'Columns with indices {candidates} can be interpreted as timelines')
                if len(counters) != 0:
                    print(f'Columns with indices {counters} look suspiciously like integer counters')

            final = list(set(candidates) - set(counters))
            time_col = cols[final[0]]

            if verbose:
                if len(final) == 1:
                    print(f'Assuming column "{time_col}" with index {final[0]} as a timeline')
                else:
                    print(f'Assuming the first column "{time_col}" with index {final[0]} out of possible index set {final} as a timeline, this may cause errors')

    return time_col


def _calc_fps(vals, verbose):
    diffs = np.diff(vals)
    unique_vals = np.unique(diffs)
    sc = 1.0*len(unique_vals)/len(diffs)
    if sc > MAX_UNIQUE_VALS and verbose:
        print('timestamp differences seem too diverse, auto fps detection may be erroneous')

    fps = -1
    decimal_multipliers = [10**_ for _ in range(8)]
    for dm in decimal_multipliers:
        fps = np.round(np.mean([dm/d for d in diffs]),2)
        if (min(ALLOWED_FPS) <= fps) and (fps < max(ALLOWED_FPS)):
            break

    return fps


def calc_fps(df, time_col, verbose=True):
    try:
        if time_col is not None:
            fps = _calc_fps(df[time_col].values, verbose=verbose)
            if verbose:
                print(f'Automatically determined fps: {fps}')
                if fps not in ALLOWED_FPS:
                    print(f'Automatically determined fps {fps} is suspicious, consider manual check!')

            return fps

    except Exception:
        print('fps computation failed')
        return None


def get_fps(df, verbose=True):
    time_col = find_time_column(df, verbose=verbose)
    fps = calc_fps(df, time_col, verbose=verbose)
    return fps
