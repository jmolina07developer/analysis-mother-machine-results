"""Per-mother trajectories on a regular time grid."""
import numpy as np
import pandas as pd

KEYS = ['condition', 'experiment', 'area', 'channel', 'mother']


def time_grid(t_end, step):
    """grid points step, 2·step, ..., t_end (min)"""
    return np.arange(step, t_end + 1e-9, step)


def default_t_end(LM, step):
    """the last grid point that every entry (condition × experiment) still has images for"""
    last = LM.groupby(['condition', 'experiment']).t.max().min()
    return float(np.floor(last / step) * step)


def _flags(g, grid, empty_start_images):
    t = g.t.to_numpy()
    return dict(empty_start=bool(g['empty'].iloc[:empty_start_images].all()),
                lost=bool(g.lost.to_numpy()[t <= grid[-1]].any()),
                first_image=t[0])


def division_vectors(LM, grid, empty_start_images=3):
    """
    One row per mother:
        N            cumulative number of divisions seen up to each grid point, e.g. [0 0 1 1 1 2 2 ...]
        lost         lost by the tracker (or channel emptied) at or before the end of the grid
        empty_start  channel empty in its first `empty_start_images` images (no mother: left out)
        first_image  time of the first image (min)
    A division is counted at the first image where it is seen (the daughter's length is written there).
    Divisions after the mother is lost are not counted.
    """
    rows = []
    for key, g in LM.groupby(KEYS, sort=False):
        g = g.sort_values('image')
        td = g.t.to_numpy()[(g.divides & ~g.lost).to_numpy()]
        N = (td[None, :] <= grid[:, None]).sum(1)
        rows.append(dict(zip(KEYS, key), **_flags(g, grid, empty_start_images), N=N.astype(float)))
    return pd.DataFrame(rows)


def length_vectors(LM, grid, max_age=12, empty_start_images=3):
    """
    One row per mother:
        L        length of the mother (px) at each grid point: that of the last image at or before it, if it is at
                 most `max_age` min older; NaN if there is no such image, no reliable length, the mother is past
                 the exit or already lost
        exit     bool per grid point: the mother is past the channel exit there (it has no length, it is longer
                 than the channel)
        lost, empty_start, first_image   as in division_vectors
    """
    rows = []
    for key, g in LM.groupby(KEYS, sort=False):
        g = g.sort_values('image')
        t = g.t.to_numpy()
        k = np.searchsorted(t, grid, side='right') - 1
        kk = np.clip(k, 0, None)
        ok = (k >= 0) & (grid - t[kk] <= max_age)
        lost = g.lost.to_numpy()
        ex = (g.past_exit.to_numpy() & ~lost)[kk] & ok
        L = np.where(ok & ~ex & ~lost[kk], g.L.to_numpy()[kk], np.nan)
        rows.append(dict(zip(KEYS, key), **_flags(g, grid, empty_start_images), L=L.astype(float), exit=ex,
                         ever_exit=bool(ex.any())))
    return pd.DataFrame(rows)


def fill_gaps(x, max_gap):
    """
    Fill runs of NaN no longer than `max_gap` points: linear interpolation inside, nearest value at the two
    ends. Returns None if a longer run is missing (or everything is).
    """
    x = np.array(x, float); bad = np.isnan(x)
    if not bad.any():
        return x
    if bad.all():
        return None
    runs = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    if (runs[1::2] - runs[::2]).max() > max_gap:
        return None
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def as_word(n):
    """[0, 0, 1, 1, 2] -> '00112' (10 and more are written in brackets)"""
    return ''.join(str(int(x)) if x < 10 else f'[{int(x)}]' for x in n)
