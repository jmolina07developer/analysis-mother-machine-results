"""Per-mother trajectories on a regular time grid."""
import numpy as np
import pandas as pd


def time_grid(t_end, step):
    """grid points step, 2·step, ..., t_end (min)"""
    return np.arange(step, t_end + 1e-9, step)


def default_t_end(LM, step):
    """the last grid point that every entry (condition × experiment) still has images for"""
    last = LM.groupby(['condition', 'experiment']).t.max().min()
    return float(np.floor(last / step) * step)


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
    for (c, e, a, ch, m), g in LM.groupby(['condition', 'experiment', 'area', 'channel', 'mother'], sort=False):
        g = g.sort_values('image')
        t = g.t.to_numpy()
        td = t[(g.divides & ~g.lost).to_numpy()]
        N = (td[None, :] <= grid[:, None]).sum(1)
        rows.append(dict(condition=c, experiment=e, area=a, channel=ch, mother=m,
                         empty_start=bool(g['empty'].iloc[:empty_start_images].all()),
                         lost=bool(g.lost.to_numpy()[t <= grid[-1]].any()),
                         first_image=t[0], N=N.astype(float)))
    return pd.DataFrame(rows)


def as_word(n):
    """[0, 0, 1, 1, 2] -> '00112' (10 and more are written in brackets)"""
    return ''.join(str(int(x)) if x < 10 else f'[{int(x)}]' for x in n)
