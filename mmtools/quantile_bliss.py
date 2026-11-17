"""
Bliss independence on whole distributions (quantile functions) instead of means.

For a variable x (velocity v, rate λ or length L of each mother in a window, see growth.py), each condition c is
described by its quantile function Q_c(u) on N levels u = (k - 1/2)/N, in units of the control median.

    Bliss:        Q_Bliss(u) = Q_ctrl(u) ρ_A(u) ρ_B(u),   ρ_c(u) = Q_c(u) / Q_ctrl(u)
    plane fit:    Q_AB(u) ≈ Q_ctrl(u) ρ_A(u)^α ρ_B(u)^β       (α, β) = (1, 1) Bliss, (0, 1) B alone, (1, 0) A alone

Vectors (linear effects): e_c(u) = Q_c(u) - Q_ctrl(u), inner product <v, w> = (1/N) Σ_u v(u) w(u), on the levels where
Q_ctrl, Q_A and Q_B are all ≥ v_min. In linear units Bliss is e_A + e_B + e_A e_B / Q_ctrl (a cross term), so the three
vectors e_A, e_B, e_Bliss span a 3-d space where the observed effect is projected.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

VARIABLES = {'v': 'velocity v', 'lam': 'rate λ', 'L': 'length L'}
REFS = {'Bliss': (1, 1), 'B alone': (0, 1), 'A alone': (1, 0), 'no effect': (0, 0)}


def levels(n):
    return (np.arange(n) + 0.5) / n


# ============================================================================ quantile functions
def _area_arrays(W, var, roles):
    """condition role -> list of arrays (one per area) of the per-mother values"""
    out = {}
    for r in ('control', 'A', 'B', 'combination'):
        d = W[W.condition == roles[r]]
        unit = d.experiment.astype(str) + '|' + d.area.astype(str)
        out[r] = [g[var].to_numpy() for _, g in d.groupby(unit)]
    return out


def quantile_functions(samples, n_levels=100):
    """samples: role -> array of values. Returns (Q: role -> quantile function in control-median units, scale)"""
    lev = levels(n_levels)
    scale = float(np.median(samples['control']))
    return {r: np.quantile(x, lev) / scale for r, x in samples.items()}, scale


# ============================================================================ plane fit
def rms(x, n):
    return float(np.sqrt(np.sum(np.asarray(x) ** 2) / n))


def plane_fit(Q, v_min=0.05, min_effect=0.1, free=None, bounds=(-3.0, 5.0)):
    """
    Least squares in the units of the variable: S(α, β) = (1/N) Σ_u [Q_ctrl ρ_A^α ρ_B^β - Q_AB]², on the levels where
    Q_ctrl, Q_A, Q_B ≥ v_min. A drug whose effect alone (RMS of Q_drug - Q_ctrl) is below min_effect has its exponent
    fixed at 1 ('free' decides which exponents are estimated; None = decide on these data).
    """
    q0, qA, qB, qO = Q['control'], Q['A'], Q['B'], Q['combination']
    N = len(q0)
    m = (q0 >= v_min) & (qA >= v_min) & (qB >= v_min)
    base, y = q0[m], qO[m]
    lA, lB = np.log(qA[m] / base), np.log(qB[m] / base)
    size_A, size_B = rms(qA[m] - base, N), rms(qB[m] - base, N)
    if free is None:
        free = (size_A >= min_effect, size_B >= min_effect)
    free = tuple(bool(f) for f in free)

    def coef(th):
        it = iter(th)
        return [next(it) if f else 1.0 for f in free]

    model = lambda a, b: base * np.exp(a * lA + b * lB)
    k = sum(free)
    if k:
        th = least_squares(lambda th: model(*coef(th)) - y, np.ones(k), bounds=([bounds[0]] * k, [bounds[1]] * k)).x
        al, be = coef(th)
    else:
        al, be = 1.0, 1.0
    fit = model(al, be)
    JA, JB = fit * lA, fit * lB
    return dict(free=free, alpha=al, beta=be, size_A=size_A, size_B=size_B, size_obs=rms(y - base, N),
                levels_used=float(m.mean()), m=m, fit=fit, bliss=model(1, 1),
                dist_fit=rms(y - fit, N), dist_bliss=rms(y - model(1, 1), N), dist_A=rms(y - qA[m], N), dist_B=rms(y - qB[m], N),
                corr_fixed=-float(np.sum(JA * JB) / np.sqrt(np.sum(JA ** 2) * np.sum(JB ** 2))) if all(free) else np.nan)


def maha2(draws, free, point):
    """squared Mahalanobis distance of a reference point (α, β) to the bootstrap cloud of the estimated exponents"""
    if any((not f) and abs(p - 1) > 1e-9 for f, p in zip(free, point)):
        return np.nan                                     # the point needs another value of an exponent fixed at 1
    idx = [i for i, f in enumerate(free) if f]
    if not idx:
        return np.nan
    X = draws[:, idx]; d = np.array(point, float)[idx] - X.mean(0)
    S = np.atleast_2d(np.cov(X.T))
    return float(d @ np.linalg.solve(S, d))


# ============================================================================ 3-d vectors
def vectors_3d(Q, est):
    """
    Linear effects e_c = Q_c - Q_ctrl on the levels of the fit, and their coordinates in the orthonormal basis
    u1 ∥ e_A, u2 completing the plane (e_A, e_B), u3 = part of e_Bliss outside that plane (Gram-Schmidt).
    Returns dict with coordinates (A, B, sum, Bliss, obs, fit), the part of e_obs outside the 3-d space, and a table.
    """
    m = est['m']; N = len(Q['control'])
    q0 = Q['control'][m]
    e = {'A': Q['A'][m] - q0, 'B': Q['B'][m] - q0, 'obs': Q['combination'][m] - q0,
         'Bliss': est['bliss'] - q0, 'fit': est['fit'] - q0}
    e['sum'] = e['A'] + e['B']
    ip = lambda a, b: float(np.sum(a * b) / N)
    nr = lambda a: np.sqrt(max(ip(a, a), 0.0))
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(ip(a, b) / (nr(a) * nr(b)), -1, 1)))) if nr(a) * nr(b) > 0 else np.nan
    U = []
    for v in (e['A'], e['B'], e['Bliss']):
        w = v - sum(ip(v, u) * u for u in U)
        if nr(w) > 1e-9 * max(nr(v), 1e-12):
            U.append(w / nr(w))
    xyz = lambda v: np.array([ip(v, u) for u in U] + [0.0] * (3 - len(U)))
    P = {k: xyz(v) for k, v in e.items()}
    out_obs = float(np.sqrt(max(ip(e['obs'], e['obs']) - P['obs'] @ P['obs'], 0.0)))
    table = {'|e_A|': nr(e['A']), '|e_B|': nr(e['B']), 'angle A–B (°)': ang(e['A'], e['B']),
             '|cross term|': nr(e['Bliss'] - e['sum']), 'cross term out of plane (A,B)': abs(P['Bliss'][2]),
             '|e_Bliss|': nr(e['Bliss']), '|e_obs|': nr(e['obs']),
             'angle obs–A (°)': ang(e['obs'], e['A']), 'angle obs–B (°)': ang(e['obs'], e['B']),
             'angle obs–Bliss (°)': ang(e['obs'], e['Bliss']),
             'dist obs–A': nr(e['obs'] - e['A']), 'dist obs–B': nr(e['obs'] - e['B']),
             'dist obs–Bliss': nr(e['obs'] - e['Bliss']), 'dist obs–plane fit': nr(e['obs'] - e['fit']),
             'obs outside the 3-d space': out_obs}
    return dict(P=P, e=e, out_obs=out_obs, table=table)


# ============================================================================ one variable, with bootstrap
@dataclass
class QuantileBliss:
    var: str
    roles: dict
    lev: np.ndarray
    Q: dict                       # role -> quantile function (control-median units)
    scale: float                  # control median in the units of the variable
    est: dict                     # plane fit, exponents of negligible drugs fixed at 1
    est_free: dict                # plane fit, both exponents free
    draws: np.ndarray             # bootstrap (α, β), same 'free' as est
    draws_free: np.ndarray        # bootstrap (α, β), both free
    vec: dict                     # 3-d vectors
    settings: dict = field(default_factory=dict)

    def summary(self):
        e, ef, d, df = self.est, self.est_free, self.draws, self.draws_free
        A, B = self.roles['A'], self.roles['B']
        ci = lambda x, s, f=True: (f'{x:+.2f} [{np.percentile(s, 2.5):+.2f}, {np.percentile(s, 97.5):+.2f}]'
                                   if f else 'not estimated (fixed at 1)')
        thr = chi2.ppf(0.95, sum(e['free'])) if any(e['free']) else np.nan
        row = {'variable': VARIABLES[self.var], f'‖e_{A}‖': e['size_A'], f'‖e_{B}‖': e['size_B'],
               f'α ({A})': ci(e['alpha'], d[:, 0], e['free'][0]), f'β ({B})': ci(e['beta'], d[:, 1], e['free'][1]),
               'α, both free': ci(ef['alpha'], df[:, 0]), 'β, both free': ci(ef['beta'], df[:, 1]),
               'corr(α, β), both free': float(np.corrcoef(df.T)[0, 1]),
               'D² Bliss (1,1)': maha2(d, e['free'], (1, 1)), f'D² {B} alone (0,1)': maha2(d, e['free'], (0, 1)),
               f'D² {A} alone (1,0)': maha2(d, e['free'], (1, 0)), 'D² threshold (95 %)': thr,
               'dist to Bliss': e['dist_bliss'], f'dist to {A} alone': e['dist_A'], f'dist to {B} alone': e['dist_B'],
               'dist to plane fit': e['dist_fit'], 'levels used': e['levels_used']}
        return row


def quantile_bliss(W, var, roles, n_levels=100, v_min=0.05, min_effect=0.1, n_boot=500, bounds=(-3.0, 5.0), seed=2,
                   progress=None):
    """Bliss on the quantile functions of one variable, (α, β) plane with bootstrap over areas, 3-d vectors."""
    arrays = _area_arrays(W, var, roles)
    Q, scale = quantile_functions({r: np.concatenate(a) for r, a in arrays.items()}, n_levels)
    est = plane_fit(Q, v_min, min_effect, bounds=bounds)
    est_free = plane_fit(Q, v_min, min_effect, free=(True, True), bounds=bounds)
    rng = np.random.default_rng(seed)
    dr, drf = [], []
    for b in range(n_boot):
        if progress: progress(b, n_boot)
        smp = {r: np.concatenate([a[i] for i in rng.integers(0, len(a), len(a))]) for r, a in arrays.items()}
        Qb, _ = quantile_functions(smp, n_levels)
        fb = plane_fit(Qb, v_min, min_effect, free=est['free'], bounds=bounds)
        fbf = plane_fit(Qb, v_min, min_effect, free=(True, True), bounds=bounds)
        dr.append((fb['alpha'], fb['beta'])); drf.append((fbf['alpha'], fbf['beta']))
    if progress: progress(n_boot, n_boot)
    return QuantileBliss(var, dict(roles), levels(n_levels), Q, scale, est, est_free, np.array(dr), np.array(drf),
                         vectors_3d(Q, est_free),
                         dict(n_levels=n_levels, v_min=v_min, min_effect=min_effect, n_boot=n_boot, bounds=bounds))
