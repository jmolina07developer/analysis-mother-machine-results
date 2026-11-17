"""
Elongation of the mother cells split into rate × length, and Bliss independence of each part.

Between two consecutive images of a mother, with L1 the length in the first image and X2 = L2 + (daughter
released in between) the length it would have in the second without dividing:

    velocity  v = (X2 - L1) / Δt                       new length per minute
    rate      λ = ln(X2 / L1) / Δt                     new length per minute per unit of length (1/min)
    length    L̃ = (X2 - L1) / ln(X2 / L1)              logarithmic mean length during the interval

so that v = λ · L̃ exactly in every interval.

One value per mother and window of time (time-weighted over its intervals in the window):
    v_m = ΣΔX / ΣΔt,   λ_m = Σ ln(X2/L1) / ΣΔt,   L_m = Σ L̃ Δt / ΣΔt

Population means: <v> = <λ><L> κ, κ = <v> / (<λ><L>) (coupling: covariance between mothers and within mothers),
so the log-effect relative to the control splits exactly:  ε^v = ε^λ + ε^L + ε^κ,  ε^x_c = ln(<x>_c / <x>_control).
Bliss independence: ε_Bliss = ε_A + ε_B, for each term separately.
"""
import numpy as np
import pandas as pd

KEYS = ['condition', 'experiment', 'area', 'channel', 'mother']
PARTS = ['v', 'lam', 'L', 'kappa']
NAMES = {'v': 'velocity ε^v', 'lam': 'rate ε^λ', 'L': 'length ε^L', 'kappa': 'coupling ε^κ'}


def labels(unit='px'):
    return {'v': f'elongation velocity v = ΔL/Δt ({unit}/min)', 'lam': 'elongation rate λ = Δln L/Δt (1/min)',
            'L': f'mother length L ({unit})', 'div_per_h': 'divisions per mother and hour'}


# ============================================================================ intervals and windows
def growth_intervals(LM, dt_range=(3, 25), um_per_px=None):
    """
    One row per pair of consecutive images of a mother:
        t_mid (min), dt, dX (= X2 - L1), lr (= ln(X2/L1)), Lt (logarithmic mean length), div (1 if it divided)
    An interval is not used if one of the two lengths is missing (no reliable length, past the exit, lost), if it
    contains a division whose daughter has no length, or if Δt is outside dt_range (strictly).
    """
    d = LM.sort_values(['mother', 'image'])
    valid = ~d.lost & ~d.past_exit
    L = d.L.where(valid)
    D = np.where((d.divides & ~d.lost).to_numpy(), d.L_daughter.to_numpy(), 0.0)
    same = d.mother.eq(d.mother.shift()).to_numpy()
    L1, X2 = L.shift().to_numpy(), L.to_numpy() + D
    t1, t2 = d.t.shift().to_numpy(), d.t.to_numpy()
    dt = t2 - t1
    with np.errstate(invalid='ignore'):
        ok = same & np.isfinite(L1) & np.isfinite(X2) & (L1 > 0) & (X2 > 0) & (dt > dt_range[0]) & (dt < dt_range[1])
    s = float(um_per_px or 1.0)
    L1, X2, dt = L1[ok] * s, X2[ok] * s, dt[ok]
    lr, dX = np.log(X2 / L1), X2 - L1
    small = np.abs(lr) < 1e-9
    Lt = np.where(small, L1, dX / np.where(small, 1.0, lr))
    out = d.loc[ok, KEYS].reset_index(drop=True)
    return out.assign(t_mid=(t1[ok] + t2[ok]) / 2, dt=dt, dX=dX, lr=lr, Lt=Lt, div=(D[ok] > 0).astype(float))


def window_means(IV, lo, hi, cover=0.6):
    """
    One row per mother with intervals (assigned by their mid time) in [lo, hi) covering at least `cover` of it:
        v, lam, L (time-weighted means), div_per_h
    """
    d = IV[(IV.t_mid >= lo) & (IV.t_mid < hi)]
    s = d.assign(LtT=d.Lt * d.dt).groupby(KEYS, sort=False)[['dt', 'dX', 'lr', 'LtT', 'div']].sum().reset_index()
    s = s[s.dt >= cover * (hi - lo)]
    return s[KEYS].assign(v=(s.dX / s.dt).to_numpy(), lam=(s.lr / s.dt).to_numpy(), L=(s.LtT / s.dt).to_numpy(),
                          div_per_h=(60 * s['div'] / s.dt).to_numpy()).reset_index(drop=True)


def window_summary(W, conditions):
    """mothers, areas, means and medians of v, λ, L, doubling time, divisions per hour, coupling κ"""
    g = W.groupby('condition')
    t = pd.DataFrame(dict(mothers=g.size(), areas=g.apply(lambda x: x[['experiment', 'area']].drop_duplicates().shape[0]),
                          v_mean=g.v.mean(), lam_mean=g.lam.mean(), L_mean=g.L.mean(),
                          v_median=g.v.median(), lam_median=g.lam.median(), L_median=g.L.median(),
                          doubling_min=np.log(2) / g.lam.mean(), div_per_h=g.div_per_h.mean()))
    t['coupling κ'] = t.v_mean / (t.lam_mean * t.L_mean)
    return t.reindex(conditions)


# ============================================================================ log-effects and Bliss
def check_roles(roles, conditions):
    """roles: dict(control=..., A=..., B=..., combination=...) -> list of problems"""
    need = ['control', 'A', 'B', 'combination']
    probs = [f"missing role '{r}'" for r in need if r not in roles]
    probs += [f"role {r} = '{roles[r]}' is not one of the conditions {conditions}" for r in need
              if r in roles and roles[r] not in conditions]
    if len({roles.get(r) for r in need}) < 4:
        probs.append('the four roles must be four different conditions')
    return probs


def kappa_parts(W, conditions):
    """κ = <v>/(<λ><L>) = 1 + between mothers (r·CV_λ·CV_L) + within mothers (<v - λL>/(<λ><L>))"""
    rows = []
    for c in conditions:
        d = W[W.condition == c]
        ml, mL = d.lam.mean(), d.L.mean()
        rows.append({'condition': c, 'κ = ⟨v⟩/(⟨λ⟩⟨L⟩)': d.v.mean() / (ml * mL),
                     'between mothers r·CV_λ·CV_L': np.cov(d.lam, d.L, ddof=0)[0, 1] / (ml * mL),
                     'within mothers ⟨v − λL⟩/(⟨λ⟩⟨L⟩)': (d.v - d.lam * d.L).mean() / (ml * mL),
                     'corr(λ, L)': np.corrcoef(d.lam, d.L)[0, 1],
                     'CV_λ': d.lam.std(ddof=0) / ml, 'CV_L': d.L.std(ddof=0) / mL})
    t = pd.DataFrame(rows).set_index('condition')
    return t


ROWS = ['A', 'B', 'combination', 'Bliss', 'combination − Bliss', 'combination − A', 'combination − B']


def log_effects(W, roles):
    """Series indexed by (part, row): part in v, lam, L, kappa; row in ROWS"""
    m = W.groupby('condition')[['v', 'lam', 'L']].mean()
    m['kappa'] = m.v / (m.lam * m.L)
    e = np.log(m / m.loc[roles['control']])
    A, B, AB = e.loc[roles['A']], e.loc[roles['B']], e.loc[roles['combination']]
    out = {}
    for k in PARTS:
        out.update({(k, 'A'): A[k], (k, 'B'): B[k], (k, 'combination'): AB[k], (k, 'Bliss'): A[k] + B[k],
                    (k, 'combination − Bliss'): AB[k] - A[k] - B[k],
                    (k, 'combination − A'): AB[k] - A[k], (k, 'combination − B'): AB[k] - B[k]})
    return pd.Series(out)


def resample_areas(W, rng):
    """areas (experiment × area) resampled with replacement within each condition"""
    parts = []
    for c, g in W.groupby('condition'):
        unit = g.experiment.astype(str) + '|' + g.area.astype(str)
        by = {u: x for u, x in g.groupby(unit)}
        us = list(by)
        parts += [by[u] for u in rng.choice(us, len(us), replace=True)]
    return pd.concat(parts, ignore_index=True)


def bootstrap(W, fun, n=1000, seed=1, progress=None):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n):
        if progress: progress(b, n)
        out.append(fun(resample_areas(W, rng)))
    if progress: progress(n, n)
    return pd.DataFrame(out)


def bootstrap_effects(W, roles, n=1000, seed=1, progress=None):
    """
    Bootstrap of log_effects (areas resampled with replacement within each condition), fast: the means only need
    the sums of v, λ, L and the number of mothers of each area. Returns a DataFrame like bootstrap(W, log_effects).
    """
    rng = np.random.default_rng(seed)
    means = {}
    for c in dict.fromkeys(roles.values()):
        d = W[W.condition == c]
        S = d.assign(n=1).groupby([d.experiment.astype(str), d.area])[['n', 'v', 'lam', 'L']].sum().to_numpy()
        idx = rng.integers(0, len(S), size=(n, len(S)))
        T = S[idx].sum(1)                                              # n × [n, v, lam, L]
        m = T[:, 1:] / T[:, :1]
        means[c] = np.column_stack([m, m[:, 0] / (m[:, 1] * m[:, 2])])     # v, lam, L, kappa
    e = {c: np.log(means[c] / means[roles['control']]) for c in means}
    A, B, AB = e[roles['A']], e[roles['B']], e[roles['combination']]
    out = {}
    for i, k in enumerate(PARTS):
        out.update({(k, 'A'): A[:, i], (k, 'B'): B[:, i], (k, 'combination'): AB[:, i], (k, 'Bliss'): A[:, i] + B[:, i],
                    (k, 'combination − Bliss'): AB[:, i] - A[:, i] - B[:, i],
                    (k, 'combination − A'): AB[:, i] - A[:, i], (k, 'combination − B'): AB[:, i] - B[:, i]})
    if progress: progress(n, n)
    return pd.DataFrame(out)


def effects_table(E0, EB, roles):
    """log-effects with 95 % bootstrap intervals, one column per part, rows named with the conditions"""
    A, B, AB = roles['A'], roles['B'], roles['combination']
    names = {'A': f'{A} alone', 'B': f'{B} alone', 'combination': f'{AB} observed', 'Bliss': f'Bliss ({A} + {B})',
             'combination − Bliss': 'observed − Bliss', 'combination − A': f'observed − {A} alone',
             'combination − B': f'observed − {B} alone'}
    f = lambda k: f'{E0[k]:+.3f} [{EB[k].quantile(0.025):+.3f}, {EB[k].quantile(0.975):+.3f}]'
    t = pd.DataFrame({NAMES[p]: [f((p, r)) for r in ROWS] for p in PARTS}, index=[names[r] for r in ROWS])
    return t


def effects_long(E0, EB):
    """the same as effects_table, as numbers: part, row, effect, lo, hi, relative change (%)"""
    rows = [dict(part=p, row=r, effect=E0[(p, r)], lo=EB[(p, r)].quantile(0.025), hi=EB[(p, r)].quantile(0.975),
                 p_value=min(1.0, 2 * min((EB[(p, r)] <= 0).mean(), (EB[(p, r)] >= 0).mean())))
            for p in PARTS for r in ROWS]
    t = pd.DataFrame(rows)
    t['change_%'] = 100 * (np.exp(t.effect) - 1)
    return t


def bliss_prediction(W, roles, E0, EB, unit='px'):
    """
    Bliss prediction of the combination in natural units: <x>_Bliss = <x>_control · (<x>_A/<x>_control) · (<x>_B/<x>_control),
    against the observed <x>_combination; ratio observed / Bliss with its 95 % interval.
    """
    m = W.groupby('condition')[['v', 'lam', 'L']].mean()
    m['kappa'] = m.v / (m.lam * m.L)
    c = roles['control']
    units = {'v': f'{unit}/min', 'lam': '1/min', 'L': unit, 'kappa': '–'}
    rows = []
    for p in PARTS:
        pred = m.at[c, p] * np.exp(E0[(p, 'Bliss')])
        d = EB[(p, 'combination − Bliss')]
        rows.append({'part': NAMES[p].split(' ')[0], 'unit': units[p], roles['control']: m.at[c, p],
                     roles['A']: m.at[roles['A'], p], roles['B']: m.at[roles['B'], p],
                     'Bliss prediction': pred, f"{roles['combination']} observed": m.at[roles['combination'], p],
                     'observed / Bliss': np.exp(E0[(p, 'combination − Bliss')]),
                     '95 % CI': f'[{np.exp(d.quantile(0.025)):.3f}, {np.exp(d.quantile(0.975)):.3f}]'})
    return pd.DataFrame(rows).set_index('part')


# ============================================================================ along time
def time_course(IV, conditions, t_end, width=15, cover=0.6):
    """consecutive windows of `width` min ending at t_end: medians per condition (TC) and every mother (PTS)"""
    tc, pts = [], []
    for lo in np.arange(t_end - width * np.floor(t_end / width), t_end - 1e-9, width):
        w = window_means(IV, lo, lo + width, cover)
        if not len(w): continue
        g = w.groupby('condition')
        tc.append(pd.DataFrame(dict(t=lo + width / 2, n=g.v.size(), v=g.v.median(), lam=g.lam.median(), L=g.L.median(),
                                    div_per_h=g.div_per_h.mean())).reset_index())
        pts.append(w.assign(t=lo + width / 2))
    return pd.concat(tc, ignore_index=True), pd.concat(pts, ignore_index=True)


def effects_over_time(IV, roles, t_end, width=30, step=15, cover=0.6, n_boot=1000, min_mothers=30, progress=None):
    """log-effects in sliding windows (width min, moved every step min) ending by t_end"""
    rows = []
    starts = np.arange((t_end - width) % step, t_end - width + 1e-9, step)    # the last window ends at t_end
    for i, lo in enumerate(starts):
        if progress: progress(i, len(starts))
        w = window_means(IV, lo, lo + width, cover)
        w = w[w.condition.isin(roles.values())]
        n = w.groupby('condition').size().reindex(list(roles.values()))
        if n.isna().any() or n.min() < min_mothers: continue
        e0, eb = log_effects(w, roles), bootstrap_effects(w, roles, n=n_boot, seed=int(lo))
        for (p, r), val in e0.items():
            rows.append(dict(t=lo + width / 2, part=p, row=r, effect=val,
                             lo=eb[(p, r)].quantile(0.025), hi=eb[(p, r)].quantile(0.975)))
    if progress: progress(len(starts), len(starts))
    return pd.DataFrame(rows)
