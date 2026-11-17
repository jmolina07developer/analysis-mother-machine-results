"""
The analyses, as functions that take the experiments and the settings and return tables and figures.

Two levels, so that the notebooks and a graphical interface use exactly the same code:

* steps, used by the notebooks to explain each part:
      ds = load_dataset(EXPERIMENTS, step=10, t_end=170)
      P  = division_phenotypes(ds)          # or length_phenotypes(ds, ...)
      P.fig_heatmap(); P.fractions; ...

* one call per analysis, for an interface or a script; nothing is printed, everything is returned:
      res = run_division_phenotypes(EXPERIMENTS, step=10, t_end=170)
      res.figures   {name: matplotlib Figure}      res.tables    {name: DataFrame}
      res.messages  [warnings for the user]        res.save('results/...')
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import adjusted_rand_score

from . import plots as mp
from .config import as_entries, save_config
from .excel import load_experiments
from .trajectories import time_grid, default_t_end, division_vectors, length_vectors, fill_gaps, as_word
from .clustering import cluster, choose_k, numbered_clusters, fractions
from . import growth as gr
from . import quantile_bliss as qbl


# ============================================================================ data
@dataclass
class Dataset:
    """The Excel files of all the experiments, read, with the time grid of the analysis."""
    experiments: list
    LM: pd.DataFrame            # one row per channel and image (see excel.load_experiments)
    summary: pd.DataFrame       # one row per entry
    grid: np.ndarray            # min
    step: float
    t_end: float
    conditions: list            # in the order they were given
    colors: dict                # condition -> colour
    messages: list = field(default_factory=list)


def load_dataset(experiments, step=10, t_end=None, condition_colors=None, base=None):
    """
    Read every Excel file of the experiments (see config.Entry) and set the time grid
    step, 2·step, ..., t_end (min since time 0). t_end None -> the latest time all the entries have images for.
    """
    entries = as_entries(experiments)
    LM, summary = load_experiments(entries, base=base)
    conditions = list(dict.fromkeys(e.condition for e in entries))
    t_end = default_t_end(LM, step) if t_end is None else float(t_end)
    grid = time_grid(t_end, step)
    msgs = [f'{LM.mother.nunique()} channels in {summary.n_areas.sum()} areas; analysis up to {t_end:g} min']
    short = summary[summary.last_image_min < t_end]
    for r in short.itertuples():
        msgs.append(f'WARNING: {r.condition} ({r.experiment}) has images only up to {r.last_image_min:g} min, '
                    f'before the end of the window ({t_end:g} min)')
    return Dataset(entries, LM, summary, grid, step, t_end, conditions,
                   mp.condition_colors(conditions, condition_colors), msgs)


def _counts(TR, conditions, extra=()):
    """mothers per condition: channels, left out by each rule, used"""
    agg = dict(channels=('mother', 'size'), areas=('area', 'nunique'), empty_channel=('empty_start', 'sum'),
               lost=('lost_only', 'sum'))
    agg.update({name: (col, 'sum') for name, col in extra})
    c = TR.assign(lost_only=TR.lost & ~TR.empty_start).groupby('condition', sort=False).agg(**agg, mothers_used=('use', 'sum'))
    c = c.reindex(conditions)
    c['used (%)'] = (100 * c.mothers_used / (c.channels - c.empty_channel)).round(0)
    return c


def division_matrix(ds):
    """
    N(t) of every mother. Returns (TR, counts): TR one row per channel (column `use` = kept), counts per condition.
    Left out: channels empty in the first 3 images, mothers lost before the end of the window.
    """
    TR = division_vectors(ds.LM, ds.grid)
    TR['use'] = ~TR.empty_start & ~TR.lost
    return TR, _counts(TR, ds.conditions)


def length_matrix(ds, max_age=12, max_gap=2, past_exit='exclude', channel_px=None, um_per_px=None):
    """
    L(t) of every mother. Returns (TR, counts), TR with columns L (raw, NaN where missing), L_filled (used), use.

    max_age     a grid point takes the last image at or before it, if at most this many min older
    max_gap     runs of up to this many missing grid points are filled (linear interpolation; nearest value
                at the two ends); mothers with longer gaps are left out
    past_exit   'exclude': mothers past the channel exit at some grid point are left out (they have no length);
                'cap': they are given the channel length while they are past the exit
    channel_px  channel length (px) for past_exit='cap'; None -> the longest length measured in the data
    um_per_px   if given, lengths are converted to µm
    """
    TR = length_vectors(ds.LM, ds.grid, max_age=max_age)
    cap = float(channel_px) if channel_px else float(np.nanmax(np.vstack(TR.L.to_numpy())))
    filled = []
    for r in TR.itertuples():
        if r.empty_start or r.lost or (r.ever_exit and past_exit == 'exclude'):
            filled.append(None); continue
        x = np.where(r.exit, cap, r.L) if past_exit == 'cap' else r.L
        filled.append(fill_gaps(x, max_gap))
    scale = um_per_px or 1.0
    TR['L'] = [x * scale for x in TR.L]
    TR['L_filled'] = [None if x is None else x * scale for x in filled]
    TR['use'] = TR.L_filled.notna()
    TR['gaps'] = ~TR.empty_start & ~TR.lost & ~(TR.ever_exit & (past_exit == 'exclude')) & ~TR.use
    TR['exit_left_out'] = ~TR.empty_start & ~TR.lost & TR.ever_exit & (past_exit == 'exclude')
    counts = _counts(TR, ds.conditions, extra=[('past_exit', 'exit_left_out'), ('gaps', 'gaps')])
    return TR, counts


# ============================================================================ phenotypes
@dataclass
class Phenotypes:
    """A clustering of the mothers' trajectories, with everything needed for its tables and figures."""
    kind: str                     # 'divisions' or 'length'
    title: str                    # e.g. 'Divisions N(t)'
    value_label: str              # e.g. 'divisions N(t)', colour bar and y axis
    X: np.ndarray                 # mothers × grid
    ID: pd.DataFrame              # one row per mother used (condition, experiment, area, channel, phenotype)
    grid: np.ndarray
    conditions: list
    colors: dict
    counts: pd.DataFrame
    D: np.ndarray                 # condensed distance matrix
    Z: np.ndarray                 # linkage
    selection: pd.DataFrame       # silhouette and stability for each k
    k: int
    ph: np.ndarray                # phenotype (1..k) of each mother
    order: np.ndarray             # rows in dendrogram order (top to bottom)
    fractions: pd.DataFrame
    cmap: object = None
    norm: object = None
    cbar_ticks: object = None
    word: bool = False            # write the vectors as words (divisions)

    # -------------------------------------------------- tables
    def phenotype_summary(self, n_marks=4):
        marks = sorted({self.grid[np.argmin(abs(self.grid - t))] for t in np.linspace(self.grid[0], self.grid[-1], n_marks)})
        col = lambda t: list(self.grid).index(t)
        return pd.DataFrame({f'P{j}': dict(mothers=int((self.ph == j).sum()),
                                           **{f'mean at {t:g} min': self.X[self.ph == j][:, col(t)].mean() for t in marks})
                             for j in range(1, self.k + 1)}).T

    def fraction_table(self):
        return (self.fractions.assign(**{'% (95 % CI)': lambda d: [f'{100*f:.0f} ({100*l:.0f}–{100*h:.0f})'
                                                                   for f, l, h in zip(d.fraction, d.lo, d.hi)]})
                .pivot(index='condition', columns='phenotype', values='% (95 % CI)').reindex(self.conditions))

    def mothers_table(self):
        prefix = 'N' if self.kind == 'divisions' else 'L'
        cols = {f'{prefix}_{t:g}min': self.X[:, i] for i, t in enumerate(self.grid)}
        out = self.ID[['condition', 'experiment', 'area', 'channel', 'phenotype']].copy()
        if self.word:
            out['word'] = [as_word(n) for n in self.X]
        return out.assign(**cols)

    def tables(self):
        return {'mothers': self.mothers_table(), 'fractions': self.fractions, 'k_selection': self.selection,
                'phenotype_summary': self.phenotype_summary(), 'mothers_used': self.counts}

    # -------------------------------------------------- figures (each returns a matplotlib Figure)
    def fig_k(self):
        return mp.plot_k(self.selection, self.k, f'{self.title}: number of phenotypes')

    def fig_heatmap(self):
        fig, _ = mp.heatmap_dendrogram(self.X, self.grid, self.ID.condition.to_numpy(), self.Z, self.ph, self.k,
                                       self.conditions, self.colors, self.cmap, self.norm, self.value_label,
                                       f'{self.title} of each mother', cbar_ticks=self.cbar_ticks)
        return fig

    def fig_by_condition(self):
        return mp.heatmaps_by_condition(self.X, self.grid, self.ID.condition.to_numpy(), self.order, self.ph, self.k,
                                        self.conditions, self.colors, self.cmap, self.norm, self.value_label,
                                        cbar_ticks=self.cbar_ticks)

    def fig_mean_phenotype(self):
        return mp.mean_trajectories(self.X, self.grid, self.ph, self.k, self.value_label, f'Mean {self.title} of each phenotype')

    def fig_mean_condition(self):
        return mp.mean_by_condition(self.X, self.grid, self.ID.condition.to_numpy(), self.conditions, self.colors,
                                    self.value_label, f'Mean {self.title} of each condition')

    def fig_fractions(self):
        return mp.plot_fractions(self.fractions, self.k, self.conditions, f'{self.title} phenotypes')

    def fig_fractions_ci(self):
        return mp.plot_fraction_ci(self.fractions, self.k, self.conditions, self.colors,
                                   f'{self.title} phenotypes: fraction in each condition')

    def figures(self):
        """all the figures, in order"""
        return {'k_selection': self.fig_k(), 'heatmap_dendrogram': self.fig_heatmap(),
                'heatmaps_by_condition': self.fig_by_condition(), 'mean_by_phenotype': self.fig_mean_phenotype(),
                'mean_by_condition': self.fig_mean_condition(), 'fractions': self.fig_fractions(),
                'fractions_CI': self.fig_fractions_ci()}


def find_phenotypes(kind, title, value_label, X, ID, counts, ds, order_key, n_phenotypes='auto', ks=range(2, 9),
                    n_boot=100, progress=None):
    """
    Ward clustering of the rows of X (root-mean-square distance), choice of k (highest silhouette if
    n_phenotypes='auto'), phenotypes P1..Pk numbered by increasing order_key, fractions per condition.
    """
    ID = ID.reset_index(drop=True).copy()
    D, Z = cluster(X)
    units = (ID.condition + '|' + ID.experiment.astype(str) + '|' + ID.area.astype(str)).to_numpy()
    SEL = choose_k(D, Z, units, ks=ks, n_boot=n_boot, progress=progress)
    k = int(SEL.silhouette.idxmax()) if str(n_phenotypes) == 'auto' else int(n_phenotypes)
    ph = numbered_clusters(Z, k, order_key)
    ID['phenotype'] = [f'P{j}' for j in ph]
    order = np.asarray(dendrogram(Z, no_plot=True)['leaves'][::-1])
    FR = fractions(ID, ph, k, ds.conditions)
    return Phenotypes(kind, title, value_label, X, ID, ds.grid, ds.conditions, ds.colors, counts,
                      D, Z, SEL, k, ph, order, FR)


def division_phenotypes(ds, n_phenotypes='auto', ks=range(2, 9), n_boot=100, progress=None):
    """phenotypes of the cumulative number of divisions N(t); P1 = the most divisions at the end of the window"""
    TR, counts = division_matrix(ds)
    ID = TR[TR.use].reset_index(drop=True)
    X = np.vstack(ID.N.to_numpy())
    P = find_phenotypes('divisions', 'Divisions N(t)', 'divisions N(t)', X, ID.drop(columns='N'), counts, ds,
                        -X[:, -1], n_phenotypes, ks, n_boot, progress)
    nmax = int(X.max())
    P.cmap = plt.get_cmap('viridis', nmax + 1)
    P.norm = BoundaryNorm(np.arange(-0.5, nmax + 1.5), nmax + 1)
    P.cbar_ticks = range(0, nmax + 1, max(1, (nmax + 1) // 10))
    P.word = True
    return P


def length_phenotypes(ds, n_phenotypes='auto', ks=range(2, 9), n_boot=100, max_age=12, max_gap=2,
                      past_exit='exclude', channel_px=None, um_per_px=None, saturate_pct=98, progress=None):
    """phenotypes of the length of the mother L(t); P1 = the shortest at the end of the window"""
    TR, counts = length_matrix(ds, max_age, max_gap, past_exit, channel_px, um_per_px)
    ID = TR[TR.use].reset_index(drop=True)
    X = np.vstack(ID.L_filled.to_numpy())
    unit = 'µm' if um_per_px else 'px'
    P = find_phenotypes('length', 'Length L(t)', f'length of the mother ({unit})', X,
                        ID.drop(columns=['L', 'L_filled', 'exit']), counts, ds, X[:, -1], n_phenotypes, ks, n_boot, progress)
    P.cmap = plt.get_cmap('viridis')
    P.norm = Normalize(0, np.percentile(X, saturate_pct))
    return P


# ============================================================================ extra figures for the lengths
def lengths_in_division_order(ds, PN, max_age=12, um_per_px=None, saturate_pct=98):
    """
    Heatmaps of the length of the mothers used in the division clustering PN, with the rows in the order of the
    division dendrogram and PN's phenotype strip. Missing lengths are grey; past the exit, the top colour.
    """
    TR = length_vectors(ds.LM, ds.grid, max_age=max_age).set_index('mother')
    L = np.vstack([np.where(TR.at[m, 'exit'], 1e9, TR.at[m, 'L']) for m in PN.ID.mother])
    L = L * (um_per_px or 1.0)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('0.82'); cmap.set_over(cmap(1.0))
    vmax = np.nanpercentile(np.where(L >= 1e8, np.nan, L), saturate_pct)
    fig = mp.heatmaps_by_condition(L, ds.grid, PN.ID.condition.to_numpy(), PN.order, PN.ph, PN.k, PN.conditions,
                                   PN.colors, cmap, Normalize(0, vmax),
                                   f"length of the mother ({'µm' if um_per_px else 'px'})")
    fig.suptitle('Length of the mothers, rows ordered by their divisions (strip: division phenotype)')
    return fig


def compare_phenotypes(PA, PB):
    """
    Cross-table of the mothers clustered in both PA and PB (rows: PA, columns: PB), and the adjusted Rand index.
    Returns (table, ari, figure).
    """
    key = lambda d: pd.MultiIndex.from_frame(d[['condition', 'experiment', 'area', 'channel']])
    b = pd.Series(PB.ph, index=key(PB.ID)).reindex(key(PA.ID))
    both = pd.DataFrame({'A': PA.ph, 'B': b.to_numpy()}).dropna().astype(int)
    ari = adjusted_rand_score(both.A, both.B)
    ct = pd.crosstab(both.A.map(lambda j: f'P{j} ({PA.kind})'), both.B.map(lambda j: f'P{j} ({PB.kind})'))
    fig, ax = plt.subplots(figsize=(mp.HALF_W, 4.6), constrained_layout=True)
    ax.imshow(ct.to_numpy(), cmap='Greys'); ax.grid(False)
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index)
    vmax = ct.to_numpy().max()
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iat[i, j]
            ax.text(j, i, v, ha='center', va='center', fontsize=13, color='white' if v > vmax / 2 else mp.INK)
    ax.set(xlabel=f'{PB.kind} phenotype', ylabel=f'{PA.kind} phenotype',
           title=f'Mothers in each pair of phenotypes (ARI = {ari:.2f})')
    return ct, ari, fig


# ============================================================================ one call per analysis
@dataclass
class AnalysisResult:
    name: str
    settings: dict
    experiments: list
    figures: dict = field(default_factory=dict)     # name -> matplotlib Figure (in order)
    tables: dict = field(default_factory=dict)      # name -> DataFrame
    messages: list = field(default_factory=list)
    objects: dict = field(default_factory=dict)     # Dataset, Phenotypes... for further work

    def save(self, folder, dpi=200, close=False):
        """figures as numbered PNG, tables as CSV, settings.json (reloadable with config.load_config), messages.txt"""
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        for i, (name, fig) in enumerate(self.figures.items(), 1):
            fig.savefig(folder / f'{i:02d}_{name}.png', dpi=dpi, bbox_inches='tight')
            if close: plt.close(fig)
        for name, t in self.tables.items():
            t.to_csv(folder / f'{name}.csv', index=not isinstance(t.index, pd.RangeIndex))
        save_config(folder / 'settings.json', self.experiments, analysis=self.name, **self.settings)
        (folder / 'messages.txt').write_text('\n'.join(self.messages) + '\n')
        return folder


def run_division_phenotypes(experiments, step=10, t_end=None, n_phenotypes='auto', ks=range(2, 9), n_boot=100,
                            condition_colors=None, base=None, progress=None):
    """Notebook 01 in one call."""
    ds = load_dataset(experiments, step, t_end, condition_colors, base)
    P = division_phenotypes(ds, n_phenotypes, ks, n_boot, progress)
    settings = dict(step=step, t_end=ds.t_end, n_phenotypes=n_phenotypes, k_used=P.k, n_boot=n_boot)
    return AnalysisResult('01_division_phenotypes', settings, ds.experiments, P.figures(),
                          {**P.tables(), 'experiments': ds.summary}, ds.messages + [f'{P.k} phenotypes'],
                          dict(dataset=ds, divisions=P))


def run_length_phenotypes(experiments, step=10, t_end=None, n_phenotypes='auto', ks=range(2, 9), n_boot=100,
                          max_age=12, max_gap=2, past_exit='exclude', channel_px=None, um_per_px=None,
                          compare_with_divisions=True, n_phenotypes_divisions='auto',
                          condition_colors=None, base=None, progress=None):
    """Notebook 02 in one call."""
    ds = load_dataset(experiments, step, t_end, condition_colors, base)
    P = length_phenotypes(ds, n_phenotypes, ks, n_boot, max_age, max_gap, past_exit, channel_px, um_per_px,
                          progress=progress)
    figs, tabs = P.figures(), {**P.tables(), 'experiments': ds.summary}
    objects, msgs = dict(dataset=ds, length=P), ds.messages + [f'{P.k} length phenotypes']
    if compare_with_divisions:
        PN = division_phenotypes(ds, n_phenotypes_divisions, ks, n_boot=0)
        figs['lengths_in_division_order'] = lengths_in_division_order(ds, PN, max_age, um_per_px)
        ct, ari, figs['divisions_vs_length'] = compare_phenotypes(PN, P)
        tabs['divisions_vs_length'] = ct
        objects['divisions'] = PN
        msgs.append(f'{PN.k} division phenotypes; agreement with the length phenotypes: ARI = {ari:.2f}')
    settings = dict(step=step, t_end=ds.t_end, n_phenotypes=n_phenotypes, k_used=P.k, n_boot=n_boot,
                    max_age=max_age, max_gap=max_gap, past_exit=past_exit, channel_px=channel_px, um_per_px=um_per_px)
    return AnalysisResult('02_length_phenotypes', settings, ds.experiments, figs, tabs, msgs, objects)


# ============================================================================ growth: rate × length and Bliss
@dataclass
class GrowthSplit:
    """The elongation of the mothers in a window, split into rate × length, and the Bliss analysis of each part."""
    roles: dict                   # control, A, B, combination -> condition names
    window: tuple                 # (lo, hi) min
    conditions: list
    colors: dict
    unit: str
    IV: pd.DataFrame              # one row per interval between consecutive images
    W: pd.DataFrame               # one row per mother in the window: v, lam, L, div_per_h
    summary: pd.DataFrame
    kappa: pd.DataFrame
    E0: pd.Series                 # log-effects (part, row)
    EB: pd.DataFrame              # their bootstrap replicates
    effects: pd.DataFrame         # formatted table with 95 % intervals
    effects_numbers: pd.DataFrame
    bliss: pd.DataFrame           # Bliss prediction in natural units
    t_end: float
    ET: pd.DataFrame = None       # effects along time (filled by along_time)

    @property
    def labels(self):
        return gr.labels(self.unit)

    def along_time(self, width=30, step=15, cover=0.6, n_boot=1000, min_mothers=30, progress=None):
        self.ET = gr.effects_over_time(self.IV, self.roles, self.t_end, width, step, cover, n_boot, min_mothers, progress)
        return self.ET

    def fig_time_course(self, width=15, cover=0.6):
        TC, PTS = gr.time_course(self.IV, self.conditions, self.t_end, width, cover)
        return mp.growth_time_course(TC, PTS, self.conditions, self.colors, self.window, self.labels, t_end=self.t_end)

    def fig_rate_vs_length(self):
        order = [self.roles[r] for r in ('control', 'A', 'B', 'combination')]
        order += [c for c in self.conditions if c not in order]
        return mp.rate_vs_length(self.W, order, self.colors, self.roles['control'], self.window, self.unit)

    def fig_effects(self):
        return mp.effect_bars(self.E0, self.EB, self.roles, self.window, self.colors)

    def fig_effects_time(self):
        if self.ET is None: self.along_time()
        return mp.effects_time(self.ET, self.roles, self.colors, self.window)

    def tables(self):
        t = {'window_summary': self.summary, 'coupling': self.kappa, 'log_effects': self.effects,
             'log_effects_numbers': self.effects_numbers, 'bliss_prediction': self.bliss,
             'mothers_in_window': self.W.drop(columns='mother')}
        if self.ET is not None: t['log_effects_along_time'] = self.ET
        return t


def growth_split(ds, roles, window, cover=0.6, dt_range=(3, 25), um_per_px=None, n_boot=1000, progress=None):
    """
    Rate × length split of the elongation velocity in `window` (min) and Bliss of each part.
    roles: dict(control='LB', A='GEN', B='NIT', combination='NIT+GEN') (names of conditions of ds).
    """
    probs = gr.check_roles(roles, ds.conditions)
    lo, hi = window
    if not (0 <= lo < hi <= ds.t_end + 1e-9):
        probs.append(f'the window {lo}–{hi} min must be inside 0–{ds.t_end:g} min (T_END)')
    if probs:
        raise ValueError('\n'.join(probs))
    IV = gr.growth_intervals(ds.LM, dt_range, um_per_px)
    IV = IV[IV.t_mid < ds.t_end].reset_index(drop=True)
    W = gr.window_means(IV, lo, hi, cover)
    n = W.groupby('condition').size().reindex(list(roles.values()))
    if n.isna().any() or n.min() < 3:
        raise ValueError(f'too few mothers in the window {lo}–{hi} min: ' + ', '.join(f'{c}: {0 if v != v else int(v)}' for c, v in n.items()))
    Wr = W[W.condition.isin(roles.values())]
    E0 = gr.log_effects(Wr, roles)
    EB = gr.bootstrap_effects(Wr, roles, n=n_boot, progress=progress)
    unit = 'µm' if um_per_px else 'px'
    return GrowthSplit(dict(roles), (lo, hi), ds.conditions, ds.colors, unit, IV, W,
                       gr.window_summary(W, ds.conditions), gr.kappa_parts(W, ds.conditions), E0, EB,
                       gr.effects_table(E0, EB, roles), gr.effects_long(E0, EB),
                       gr.bliss_prediction(Wr, roles, E0, EB, unit), ds.t_end)


def run_growth_bliss(experiments, roles, window, t_end=None, cover=0.6, dt_range=(3, 25), um_per_px=None,
                     n_boot=1000, along_time=True, time_width=30, time_step=15, condition_colors=None, base=None,
                     progress=None):
    """Notebook 03 in one call."""
    ds = load_dataset(experiments, step=1, t_end=t_end, condition_colors=condition_colors, base=base)
    G = growth_split(ds, roles, window, cover, dt_range, um_per_px, n_boot, progress)
    figs = {'time_course': G.fig_time_course(), 'rate_vs_length': G.fig_rate_vs_length(), 'log_effects': G.fig_effects()}
    if along_time:
        G.along_time(time_width, time_step, cover)
        figs['log_effects_along_time'] = G.fig_effects_time()
    settings = dict(roles=roles, window=list(window), t_end=ds.t_end, cover=cover, dt_range=list(dt_range),
                    um_per_px=um_per_px, n_boot=n_boot)
    return AnalysisResult('03_growth_rate_length_bliss', settings, ds.experiments, figs,
                          {**G.tables(), 'experiments': ds.summary}, ds.messages, dict(dataset=ds, growth=G))


# ============================================================================ Bliss on quantile functions
@dataclass
class QuantileBlissSet:
    """Bliss on the quantile functions of v, λ and L in a window, (α, β) planes and 3-d vectors."""
    roles: dict
    window: tuple
    W: pd.DataFrame
    results: list                  # one quantile_bliss.QuantileBliss per variable
    colors: dict
    unit: str
    settings: dict

    @property
    def titles(self):
        u = self.unit
        return {'v': f'velocity v ({u}/min)', 'lam': 'rate λ (1/min)', 'L': f'length L ({u})'}

    def table(self):
        return pd.DataFrame([R.summary() for R in self.results]).set_index('variable')

    def vectors_table(self):
        return pd.DataFrame({qbl.VARIABLES[R.var]: R.vec['table'] for R in self.results}).T

    def quantiles_table(self):
        rows = {}
        for R in self.results:
            for r in ('control', 'A', 'B', 'combination'):
                rows[(qbl.VARIABLES[R.var], self.roles[r])] = R.Q[r] * R.scale
            bl = np.full(len(R.lev), np.nan); bl[R.est['m']] = R.est['bliss'] * R.scale
            rows[(qbl.VARIABLES[R.var], 'Bliss')] = bl
        t = pd.DataFrame(rows, index=pd.Index(np.round(R.lev, 4), name='rank u')).T
        return t

    def fig_quantiles(self):
        return mp.qb_quantiles(self.results, self.roles, self.colors, self.titles)

    def fig_vectors(self, var):
        R = next(R for R in self.results if R.var == var)
        return mp.qb_vectors3d(R, self.roles, self.colors, self.titles[var])

    def fig_alpha_beta(self):
        return mp.qb_alpha_beta(self.results, self.roles, self.colors, self.titles)

    def figures(self):
        figs = {'quantile_functions': self.fig_quantiles()}
        figs.update({f'vectors_3d_{R.var}': self.fig_vectors(R.var) for R in self.results})
        figs['alpha_beta'] = self.fig_alpha_beta()
        return figs

    def tables(self):
        return {'alpha_beta': self.table(), 'vectors_3d': self.vectors_table(), 'quantile_functions': self.quantiles_table()}


def quantile_bliss_analysis(ds, roles, window, variables=('v', 'lam', 'L'), cover=0.6, dt_range=(3, 25), um_per_px=None,
                            n_levels=100, v_min=0.05, min_effect=0.1, n_boot=500, IV=None, progress=None):
    """
    Bliss on the quantile functions of the per-mother values in `window` (see growth.window_means).
    roles: dict(control=..., A=..., B=..., combination=...).
    """
    probs = gr.check_roles(roles, ds.conditions)
    lo, hi = window
    if not (0 <= lo < hi <= ds.t_end + 1e-9):
        probs.append(f'the window {lo}–{hi} min must be inside 0–{ds.t_end:g} min (T_END)')
    bad = [v for v in variables if v not in qbl.VARIABLES]
    if bad:
        probs.append(f'unknown variables {bad}: use v, lam, L')
    if probs:
        raise ValueError('\n'.join(probs))
    if IV is None:
        IV = gr.growth_intervals(ds.LM, dt_range, um_per_px)
        IV = IV[IV.t_mid < ds.t_end].reset_index(drop=True)
    W = gr.window_means(IV, lo, hi, cover)
    W = W[W.condition.isin(roles.values())].reset_index(drop=True)
    n = W.groupby('condition').size().reindex(list(roles.values()))
    if n.isna().any() or n.min() < 10:
        raise ValueError(f'too few mothers in the window {lo}–{hi} min: ' + ', '.join(f'{c}: {0 if v != v else int(v)}' for c, v in n.items()))
    res = []
    for i, v in enumerate(variables):
        sub = (lambda d, t, i=i: progress(i * t + d, len(variables) * t)) if progress else None
        res.append(qbl.quantile_bliss(W, v, roles, n_levels, v_min, min_effect, n_boot, progress=sub))
    settings = dict(n_levels=n_levels, v_min=v_min, min_effect=min_effect, n_boot=n_boot, cover=cover,
                    dt_range=list(dt_range), um_per_px=um_per_px)
    return QuantileBlissSet(dict(roles), (lo, hi), W, res, ds.colors, 'µm' if um_per_px else 'px', settings)


def quantile_bliss_windows(ds, roles, windows, n_boot=300, **kw):
    """the (α, β) table for several windows (sensitivity to the choice of the window)"""
    IV = gr.growth_intervals(ds.LM, kw.get('dt_range', (3, 25)), kw.get('um_per_px'))
    IV = IV[IV.t_mid < ds.t_end].reset_index(drop=True)
    out = []
    for w in windows:
        S = quantile_bliss_analysis(ds, roles, w, n_boot=n_boot, IV=IV, **kw)
        t = S.table().reset_index()
        t.insert(0, 'window', f'{w[0]:g}–{w[1]:g}')
        t.insert(1, 'mothers', '/'.join(str((S.W.condition == roles[r]).sum()) for r in ('control', 'A', 'B', 'combination')))
        out.append(t)
    return pd.concat(out, ignore_index=True).set_index(['window', 'variable'])


def run_quantile_bliss(experiments, roles, window, t_end=None, variables=('v', 'lam', 'L'), other_windows=(),
                       cover=0.6, dt_range=(3, 25), um_per_px=None, n_levels=100, v_min=0.05, min_effect=0.1,
                       n_boot=500, condition_colors=None, base=None, progress=None):
    """Notebook 04 in one call."""
    ds = load_dataset(experiments, step=1, t_end=t_end, condition_colors=condition_colors, base=base)
    S = quantile_bliss_analysis(ds, roles, window, variables, cover, dt_range, um_per_px, n_levels, v_min, min_effect,
                                n_boot, progress=progress)
    tabs = {**S.tables(), 'experiments': ds.summary}
    if other_windows:
        tabs['alpha_beta_windows'] = quantile_bliss_windows(ds, roles, list(other_windows), variables=variables, cover=cover,
                                                            dt_range=dt_range, um_per_px=um_per_px, n_levels=n_levels,
                                                            v_min=v_min, min_effect=min_effect)
    settings = dict(roles=roles, window=list(window), t_end=ds.t_end, variables=list(variables),
                    other_windows=[list(w) for w in other_windows], **S.settings)
    return AnalysisResult('04_quantile_bliss', settings, ds.experiments, S.figures(), tabs, ds.messages,
                          dict(dataset=ds, quantile_bliss=S))


ANALYSES = {
    '01_division_phenotypes': run_division_phenotypes,
    '02_length_phenotypes': run_length_phenotypes,
    '03_growth_rate_length_bliss': run_growth_bliss,
    '04_quantile_bliss': run_quantile_bliss,
}
