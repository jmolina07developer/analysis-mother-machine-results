"""Figures shared by the notebooks (sized for slides: 13.3 in = a full 16:9 slide, 6.5 in = half)."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

FULL_W, HALF_W = 13.3, 6.5
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
# phenotypes (P1, P2, ...) and conditions use two different palettes so they are never confused
PHENO_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#c0407a', '#7a5c3a', '#00a3b4', '#8a8a8a', '#b8a000', '#5b4bd1', '#d43d3d']
COND_COLORS = ['#52514e', '#3a9d5d', '#d9a21b', '#8a5cc2', '#2f6f9f', '#c4573a', '#7fa63a', '#b0679c', '#3aa6a0', '#9c8a5a']


def set_style():
    plt.rcParams.update({
        'figure.dpi': 110, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'lines.linewidth': 2, 'legend.frameon': False, 'axes.titleweight': 'bold',
        'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 11, 'figure.titlesize': 15})


def condition_colors(conditions, custom=None):
    col = {c: COND_COLORS[i % len(COND_COLORS)] for i, c in enumerate(conditions)}
    col.update(custom or {})
    return col


def save(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder) / f'{name}.png', dpi=200, bbox_inches='tight')


def plot_k(SEL, k_sel, title):
    fig, ax = plt.subplots(figsize=(HALF_W, 4.6), constrained_layout=True)
    ks = SEL.index.to_numpy()
    ax.fill_between(ks, SEL['ARI 10th pct'], SEL['ARI 90th pct'], color='0.6', alpha=0.2, lw=0)
    ax.plot(ks, SEL['stability (median ARI)'], 's-', color='0.55', label='stability (median ARI)')
    ax.plot(ks, SEL.silhouette, 'o-', color=INK, lw=2.4, ms=7, label='silhouette')
    ax.axvline(k_sel, color=PHENO_COLORS[0], ls='--', lw=1.4)
    ax.text(k_sel, 1.02, f' k = {k_sel}', color=PHENO_COLORS[0], transform=ax.get_xaxis_transform(), fontsize=12)
    ax.set(xlabel='number of phenotypes k', ylabel='score', ylim=(0, 1), title=title)
    ax.set_xticks(ks)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
    return fig


def heatmap_dendrogram(X, grid, conds, Z, ph, k, cond_order, cond_col, cmap, norm, cbar_label, title, cbar_ticks=None):
    """all the mothers together: dendrogram | heatmap | phenotype strip | condition strip. Returns (fig, row order)."""
    step = grid[1] - grid[0] if len(grid) > 1 else grid[0]
    fig = plt.figure(figsize=(FULL_W, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 7, 0.28, 0.28, 0.25])
    axd, axh, axp, axc, axb = [fig.add_subplot(gs[i]) for i in range(5)]
    n = len(X)
    members = {i: {ph[i]} for i in range(n)}                       # phenotypes under every node of the tree
    for j, (a, b) in enumerate(Z[:, :2].astype(int)):
        members[n + j] = members[a] | members[b]
    link_col = lambda node: PHENO_COLORS[next(iter(members[node])) - 1] if len(members[node]) == 1 else '0.6'
    dn = dendrogram(Z, orientation='left', no_labels=True, ax=axd, link_color_func=link_col)
    order = np.asarray(dn['leaves'][::-1])                         # leaves are drawn bottom-up; heatmap rows top-down
    axd.set_ylim(0, 10 * n); axd.axis('off')
    im = axh.imshow(X[order], aspect='auto', interpolation='nearest', cmap=cmap, norm=norm,
                    extent=(grid[0] - step / 2, grid[-1] + step / 2, n, 0))
    axh.set(xlabel='time (min)', yticks=[], title=f'{title} ({n} mothers)'); axh.grid(False)
    axp.imshow(ph[order][:, None], aspect='auto', interpolation='nearest',
               cmap=ListedColormap(PHENO_COLORS[:k]), vmin=0.5, vmax=k + 0.5)
    ci = np.array([cond_order.index(c) for c in conds])
    axc.imshow(ci[order][:, None], aspect='auto', interpolation='nearest',
               cmap=ListedColormap([cond_col[c] for c in cond_order]), vmin=-0.5, vmax=len(cond_order) - 0.5)
    for ax, t in ((axp, 'phenotype'), (axc, 'condition')):
        ax.set_xticks([0]); ax.set_xticklabels([t], rotation=90); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, cax=axb, label=cbar_label)
    if cbar_ticks is not None: cb.set_ticks(cbar_ticks)
    fig.legend(handles=[Patch(color=PHENO_COLORS[j], label=f'P{j + 1}') for j in range(k)]
               + [Patch(color='none', label='   ')] + [Patch(color=cond_col[c], label=c) for c in cond_order],
               loc='outside lower center', ncol=min(k + 1 + len(cond_order), 12))
    return fig, order


def heatmaps_by_condition(X, grid, conds, order, ph, k, cond_order, cond_col, cmap, norm, cbar_label, cbar_ticks=None,
                          ncols=4):
    """one heatmap per condition, rows in the order of the global dendrogram, phenotype strip on the left"""
    step = grid[1] - grid[0] if len(grid) > 1 else grid[0]
    nc = len(cond_order); ncols = min(ncols, nc); nrows = int(np.ceil(nc / ncols))
    fig = plt.figure(figsize=(FULL_W, 6.6 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, 2 * ncols + 1, width_ratios=[0.25, 4] * ncols + [0.2])
    conds = np.asarray(conds)
    for i, c in enumerate(cond_order):
        r, q = divmod(i, ncols)
        axp, axh = fig.add_subplot(gs[r, 2 * q]), fig.add_subplot(gs[r, 2 * q + 1])
        rows = order[conds[order] == c]
        im = axh.imshow(X[rows], aspect='auto', interpolation='nearest', cmap=cmap, norm=norm,
                        extent=(grid[0] - step / 2, grid[-1] + step / 2, len(rows), 0))
        axh.set_title(f'{c} ({len(rows)})', color=cond_col[c]); axh.set_xlabel('time (min)'); axh.set_yticks([]); axh.grid(False)
        axp.imshow(ph[rows][:, None], aspect='auto', interpolation='nearest',
                   cmap=ListedColormap(PHENO_COLORS[:k]), vmin=0.5, vmax=k + 0.5)
        axp.set_xticks([]); axp.set_yticks([]); axp.grid(False)
        for s in axp.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, -1]), label=cbar_label)
    if cbar_ticks is not None: cb.set_ticks(cbar_ticks)
    fig.legend(handles=[Patch(color=PHENO_COLORS[j], label=f'P{j + 1}') for j in range(k)],
               loc='outside lower center', ncol=k)
    return fig


def mean_trajectories(X, grid, ph, k, ylabel, title):
    """mean ± SD of each phenotype"""
    fig, ax = plt.subplots(figsize=(HALF_W, 4.6), constrained_layout=True)
    for j in range(1, k + 1):
        x = X[ph == j]; m, s = x.mean(0), x.std(0)
        ax.fill_between(grid, m - s, m + s, color=PHENO_COLORS[j - 1], alpha=0.15, lw=0)
        ax.plot(grid, m, color=PHENO_COLORS[j - 1], label=f'P{j} ({len(x)})')
    ax.set(xlabel='time (min)', ylabel=ylabel, title=title)
    ax.legend(loc='upper left')
    return fig


def mean_by_condition(X, grid, conds, cond_order, cond_col, ylabel, title):
    """mean ± SD of each condition"""
    fig, ax = plt.subplots(figsize=(HALF_W, 4.6), constrained_layout=True)
    conds = np.asarray(conds)
    for c in cond_order:
        x = X[conds == c]; m, s = x.mean(0), x.std(0)
        ax.fill_between(grid, m - s, m + s, color=cond_col[c], alpha=0.12, lw=0)
        ax.plot(grid, m, color=cond_col[c], label=f'{c} ({len(x)})')
    ax.set(xlabel='time (min)', ylabel=ylabel, title=title)
    ax.legend(loc='upper left')
    return fig


def plot_fractions(FR, k, cond_order, title):
    """stacked bars: composition of each condition"""
    fig, ax = plt.subplots(figsize=(HALF_W, 4.6), constrained_layout=True)
    bottom = np.zeros(len(cond_order))
    for j in range(1, k + 1):
        f = FR[FR.phenotype == f'P{j}'].set_index('condition').fraction.reindex(cond_order).to_numpy() * 100
        ax.bar(range(len(cond_order)), f, bottom=bottom, color=PHENO_COLORS[j - 1], label=f'P{j}', width=0.7)
        for x, (b, v) in enumerate(zip(bottom, f)):
            if v >= 8: ax.text(x, b + v / 2, f'{v:.0f}', ha='center', va='center', color='white', fontsize=11)
        bottom += f
    ax.set_xticks(range(len(cond_order))); ax.set_xticklabels(cond_order, rotation=0 if len(cond_order) <= 4 else 30, ha='center' if len(cond_order) <= 4 else 'right')
    ax.set(ylabel='mothers (%)', ylim=(0, 100), title=title); ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    return fig


def plot_fraction_ci(FR, k, cond_order, cond_col, title):
    """fraction of each phenotype in each condition with 95 % bootstrap intervals (over areas)"""
    fig, axs = plt.subplots(1, k, figsize=(FULL_W, 4.4), constrained_layout=True, sharey=True, squeeze=False)
    for j, ax in enumerate(axs[0]):
        d = FR[FR.phenotype == f'P{j + 1}'].set_index('condition').reindex(cond_order)
        for x, c in enumerate(cond_order):
            ax.errorbar(x, 100 * d.fraction[c], yerr=[[100 * (d.fraction[c] - d.lo[c])], [100 * (d.hi[c] - d.fraction[c])]],
                        fmt='o', color=cond_col[c], ms=8, capsize=4, lw=2)
        ax.set_xticks(range(len(cond_order))); ax.set_xticklabels(cond_order, rotation=30, ha='right')
        ax.set_title(f'P{j + 1}', color=PHENO_COLORS[j]); ax.set_xlim(-0.5, len(cond_order) - 0.5)
    axs[0, 0].set_ylabel('mothers in the phenotype (%)')
    fig.suptitle(title)
    return fig


# ================================================================================ growth: rate × length
BLISS_COLOR = '#2a78d6'
PART_COLORS = {'lam': '#2a78d6', 'L': '#eb6834', 'kappa': '#b9b8b2', 'v': INK}
PART_NAMES = {'lam': 'rate λ', 'L': 'length L', 'kappa': 'coupling κ', 'v': 'velocity v'}


def growth_time_course(TC, PTS, conditions, colors, window, labels, min_n=30, t_end=None):
    """v, λ, L and divisions per hour along time: every mother (dots) and the median of each condition (lines)"""
    fig, axs = plt.subplots(2, 2, figsize=(FULL_W, 8.4), constrained_layout=True)
    axs = axs.ravel()
    width = np.diff(np.unique(TC.t))[0] if TC.t.nunique() > 1 else 15
    dodge = {c: (i - (len(conditions) - 1) / 2) * width * 0.8 / max(len(conditions), 1) for i, c in enumerate(conditions)}
    r = np.random.default_rng(7)
    x = PTS.t + PTS.condition.map(dodge).fillna(0) + r.uniform(-0.06, 0.06, len(PTS)) * width
    for ax, k in zip(axs, ['v', 'lam', 'L', 'div_per_h']):
        for c in conditions:
            if k != 'div_per_h':
                sel = (PTS.condition == c).to_numpy()
                ax.scatter(x[sel], PTS[k][sel], s=2.5, color=colors[c], alpha=0.22, lw=0, zorder=1)
            d = TC[(TC.condition == c) & (TC.n >= min_n)]
            ax.plot(d.t + (dodge[c] if k != 'div_per_h' else 0), d[k], color=colors[c], label=c, marker='o', ms=3, zorder=3)
        ax.axvspan(*window, color=BLISS_COLOR, alpha=0.07, lw=0)
        ax.set(xlabel='time (min)', title=labels[k], xlim=(0, t_end or TC.t.max() + width / 2))
    for ax, k in zip(axs[:3], ['v', 'lam', 'L']):
        lo, hi = np.nanpercentile(PTS[k], [1, 99])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    axs[2].set_yscale('log'); axs[2].set_ylim(max(np.nanpercentile(PTS.L, 0.5) * 0.8, 1e-3), np.nanpercentile(PTS.L, 99.5) * 1.2)
    axs[0].legend(loc='upper left')
    return fig


def rate_vs_length(W, conditions, colors, control, window, unit='px'):
    """one dot per mother; dashed curves of constant velocity λ = v / L; crosses: means of the panel and of the control"""
    n = len(conditions); ncols = 2 if n <= 4 else 3; nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(FULL_W, 4.5 * nrows), sharex=True, sharey=True,
                            constrained_layout=True, squeeze=False)
    axs = axs.ravel()
    Lmin, Lmax = np.nanpercentile(W.L, 0.5) * 0.7, np.nanpercentile(W.L, 99.5) * 1.3
    lam_lo, lam_hi = np.nanpercentile(W.lam, [0.5, 99.5])
    Lg = np.geomspace(Lmin, Lmax, 200)
    ctrl = W[W.condition == control]
    base = 10 ** np.floor(np.log10(max(np.nanmedian(W.v), 1e-9)))
    levels = base * np.array([0.25, 0.5, 1, 2, 4, 8])                   # round values of the velocity
    for ax, c in zip(axs, conditions):
        d = W[W.condition == c]
        for vv in levels:
            ax.plot(Lg, vv / Lg, ls='--', color='0.75', lw=0.9, zorder=0)
            if lam_lo < vv / Lmax * 1.1 < lam_hi:
                ax.text(Lmax, vv / Lmax, f' v={vv:g}', fontsize=10, color='0.5', va='center')
        if c != control:
            ax.scatter(ctrl.L, ctrl.lam, s=10, color='0.75', alpha=0.6, lw=0, zorder=1)
        ax.scatter(d.L, d.lam, s=10, color=colors[c], alpha=0.65, lw=0, zorder=2)
        ax.plot(ctrl.L.mean(), ctrl.lam.mean(), 'P', color='0.55', ms=11, mec='white', zorder=4)
        ax.plot(d.L.mean(), d.lam.mean(), 'P', color=INK, ms=11, mec='white', zorder=5)
        ax.annotate('', xy=(d.L.mean(), d.lam.mean()), xytext=(ctrl.L.mean(), ctrl.lam.mean()),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=1.2), zorder=5)
        rho = np.corrcoef(d.lam, d.L)[0, 1] if len(d) > 2 else np.nan
        ax.set(xscale='log', xlim=(Lmin, Lmax), ylim=(lam_lo - 0.1 * (lam_hi - lam_lo), lam_hi + 0.1 * (lam_hi - lam_lo)),
               title=f'{c} ({len(d)} mothers, corr(λ, L) = {rho:.2f})')
        ax.title.set_color(colors[c])
    for ax in axs[len(conditions):]:
        ax.axis('off')
    for ax in axs[(nrows - 1) * ncols:]:
        ax.set_xlabel(f'mean length in the window ({unit})')
    for ax in axs[::ncols]:
        ax.set_ylabel('mean rate λ (1/min)')
    fig.suptitle(f'One dot per mother, {window[0]:g}–{window[1]:g} min (grey: {control}; dashed: constant velocity)')
    return fig


def effect_bars(E0, EB, roles, window, colors=None):
    """
    left: log-effect on the velocity of each drug, the combination and Bliss, split into rate + length + coupling;
    right: observed − Bliss and observed − each drug alone, for each part, with 95 % intervals
    """
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 5.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    rows = ['A', 'B', 'combination', 'Bliss']
    for i, r in enumerate(rows):
        pos = neg = 0.0
        for k in ['lam', 'L', 'kappa']:
            val = E0[(k, r)]; base = pos if val >= 0 else neg
            ax.bar(i, val, bottom=base, color=PART_COLORS[k], width=0.6, edgecolor='white',
                   hatch='//' if r == 'Bliss' else None, label=PART_NAMES[k] if i == 0 else None)
            if val >= 0: pos += val
            else: neg += val
        lo, hi = EB[('v', r)].quantile([0.025, 0.975])
        ax.errorbar(i + 0.38, E0[('v', r)], yerr=[[E0[('v', r)] - lo], [hi - E0[('v', r)]]], fmt='D', color=INK, ms=5,
                    capsize=3, label='velocity (the sum)' if i == 0 else None)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([roles['A'], roles['B'], f"{roles['combination']}\n(observed)",
                                                 f"Bliss\n({roles['A']} + {roles['B']})"])
    ax.set(ylabel=f"log-effect relative to {roles['control']}",
           title=f'velocity = rate + length + coupling ({window[0]:g}–{window[1]:g} min)')
    ax.legend(ncol=2, loc='best'); ax.grid(axis='x', visible=False)
    ax = axs[1]
    colors = colors or {}
    comps = [('combination − Bliss', 'obs − Bliss', BLISS_COLOR),
             ('combination − A', f"obs − {roles['A']}", colors.get(roles['A'], '#3a9d5d')),
             ('combination − B', f"obs − {roles['B']}", colors.get(roles['B'], '#d9a21b'))]
    parts = ['v', 'lam', 'L', 'kappa']
    for j, (r, lab, col) in enumerate(comps):
        for i, k in enumerate(parts):
            y = i + (j - 1) * 0.22
            lo, hi = EB[(k, r)].quantile([0.025, 0.975])
            ax.errorbar(E0[(k, r)], y, xerr=[[E0[(k, r)] - lo], [hi - E0[(k, r)]]], fmt='o', color=col, ms=6, capsize=3,
                        label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(parts))); ax.set_yticklabels([PART_NAMES[k] for k in parts]); ax.invert_yaxis()
    ax.set(xlabel='difference of log-effects', title='combination against Bliss and each drug alone')
    ax.legend(loc='best'); ax.grid(axis='y', visible=False)
    return fig


def effects_time(ET, roles, colors, window):
    """log-effects of each drug, the combination and Bliss along time, for rate, length and velocity"""
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 4.8), sharey=True, constrained_layout=True)
    styles = [('A', roles['A'], colors[roles['A']], '-'), ('B', roles['B'], colors[roles['B']], '-'),
              ('combination', f"{roles['combination']} (observed)", colors[roles['combination']], '-'),
              ('Bliss', f"Bliss ({roles['A']} + {roles['B']})", BLISS_COLOR, '--')]
    for ax, k in zip(axs, ['lam', 'L', 'v']):
        for r, lab, col, ls in styles:
            d = ET[(ET.part == k) & (ET.row == r)]
            ax.fill_between(d.t, d.lo, d.hi, color=col, alpha=0.13, lw=0)
            ax.plot(d.t, d.effect, color=col, ls=ls, label=lab)
        ax.axhline(0, color=INK2, lw=0.8); ax.axvspan(*window, color=BLISS_COLOR, alpha=0.06, lw=0)
        ax.set(xlabel='centre of the window (min)', title={'lam': 'rate ε^λ', 'L': 'length ε^L', 'v': 'velocity ε^v'}[k])
    axs[0].set_ylabel(f"log-effect relative to {roles['control']}"); axs[0].legend(loc='best')
    return fig


# ================================================================================ Bliss on quantiles
FIT_COLOR = '#c0407a'
DEV_COLOR = '#b23a48'


def qb_quantiles(results, roles, colors, var_titles):
    """one row per variable: quantile functions with Bliss and the plane fit | log-effect at each rank"""
    n = len(results)
    fig, axs = plt.subplots(n, 2, figsize=(FULL_W, 4.3 * n), constrained_layout=True, squeeze=False)
    col = {r: colors[roles[r]] for r in ('control', 'A', 'B', 'combination')}
    for row, R in zip(axs, results):
        Q, e, ef, lev = R.Q, R.est, R.est_free, R.lev
        m = e['m']
        ax = row[0]
        for r in ('control', 'A', 'B', 'combination'):
            ax.plot(lev, Q[r], color=col[r], lw=2.6 if r == 'combination' else 1.7,
                    label=roles[r] + (' (observed)' if r == 'combination' else ''))
        ax.plot(lev[m], e['bliss'], ':', color=INK, lw=2.2, label='Bliss (1, 1)')
        ax.plot(lev[m], ef['fit'], '--', color=FIT_COLOR, lw=1.8, label=f"plane fit ({ef['alpha']:.2f}, {ef['beta']:.2f})")
        ax.set(xlabel='rank u (0 = lowest mother)', ylabel=f"units of the {roles['control']} median",
               title=var_titles[R.var])
        if R.var == 'L':
            from matplotlib.ticker import ScalarFormatter
            ax.set_yscale('log'); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.tick_params(axis='y', which='minor', labelsize=9)
        ax.legend(fontsize=10, loc='best')
        ax = row[1]
        pos = m & (Q['combination'] > 0)
        lg = lambda r: np.log(Q[r][pos] / Q['control'][pos])
        ax.plot(lev[pos], lg('A'), color=col['A'], label=roles['A'])
        ax.plot(lev[pos], lg('B'), color=col['B'], label=roles['B'])
        ax.plot(lev[pos], lg('A') + lg('B'), ':', color=INK, lw=2.2, label=f"Bliss ({roles['A']} + {roles['B']})")
        ax.plot(lev[pos], lg('combination'), color=col['combination'], lw=2.6, label=f"{roles['combination']} (observed)")
        ax.axhline(0, color=INK2, lw=0.8)
        y = np.r_[lg('A'), lg('B'), lg('A') + lg('B'), lg('combination')]
        lo, hi = np.percentile(y, [1, 99]); pad = 0.15 * (hi - lo)
        ax.set(xlabel='rank u', ylabel='log-effect ln ρ(u)', title='effect at each rank', ylim=(lo - pad, hi + pad))
        ax.legend(fontsize=10, loc='best')
    return fig


def qb_vectors3d(R, roles, colors, var_title):
    """left: the effects as vectors in the 3-d space of e_A, e_B, e_Bliss; right: the same vectors along the rank"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    V, P, e = R.vec, R.vec['P'], R.vec['e']
    col = {'A': colors[roles['A']], 'B': colors[roles['B']], 'obs': colors[roles['combination']]}
    fig = plt.figure(figsize=(FULL_W, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax3 = fig.add_subplot(gs[0], projection='3d'); ax = fig.add_subplot(gs[1])
    pts = np.array([np.zeros(3)] + [P[k] for k in ('A', 'B', 'sum', 'Bliss', 'obs', 'fit')])
    lo, hi = pts.min(0), pts.max(0); span = max((hi - lo).max(), 1e-9); pad = 0.12 * span
    gx, gy = np.meshgrid([lo[0] - pad, hi[0] + pad], [lo[1] - pad, hi[1] + pad])
    ax3.plot_surface(gx, gy, np.zeros_like(gx), color='0.6', alpha=0.08, lw=0)
    arrow = lambda p, c, lw, ls='-', lab=None: ax3.quiver(0, 0, 0, *p, color=c, lw=lw, ls=ls, label=lab,
                                                         arrow_length_ratio=0.07 * span / max(np.linalg.norm(p), 1e-9))
    for a, b in ((P['A'], P['sum']), (P['B'], P['sum'])):
        ax3.plot(*np.c_[a, b], color='0.6', ls=':', lw=1.2)
    arrow(P['A'], col['A'], 2.4, lab=roles['A']); arrow(P['B'], col['B'], 2.4, lab=roles['B'])
    arrow(P['sum'], '0.6', 1.6, '--', lab='plain sum')
    arrow(P['Bliss'], INK, 2.4, lab='Bliss'); arrow(P['fit'], FIT_COLOR, 1.8, '--', lab='plane fit')
    arrow(P['obs'], col['obs'], 3.0, lab=f"{roles['combination']} (observed)")
    ax3.plot(*np.c_[P['sum'], P['Bliss']], color=INK, ls=(0, (1, 1.5)), lw=1.4)          # cross term
    ax3.plot(*np.c_[P['Bliss'], P['obs']], color=DEV_COLOR, lw=2.0, label='observed − Bliss')
    for p in (P['Bliss'], P['obs']):                                                      # height above the plane
        ax3.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color='0.6', ls='--', lw=0.9)
    c = (lo + hi) / 2; h = span / 2 + pad
    ax3.set_xlim(c[0] - h, c[0] + h); ax3.set_ylim(c[1] - h, c[1] + h); ax3.set_zlim(c[2] - h, c[2] + h)
    try: ax3.set_box_aspect((1, 1, 1))
    except Exception: pass
    ax3.set_xlabel(f"u1 (along {roles['A']})", fontsize=10); ax3.set_ylabel(f"u2 (plane {roles['A']}, {roles['B']})", fontsize=10)
    ax3.set_zlabel('u3 (out of plane)', fontsize=10)
    ax3.tick_params(labelsize=8); ax3.view_init(elev=22, azim=-60)
    ax3.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0.0, 1.0))
    ax3.set_title(f'{var_title}: effects as vectors (obs outside this space: {V["out_obs"]:.3f})', fontsize=12)
    lev = R.lev[R.est['m']]
    ax.plot(lev, e['A'], color=col['A'], label=roles['A']); ax.plot(lev, e['B'], color=col['B'], label=roles['B'])
    ax.plot(lev, e['sum'], '--', color='0.6', label='plain sum')
    ax.plot(lev, e['Bliss'], ':', color=INK, lw=2.2, label='Bliss')
    ax.plot(lev, e['fit'], '--', color=FIT_COLOR, lw=1.6, label='plane fit')
    ax.plot(lev, e['obs'], color=col['obs'], lw=2.6, label=f"{roles['combination']} (observed)")
    ax.axhline(0, color=INK2, lw=0.8)
    y = np.r_[e['A'], e['B'], e['Bliss'], e['obs']]
    ylo, yhi = np.percentile(y, [1, 99]); ypad = 0.15 * (yhi - ylo)
    ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set(xlabel='rank u', ylabel=f"e_c(u) = Q_c − Q_{roles['control']} ({roles['control']}-median units)",
           title='the same vectors along the rank')
    ax.legend(fontsize=10, loc='best')
    return fig


def qb_alpha_beta(results, roles, colors, var_titles):
    """one panel per variable: bootstrap cloud of (α, β) with both free, 95 % ellipse, estimate and reference points"""
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    n = len(results)
    fig, axs = plt.subplots(1, n, figsize=(FULL_W, 4.9), constrained_layout=True, squeeze=False)
    cO = colors[roles['combination']]
    refs = {'Bliss (1,1)': ((1, 1), '*'), f"{roles['B']} alone (0,1)": ((0, 1), 's'),
            f"{roles['A']} alone (1,0)": ((1, 0), '^'), f"{roles['control']} (0,0)": ((0, 0), 'X')}
    for ax, R in zip(axs[0], results):
        D, ef, e = R.draws_free, R.est_free, R.est
        ax.scatter(D[:, 0], D[:, 1], s=5, color='0.6', alpha=0.35, lw=0, zorder=2)
        mu, S = D.mean(0), np.cov(D.T); w, V = np.linalg.eigh(S); thr = chi2.ppf(0.95, 2)
        ax.add_patch(Ellipse(mu, 2 * np.sqrt(w[1] * thr), 2 * np.sqrt(w[0] * thr),
                             angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fill=False, ec=cO, lw=1.8, zorder=3))
        ax.plot(ef['alpha'], ef['beta'], 'o', color=cO, ms=9, mec=INK, mew=1.1, zorder=5, label='estimate')
        for lab, (p, mk) in refs.items():
            ax.plot(*p, mk, color=INK, mfc='white' if mk != 'X' else INK, ms=10 if mk == '*' else 7, zorder=4, label=lab)
        pts = np.r_[np.percentile(D, [1, 99], axis=0), np.array([p for p, _ in refs.values()])]
        span = np.ptp(pts, axis=0).max(); pad = 0.12 * span
        cx, cy = (pts.min(0) + pts.max(0)) / 2
        ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad); ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
        ax.set_aspect('equal', adjustable='box')
        note = ''
        if not all(e['free']):
            which = roles['A'] if not e['free'][0] else roles['B']
            note = f'\n({which} effect < MIN_EFFECT: its exponent is not estimable)'
        ax.set(xlabel=f"α (exponent of {roles['A']})", ylabel=f"β (exponent of {roles['B']})",
               title=f"{var_titles[R.var]}: ({ef['alpha']:.2f}, {ef['beta']:.2f}){note}")
        ax.title.set_fontsize(12)
    axs[0, -1].legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    return fig
