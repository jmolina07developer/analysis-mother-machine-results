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
