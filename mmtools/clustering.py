"""Hierarchical clustering of trajectories, choice of the number of clusters, fractions per condition."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score


def rms_distance(X):
    """root-mean-square difference between every pair of rows (condensed form)"""
    return pdist(X, 'euclidean') / np.sqrt(X.shape[1])


def cluster(X, method='ward'):
    D = rms_distance(X)
    return D, linkage(D, method)


def choose_k(D, Z, units, ks=range(2, 9), n_boot=100, method='ward', seed=0, progress=None):
    """
    For each k: silhouette of the partition, and its stability when the units (areas) of each
    condition are resampled with replacement and the clustering is repeated (adjusted Rand index
    with the original partition, on the mothers drawn at least once).
    units: array with one label per mother, e.g. 'condition|experiment|area'; resampling is done
    within each condition (the part of the label before the first '|').
    n_boot = 0 skips the stability (NaN columns). progress: optional function called as progress(done, total).
    """
    ks = [k for k in ks if k < len(units)]
    rng = np.random.default_rng(seed)
    DM = squareform(D)
    ref = {k: fcluster(Z, k, 'maxclust') for k in ks}
    sil = {k: silhouette_score(DM, ref[k], metric='precomputed') if len(np.unique(ref[k])) > 1 else np.nan for k in ks}
    units = np.asarray(units)
    cond = np.array([u.split('|')[0] for u in units])
    by_cond = {c: np.unique(units[cond == c]) for c in np.unique(cond)}
    members = {u: np.flatnonzero(units == u) for u in np.unique(units)}
    ari = {k: [] for k in ks}
    for b in range(n_boot):
        if progress: progress(b, n_boot)
        idx = np.concatenate([members[u] for c, us in by_cond.items() for u in rng.choice(us, len(us), replace=True)])
        Zb = linkage(squareform(DM[np.ix_(idx, idx)], checks=False), method)
        _, first = np.unique(idx, return_index=True)
        for k in ks:
            ari[k].append(adjusted_rand_score(ref[k][idx[first]], fcluster(Zb, k, 'maxclust')[first]))
    if progress: progress(n_boot, n_boot)
    q = lambda v, f: f(v) if len(v) else np.nan
    return pd.DataFrame({
        'silhouette': sil,
        'stability (median ARI)': {k: q(v, np.median) for k, v in ari.items()},
        'ARI 10th pct': {k: q(v, lambda x: np.percentile(x, 10)) for k, v in ari.items()},
        'ARI 90th pct': {k: q(v, lambda x: np.percentile(x, 90)) for k, v in ari.items()},
        'smallest cluster (mothers)': {k: int(np.bincount(ref[k])[1:].min()) for k in ks},
    }).rename_axis('k')


def numbered_clusters(Z, k, key):
    """cut the tree in k clusters and number them 1..k by increasing mean of `key`"""
    lab = fcluster(Z, k, 'maxclust')
    order = pd.Series(key).groupby(lab).mean().sort_values().index
    return pd.Series(lab).map({c: i + 1 for i, c in enumerate(order)}).to_numpy()


def fractions(ID, ph, k, conditions, n_boot=1000, seed=1):
    """
    Share of the mothers of each condition in each phenotype, with a 95 % bootstrap interval
    over the areas (experiment × area) of the condition.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in conditions:
        sel = (ID.condition == c).to_numpy()
        d = ID[sel].assign(ph=ph[sel])
        unit = (d.experiment.astype(str) + '|' + d.area.astype(str)).to_numpy()
        us = np.unique(unit)
        counts = np.array([[(d.ph[unit == u] == j).sum() for j in range(1, k + 1)] for u in us])
        boot = np.array([counts[rng.integers(0, len(us), len(us))].sum(0) for _ in range(n_boot)])
        boot = boot / boot.sum(1, keepdims=True)
        tot = counts.sum(0)
        for j in range(k):
            rows.append(dict(condition=c, phenotype=f'P{j + 1}', mothers=int(tot[j]), fraction=tot[j] / tot.sum(),
                             lo=np.percentile(boot[:, j], 2.5), hi=np.percentile(boot[:, j], 97.5)))
    return pd.DataFrame(rows)
