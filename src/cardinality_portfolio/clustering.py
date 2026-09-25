"""Hierarchical clustering of assets by correlation distance (from original notebook)."""

from __future__ import annotations

import numpy as np


def hierarchical_cluster_labels(returns: np.ndarray) -> np.ndarray:
    """Agglomerative clustering on correlation distance with largest-gap cut.

    Parameters
    ----------
    returns : array (T, N)
        Asset return matrix.

    Returns
    -------
    labels : array (N,)
        Integer cluster id per asset.
    """
    R = np.asarray(returns, dtype=float)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("returns must be (T, N) with N >= 2")

    tau = np.corrcoef(R, rowvar=False)
    tau = np.clip(tau, -1.0, 1.0)
    D = 1.0 - tau
    n = R.shape[1]
    C: list[list[int]] = [[i] for i in range(n)]

    def cluster_dist(Ci: list[int], Cj: list[int]) -> float:
        cent_i = D[Ci, :].mean(axis=0)
        cent_j = D[Cj, :].mean(axis=0)
        return (len(Ci) * len(Cj)) / (len(Ci) + len(Cj)) * float(np.linalg.norm(cent_i - cent_j) ** 2)

    merge_distances: list[float] = []
    merge_log: list[tuple] = []

    while len(C) > 1:
        min_d = float("inf")
        best = (0, 1)
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                d = cluster_dist(C[i], C[j])
                if d < min_d:
                    min_d = d
                    best = (i, j)
        i, j = best
        new = C[i] + C[j]
        merge_log.append((list(C[i]), list(C[j]), new, min_d))
        for idx in sorted([i, j], reverse=True):
            del C[idx]
        C.append(new)
        merge_distances.append(min_d)

    if len(merge_distances) < 2:
        return np.zeros(n, dtype=int)

    gaps = np.diff(merge_distances)
    largest_gap = int(np.argmax(gaps)) + 1
    threshold = merge_distances[largest_gap]

    clusters: list[set[int]] = [{i} for i in range(n)]
    for Ci, Cj, _, dist in merge_log:
        if dist > threshold:
            continue
        to_merge = [c for c in clusters if any(a in c for a in Ci) or any(a in c for a in Cj)]
        if len(to_merge) >= 2:
            merged = set().union(*to_merge)
            for c in to_merge:
                clusters.remove(c)
            clusters.append(merged)

    labels = np.empty(n, dtype=int)
    for cid, aset in enumerate(clusters):
        for a in aset:
            labels[a] = cid
    return labels
