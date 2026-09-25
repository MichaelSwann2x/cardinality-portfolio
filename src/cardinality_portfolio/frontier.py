"""Efficient frontier sweeps over cardinality K (new feature)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .optimize import solve_cardinality_portfolio, unconstrained_min_variance


def efficient_frontier_cardinality(
    mu: np.ndarray,
    Sigma: np.ndarray,
    K_values: Sequence[int],
    target_returns: Optional[Sequence[float]] = None,
) -> dict:
    """For each cardinality K, solve min-var (and optional target-return points)."""
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    w_unc = unconstrained_min_variance(mu, Sigma)
    unc_risk = float(w_unc @ Sigma @ w_unc)
    unc_ret = float(mu @ w_unc)

    results = {
        "K": [],
        "weights": [],
        "risk": [],
        "ret": [],
        "unconstrained_risk": unc_risk,
        "unconstrained_ret": unc_ret,
        "n_assets": n,
    }

    for K in K_values:
        K = int(K)
        if target_returns is None:
            w = solve_cardinality_portfolio(mu, Sigma, K)
            results["K"].append(K)
            results["weights"].append(w)
            results["risk"].append(float(w @ Sigma @ w))
            results["ret"].append(float(mu @ w))
        else:
            for tr in target_returns:
                w = solve_cardinality_portfolio(mu, Sigma, K, target_return=float(tr))
                results["K"].append(K)
                results["weights"].append(w)
                results["risk"].append(float(w @ Sigma @ w))
                results["ret"].append(float(mu @ w))

    return results
