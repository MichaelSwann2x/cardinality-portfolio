"""Cardinality-constrained and unconstrained mean-variance optimizers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


def unconstrained_min_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: Optional[float] = None,
    long_only: bool = True,
) -> np.ndarray:
    """Classical Markowitz min-variance (optionally with target return)."""
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    def obj(w):
        return float(w @ Sigma @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        cons.append({"type": "eq", "fun": lambda w: float(mu @ w) - target_return})

    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return w0
    w = res.x
    w = np.clip(w, 0, None) if long_only else w
    s = w.sum()
    return w / s if s > 0 else w0


def greedy_cardinality_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    K: int,
    target_return: Optional[float] = None,
) -> np.ndarray:
    """Greedy cardinality-constrained min-variance (always available).

    Rank by mu/vol, keep top K, solve MV on the subset.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    K = int(max(1, min(K, n)))

    vol = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    score = mu / vol
    top = np.argsort(score)[::-1][:K]

    mu_s = mu[top]
    Sig_s = Sigma[np.ix_(top, top)]
    w_s = unconstrained_min_variance(mu_s, Sig_s, target_return=target_return, long_only=True)

    w = np.zeros(n)
    w[top] = w_s
    return w


def solve_cardinality_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    K: int,
    target_return: Optional[float] = None,
    prefer_cvxpy: bool = True,
) -> np.ndarray:
    """Solve cardinality-constrained long-only min-variance.

    Tries cvxpy MIQP when available; otherwise greedy.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    K = int(max(1, min(K, n))

    if prefer_cvxpy:
        try:
            import cvxpy as cp

            w = cp.Variable(n)
            z = cp.Variable(n, boolean=True)
            risk = cp.quad_form(w, Sigma)
            cons = [cp.sum(w) == 1, w >= 0, w <= z, cp.sum(z) <= K]
            if target_return is not None:
                cons.append(mu @ w >= target_return)
            prob = cp.Problem(cp.Minimize(risk), cons)
            for solver in (cp.ECOS_BB, cp.SCIPY, cp.GLPK_MI, None):
                try:
                    if solver is None:
                        prob.solve(verbose=False)
                    else:
                        prob.solve(solver=solver, verbose=False)
                    if w.value is not None:
                        wv = np.asarray(w.value, dtype=float).ravel()
                        wv = np.clip(wv, 0, None)
                        s = wv.sum()
                        return wv / s if s > 0 else greedy_cardinality_portfolio(mu, Sigma, K, target_return)
                except Exception:
                    continue
        except ImportError:
            pass

    return greedy_cardinality_portfolio(mu, Sigma, K, target_return)
