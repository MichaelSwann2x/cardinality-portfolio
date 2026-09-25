"""Tests for cardinality-portfolio."""

import numpy as np
import pytest

from cardinality_portfolio import (
    hierarchical_cluster_labels,
    unconstrained_min_variance,
    greedy_cardinality_portfolio,
    solve_cardinality_portfolio,
    efficient_frontier_cardinality,
)


def _synth(n_assets=8, t=200, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(t, 1))
    loads = rng.normal(size=(1, n_assets))
    noise = rng.normal(scale=0.5, size=(t, n_assets))
    R = f @ loads + noise
    mu = R.mean(axis=0)
    Sigma = np.cov(R, rowvar=False)
    Sigma = Sigma + 1e-6 * np.eye(n_assets)
    return R, mu, Sigma


def test_clustering():
    R, _, _ = _synth()
    labels = hierarchical_cluster_labels(R)
    assert labels.shape == (R.shape[1],)
    assert labels.min() >= 0


def test_unconstrained():
    _, mu, Sigma = _synth()
    w = unconstrained_min_variance(mu, Sigma)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-8).all()


def test_greedy_cardinality():
    _, mu, Sigma = _synth(n_assets=10)
    w = greedy_cardinality_portfolio(mu, Sigma, K=3)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.sum(w > 1e-6) <= 3


def test_solve_falls_back():
    _, mu, Sigma = _synth()
    w = solve_cardinality_portfolio(mu, Sigma, K=4, prefer_cvxpy=False)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.sum(w > 1e-6) <= 4


def test_frontier():
    _, mu, Sigma = _synth(n_assets=6)
    fr = efficient_frontier_cardinality(mu, Sigma, K_values=[2, 3, 5])
    assert len(fr["K"]) == 3
    assert fr["unconstrained_risk"] > 0
