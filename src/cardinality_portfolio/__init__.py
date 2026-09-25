"""Cardinality-constrained portfolio optimization — improved package.

Hierarchical clustering of assets, cluster-level cardinality allocation,
and intra-cluster selection (MIQP when cvxpy is available, greedy otherwise).
Unconstrained Markowitz baseline for comparison.
"""

from .clustering import hierarchical_cluster_labels
from .optimize import (
    unconstrained_min_variance,
    greedy_cardinality_portfolio,
    solve_cardinality_portfolio,
)
from .frontier import efficient_frontier_cardinality

__version__ = "0.2.0"
__all__ = [
    "hierarchical_cluster_labels",
    "unconstrained_min_variance",
    "greedy_cardinality_portfolio",
    "solve_cardinality_portfolio",
    "efficient_frontier_cardinality",
]
