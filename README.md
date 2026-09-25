# Cardinality Portfolio (improved)

Clean packaging of cardinality-constrained mean-variance optimization, based on
the research notebook at
[mirkovicdev/Cardinality-Constrained-Portfolio-Optimization](https://github.com/mirkovicdev/Cardinality-Constrained-Portfolio-Optimization).

## What changed

| Area | Original | This package |
|------|----------|--------------|
| Structure | Single notebook | `src/` package + tests |
| Solver | cvxpy MIQP (when available) | cvxpy optional; **greedy always works** |
| Clustering | Hierarchical correlation distance | Same, cleaned |
| New | — | `efficient_frontier_cardinality`, pure-scipy unconstrained MV |
| Tooling | — | pyproject, ruff, pytest |

## Install

```bash
uv pip install -e ".[dev]"
# optional MIQP:
uv pip install -e ".[dev,solver]"
```

## Quick start

```python
import numpy as np
from cardinality_portfolio import (
    hierarchical_cluster_labels,
    unconstrained_min_variance,
    solve_cardinality_portfolio,
    efficient_frontier_cardinality,
)

labels = hierarchical_cluster_labels(returns)
mu = returns.mean(axis=0)
Sigma = np.cov(returns, rowvar=False)

w_free = unconstrained_min_variance(mu, Sigma)
w_k3 = solve_cardinality_portfolio(mu, Sigma, K=3)

frontier = efficient_frontier_cardinality(mu, Sigma, K_values=[2, 3, 5, 8])
```

## New features

1. **Greedy cardinality portfolio** — rank by mu/vol, keep top K, solve MV on the subset.
2. **efficient_frontier_cardinality** — sweep over K.
3. **Optional cvxpy MIQP** when installed.

## License

MIT.
