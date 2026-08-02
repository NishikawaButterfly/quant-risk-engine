"""Reproducible portfolio-analytics and market-risk engine.

The package computes risk and performance evidence from price data the
caller supplies. It does not forecast anything.
"""

from quantrisk.backtest import (
    BacktestResult,
    PolicyWindow,
    RebalanceRecord,
    WeightPolicy,
    run_backtest,
)
from quantrisk.benchmark import BenchmarkComparison, compare_to_benchmark
from quantrisk.frontier import (
    FrontierPoint,
    annualized_mean_returns,
    efficient_frontier,
    minimum_variance_portfolio,
)
from quantrisk.metrics import (
    Drawdown,
    annualized_volatility,
    historical_cvar,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)
from quantrisk.portfolio import Portfolio, correlation_matrix, covariance_matrix
from quantrisk.series import PriceSeries, align

__version__ = "0.0.1"

__all__ = [
    "BacktestResult",
    "BenchmarkComparison",
    "Drawdown",
    "FrontierPoint",
    "PolicyWindow",
    "Portfolio",
    "PriceSeries",
    "RebalanceRecord",
    "WeightPolicy",
    "align",
    "annualized_mean_returns",
    "annualized_volatility",
    "compare_to_benchmark",
    "correlation_matrix",
    "covariance_matrix",
    "efficient_frontier",
    "historical_cvar",
    "historical_var",
    "max_drawdown",
    "minimum_variance_portfolio",
    "parametric_var",
    "run_backtest",
    "sharpe_ratio",
    "sortino_ratio",
]
