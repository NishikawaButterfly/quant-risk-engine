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
from quantrisk.montecarlo import (
    MonteCarloResult,
    Percentiles,
    run_bootstrap,
    run_parametric_normal,
)
from quantrisk.portfolio import (
    CovarianceDiagnostics,
    Portfolio,
    correlation_matrix,
    covariance_matrix,
    validate_covariance,
)
from quantrisk.series import PriceSeries, align
from quantrisk.stress import (
    ShockStressResult,
    StressWindow,
    WindowStressResult,
    historical_stress,
    shock_stress,
)

__version__ = "0.2.0"

__all__ = [
    "BacktestResult",
    "BenchmarkComparison",
    "CovarianceDiagnostics",
    "Drawdown",
    "FrontierPoint",
    "MonteCarloResult",
    "Percentiles",
    "PolicyWindow",
    "Portfolio",
    "PriceSeries",
    "RebalanceRecord",
    "ShockStressResult",
    "StressWindow",
    "WeightPolicy",
    "WindowStressResult",
    "align",
    "annualized_mean_returns",
    "annualized_volatility",
    "compare_to_benchmark",
    "correlation_matrix",
    "covariance_matrix",
    "efficient_frontier",
    "historical_cvar",
    "historical_stress",
    "historical_var",
    "max_drawdown",
    "minimum_variance_portfolio",
    "parametric_var",
    "run_backtest",
    "run_bootstrap",
    "run_parametric_normal",
    "sharpe_ratio",
    "shock_stress",
    "sortino_ratio",
    "validate_covariance",
]
