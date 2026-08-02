"""Reproducible portfolio-analytics and market-risk engine.

The package computes risk and performance evidence from price data the
caller supplies. It does not forecast anything.
"""

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
from quantrisk.series import PriceSeries, align

__version__ = "0.0.1"

__all__ = [
    "Drawdown",
    "PriceSeries",
    "align",
    "annualized_volatility",
    "historical_cvar",
    "historical_var",
    "max_drawdown",
    "parametric_var",
    "sharpe_ratio",
    "sortino_ratio",
]
