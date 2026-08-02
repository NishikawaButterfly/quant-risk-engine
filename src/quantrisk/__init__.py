"""Reproducible portfolio-analytics and market-risk engine.

The package computes risk and performance evidence from price data the
caller supplies. It does not forecast anything.
"""

from quantrisk.series import PriceSeries, align

__version__ = "0.0.1"

__all__ = [
    "PriceSeries",
    "align",
]
