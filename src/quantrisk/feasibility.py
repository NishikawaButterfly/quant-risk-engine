"""Feasibility: prove a loaded spec can be evaluated before promising it.

:func:`~quantrisk.io.load_spec` checks that a spec is well formed and
that its cross-references hold, but a well-formed spec can still be
unevaluable: a constant series has no Sharpe ratio, a three-date grid
yields too few paired returns for a benchmark comparison, a short
position can drive the compounded value path to zero, and extreme
price ratios can overflow into non-finite figures that neither the
metrics layer nor JSON serialization accepts. Before this module,
``quantrisk validate`` accepted such specs and ``quantrisk run``
failed on them — validation did not imply runnability.

:func:`check_feasibility` closes that gap without restating a single
downstream rule. It evaluates the spec through
:func:`~quantrisk.report.evaluate_spec`, serializes the payload with
the run's own ``allow_nan=False`` discipline, and renders the report —
the exact code paths ``quantrisk run`` executes — and discards every
result. Whatever the run would refuse, the check refuses first, with
the same message; there is no second list of rules to fall out of sync.

The one deliberately special case is Monte Carlo. Its input caps are
enforced at parse time and a bankrupt path is absorbed at zero by
policy rather than raised, so the only way a simulation can fail after
validation is a terminal value overflowing to a non-finite float. For
the bootstrap mode that possibility is decided by arithmetic: every
draw is a historical portfolio return, so the terminal values are
bounded by ``(1 + max return) ** horizon`` and a cheap logarithmic
bound proves most specs safe without simulating. When the bound cannot
prove safety — and always for the parametric mode, whose normal draws
are unbounded — the check runs the simulation itself. The seed makes
that dry run bit-identical to the real one, and the parse-time caps
bound its cost, so feasibility stays decidable rather than assumed.

What the guarantee does not cover is publication: ``quantrisk run``
can still fail writing its artifacts — an output directory that
already holds results and no ``--force``, a permissions error — and no
spec check can rule that out. Validation implies evaluability, not a
writable filesystem.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace

from quantrisk.io import RunSpec
from quantrisk.report import evaluate_spec, render_report, results_payload


class SpecFeasibilityError(ValueError):
    """Raised when a well-formed spec cannot be evaluated to a report."""


def _bootstrap_certainly_finite(spec: RunSpec) -> bool:
    """True when arithmetic alone proves the bootstrap cannot overflow.

    Every bootstrap draw is one of the portfolio's historical daily
    returns. A run containing a factor at or below zero is absorbed at
    exactly 0.0 (the bankruptcy policy), so every surviving run is a
    product of factors in ``(0, 1 + max return]`` and every terminal
    value is at most ``(1 + max return) ** horizon``. The summary
    statistics square those values and sum them over the runs, so the
    whole simulation is finite whenever
    ``runs * ((1 + max) ** horizon) ** 2`` fits in a float — checked in
    logarithms so the check itself cannot overflow. A ``False`` here
    does not mean the simulation fails; it means safety needs the dry
    run to decide.
    """

    block = spec.monte_carlo
    if block is None:
        return True
    top = max(spec.portfolio.return_series(spec.asset_series))
    if top <= 0.0:
        # Every factor is in (0, 1]; products, sums, and squares all
        # stay bounded by the run count.
        return True
    budget = (math.log(sys.float_info.max) - math.log(block.runs)) / 2.0
    return block.horizon_days * math.log1p(top) <= budget


def _needs_monte_carlo_dry_run(spec: RunSpec) -> bool:
    block = spec.monte_carlo
    if block is None:
        return False
    if block.mode == "bootstrap":
        return not _bootstrap_certainly_finite(spec)
    # Parametric draws come from a normal distribution with unbounded
    # support, so no static bound can prove the terminals finite; only
    # the (seeded, capped, deterministic) simulation itself decides.
    return True


def check_feasibility(spec: RunSpec) -> None:
    """Raise :class:`SpecFeasibilityError` unless the spec can be evaluated.

    The check is the run itself with the results discarded: the spec is
    evaluated through :func:`~quantrisk.report.evaluate_spec`, the
    payload is serialized with ``allow_nan=False`` exactly as
    ``quantrisk run`` serializes it, and the report is rendered. The
    Monte Carlo block is skipped only when the bootstrap overflow bound
    proves it cannot produce a non-finite figure; otherwise it is
    simulated for real, which the mandatory seed makes bit-identical to
    the run's own simulation. A spec that passes therefore evaluates,
    serializes, and renders without error when run — the remaining
    failure modes of ``quantrisk run`` are filesystem ones (an occupied
    output directory, a permissions error), which no spec check can
    exclude.
    """

    candidate = spec if _needs_monte_carlo_dry_run(spec) else replace(spec, monte_carlo=None)
    try:
        evaluation = evaluate_spec(candidate)
        json.dumps(results_payload(evaluation), allow_nan=False)
        render_report(evaluation, source_name=spec.prices_csv_name)
    except (ValueError, ArithmeticError) as exc:
        raise SpecFeasibilityError(f"spec cannot be evaluated: {exc}") from exc
