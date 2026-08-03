"""Minimum-variance and efficient-frontier portfolios.

Everything here is plain Markowitz mean-variance machinery over two
caller-supplied inputs: a vector of expected *annual* returns and the
*daily* sample covariance matrix from
:func:`~quantrisk.portfolio.covariance_matrix`. Volatilities are
annualized inside with the usual 252-day convention; scaling the
objective by a positive constant never moves a minimizer, so the daily
matrix can be optimized directly and only the reported volatility needs
the factor.

Expected returns are deliberately an explicit argument, never a
default. The obvious candidate — the historical mean return, offered
here as :func:`annualized_mean_returns` — is a notoriously noisy
estimator: its standard error is the volatility divided by the square
root of the sample size, which for daily equity data is roughly the
size of the mean itself over any realistic window. Feeding historical
means into a frontier is therefore a modelling decision the caller must
make visibly, not something this module smuggles in.

The unconstrained minimum-variance portfolio has a closed form. With
the sum-to-one constraint alone, the Lagrangian condition is
``Σw = λ1``, so the weights are ``Σ⁻¹1`` rescaled to sum to one. The
implementation calls :func:`numpy.linalg.solve` on that linear system
rather than forming ``Σ⁻¹`` explicitly: an explicit inverse costs more,
loses accuracy roughly with the square of the matrix's condition
number, and is never needed when only ``Σ⁻¹1`` is wanted. Everything
without a closed form — long-only bounds, target-return equality — goes
to SciPy's SLSQP, and the solver's answer is validated (success flag,
constraint residuals, bounds) rather than trusted.

Every covariance entering this module runs through the shared
:func:`~quantrisk.portfolio.validate_covariance` path first, so an
asymmetric or indefinite matrix is rejected at entry with its
eigenvalue named. Because both solvers pass the matrix through a
linear solve — which amplifies relative input noise by the matrix's
condition number — a matrix conditioned beyond
:data:`~quantrisk.portfolio.CONDITION_NUMBER_REFUSE_LIMIT` is refused
outright, and one beyond
:data:`~quantrisk.portfolio.CONDITION_NUMBER_WARN_LIMIT` proceeds with
a ``conditioning_warning`` carried on every resulting
:class:`FrontierPoint`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from quantrisk._validation import require_finite_numbers
from quantrisk.metrics import TRADING_DAYS_PER_YEAR
from quantrisk.portfolio import (
    CONDITION_NUMBER_REFUSE_LIMIT,
    CovarianceDiagnostics,
    _validate_aligned,
    validate_covariance,
)
from quantrisk.series import PriceSeries

#: Absolute tolerance on the solver's constraint residuals and bound
#: violations. SLSQP satisfies equality constraints far more tightly in
#: practice; anything past this tolerance means the result is not a
#: portfolio and is rejected rather than returned.
SOLVER_TOLERANCE = 1e-8


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One efficient-frontier portfolio.

    ``weights`` are fractions summing to one, in asset order.
    ``expected_return`` is the annual expected return actually attained
    (equal to the requested target up to solver tolerance) and
    ``volatility`` is annualized with the 252-day convention.
    ``condition_number`` is the covariance matrix's 2-norm condition
    number and ``conditioning_warning`` its disclosure when that number
    exceeds :data:`~quantrisk.portfolio.CONDITION_NUMBER_WARN_LIMIT`
    (``None`` otherwise); both describe the input matrix, so every
    point of one :func:`efficient_frontier` call carries the same pair.
    """

    weights: tuple[float, ...]
    expected_return: float
    volatility: float
    condition_number: float
    conditioning_warning: str | None


def _solvable_covariance(
    covariance: Sequence[Sequence[float]],
) -> tuple[npt.NDArray[np.float64], CovarianceDiagnostics]:
    """Validate a covariance for use in a linear solve, or refuse.

    Runs the shared entry validation, then applies this module's
    stricter standard: a solve against a matrix with 2-norm condition
    number beyond
    :data:`~quantrisk.portfolio.CONDITION_NUMBER_REFUSE_LIMIT`
    amplifies relative input noise by that factor, so the weights it
    would produce are noise presented as precision, and refusing beats
    returning them.
    """

    diagnostics = validate_covariance(covariance)
    if diagnostics.condition_number > CONDITION_NUMBER_REFUSE_LIMIT:
        raise ValueError(
            "covariance matrix is numerically singular: its 2-norm condition number "
            f"{diagnostics.condition_number:.3e} exceeds "
            f"{CONDITION_NUMBER_REFUSE_LIMIT:.0e}, so solved weights would amplify "
            "relative input noise by that factor; remove the redundant asset or "
            "shrink the covariance estimate"
        )
    matrix = np.array([tuple(row) for row in covariance], dtype=np.float64)
    return matrix, diagnostics


def _validate_expected_returns(
    expected_returns: Sequence[float], size: int
) -> npt.NDArray[np.float64]:
    values = tuple(expected_returns)
    if len(values) != size:
        raise ValueError(
            f"got {len(values)} expected returns for a {size}-asset covariance matrix"
        )
    return np.array(require_finite_numbers(values, "expected return"), dtype=np.float64)


def _annualized_volatility(
    matrix: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> float:
    variance = float(weights @ matrix @ weights)
    # Defense in depth: entry validation already rejects indefinite
    # matrices, so a negative variance here means a bug, not bad input.
    if variance < -SOLVER_TOLERANCE:
        raise ValueError(
            "covariance matrix is not positive semidefinite: "
            f"portfolio variance came out {variance!r}"
        )
    # Rounding can leave a zero variance infinitesimally negative.
    return math.sqrt(max(variance, 0.0) * TRADING_DAYS_PER_YEAR)


def _validated_weights(
    raw: npt.NDArray[np.float64], *, long_only: bool
) -> npt.NDArray[np.float64]:
    """Check solver output against the constraints, then tidy it.

    The weight sum must be one and, long-only, no weight may sit more
    than :data:`SOLVER_TOLERANCE` below zero. Long-only weights within
    the tolerance of zero are float dust from an active bound: they
    snap to exactly zero and the vector is renormalized, so an excluded
    asset reads as excluded rather than as ``1e-17`` of the portfolio.
    """

    if abs(float(raw.sum()) - 1.0) > SOLVER_TOLERANCE:
        raise ValueError(f"solver weights sum to {float(raw.sum())!r}, not 1")
    if not long_only:
        return raw
    lowest = float(raw.min())
    if lowest < -SOLVER_TOLERANCE:
        raise ValueError(f"solver produced a negative long-only weight {lowest!r}")
    snapped = np.where(raw < SOLVER_TOLERANCE, 0.0, raw)
    return snapped / float(snapped.sum())


def _solve_qp(
    matrix: npt.NDArray[np.float64],
    initial: npt.NDArray[np.float64],
    constraints: list[dict[str, object]],
    *,
    long_only: bool,
) -> npt.NDArray[np.float64]:
    """Minimize ``w'Σw`` under the given equality constraints with SLSQP."""

    result = minimize(
        lambda weights: float(weights @ matrix @ weights),
        initial,
        jac=lambda weights: 2.0 * (matrix @ weights),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(initial) if long_only else None,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not result.success:
        raise ValueError(f"SLSQP did not converge: {result.message}")
    return _validated_weights(np.asarray(result.x, dtype=np.float64), long_only=long_only)


def _sum_to_one_constraint(size: int) -> dict[str, object]:
    return {
        "type": "eq",
        "fun": lambda weights: float(weights.sum() - 1.0),
        "jac": lambda weights: np.ones(size, dtype=np.float64),
    }


def annualized_mean_returns(aligned_series: Sequence[PriceSeries]) -> tuple[float, ...]:
    """Mean daily simple return times 252 for each series, in input order.

    A convenience for feeding :func:`efficient_frontier`, and nothing
    more. The sample mean is a *noisy* estimator of expected return —
    its standard error is the volatility over the square root of the
    sample size, comparable to the mean itself on realistic daily
    windows — which is exactly why the frontier functions demand
    expected returns explicitly instead of defaulting to this helper.
    The series must already share one date grid, so these means describe
    the same period the covariance matrix does.
    """

    items = _validate_aligned(aligned_series)
    means = []
    for item in items:
        returns = item.simple_returns()
        means.append(sum(returns) / len(returns) * TRADING_DAYS_PER_YEAR)
    return tuple(means)


def minimum_variance_portfolio(
    covariance: Sequence[Sequence[float]], *, long_only: bool = False
) -> tuple[float, ...]:
    """Weights of the minimum-variance portfolio, in asset order.

    ``covariance`` is the daily sample covariance matrix from
    :func:`~quantrisk.portfolio.covariance_matrix`. Unconstrained, the
    answer is closed-form: minimizing ``w'Σw`` subject only to
    ``1'w = 1`` gives ``Σw = λ1``, so the weights are the solution of
    ``Σx = 1`` rescaled to sum to one. That system goes through
    :func:`numpy.linalg.solve` — never an explicit inverse, which costs
    more and amplifies conditioning error for no benefit. With
    ``long_only=True`` there is no closed form and SLSQP minimizes the
    same quadratic under bounds; the solver result is validated, not
    trusted. A singular or near-singular matrix (a redundant asset) has
    no meaningful answer and is refused at entry by the conditioning
    gate; callers wanting the diagnosis itself should use
    :func:`~quantrisk.portfolio.validate_covariance`, since this
    function's return type is the bare weights tuple.
    """

    matrix, _ = _solvable_covariance(covariance)
    size = len(matrix)
    if long_only:
        initial = np.full(size, 1.0 / size)
        weights = _solve_qp(matrix, initial, [_sum_to_one_constraint(size)], long_only=True)
        return tuple(float(value) for value in weights)
    # Defense in depth: the eigenvalue validation and the conditioning
    # gate already reject singular and indefinite matrices at entry, so
    # these two guards should be unreachable; they stay because the
    # closed form must never return weights from a solve that failed.
    try:
        base = np.linalg.solve(matrix, np.ones(size, dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "covariance matrix is singular; the minimum-variance weights are not unique"
        ) from error
    total = float(base.sum())
    if total <= 0.0:
        raise ValueError(
            "covariance matrix is not positive definite; "
            "it cannot be a covariance matrix of non-redundant assets"
        )
    return tuple(float(value) for value in base / total)


def _check_targets_reachable(
    targets: tuple[float, ...], mu: npt.NDArray[np.float64], *, long_only: bool
) -> None:
    lowest = float(mu.min())
    highest = float(mu.max())
    for target in targets:
        if long_only and not lowest <= target <= highest:
            raise ValueError(
                f"target return {target} is unreachable long-only; asset expected "
                f"returns span [{lowest}, {highest}]"
            )
        if lowest == highest and target != lowest:
            raise ValueError(
                f"every asset has expected return {lowest}, so no combination of "
                f"weights summing to one can reach {target}"
            )


def _frontier_initial(mu: npt.NDArray[np.float64], target: float) -> npt.NDArray[np.float64]:
    """A start satisfying both equality constraints exactly.

    Interpolate (or, unconstrained, extrapolate) between the lowest- and
    highest-return assets; for a long-only-reachable target the start is
    feasible for the bounds too.
    """

    lowest = int(np.argmin(mu))
    highest = int(np.argmax(mu))
    spread = float(mu[highest] - mu[lowest])
    fraction = (target - float(mu[lowest])) / spread if spread > 0.0 else 0.0
    initial = np.zeros(len(mu), dtype=np.float64)
    initial[lowest] = 1.0 - fraction
    initial[highest] += fraction
    return initial


def efficient_frontier(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    target_returns: Sequence[float],
    *,
    long_only: bool = False,
) -> tuple[FrontierPoint, ...]:
    """The minimum-variance portfolio for each target return, in order.

    ``expected_returns`` are annual and supplied by the caller — see the
    module docstring for why there is no historical-mean default —
    while ``covariance`` is the daily matrix from
    :func:`~quantrisk.portfolio.covariance_matrix`; the reported
    volatility is annualized inside with the 252-day convention. Each
    target return becomes one SLSQP problem: minimize ``w'Σw`` under
    ``1'w = 1`` and ``μ'w = target``, plus ``0 ≤ w ≤ 1`` when
    ``long_only``. Long-only, a target outside the range of asset
    expected returns is unreachable and rejected up front; without
    bounds any target is reachable through leverage unless every asset
    has the same expected return, in which case only that value is.
    Volatility along the returned points falls until the global
    minimum-variance return and rises after it — the frontier is convex
    in the target — and the tests assert exactly that shape. Every
    returned point carries the covariance's 2-norm condition number
    and, past the warn limit, its ``conditioning_warning``; past the
    refuse limit the whole call is refused at entry.
    """

    matrix, diagnostics = _solvable_covariance(covariance)
    mu = _validate_expected_returns(expected_returns, len(matrix))
    targets = require_finite_numbers(tuple(target_returns), "target return")
    if not targets:
        raise ValueError("need at least one target return")
    _check_targets_reachable(targets, mu, long_only=long_only)
    # With every expected return equal, the (already validated) return
    # constraint is the budget constraint scaled — a redundant row that
    # stalls SLSQP — so it is dropped rather than handed to the solver.
    degenerate = float(mu.min()) == float(mu.max())
    points = []
    for target in targets:
        constraints = [_sum_to_one_constraint(len(matrix))]
        if not degenerate:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda weights, level=target: float(weights @ mu - level),
                    "jac": lambda weights: mu,
                }
            )
        weights = _solve_qp(
            matrix, _frontier_initial(mu, target), constraints, long_only=long_only
        )
        attained = float(weights @ mu)
        if abs(attained - target) > SOLVER_TOLERANCE:
            raise ValueError(f"solver missed the target return {target} by {attained - target!r}")
        points.append(
            FrontierPoint(
                weights=tuple(float(value) for value in weights),
                expected_return=float(weights @ mu),
                volatility=_annualized_volatility(matrix, weights),
                condition_number=diagnostics.condition_number,
                conditioning_warning=diagnostics.conditioning_warning,
            )
        )
    return tuple(points)
