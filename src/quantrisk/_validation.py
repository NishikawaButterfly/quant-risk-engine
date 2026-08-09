"""The engine's single validation path for numbers crossing a boundary.

Every public entry point that accepts a number — prices, weights,
rates, shocks, confidence levels, expected and target returns, spec
fields — funnels it through :func:`require_finite_number` (or the
sequence form :func:`require_finite_numbers`), so the whole engine
rejects the same inputs with the same message shape: values that are
not real numbers, ``bool``, and the non-finite floats ``nan``,
``inf``, and ``-inf``.

``bool`` is rejected explicitly, and first, because Python's ``bool``
subclasses ``int``: without the check, ``True`` would sail through any
numeric type test and silently compute as ``1.0`` — a corrupted spec
or a comparison typo becoming a weight, a rate, or a shock.

Acceptance is decided by :class:`numbers.Real`, not by
``isinstance(value, int | float)``. NumPy's scalar types register with
the :mod:`numbers` tower, so ``numpy.float64`` (which subclasses
``float``) and ``numpy.int64`` (which does *not* subclass ``int``)
both pass and are normalized to a built-in ``float`` on the way in; a
check against the concrete built-ins would accept the former and
reject the latter for no reason a caller could predict. The trade-off
is stated rather than hidden: :class:`decimal.Decimal` deliberately
never registered with ``Real`` (mixing it with binary floats loses the
precision it exists for), so a ``Decimal`` is rejected here and must
be converted by the caller, visibly.

This module also owns :data:`MIN_VOLATILITY`, the engine's single
near-zero-variance tolerance, and its enforcement path
:func:`require_meaningful_volatility`: every place a volatility (or
its square, a variance) sits in a denominator guards it against this
one constant, so "too close to zero to divide by" means the same
thing at every site.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

#: The engine's single near-zero-variance tolerance, stated as a daily
#: volatility (sample standard deviation of daily simple returns).
#: Everywhere a volatility is a denominator — the Sharpe and Sortino
#: ratios, the correlation matrix, beta, risk contributions, the
#: information ratio — the guard compares against this one constant;
#: sites whose denominator is a *variance* compare against its square,
#: never against a second number.
#:
#: Why 1e-12. Prices enter the engine as float64, carrying relative
#: representation error up to eps/2 with eps = 2**-52 ≈ 2.2e-16, so a
#: simple return ``p1/p0 - 1`` computed from a truly constant price
#: path can come out nonzero by rounding alone, with magnitude of
#: order eps. The sample standard deviation of such noise returns is
#: bounded by their largest deviation — order 1e-16 — so any computed
#: daily volatility near that level can be an artifact of rounding
#: and proves nothing about the data. On the other side, the smallest
#: economically meaningful volatility is bounded below by the price
#: grid itself: a single move of one part in 1e-8 (a hundredth of a
#: cent on a $1000 price — finer than any traded tick) once in ten
#: thousand sessions already leaves a daily stddev around 1e-10.
#: 1e-12 sits in the gap between those two bounds: four orders of
#: magnitude above the largest rounding artifact (headroom for noise
#: accumulated through alignment, mean subtraction, and covariance
#: sums) and two below the smallest volatility a real price grid can
#: express. Data on either side of the line is classified by
#: arithmetic, not by tuning. See "Degenerate variance" in
#: ``docs/methodology.md``.
MIN_VOLATILITY = 1e-12


def require_meaningful_volatility(value: float, description: str) -> float:
    """``value`` back unless it lies below :data:`MIN_VOLATILITY`.

    ``value`` is a daily volatility (variance sites pass its square
    root). ``description`` states, in the caller's domain terms, what
    became undefined — it leads the message, and the uniform tail
    names the constant and the reason, so every rejection across the
    engine reads the same way and points at the same documentation.
    """

    if value < MIN_VOLATILITY:
        raise ValueError(
            f"{description}: {value!r} is below MIN_VOLATILITY ({MIN_VOLATILITY:.0e}), "
            "the near-zero-variance tolerance; a daily volatility this small is "
            "indistinguishable from float64 rounding noise (see 'Degenerate "
            "variance' in docs/methodology.md)"
        )
    return value


def require_finite_number(value: object, description: str) -> float:
    """``value`` as a built-in float, or ``ValueError`` naming ``description``.

    Rejects ``bool`` explicitly, rejects anything that is not a
    :class:`numbers.Real`, and rejects non-finite values. NumPy
    scalars (``float64`` and ``int64`` alike) pass through the numbers
    tower and come back as built-in floats.
    """

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{description} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} is {number!r}; it must be finite")
    return number


def require_finite_numbers(values: Sequence[object], description: str) -> tuple[float, ...]:
    """Every element as a built-in float, each error naming its index.

    The description carries the element's position, so a bad entry in
    a long sequence is reported as, say, ``weight [3] is not a
    number`` rather than as an anonymous failure somewhere in the
    input.
    """

    return tuple(
        require_finite_number(value, f"{description} [{index}]")
        for index, value in enumerate(values)
    )
