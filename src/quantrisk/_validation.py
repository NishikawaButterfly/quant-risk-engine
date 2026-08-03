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
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence


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
