from __future__ import annotations

import math
import unittest

from quantrisk.metrics import annualized_volatility
from quantrisk.series import MIN_PRICES, PriceSeries, align

DATES = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12")
PRICES = (100.0, 102.0, 99.0, 101.0, 98.0, 100.0)


def make_series(name: str = "AAA") -> PriceSeries:
    return PriceSeries(name=name, dates=DATES, prices=PRICES)


class PriceSeriesValidationTests(unittest.TestCase):
    def test_a_valid_series_constructs_and_reports_its_length(self) -> None:
        series = make_series()
        self.assertEqual(len(series), 6)
        self.assertEqual(series.name, "AAA")

    def test_minimum_length_is_three_prices(self) -> None:
        self.assertEqual(MIN_PRICES, 3)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            PriceSeries("AAA", ("2026-01-05", "2026-01-06"), (100.0, 101.0))

    def test_empty_and_blank_names_are_rejected(self) -> None:
        for name in ("", "   "):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                PriceSeries(name, DATES, PRICES)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "6 dates but 5 prices"):
            PriceSeries("AAA", DATES, PRICES[:5])

    def test_malformed_dates_are_rejected(self) -> None:
        for bad in ("2026-1-05", "not-a-date", "2026-13-01", "20260105", "2026-01-05T00:00"):
            with self.assertRaisesRegex(ValueError, "ISO|canonical"):
                PriceSeries("AAA", (bad, "2026-01-06", "2026-01-07"), (1.0, 2.0, 3.0))

    def test_malformed_date_errors_name_the_series_and_the_date(self) -> None:
        # A rejection must identify both the owning series and the
        # offending value: in a multi-series file, "date '2026-13-01'
        # is malformed" without the series name is a hunt.
        with self.assertRaisesRegex(ValueError, r"series 'AAA'.*'2026-13-01'"):
            PriceSeries("AAA", ("2026-13-01", "2026-01-06", "2026-01-07"), (1.0, 2.0, 3.0))
        with self.assertRaisesRegex(ValueError, r"series 'BBB'.*'2026-1-05'"):
            PriceSeries("BBB", ("2026-1-05", "2026-01-06", "2026-01-07"), (1.0, 2.0, 3.0))

    def test_unsorted_and_duplicate_dates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PriceSeries("AAA", ("2026-01-06", "2026-01-05", "2026-01-07"), (1.0, 2.0, 3.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PriceSeries("AAA", ("2026-01-05", "2026-01-05", "2026-01-07"), (1.0, 2.0, 3.0))

    def test_nonpositive_and_nonfinite_prices_are_rejected(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                PriceSeries("AAA", DATES[:3], (100.0, bad, 101.0))
        for bad in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                PriceSeries("AAA", DATES[:3], (100.0, bad, 101.0))

    def test_boolean_prices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a number"):
            PriceSeries("AAA", DATES[:3], (100.0, True, 101.0))


class ReturnTests(unittest.TestCase):
    def test_simple_returns_match_hand_values(self) -> None:
        returns = make_series().simple_returns()
        self.assertEqual(len(returns), 5)
        self.assertAlmostEqual(returns[0], 0.02, places=15)
        self.assertAlmostEqual(returns[1], 99.0 / 102.0 - 1.0, places=15)
        self.assertAlmostEqual(returns[4], 100.0 / 98.0 - 1.0, places=15)

    def test_log_returns_match_hand_values(self) -> None:
        returns = make_series().log_returns()
        self.assertEqual(len(returns), 5)
        self.assertAlmostEqual(returns[0], math.log(1.02), places=15)
        self.assertAlmostEqual(returns[2], math.log(101.0 / 99.0), places=15)

    def test_log_returns_sum_to_the_log_price_ratio(self) -> None:
        series = make_series()
        total = sum(series.log_returns())
        self.assertAlmostEqual(total, math.log(series.prices[-1] / series.prices[0]), places=12)


class SessionModelTests(unittest.TestCase):
    """Pin the session model of docs/methodology.md.

    Each row is one trading session and consecutive rows are
    consecutive sessions; the calendar gap between two successive
    dates carries no information and never enters the arithmetic.
    """

    def test_weekend_dates_are_accepted(self) -> None:
        # 2026-01-10 is a Saturday, 2026-01-11 a Sunday. Markets that
        # trade seven days a week (cryptocurrencies, some futures) are
        # legal input; under the session model the weekday of a row is
        # irrelevant, so no weekday check exists.
        series = PriceSeries(
            "COIN",
            ("2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12"),
            (100.0, 101.0, 99.0, 102.0),
        )
        self.assertEqual(len(series), 4)

    def test_a_weekend_gap_is_one_session_with_no_accrual(self) -> None:
        # Friday 2026-01-09 close to Monday 2026-01-12 close: three
        # calendar days, exactly one session return, p_mon / p_fri - 1.
        # Nothing accrues across the weekend.
        series = PriceSeries(
            "AAA", ("2026-01-08", "2026-01-09", "2026-01-12"), (100.0, 104.0, 91.0)
        )
        returns = series.simple_returns()
        self.assertEqual(len(returns), 2)
        self.assertEqual(returns[1], 91.0 / 104.0 - 1.0)

    def test_returns_and_annualization_are_calendar_invariant(self) -> None:
        # The same prices on a gapless grid and on a grid with weekend
        # and holiday gaps produce identical returns and identical
        # annualized volatility: dates label the sessions, and the 252
        # convention annualizes session counts, not calendar days.
        prices = (100.0, 102.0, 99.0, 101.0)
        gapless = PriceSeries(
            "AAA", ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"), prices
        )
        gapped = PriceSeries(
            "AAA", ("2026-01-09", "2026-01-12", "2026-01-20", "2026-02-02"), prices
        )
        self.assertEqual(gapless.simple_returns(), gapped.simple_returns())
        self.assertEqual(gapless.log_returns(), gapped.log_returns())
        self.assertEqual(
            annualized_volatility(gapless.simple_returns()),
            annualized_volatility(gapped.simple_returns()),
        )


class AlignTests(unittest.TestCase):
    def test_alignment_keeps_only_the_shared_dates(self) -> None:
        left = make_series("AAA")
        right = PriceSeries(
            "BBB",
            ("2026-01-02", "2026-01-05", "2026-01-07", "2026-01-09", "2026-01-12"),
            (50.0, 51.0, 52.0, 53.0, 54.0),
        )
        aligned_left, aligned_right = align((left, right))
        expected_dates = ("2026-01-05", "2026-01-07", "2026-01-09", "2026-01-12")
        self.assertEqual(aligned_left.dates, expected_dates)
        self.assertEqual(aligned_right.dates, expected_dates)
        self.assertEqual(aligned_left.prices, (100.0, 99.0, 98.0, 100.0))
        self.assertEqual(aligned_right.prices, (51.0, 52.0, 53.0, 54.0))
        self.assertEqual(aligned_left.name, "AAA")
        self.assertEqual(aligned_right.name, "BBB")

    def test_alignment_never_invents_prices(self) -> None:
        left = make_series("AAA")
        right = PriceSeries(
            "BBB",
            ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-13"),
            (50.0, 51.0, 52.0, 53.0),
        )
        aligned_left, aligned_right = align((left, right))
        # 2026-01-08 through 2026-01-13 are not shared; nothing is filled.
        self.assertEqual(aligned_left.dates, ("2026-01-05", "2026-01-06", "2026-01-07"))
        self.assertEqual(aligned_right.prices, (50.0, 51.0, 52.0))

    def test_alignment_requires_two_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            align((make_series(),))

    def test_alignment_rejects_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            align((make_series("AAA"), make_series("AAA")))

    def test_alignment_rejects_too_small_an_overlap(self) -> None:
        left = make_series("AAA")
        right = PriceSeries(
            "BBB",
            ("2026-01-05", "2026-01-06", "2026-02-02"),
            (50.0, 51.0, 52.0),
        )
        with self.assertRaisesRegex(ValueError, "share only 2 common dates"):
            align((left, right))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
