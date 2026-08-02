from __future__ import annotations

import math
import unittest

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

    def test_unsorted_and_duplicate_dates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PriceSeries("AAA", ("2026-01-06", "2026-01-05", "2026-01-07"), (1.0, 2.0, 3.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PriceSeries("AAA", ("2026-01-05", "2026-01-05", "2026-01-07"), (1.0, 2.0, 3.0))

    def test_nonpositive_and_nonfinite_prices_are_rejected(self) -> None:
        for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
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
