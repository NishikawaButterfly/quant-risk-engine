from __future__ import annotations

import unittest

import quantrisk


class PackageTests(unittest.TestCase):
    def test_version_is_a_nonempty_string(self) -> None:
        self.assertIsInstance(quantrisk.__version__, str)
        self.assertTrue(quantrisk.__version__)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
