from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from scripts.fetch_yfinance import (
    ADVISORY_LOCK,
    acquire_advisory_lock,
    dataset_getters as existing_dataset_getters,
    supported_datasets,
)
from scripts.fetch_yfinance_optionals import (
    OPTIONAL_DATASETS,
    dataset_getters,
    get_option_chain,
)


class OptionalYfinanceDatasetsTests(unittest.TestCase):
    def test_advisory_lock_helper_uses_nonblocking_lock_and_commits(self) -> None:
        class FakeConnection:
            def __init__(self) -> None:
                self.queries: list[tuple[str, tuple[int, int]]] = []
                self.commits = 0

            def execute(self, query: str, params: tuple[int, int]) -> SimpleNamespace:
                self.queries.append((query, params))
                return SimpleNamespace(fetchone=lambda: (True,))

            def commit(self) -> None:
                self.commits += 1

        conn = FakeConnection()
        acquire_advisory_lock(conn)  # type: ignore[arg-type]

        self.assertEqual(conn.queries, [("SELECT pg_try_advisory_lock(%s, %s)", ADVISORY_LOCK)])
        self.assertEqual(conn.commits, 1)

    def test_optional_datasets_are_not_in_the_existing_getter_set(self) -> None:
        self.assertEqual(len(OPTIONAL_DATASETS), 13)
        self.assertFalse(set(OPTIONAL_DATASETS) & set(existing_dataset_getters(30)))
        self.assertEqual(
            supported_datasets(
                ["info", *OPTIONAL_DATASETS],
                existing_dataset_getters(30),
            ),
            ["info"],
        )

    def test_option_chain_collects_each_expiration(self) -> None:
        class FakeTicker:
            options = ("2026-12-18", "2027-01-15")

            def __init__(self) -> None:
                self.requested: list[str] = []

            def option_chain(self, expiration: str) -> SimpleNamespace:
                self.requested.append(expiration)
                return SimpleNamespace(
                    calls=pd.DataFrame([{"contractSymbol": f"{expiration}-C"}]),
                    puts=pd.DataFrame([{"contractSymbol": f"{expiration}-P"}]),
                )

        ticker = FakeTicker()
        result = get_option_chain(ticker)  # type: ignore[arg-type]

        self.assertEqual(ticker.requested, list(ticker.options))
        self.assertEqual(result["expirations"], list(ticker.options))
        self.assertEqual(
            result["chains"]["2026-12-18"]["calls"][0]["contractSymbol"],
            "2026-12-18-C",
        )

    def test_empty_option_expirations_are_recorded(self) -> None:
        ticker = SimpleNamespace(options=())
        self.assertEqual(
            get_option_chain(ticker),  # type: ignore[arg-type]
            {"expirations": [], "chains": {}},
        )

    def test_getter_set_includes_option_chain_handler(self) -> None:
        getters = dataset_getters()
        self.assertEqual(set(getters), set(OPTIONAL_DATASETS))
        self.assertIs(getters["option_chain"], get_option_chain)


if __name__ == "__main__":
    unittest.main()
