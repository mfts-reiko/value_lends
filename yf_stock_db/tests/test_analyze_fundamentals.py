from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.analyze_fundamentals import (
    add_sector_scores,
    build_sector_summary,
    cagr,
    dividend_metrics,
    passes_filters,
    price_metrics,
    save_ticker_report,
    statement_value,
    write_results,
)


class FundamentalAnalysisTests(unittest.TestCase):
    def test_statement_value_uses_latest_period_even_if_columns_are_descending(self) -> None:
        statement = pd.DataFrame(
            [[300.0, 200.0, 100.0]],
            index=["Total Revenue"],
            columns=["2024-12-31", "2023-12-31", "2022-12-31"],
        )
        self.assertEqual(statement_value(statement, ("Total Revenue",)), 300.0)

    def test_cagr_returns_none_for_non_positive_base(self) -> None:
        values = [
            (pd.Timestamp("2021-12-31"), -10.0),
            (pd.Timestamp("2022-12-31"), 2.0),
            (pd.Timestamp("2023-12-31"), 5.0),
            (pd.Timestamp("2024-12-31"), 10.0),
        ]
        self.assertIsNone(cagr(values, 3))

    def test_price_metrics_calculates_max_drawdown(self) -> None:
        history = pd.DataFrame(
            {"Close": [100.0, 120.0, 90.0, 110.0]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        metrics = price_metrics(history, 0.0)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.25)
        self.assertIsNotNone(metrics["annualized_return"])

    def test_dividend_metrics_accepts_timezone_aware_history(self) -> None:
        today = pd.Timestamp.now(tz="Asia/Tokyo").normalize()
        dividends = pd.Series(
            [2.0, 2.0],
            index=pd.DatetimeIndex(
                [today - pd.Timedelta(days=180), today - pd.Timedelta(days=30)]
            ),
        )
        metrics = dividend_metrics(
            dividends,
            price=100.0,
            net_income=1000.0,
            free_cash_flow=500.0,
            payout_ratio=0.4,
            shares_outstanding=10.0,
            cash_dividends_paid=None,
            debt_to_equity=None,
        )
        self.assertAlmostEqual(metrics["dividend_yield"], 0.04)
        self.assertEqual(metrics["dividend_growth_potential"], "高")

    def test_empty_dividend_history_is_reported_without_crashing(self) -> None:
        metrics = dividend_metrics(
            pd.Series(dtype=float),
            price=100.0,
            net_income=1000.0,
            free_cash_flow=500.0,
            payout_ratio=0.0,
            shares_outstanding=10.0,
            cash_dividends_paid=None,
            debt_to_equity=None,
        )
        self.assertEqual(metrics["dividend_yield"], 0.0)
        self.assertEqual(metrics["dividend_growth_potential"], "判定不能")

    def test_missing_filtered_metric_fails_filter(self) -> None:
        args = SimpleNamespace(
            min_operating_margin=0.05,
            min_net_margin=None,
            min_roe=None,
            min_roa=None,
            min_revenue_cagr=None,
            min_operating_income_cagr=None,
            min_free_cash_flow_cagr=None,
            min_sharpe=None,
            max_drawdown=None,
        )
        self.assertFalse(passes_filters({"operating_margin": None}, args))

    def test_incomplete_company_is_excluded_from_sector_score(self) -> None:
        frame = add_sector_scores(
            pd.DataFrame(
                [
                    {
                        "ticker": "0000.T",
                        "sector": "不明",
                        "filter_pass": False,
                        "fetch_status": "分析失敗",
                    }
                ]
            )
        )
        summary = build_sector_summary(frame)
        self.assertEqual(frame.iloc[0]["score_data_status"], "不足")
        self.assertTrue(pd.isna(frame.iloc[0]["sector_score"]))
        self.assertEqual(summary.iloc[0]["scoreable_count"], 0)
        self.assertEqual(summary.iloc[0]["data_issue_count"], 1)

    def test_write_results_creates_readable_report_for_each_ticker(self) -> None:
        row = {
            "ticker": "1301.T",
            "company": "テスト企業",
            "sector": "Industrials",
            "filter_pass": None,
            "fetch_status": "成功",
            "fetch_errors": "",
            "operating_margin": 0.1,
            "revenue_cagr": 0.05,
            "sharpe_ratio": 0.8,
            "market_excess_return": 0.02,
            "dividend_growth_potential": "中",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_results(
                [row],
                output_dir,
                SimpleNamespace(output_dir=output_dir, years=5),
            )
            report_path = output_dir / "by_ticker" / "1301.T_テスト企業.md"
            self.assertTrue(report_path.is_file())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# テスト企業（1301.T）", report)
            self.assertIn("営業利益率", report)
            self.assertTrue((output_dir / "all_companies.csv").is_file())
            self.assertTrue((output_dir / "sector_summary.csv").is_file())

    def test_save_ticker_report_replaces_report_without_temp_file(self) -> None:
        row = {
            "ticker": "0001.T",
            "company": "企業",
            "operating_margin": 0.1,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            ticker_dir = Path(temporary_directory)
            save_ticker_report(ticker_dir, row)
            report_path = ticker_dir / "0001.T_企業.md"
            self.assertTrue(report_path.is_file())
            self.assertFalse((ticker_dir / "0001.T_企業.md.tmp").exists())


if __name__ == "__main__":
    unittest.main()
