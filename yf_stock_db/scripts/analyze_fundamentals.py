from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ASSET_FILES = (
    ("tse_growth_assets.csv", "グロース"),
    ("tse_prime_assets.csv", "プライム"),
    ("tse_standard_assets.csv", "スタンダード"),
)
STATEMENT_ALIASES = {
    "revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
    "operating_income": ("Operating Income", "Total Operating Income As Reported"),
    "net_income": (
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ),
    "total_assets": ("Total Assets",),
    "equity": (
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
    ),
    "operating_cash_flow": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
    "capital_expenditure": ("Capital Expenditure", "Capital Expenditures"),
    "dividends_paid": ("Cash Dividends Paid", "Common Stock Dividend Paid"),
}


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def statement_value(statement: pd.DataFrame, aliases: tuple[str, ...], position: int = -1) -> float | None:
    values = statement_series(statement, aliases)
    if not values or abs(position) > len(values):
        return None
    return values[position][1]


def statement_series(statement: pd.DataFrame, aliases: tuple[str, ...]) -> list[tuple[pd.Timestamp, float]]:
    if statement.empty:
        return []
    normalized = {str(index).strip().casefold(): index for index in statement.index}
    for alias in aliases:
        row = normalized.get(alias.casefold())
        if row is not None:
            values: list[tuple[pd.Timestamp, float]] = []
            for column, value in statement.loc[row].items():
                number = finite_number(value)
                date_value = pd.to_datetime(column, errors="coerce")
                if number is not None and not pd.isna(date_value):
                    values.append((pd.Timestamp(date_value), number))
            return sorted(values)
    return []


def free_cash_flow_series(statement: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    operating_cash_flow = statement_series(
        statement, STATEMENT_ALIASES["operating_cash_flow"]
    )
    capital_expenditure = dict(
        statement_series(statement, STATEMENT_ALIASES["capital_expenditure"])
    )
    if not capital_expenditure:
        return []
    return [
        (period, operating_cash_flow_value + capital_expenditure[period])
        for period, operating_cash_flow_value in operating_cash_flow
        if period in capital_expenditure
    ]


def cagr(values: list[tuple[pd.Timestamp, float]], years: int) -> float | None:
    if years < 1 or len(values) < years + 1:
        return None
    start_date, start_value = values[-(years + 1)]
    end_date, end_value = values[-1]
    elapsed_years = (end_date - start_date).days / 365.25
    if start_value <= 0 or end_value <= 0 or elapsed_years <= 0:
        return None
    return (end_value / start_value) ** (1 / elapsed_years) - 1


def price_metrics(history: pd.DataFrame, risk_free_rate: float) -> dict[str, float | None]:
    result: dict[str, Any] = {
        "annualized_return": None,
        "annualized_volatility": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "price_as_of": None,
    }
    if history.empty or "Close" not in history:
        return result
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    close = close[close > 0]
    if len(close) < 2:
        return result

    returns = close.pct_change().dropna()
    if returns.empty:
        return result
    elapsed_years = (close.index[-1] - close.index[0]).days / 365.25
    if elapsed_years > 0:
        result["annualized_return"] = (close.iloc[-1] / close.iloc[0]) ** (1 / elapsed_years) - 1
    volatility = finite_number(returns.std(ddof=1) * math.sqrt(252))
    result["annualized_volatility"] = volatility
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    std_daily = finite_number(returns.std(ddof=1))
    if std_daily is not None and std_daily > 0:
        result["sharpe_ratio"] = finite_number(
            (returns.mean() - daily_risk_free) / std_daily * math.sqrt(252)
        )
    drawdown = close / close.cummax() - 1
    result["max_drawdown"] = finite_number(drawdown.min())
    result["price_as_of"] = close.index[-1].date().isoformat()
    return result


def dividend_metrics(
    dividends: pd.Series,
    price: float | None,
    net_income: float | None,
    free_cash_flow: float | None,
    payout_ratio: float | None,
    shares_outstanding: float | None,
    cash_dividends_paid: float | None,
    debt_to_equity: float | None,
) -> dict[str, Any]:
    clean = pd.to_numeric(dividends, errors="coerce").dropna()
    clean = clean[clean >= 0]
    today = pd.Timestamp.now(tz="UTC")
    if clean.empty:
        trailing_dividends = 0.0
    else:
        dividend_dates = clean.index
        if dividend_dates.tz is None:
            dividend_dates = dividend_dates.tz_localize("UTC")
        else:
            dividend_dates = dividend_dates.tz_convert("UTC")
        clean.index = dividend_dates
        trailing_dividends = finite_number(
            clean[clean.index >= today - pd.Timedelta(days=365)].sum()
        )
    dividend_yield = trailing_dividends / price if trailing_dividends is not None and price and price > 0 else None
    if payout_ratio is None and trailing_dividends is not None and net_income and net_income > 0 and shares_outstanding:
        payout_ratio = trailing_dividends * shares_outstanding / net_income
    annual: list[tuple[pd.Timestamp, float]] = []
    if not clean.empty:
        by_year = clean.groupby(clean.index.year).sum()
        current_year = today.year
        annual = [
            (pd.Timestamp(year=int(year), month=12, day=31), float(amount))
            for year, amount in by_year.items()
            if int(year) < current_year
        ]
    dividend_cagr = cagr(annual, 3)

    fcf_payout = None
    if free_cash_flow is not None and free_cash_flow > 0:
        if cash_dividends_paid is not None:
            fcf_payout = abs(cash_dividends_paid) / free_cash_flow
        elif trailing_dividends is not None and shares_outstanding:
            fcf_payout = trailing_dividends * shares_outstanding / free_cash_flow

    if (
        trailing_dividends == 0
        and (cash_dividends_paid is None or cash_dividends_paid == 0)
        and dividend_cagr is None
    ):
        potential, reason = "判定不能", "直近または過去の配当実績を確認できません"
    elif payout_ratio is None and fcf_payout is None and dividend_cagr is None:
        potential, reason = "判定不能", "配当または利益・キャッシュフローのデータ不足"
    elif (
        (payout_ratio is not None and payout_ratio > 0.75)
        or (fcf_payout is not None and fcf_payout > 0.8)
        or (free_cash_flow is not None and free_cash_flow <= 0)
        or (debt_to_equity is not None and debt_to_equity > 200)
    ):
        potential, reason = "低", "配当性向・FCF配当負担・負債水準のいずれかが高い"
    elif (
        payout_ratio is not None
        and payout_ratio <= 0.5
        and free_cash_flow is not None
        and free_cash_flow > 0
        and (dividend_cagr is None or dividend_cagr >= 0)
        and (debt_to_equity is None or debt_to_equity <= 200)
    ):
        potential, reason = "高", "配当性向に余裕があり、FCFは正。過去の減配・負債水準も確認"
    else:
        potential, reason = "中", "配当実績・配当性向・FCF・負債水準から追加確認が必要"

    return {
        "dividend_yield": dividend_yield,
        "payout_ratio": payout_ratio,
        "dividend_cagr_3y": dividend_cagr,
        "fcf_payout_ratio": fcf_payout,
        "cash_dividends_paid": cash_dividends_paid,
        "debt_to_equity": debt_to_equity,
        "dividend_growth_potential": potential,
        "dividend_growth_basis": reason,
    }


def load_assets() -> list[dict[str, str]]:
    by_ticker: dict[str, dict[str, str]] = {}
    for file_name, market_segment in ASSET_FILES:
        path = DATA_DIR / file_name
        if not path.is_file():
            raise FileNotFoundError(f"銘柄CSVがありません: {path}")
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        if not {"ticker", "company"}.issubset(frame.columns):
            raise ValueError(f"{path.name} には ticker,company 列が必要です")
        if frame.empty:
            raise ValueError(f"{path.name} に銘柄データがありません")
        for row in frame.to_dict(orient="records"):
            ticker = (row.get("ticker") or "").strip()
            company = (row.get("company") or "").strip()
            if not ticker:
                raise ValueError(f"{path.name} にtickerが空の行があります")
            if ticker in by_ticker:
                raise ValueError(f"銘柄CSV間に重複tickerがあります: {ticker}")
            by_ticker[ticker] = {
                "ticker": ticker,
                "company": company,
                "market_segment": market_segment,
                "source_file": file_name,
            }
    return list(by_ticker.values())


def fetch_market_history(ticker: str, years: int) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True, actions=False)


def analyze_asset(
    asset: dict[str, str],
    benchmark_history: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ticker = asset["ticker"]
    row: dict[str, Any] = {
        **asset,
        "sector": "不明",
        "sector_source": "yfinance",
        "industry": "不明",
        "benchmark_ticker": args.benchmark,
        "analysis_years": args.years,
        "analysis_as_of": datetime.now(timezone.utc).date().isoformat(),
        "fetch_status": "成功",
        "fetch_errors": "",
    }
    errors: list[str] = []
    company = yf.Ticker(ticker)
    info: dict[str, Any] = {}
    try:
        info = company.info or {}
        row["sector"] = str(info.get("sector") or "不明")
        row["industry"] = str(info.get("industry") or "不明")
        if row["sector"] == "不明":
            row["sector_source"] = "取得不可"
            errors.append("企業情報: セクタを取得できません")
    except Exception as error:
        row["sector_source"] = "取得失敗"
        errors.append(f"企業情報: {type(error).__name__}: {error}")

    statements: dict[str, pd.DataFrame] = {}
    for name, attr in (("income", "income_stmt"), ("balance", "balance_sheet"), ("cashflow", "cashflow")):
        try:
            statements[name] = getattr(company, attr)
            if statements[name].empty:
                errors.append(f"{attr}: データなし")
        except Exception as error:
            statements[name] = pd.DataFrame()
            errors.append(f"{attr}: {type(error).__name__}: {error}")

    income = statements.get("income", pd.DataFrame())
    balance = statements.get("balance", pd.DataFrame())
    cashflow = statements.get("cashflow", pd.DataFrame())
    revenue = statement_value(income, STATEMENT_ALIASES["revenue"])
    operating_income = statement_value(income, STATEMENT_ALIASES["operating_income"])
    net_income = statement_value(income, STATEMENT_ALIASES["net_income"])
    total_assets_values = statement_series(balance, STATEMENT_ALIASES["total_assets"])
    equity_values = statement_series(balance, STATEMENT_ALIASES["equity"])
    total_assets = total_assets_values[-1][1] if total_assets_values else None
    equity = equity_values[-1][1] if equity_values else None
    fcf = statement_value(cashflow, STATEMENT_ALIASES["operating_cash_flow"])
    capex = statement_value(cashflow, STATEMENT_ALIASES["capital_expenditure"])
    if fcf is not None and capex is not None:
        fcf += capex
    cash_dividends_paid = statement_value(cashflow, STATEMENT_ALIASES["dividends_paid"])
    if revenue is None:
        errors.append("財務指標: 売上高を取得できません")
    if operating_income is None:
        errors.append("財務指標: 営業利益を取得できません")
    if net_income is None:
        errors.append("財務指標: 純利益を取得できません")
    if total_assets is None or equity is None:
        errors.append("財務指標: 総資産または自己資本を取得できません")
    if fcf is None:
        errors.append("財務指標: フリーキャッシュフローを算出できません")

    for name, value in (
        ("revenue", revenue),
        ("operating_income", operating_income),
        ("net_income", net_income),
        ("total_assets", total_assets),
        ("equity", equity),
        ("free_cash_flow", fcf),
    ):
        row[name] = value
    row["operating_margin"] = (
        operating_income / revenue
        if operating_income is not None and revenue is not None and revenue > 0
        else None
    )
    row["net_margin"] = (
        net_income / revenue
        if net_income is not None and revenue is not None and revenue > 0
        else None
    )
    average_equity = (
        sum(value for _, value in equity_values[-2:]) / len(equity_values[-2:])
        if equity_values
        else None
    )
    row["roe"] = (
        net_income / average_equity
        if net_income is not None and average_equity and average_equity > 0
        else finite_number(info.get("returnOnEquity"))
    )
    average_assets = (
        sum(value for _, value in total_assets_values[-2:]) / len(total_assets_values[-2:])
        if total_assets_values
        else None
    )
    row["roa"] = (
        net_income / average_assets
        if net_income is not None and average_assets and average_assets > 0
        else finite_number(info.get("returnOnAssets"))
    )
    row["revenue_cagr"] = cagr(statement_series(income, STATEMENT_ALIASES["revenue"]), args.growth_years)
    row["operating_income_cagr"] = cagr(
        statement_series(income, STATEMENT_ALIASES["operating_income"]), args.growth_years
    )
    row["free_cash_flow_cagr"] = cagr(
        free_cash_flow_series(cashflow), args.growth_years
    )

    try:
        history = company.history(period=f"{args.years}y", auto_adjust=True, actions=False)
        price_values = price_metrics(history, args.risk_free_rate)
        row.update(price_values)
        close = pd.to_numeric(history.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        current_price = finite_number(close.iloc[-1]) if not close.empty else None
        if history.empty or current_price is None:
            errors.append("株価履歴: データなし")
    except Exception as error:
        current_price = None
        errors.append(f"株価履歴: {type(error).__name__}: {error}")
        row.update(price_metrics(pd.DataFrame(), args.risk_free_rate))

    benchmark_metrics = price_metrics(benchmark_history, args.risk_free_rate)
    row["benchmark_annualized_return"] = benchmark_metrics["annualized_return"]
    row["benchmark_sharpe_ratio"] = benchmark_metrics["sharpe_ratio"]
    row["benchmark_max_drawdown"] = benchmark_metrics["max_drawdown"]
    row["benchmark_as_of"] = benchmark_metrics["price_as_of"]
    stock_return = row.get("annualized_return")
    market_return = benchmark_metrics["annualized_return"]
    row["market_excess_return"] = (
        stock_return - market_return
        if stock_return is not None and market_return is not None
        else None
    )

    try:
        dividends = company.dividends
    except Exception as error:
        dividends = pd.Series(dtype=float)
        errors.append(f"配当履歴: {type(error).__name__}: {error}")
    dividend_info = dividend_metrics(
        dividends,
        current_price,
        net_income,
        fcf,
        finite_number(info.get("payoutRatio")),
        finite_number(info.get("sharesOutstanding")),
        cash_dividends_paid,
        finite_number(info.get("debtToEquity")),
    )
    row.update(dividend_info)

    row["financial_data_as_of"] = (
        max(str(column)[:10] for frame in statements.values() for column in frame.columns)
        if any(not frame.empty for frame in statements.values())
        else None
    )
    row["fetch_status"] = "一部取得失敗" if errors else "成功"
    row["fetch_errors"] = " | ".join(errors)
    row["filter_pass"] = passes_filters(row, args)
    return row


def passes_filters(row: dict[str, Any], args: argparse.Namespace) -> bool | None:
    checks: list[bool] = []
    for field, threshold, comparison in (
        ("operating_margin", args.min_operating_margin, "min"),
        ("net_margin", args.min_net_margin, "min"),
        ("roe", args.min_roe, "min"),
        ("roa", args.min_roa, "min"),
        ("revenue_cagr", args.min_revenue_cagr, "min"),
        ("operating_income_cagr", args.min_operating_income_cagr, "min"),
        ("free_cash_flow_cagr", args.min_free_cash_flow_cagr, "min"),
        ("sharpe_ratio", args.min_sharpe, "min"),
        ("max_drawdown", args.max_drawdown, "drawdown"),
    ):
        if threshold is None:
            continue
        value = row.get(field)
        if value is None:
            checks.append(False)
        elif comparison == "drawdown":
            checks.append(value >= threshold)
        else:
            checks.append(value >= threshold)
    return all(checks) if checks else None


def add_sector_scores(frame: pd.DataFrame) -> pd.DataFrame:
    score_fields = {
        "operating_margin": 0.10,
        "net_margin": 0.10,
        "roe": 0.10,
        "revenue_cagr": 0.10,
        "operating_income_cagr": 0.10,
        "free_cash_flow_cagr": 0.10,
        "sharpe_ratio": 0.25,
        "max_drawdown": 0.15,
    }
    for field in score_fields:
        if field not in frame:
            frame[field] = pd.NA
    frame["sector_score"] = pd.NA
    frame["score_metric_count"] = frame[list(score_fields)].notna().sum(axis=1)
    frame["score_data_status"] = frame["score_metric_count"].map(
        lambda count: "十分" if count == len(score_fields) else "不足"
    )
    for _, indices in frame.groupby("sector", dropna=False).groups.items():
        group = frame.loc[indices]
        group = group.loc[group["score_data_status"] == "十分"]
        if group.empty:
            continue
        weighted = pd.Series(0.0, index=group.index)
        total_weight = pd.Series(0.0, index=group.index)
        for field, weight in score_fields.items():
            values = pd.to_numeric(group[field], errors="coerce")
            if field == "max_drawdown":
                values = -values
            percentiles = values.rank(pct=True)
            valid = percentiles.notna()
            weighted.loc[valid] += percentiles.loc[valid] * weight
            total_weight.loc[valid] += weight
        valid = total_weight > 0
        frame.loc[group.index[valid], "sector_score"] = (
            weighted.loc[valid] / total_weight.loc[valid] * 100
        )
    frame["sector_rank"] = frame.groupby("sector", dropna=False)["sector_score"].rank(
        ascending=False, method="min"
    )
    return frame


def safe_filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip(" ._")
    return result or "unknown"


def safe_company_filename(value: str) -> str:
    result = re.sub(r'[\x00-\x1f/\\:*?"<>|]+', "_", value).strip(" .")
    return result or "unknown"


REPORT_SECTIONS = (
    (
        "企業情報",
        (
            "ticker",
            "company",
            "sector",
            "industry",
            "market_segment",
            "source_file",
            "sector_source",
        ),
    ),
    (
        "財務・収益性・成長性",
        (
            "revenue",
            "operating_income",
            "net_income",
            "total_assets",
            "equity",
            "free_cash_flow",
            "operating_margin",
            "net_margin",
            "roe",
            "roa",
            "revenue_cagr",
            "operating_income_cagr",
            "free_cash_flow_cagr",
        ),
    ),
    (
        "株価・リスク調整リターン",
        (
            "analysis_years",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "benchmark_ticker",
            "benchmark_annualized_return",
            "benchmark_sharpe_ratio",
            "benchmark_max_drawdown",
            "market_excess_return",
            "price_as_of",
            "benchmark_as_of",
        ),
    ),
    (
        "配当",
        (
            "dividend_yield",
            "payout_ratio",
            "dividend_cagr_3y",
            "fcf_payout_ratio",
            "cash_dividends_paid",
            "debt_to_equity",
            "dividend_growth_potential",
            "dividend_growth_basis",
        ),
    ),
    (
        "スクリーニング結果・データ状態",
        (
            "filter_pass",
            "sector_score",
            "score_data_status",
            "score_metric_count",
            "sector_rank",
            "fetch_status",
            "fetch_errors",
            "financial_data_as_of",
            "analysis_as_of",
        ),
    ),
)
REPORT_LABELS = {
    "ticker": "Ticker",
    "company": "企業名",
    "sector": "セクタ",
    "industry": "業種",
    "market_segment": "市場区分",
    "source_file": "入力CSV",
    "sector_source": "セクタ情報の取得元",
    "revenue": "売上高",
    "operating_income": "営業利益",
    "net_income": "純利益",
    "total_assets": "総資産",
    "equity": "自己資本",
    "free_cash_flow": "フリーキャッシュフロー",
    "operating_margin": "営業利益率",
    "net_margin": "純利益率",
    "roe": "ROE",
    "roa": "ROA",
    "revenue_cagr": "売上高CAGR",
    "operating_income_cagr": "営業利益CAGR",
    "free_cash_flow_cagr": "FCF CAGR",
    "analysis_years": "株価分析期間（年）",
    "annualized_return": "年率リターン",
    "annualized_volatility": "年率ボラティリティ",
    "sharpe_ratio": "Sharpe比",
    "max_drawdown": "最大ドローダウン",
    "benchmark_ticker": "市場比較ティッカー",
    "benchmark_annualized_return": "市場指数の年率リターン",
    "benchmark_sharpe_ratio": "市場指数のSharpe比",
    "benchmark_max_drawdown": "市場指数の最大ドローダウン",
    "market_excess_return": "市場超過リターン",
    "price_as_of": "株価データ基準日",
    "benchmark_as_of": "市場指数データ基準日",
    "dividend_yield": "配当利回り",
    "payout_ratio": "配当性向",
    "dividend_cagr_3y": "3年配当CAGR",
    "fcf_payout_ratio": "FCF配当負担率",
    "cash_dividends_paid": "配当支払額",
    "debt_to_equity": "負債資本倍率",
    "dividend_growth_potential": "配当上昇余地（参考評価）",
    "dividend_growth_basis": "配当評価の根拠",
    "filter_pass": "フィルタ判定",
    "sector_score": "セクタ内スコア",
    "score_data_status": "スコア用データ状態",
    "score_metric_count": "スコア計算可能な指標数",
    "sector_rank": "セクタ内順位",
    "fetch_status": "データ取得状態",
    "fetch_errors": "取得・分析時の注意事項",
    "financial_data_as_of": "財務データ基準日",
    "analysis_as_of": "分析実行日",
}


def markdown_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:,.6g}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_ticker_report(row: dict[str, Any]) -> str:
    ticker = str(row.get("ticker") or "unknown")
    company = str(row.get("company") or ticker)
    lines = [
        f"# {company}（{ticker}）",
        "",
        "このレポートはファンダメンタル分析ツールの実行時点で取得できた情報をまとめたものです。",
        "",
    ]
    for section, fields in REPORT_SECTIONS:
        lines.extend((f"## {section}", "", "| 項目 | 値 |", "|---|---|"))
        for field in fields:
            if field in row:
                label = REPORT_LABELS.get(field, field)
                lines.append(f"| {label} | {markdown_value(row[field])} |")
        lines.append("")
    lines.extend(
        (
            "## 指標の読み方",
            "",
            "- 割合は小数表記です（`0.05` は5%）。",
            "- `—` はデータ未取得または計算不能を表し、0を意味しません。",
            "- 配当上昇余地は過去データと財務状態に基づく参考評価であり、将来予測ではありません。",
            "- 市場超過リターンは企業の年率リターンから市場指数の年率リターンを差し引いた値です。",
            "",
        )
    )
    return "\n".join(lines)


def ticker_report_path(ticker_dir: Path, row: dict[str, Any]) -> Path:
    ticker = safe_filename(str(row.get("ticker") or "unknown"))
    company = safe_company_filename(str(row.get("company") or ""))
    report_name = f"{ticker}_{company}.md" if company else f"{ticker}.md"
    return ticker_dir / report_name


def save_ticker_report(ticker_dir: Path, row: dict[str, Any]) -> None:
    path = ticker_report_path(ticker_dir, row)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(render_ticker_report(row), encoding="utf-8")
    os.replace(temporary_path, path)


def build_sector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary_numeric_fields = (
        "operating_margin",
        "revenue_cagr",
        "sharpe_ratio",
        "market_excess_return",
    )
    for field in summary_numeric_fields:
        if field not in frame:
            frame[field] = pd.NA
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return (
        frame.groupby("sector", dropna=False)
        .agg(
            company_count=("ticker", "count"),
            filter_pass_count=("filter_pass", lambda values: int(values.eq(True).sum())),
            scoreable_count=("score_data_status", lambda values: int(values.eq("十分").sum())),
            average_operating_margin=("operating_margin", "mean"),
            average_revenue_cagr=("revenue_cagr", "mean"),
            average_sharpe_ratio=("sharpe_ratio", "mean"),
            average_market_excess_return=("market_excess_return", "mean"),
            data_issue_count=("fetch_status", lambda values: int(values.ne("成功").sum())),
        )
        .reset_index()
        .sort_values("sector")
    )


def write_results(
    rows: list[dict[str, Any]] | pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    scored = add_sector_scores(rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.assign(filter_pass=scored["filter_pass"].map(
        lambda value: "未設定" if pd.isna(value) else ("合格" if value else "不合格")
    )).sort_values(
        ["sector", "sector_score"], ascending=[True, False], na_position="last"
    ).to_csv(
        output_dir / "all_companies.csv", index=False, encoding="utf-8-sig"
    )
    summary = build_sector_summary(scored)
    summary.to_csv(output_dir / "sector_summary.csv", index=False, encoding="utf-8-sig")

    sector_dir = output_dir / "by_sector"
    sector_dir.mkdir(exist_ok=True)
    for sector, group in scored.groupby("sector", dropna=False):
        group.assign(filter_pass=group["filter_pass"].map(
            lambda value: "未設定" if pd.isna(value) else ("合格" if value else "不合格")
        )).sort_values("sector_score", ascending=False, na_position="last").to_csv(
            sector_dir / f"{safe_filename(str(sector))}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    ticker_dir = output_dir / "by_ticker"
    ticker_dir.mkdir(exist_ok=True)
    for _, row in scored.iterrows():
        save_ticker_report(ticker_dir, row.to_dict())

    settings = {
        key: value
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    settings["output_dir"] = str(output_dir)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "settings": settings},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="yfinanceの財務・株価・配当データから日本株をセクタ別にスクリーニングします。"
    )
    parser.add_argument("--years", type=int, default=5, help="株価分析期間（年、既定: 5）")
    parser.add_argument("--growth-years", type=int, default=3, help="財務成長率の期間（年、既定: 3）")
    parser.add_argument(
        "--benchmark",
        default="^N225",
        help="市場比較用Yahoo Financeティッカー（既定: ^N225、日経平均）",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="年率無リスク金利（小数、既定: 0）")
    parser.add_argument("--pause-seconds", type=float, default=0.5, help="銘柄間の待機秒数（既定: 0.5）")
    parser.add_argument("--limit", type=int, help="試験実行用の最大銘柄数")
    parser.add_argument("--output-dir", type=Path, help="出力先（既定: reports/fundamental_analysis/<実行時刻>）")
    parser.add_argument("--min-operating-margin", type=float, help="営業利益率の最低値（例: 0.05 = 5%%）")
    parser.add_argument("--min-net-margin", type=float, help="純利益率の最低値")
    parser.add_argument("--min-roe", type=float, help="ROEの最低値")
    parser.add_argument("--min-roa", type=float, help="ROAの最低値")
    parser.add_argument("--min-revenue-cagr", type=float, help="売上高CAGRの最低値")
    parser.add_argument("--min-operating-income-cagr", type=float, help="営業利益CAGRの最低値")
    parser.add_argument("--min-free-cash-flow-cagr", type=float, help="FCF CAGRの最低値")
    parser.add_argument("--min-sharpe", type=float, help="Sharpe比の最低値")
    parser.add_argument(
        "--max-drawdown",
        type=float,
        help="最大ドローダウンの最低許容値（例: -0.4 は40%%下落まで許容）",
    )
    args = parser.parse_args()
    if args.years < 1 or args.growth_years < 1:
        parser.error("--years と --growth-years は1以上で指定してください")
    if args.pause_seconds < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--pause-seconds は0以上、--limit は1以上で指定してください")
    if args.risk_free_rate <= -1:
        parser.error("--risk-free-rate は -1 より大きい値を指定してください")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        assets = load_assets()
        if args.limit is not None:
            assets = assets[: args.limit]
        logging.info("銘柄数=%s, benchmark=%s", len(assets), args.benchmark)
        benchmark_history = fetch_market_history(args.benchmark, args.years)
        if benchmark_history.empty:
            raise RuntimeError(f"市場指数の株価データを取得できません: {args.benchmark}")

        output_dir = args.output_dir or (
            PROJECT_ROOT
            / "report"
            / "fundamental_analysis"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        ticker_dir = output_dir / "by_ticker"
        output_dir.mkdir(parents=True, exist_ok=True)
        ticker_dir.mkdir(exist_ok=True)
        progress_path = output_dir / "analysis_progress.jsonl"
        logging.info("逐次出力先=%s", output_dir)

        with progress_path.open("w", encoding="utf-8") as progress_file:
            for index, asset in enumerate(assets, start=1):
                logging.info("[%s/%s] %s", index, len(assets), asset["ticker"])
                try:
                    row = analyze_asset(asset, benchmark_history, args)
                except Exception as error:
                    logging.exception("銘柄分析に失敗しました: %s", asset["ticker"])
                    row = {
                        **asset,
                        "sector": "不明",
                        "industry": "不明",
                        "benchmark_ticker": args.benchmark,
                        "analysis_years": args.years,
                        "analysis_as_of": datetime.now(timezone.utc).date().isoformat(),
                        "fetch_status": "分析失敗",
                        "fetch_errors": f"{type(error).__name__}: {error}",
                        "filter_pass": False,
                    }

                row["sector_score"] = None
                row["score_metric_count"] = None
                row["score_data_status"] = "全銘柄処理後に算出"
                row["sector_rank"] = None
                progress_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                progress_file.flush()
                save_ticker_report(ticker_dir, row)
                if args.pause_seconds and index < len(assets):
                    time.sleep(args.pause_seconds)

        results = pd.read_json(progress_path, lines=True)
        write_results(results, output_dir, args)
        failures = int(results["fetch_status"].ne("成功").sum())
        logging.info(
            "完了: 銘柄=%s, データ取得・分析に問題=%s, 出力先=%s",
            len(results),
            failures,
            output_dir,
        )
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
