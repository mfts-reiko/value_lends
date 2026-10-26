from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "info", "fast_info", "income_stmt", "balance_sheet", "cashflow", "dividends",
    "major_holders", "institutional_holders", "mutualfund_holders",
    "insider_transactions", "insider_purchases", "insider_roster_holders", "history",
)
FIELD_LABELS = {
    "shortName": "会社名（短縮）",
    "longName": "会社名（正式）",
    "exchange": "取引所コード",
    "fullExchangeName": "取引所",
    "market": "市場コード",
    "sector": "セクター",
    "industry": "業種",
    "industryDisp": "業種（表示名）",
    "country": "国",
    "city": "本社所在都市",
    "address1": "本社住所",
    "address2": "本社住所（続き）",
    "zip": "郵便番号",
    "phone": "電話番号",
    "website": "企業Webサイト",
    "fullTimeEmployees": "従業員数",
    "currency": "通貨",
    "marketCap": "時価総額",
    "enterpriseValue": "企業価値",
    "regularMarketPrice": "直近株価",
    "lastPrice": "直近株価",
    "previousClose": "前日終値",
    "fiftyTwoWeekLow": "52週安値",
    "fiftyTwoWeekHigh": "52週高値",
    "yearLow": "年初来安値",
    "yearHigh": "年初来高値",
    "trailingPE": "実績PER",
    "forwardPE": "予想PER",
    "priceToBook": "PBR",
    "trailingEps": "実績EPS",
    "forwardEps": "予想EPS",
    "bookValue": "1株当たり純資産（BPS）",
    "dividendRate": "年間配当予想（1株当たり）",
    "dividendYield": "配当利回り（Yahoo返却値）",
    "payoutRatio": "配当性向",
    "returnOnEquity": "ROE（Yahoo返却値）",
    "returnOnAssets": "ROA（Yahoo返却値）",
    "beta": "ベータ値",
    "sharesOutstanding": "発行済株式数",
    "floatShares": "浮動株数",
    "heldPercentInsiders": "内部関係者保有割合",
    "heldPercentInstitutions": "機関投資家保有割合",
    "lastFiscalYearEnd": "前回決算期末",
    "nextFiscalYearEnd": "次回決算期末",
    "firstTradeDateMilliseconds": "Yahoo取得可能な最初の取引日",
    "longBusinessSummary": "事業概要",
    "companyOfficers": "役員情報",
    "Total Revenue": "売上高",
    "Operating Revenue": "営業収益",
    "Operating Income": "営業利益",
    "Pretax Income": "税引前利益（経常利益とは異なる場合があります）",
    "Net Income": "当期純利益",
    "Net Income Common Stockholders": "普通株主帰属当期純利益",
    "Basic EPS": "基本的EPS",
    "Diluted EPS": "希薄化後EPS",
    "Normalized Income": "調整後当期純利益",
    "Diluted Average Shares": "希薄化後加重平均株式数",
    "Total Assets": "総資産",
    "Stockholders Equity": "株主資本",
    "Common Stock Equity": "普通株主資本",
    "Total Debt": "有利子負債",
    "Net Debt": "純有利子負債",
    "Cash And Cash Equivalents": "現金及び現金同等物",
    "Working Capital": "運転資本",
    "Investment Properties": "投資不動産",
    "Properties": "不動産",
    "Land And Improvements": "土地・土地改良",
    "Buildings And Improvements": "建物・改良施設",
    "Net PPE": "有形固定資産純額",
    "Other Properties": "その他不動産",
    "Operating Cash Flow": "営業キャッシュフロー",
    "Investing Cash Flow": "投資キャッシュフロー",
    "Financing Cash Flow": "財務キャッシュフロー",
    "Free Cash Flow": "フリーキャッシュフロー",
    "Dividends": "1株当たり配当金",
    "Open": "始値",
    "High": "高値",
    "Low": "安値",
    "Close": "終値",
    "Adj Close": "調整後終値",
    "Volume": "出来高",
}
SHAREHOLDER_DATASETS = {
    "major_holders": "株主保有割合の概要",
    "institutional_holders": "機関投資家の保有銘柄",
    "mutualfund_holders": "投資信託の保有銘柄",
    "insider_transactions": "インサイダー取引履歴",
    "insider_purchases": "インサイダー売買集計",
    "insider_roster_holders": "役員・関係者の保有情報",
}
MAJOR_HOLDER_LABELS = {
    "insidersPercentHeld": "内部関係者保有割合",
    "institutionsPercentHeld": "機関投資家保有割合",
    "institutionsFloatPercentHeld": "機関投資家の浮動株保有割合",
    "institutionsCount": "機関投資家数",
}


def database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL がありません。.env を確認してください")
    prefix = "postgresql+psycopg://"
    return "postgresql://" + url.removeprefix(prefix) if url.startswith(prefix) else url


def fetch_tasks(conn: psycopg.Connection[Any], ticker: str, run_id: int | None) -> list[tuple[Any, ...]]:
    return conn.execute(
         """SELECT task.run_id, task.dataset, task.status, task.artifact_path,
                task.row_count, task.last_error, run.started_at
            FROM yf_fetch_task AS task
            JOIN yf_fetch_run AS run USING (run_id)
           WHERE task.ticker = %s AND task.dataset = ANY(%s)
             AND (%s::bigint IS NULL OR task.run_id = %s)
           ORDER BY task.run_id, task.dataset""",
        (ticker, list(DATASETS), run_id, run_id),
    ).fetchall()


def fetch_asset_info(
    conn: psycopg.Connection[Any], ticker: str, run_id: int | None
) -> tuple[str | None, str | None]:
    row = conn.execute(
        """SELECT NULLIF(btrim(company_name), ''), market_segment
           FROM yf_asset_universe_member
           WHERE ticker = %s AND (%s::bigint IS NULL OR run_id = %s)
           ORDER BY run_id DESC LIMIT 1""",
        (ticker, run_id, run_id),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        text = str(value).strip()
        return text if text else "—"
    if not number.is_finite():
        return "—"
    return f"{number:,.10f}".rstrip("0").rstrip(".")


def format_epoch(value: Any, milliseconds: bool = False) -> str:
    try:
        epoch = float(value)
        if milliseconds:
            epoch /= 1000
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError, OSError):
        return "—"


def format_ratio(value: Any) -> str:
    try:
        ratio = Decimal(str(value)) * 100
    except (InvalidOperation, ValueError):
        return "—"
    if not ratio.is_finite():
        return "—"
    return f"{ratio:.2f}%"


def read_artifact(path_value: str) -> Any:
    path = (PROJECT_ROOT / path_value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"artifact path is outside the project: {path_value}")
    if not path.is_file():
        raise FileNotFoundError(path_value)
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return json.load(source)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported artifact format: {path_value}")


def merge_statement(rows: list[tuple[Any, ...]], dataset: str) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    values: dict[str, dict[str, Any]] = defaultdict(dict)
    periods: list[str] = []
    errors: list[str] = []
    for run_id, row_dataset, status, artifact_path, _row_count, last_error, _started_at in rows:
        if row_dataset != dataset:
            continue
        if status != "succeeded" or not artifact_path:
            errors.append(f"run {run_id} / {dataset}: {status}" + (f" ({last_error})" if last_error else ""))
            continue
        try:
            frame = read_artifact(artifact_path)
        except Exception as error:
            errors.append(f"run {run_id} / {dataset}: {type(error).__name__}: {error}")
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            errors.append(f"run {run_id} / {dataset}: データなし（0行）")
            continue
        for column in frame.columns:
            label = pd.to_datetime(column, errors="coerce")
            period = str(column) if pd.isna(label) else label.strftime("%Y-%m-%d")
            if period not in periods:
                periods.append(period)
            for item, value in frame[column].items():
                values[str(item)][period] = value
    periods.sort(reverse=True)
    return values, periods, errors


def statement_rows(
    items: list[str], values: dict[str, dict[str, Any]], periods: list[str]
) -> list[str]:
    lines = ["| 項目 | " + " | ".join(periods) + " |", "| --- | " + " | ".join("---:" for _ in periods) + " |"]
    for item in items:
        if item in values:
            lines.append(f"| {FIELD_LABELS.get(item, item)} (`{item}`) | " + " | ".join(format_value(values[item].get(period)) for period in periods) + " |")
    return lines


def shareholder_snapshot_values(dataset: str, artifact: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not isinstance(artifact, pd.DataFrame) or artifact.empty:
        return values
    if dataset == "major_holders" and "Value" in artifact.columns:
        for key, value in artifact["Value"].items():
            values[str(key)] = value
    elif dataset == "insider_purchases":
        for _, row in artifact.iterrows():
            label = str(row.iloc[0]) if len(row) else ""
            if label == "Total Insider Shares Held" and "Shares" in artifact.columns:
                values[label] = row["Shares"]
            elif label in ("Purchases", "Sales", "Net Shares Purchased (Sold)") and "Shares" in artifact.columns:
                values[label] = row["Shares"]
    return values


def render_artifact_table(dataset: str, frame: Any) -> list[str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return ["このデータセットに表示できる明細はありません。", ""]
    if dataset == "insider_purchases" and len(frame.columns) >= 3:
        labels = {
            "Purchases": "購入",
            "Sales": "売却",
            "Net Shares Purchased (Sold)": "株式取得純数（買越・売越）",
            "Total Insider Shares Held": "内部関係者保有株数",
            "% Net Shares Purchased (Sold)": "取得純数の比率",
            "% Buy Shares": "購入株数比率",
            "% Sell Shares": "売却株数比率",
        }
        lines = [
            "| 区分 | 持ち株数 | 取引数 |",
            "| --- | ---: | ---: |",
        ]
        for _, row in frame.iterrows():
            raw_label = str(row.iloc[0]) if len(row) else ""
            label = labels.get(raw_label, raw_label)
            shares = format_value(row.get("Shares"))
            transactions = format_value(row.get("Trans"))
            lines.append(f"| {label} | {shares} | {transactions} |")
        lines.append("")
        return lines
    lines = [
        f"### {SHAREHOLDER_DATASETS[dataset]}",
        "",
        "| " + " | ".join(["項目", *[str(column) for column in frame.columns]]) + " |",
        "| " + " | ".join(["---", *["---:" for _ in frame.columns]]) + " |",
    ]
    for index, row in frame.iterrows():
        cells = [format_value(index), *[format_value(value) for value in row.tolist()]]
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", "<br>") for cell in cells) + " |")
    lines.append("")
    return lines


def build_report(
    ticker: str,
    company_name: str | None,
    market_segment: str | None,
    tasks: list[tuple[Any, ...]],
) -> str:
    latest: dict[str, tuple[int, Any]] = {}
    shareholder_snapshots: list[tuple[int, Any, str, Any]] = []
    errors: list[str] = []
    for run_id, dataset, status, artifact_path, _row_count, last_error, started_at in tasks:
        if status == "succeeded" and artifact_path:
            should_read = dataset in SHAREHOLDER_DATASETS or dataset not in latest or run_id > latest[dataset][0]
            if should_read:
                try:
                    artifact = read_artifact(artifact_path)
                    if dataset in SHAREHOLDER_DATASETS:
                        shareholder_snapshots.append((run_id, started_at, dataset, artifact))
                    if dataset not in latest or run_id > latest[dataset][0]:
                        latest[dataset] = (run_id, artifact)
                except Exception as error:
                    errors.append(f"run {run_id} / {dataset}: {type(error).__name__}: {error}")
        elif status != "succeeded" or last_error:
            errors.append(f"run {run_id} / {dataset}: {status}" + (f" ({last_error})" if last_error else ""))

    info: dict[str, Any] = latest.get("info", (0, {}))[1]
    fast_info: dict[str, Any] = latest.get("fast_info", (0, {}))[1]
    if not isinstance(info, dict):
        info = {}
    if not isinstance(fast_info, dict):
        fast_info = {}

    annual_income, income_periods, income_errors = merge_statement(tasks, "income_stmt")
    balance, balance_periods, balance_errors = merge_statement(tasks, "balance_sheet")
    cashflow, cashflow_periods, cashflow_errors = merge_statement(tasks, "cashflow")
    errors.extend(income_errors + balance_errors + cashflow_errors)

    title = f"{ticker} {company_name}" if company_name else ticker
    latest_run = max((row[0] for row in tasks), default=None)
    report = [
        f"# {title} 企業分析シート",
        "",
        f"- レポート作成日時（UTC）: {datetime.now(timezone.utc).isoformat()}",
        f"- 対象run: {latest_run if latest_run is not None else '—'}",
        f"- TSE市場区分（資産CSV）: {market_segment or '取得できていません'}",
        "- 現在指標は最新の成功した`info`/`fast_info` snapshot、財務表は全runを期間ごとに統合しています。",
        "",
        "## 会社概要",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
    ]

    profile_fields = [
        ("会社名", company_name or info.get("longName") or info.get("shortName")),
        ("ticker", ticker),
        ("市場区分", market_segment),
        ("取引所", info.get("fullExchangeName") or info.get("exchange")),
        ("Yahoo市場分類", info.get("market")),
        ("セクター", info.get("sectorDisp") or info.get("sector")),
        ("Yahoo業種", info.get("industryDisp") or info.get("industry")),
        ("東証33業種", None),
        ("指数採用", None),
        ("決算期末", format_epoch(info.get("lastFiscalYearEnd"))),
        ("次回決算期末", format_epoch(info.get("nextFiscalYearEnd"))),
        ("設立年（説明文から抽出）", f"{extract_founded_year(info.get('longBusinessSummary'))}年" if extract_founded_year(info.get("longBusinessSummary")) else None),
        ("Yahoo取得可能な最初の取引日", format_epoch(info.get("firstTradeDateMilliseconds"), milliseconds=True)),
        ("本社", " / ".join(str(info[key]) for key in ("address1", "address2", "city", "country") if info.get(key)) or None),
        ("電話番号", info.get("phone")),
        ("Webサイト", info.get("website")),
        ("従業員数", info.get("fullTimeEmployees")),
        ("事業概要", info.get("longBusinessSummary")),
    ]
    for label, value in profile_fields:
        report.append(f"| {label} | {format_value(value) if value is not None else '取得できていません'} |")

    report.extend(["", "## 株主・持ち株数", ""])
    snapshot_by_run: dict[int, dict[str, Any]] = defaultdict(dict)
    snapshot_dates: dict[int, Any] = {}
    latest_holder_frames: dict[str, tuple[int, pd.DataFrame]] = {}
    for run_id, started_at, dataset, artifact in shareholder_snapshots:
        snapshot_dates[run_id] = started_at
        snapshot_by_run[run_id].update(shareholder_snapshot_values(dataset, artifact))
        if isinstance(artifact, pd.DataFrame) and not artifact.empty:
            if dataset not in latest_holder_frames or run_id > latest_holder_frames[dataset][0]:
                latest_holder_frames[dataset] = (run_id, artifact)

    if snapshot_by_run:
        report.extend([
            "取得runごとの株主構成スナップショットです。過去分はこのシステムで取得・保存したrun以降に限ります。",
            "",
            "| Run | 取得日時 | 内部関係者保有割合 | 機関投資家保有割合 | 機関投資家の浮動株保有割合 | 機関数 | 内部関係者保有株数 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for run_id in sorted(snapshot_by_run):
            snapshot = snapshot_by_run[run_id]
            report.append(
                "| " + " | ".join([
                    str(run_id),
                    format_value(snapshot_dates.get(run_id)),
                    format_ratio(snapshot["insidersPercentHeld"]) if snapshot.get("insidersPercentHeld") is not None else "—",
                    format_ratio(snapshot["institutionsPercentHeld"]) if snapshot.get("institutionsPercentHeld") is not None else "—",
                    format_ratio(snapshot["institutionsFloatPercentHeld"]) if snapshot.get("institutionsFloatPercentHeld") is not None else "—",
                    format_value(snapshot.get("institutionsCount")),
                    format_value(snapshot.get("Total Insider Shares Held")),
                ]) + " |"
            )
        report.append("")
    else:
        report.extend(["株主構成のデータは取得できていません。", ""])

    report.extend([
        "### 最新の取得成功データに含まれる株主・保有明細",
        "",
        "名簿データがYahoo Financeから返された場合に表示します。空表は取得対象に対する名簿データの返却がなかったことを示します。",
        "",
    ])
    for dataset in ("institutional_holders", "mutualfund_holders", "insider_transactions", "insider_purchases", "insider_roster_holders"):
        if dataset in latest_holder_frames:
            holder_run, holder_frame = latest_holder_frames[dataset]
            report.extend([f"参照run: {holder_run}", ""])
            report.extend(render_artifact_table(dataset, holder_frame))
        else:
            report.extend([f"### {SHAREHOLDER_DATASETS[dataset]}", "", "取得データなし。", ""])

    report.extend(["", "## 現在の株価・企業価値・評価指標", "", "| 指標 | 値 |", "| --- | ---: |"])
    current_fields = [
        ("株価", fast_info.get("lastPrice", info.get("regularMarketPrice"))),
        ("前日終値", fast_info.get("previousClose", info.get("regularMarketPreviousClose"))),
        ("時価総額", info.get("marketCap", fast_info.get("marketCap"))),
        ("企業価値", info.get("enterpriseValue")),
        ("通貨", info.get("currency", fast_info.get("currency"))),
        ("実績PER", info.get("trailingPE")),
        ("予想PER", info.get("forwardPE")),
        ("PBR", info.get("priceToBook")),
        ("実績EPS", info.get("trailingEps")),
        ("予想EPS", info.get("forwardEps")),
        ("BPS", info.get("bookValue")),
        ("配当利回り（Yahoo返却値）", info.get("dividendYield")),
        ("年間配当予想（1株当たり）", info.get("dividendRate")),
        ("配当性向", format_ratio(info.get("payoutRatio")) if info.get("payoutRatio") is not None else None),
        ("ROE（Yahoo返却値）", format_ratio(info.get("returnOnEquity")) if info.get("returnOnEquity") is not None else None),
        ("ROA（Yahoo返却値）", format_ratio(info.get("returnOnAssets")) if info.get("returnOnAssets") is not None else None),
        ("ベータ", info.get("beta")),
        ("52週高値", info.get("fiftyTwoWeekHigh", fast_info.get("yearHigh"))),
        ("52週安値", info.get("fiftyTwoWeekLow", fast_info.get("yearLow"))),
        ("発行済株式数", info.get("sharesOutstanding", fast_info.get("shares"))),
        ("浮動株数", info.get("floatShares")),
        ("内部関係者保有割合", format_ratio(info.get("heldPercentInsiders")) if info.get("heldPercentInsiders") is not None else None),
        ("機関投資家保有割合", format_ratio(info.get("heldPercentInstitutions")) if info.get("heldPercentInstitutions") is not None else None),
    ]
    for label, value in current_fields:
        report.append(f"| {label} | {format_value(value) if value is not None else '取得できていません'} |")
    report.extend(["", "※配当利回りはYahoo返却値の単位を加工せず表示しています。", ""])

    report.extend(["## 業績推移（年次）", ""])
    income_items = [
        "Total Revenue", "Operating Revenue", "Operating Income", "Pretax Income",
        "Net Income", "Net Income Common Stockholders", "Basic EPS", "Diluted EPS",
    ]
    present_income_items = [item for item in income_items if item in annual_income]
    if income_periods and present_income_items:
        report.extend(statement_rows(present_income_items, annual_income, income_periods))
    else:
        report.append("取得済みの年次損益データがありません。")
    report.append("")

    report.extend(["## 財務・資本効率の推移", ""])
    equity_item = "Stockholders Equity" if "Stockholders Equity" in balance else "Common Stock Equity"
    ratio_rows: dict[str, dict[str, Any]] = defaultdict(dict)
    sorted_periods = sorted(income_periods)
    net_income_item = "Net Income" if "Net Income" in annual_income else "Net Income Common Stockholders"
    if net_income_item in annual_income:
        for position, period in enumerate(sorted_periods):
            net_income = decimal_or_none(annual_income[net_income_item].get(period))
            if net_income is None:
                continue
            previous_period = sorted_periods[position - 1] if position else None
            for balance_item, label in (("Total Assets", "ROA推計"), (equity_item, "ROE推計")):
                current_value = decimal_or_none(balance.get(balance_item, {}).get(period))
                previous_value = decimal_or_none(balance.get(balance_item, {}).get(previous_period)) if previous_period else None
                if current_value is None:
                    continue
                denominator = (current_value + previous_value) / 2 if previous_value is not None else current_value
                if denominator:
                    ratio_rows[label][period] = net_income / denominator * Decimal(100)
            normalized_income = decimal_or_none(annual_income.get("Normalized Income", {}).get(period))
            diluted_shares = decimal_or_none(annual_income.get("Diluted Average Shares", {}).get(period))
            if normalized_income is not None and diluted_shares:
                ratio_rows["調整EPS推計（調整後純利益÷希薄化後平均株式数）"][period] = normalized_income / diluted_shares
            equity_value = decimal_or_none(balance.get(equity_item, {}).get(period))
            shares_value = decimal_or_none(balance.get("Ordinary Shares Number", {}).get(period))
            if equity_value is not None and shares_value:
                ratio_rows["BPS推計（株主資本÷普通株式数）"][period] = equity_value / shares_value
    ratio_periods = sorted({period for row in ratio_rows.values() for period in row}, reverse=True)
    if ratio_periods:
        report.extend(["| 指標（推計） | " + " | ".join(ratio_periods) + " |", "| --- | " + " | ".join("---:" for _ in ratio_periods) + " |"])
        for label, period_values in ratio_rows.items():
            report.append(f"| {label} | " + " | ".join(format_value(period_values.get(period)) + ("%" if label in ("ROA推計", "ROE推計") and period_values.get(period) is not None else "") for period in ratio_periods) + " |")
    yahoo_ratio_rows = [
        ("ROE（最新Yahoo返却値）", info.get("returnOnEquity")),
        ("ROA（最新Yahoo返却値）", info.get("returnOnAssets")),
    ]
    report.append("")
    report.extend(["Yahoo現在値: " + "、".join(f"{label} {format_ratio(value)}" for label, value in yahoo_ratio_rows if value is not None) if any(value is not None for _, value in yahoo_ratio_rows) else "Yahooの現在ROE/ROA値はありません。", ""])

    report.extend(["## 配当推移", ""])
    dividend_events = latest.get("dividends", (0, None))[1]
    if isinstance(dividend_events, pd.DataFrame) and not dividend_events.empty:
        events = dividend_events.copy()
        events.index = pd.to_datetime(events.index, errors="coerce")
        events = events.loc[events.index.notna()]
        if not events.empty and "Dividends" in events.columns:
            annual_dividends = events["Dividends"].groupby(events.index.year).sum()
            report.extend(["暦年合計（1株当たり、配当支払日ベース。決算期別ではありません）", "", "| 年 | 配当合計 |", "| ---: | ---: |"])
            for year, amount in annual_dividends.items():
                report.append(f"| {year} | {format_value(amount)} |")
            report.extend(["", "配当イベント一覧", "", "| 支払日 | 1株当たり配当 |", "| --- | ---: |"])
            for event_date, row in events.iterrows():
                report.append(f"| {event_date.strftime('%Y-%m-%d')} | {format_value(row['Dividends'])} |")
        else:
            report.append("配当イベント列が取得されていません。")
    else:
        report.append("配当履歴は取得できていません。")
    report.append("")

    report.extend(["## 保有不動産・設備の参考項目", "", "貸借対照表で返された関連項目を掲載します。網羅的な土地・建物台帳や時価評価ではありません。", ""])
    property_items = ["Investment Properties", "Properties", "Land And Improvements", "Buildings And Improvements", "Other Properties", "Net PPE", "Gross PPE"]
    available_property = [item for item in property_items if item in balance]
    if balance_periods and available_property:
        report.extend(statement_rows(available_property, balance, balance_periods))
    else:
        report.append("個別不動産の項目は取得できていません。")
    report.append("")

    officers = info.get("companyOfficers")
    report.extend(["## 役員情報", ""])
    if isinstance(officers, list) and officers:
        report.extend(["| 氏名 | 役職 | 年齢 | 報酬（Yahoo返却値） |", "| --- | --- | ---: | ---: |"])
        for officer in officers:
            report.append("| " + " | ".join(format_value(officer.get(key)) for key in ("name", "title", "age", "totalPay")) + " |")
    else:
        report.append("役員情報は取得できていません。")
    report.append("")

    limitations = [
        "指数採用状況（Yahooの基本情報に指数構成銘柄の一覧がありません）",
        "東証33業種コード（Yahoo業種とは分類体系が異なり、対応表をこのデータだけでは確定できません）",
        "主取引銀行",
        "主要仕入先・販売先",
        "保有不動産の網羅的な所在地・面積・簿価・時価",
    ]
    report.extend(["## 取得できない項目・注意事項", ""])
    report.extend(f"- {item}" for item in limitations)
    report.extend([
        "- 経常利益はYahooの損益計算書で同名の項目を確認できないため、税引前利益で代用せず、元項目名どおりに掲載しています。",
        "- 設立年は企業説明文に明記された場合だけ抽出しています。上場日はYahooの最初の取引日であり、正式な上場日とは限りません。",
        "- ROE/ROA推計は当期純利益÷期首・期末平均の株主資本/総資産です。初年度は期末残高のみを分母に使用します。Yahoo返却値とは定義が異なる場合があります。",
        "- 配当利回りはYahoo返却値をそのまま表示します。Yahoo側の単位表現が一定でない可能性があるため、百分率変換していません。",
        "- このレポートは投資判断を自動化するものではありません。開示資料や会社発表との照合が必要です。",
        "",
    ])
    if errors:
        report.extend(["## 取得エラー・空データ", ""])
        report.extend(f"- {error}" for error in errors)
        report.append("")
    return "\n".join(report)


def decimal_or_none(value: Any) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def extract_founded_year(summary: Any) -> str | None:
    if not isinstance(summary, str):
        return None
    match = re.search(r"(?:founded|established)\s+(?:in\s+)?(1[5-9]\d{2}|20\d{2})", summary, re.IGNORECASE)
    return match.group(1) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a company-analysis report from saved Yahoo Finance artifacts.")
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. 7013.T")
    parser.add_argument("--run-id", type=int, help="Use this run; default uses latest current snapshot and all annual periods")
    parser.add_argument("--output", help="Output Markdown path")
    args = parser.parse_args()
    if args.run_id is not None and args.run_id < 1:
        parser.error("run-id must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        with psycopg.connect(database_url(), connect_timeout=10) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            tasks = fetch_tasks(conn, args.ticker, args.run_id)
            company_name, market_segment = fetch_asset_info(conn, args.ticker, args.run_id)
        if not tasks:
            print(f"取得履歴がありません: {args.ticker}", file=sys.stderr)
            return 1
        report = build_report(args.ticker, company_name, market_segment, tasks)
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", args.ticker)
        suffix = f"run-{args.run_id:06d}" if args.run_id else "latest"
        output = Path(args.output).expanduser() if args.output else PROJECT_ROOT / "data" / "reports" / "yfinance" / f"{safe_ticker}-company-analysis-{suffix}.md"
        if not output.is_absolute():
            output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(output)
        print(f"企業分析レポートを作成しました: {output}")
        print(f"参照データセット: {', '.join(sorted({row[1] for row in tasks}))}")
        return 0
    except (OSError, RuntimeError, psycopg.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
