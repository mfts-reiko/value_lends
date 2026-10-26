from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cashflow": "年次キャッシュフロー計算書",
    "quarterly_cashflow": "四半期キャッシュフロー計算書",
}
ITEM_LABELS = {
    "Operating Cash Flow": "営業活動によるキャッシュフロー",
    "Investing Cash Flow": "投資活動によるキャッシュフロー",
    "Financing Cash Flow": "財務活動によるキャッシュフロー",
    "Free Cash Flow": "フリーキャッシュフロー",
    "Net Income From Continuing Operations": "継続事業当期純利益",
    "Depreciation And Amortization": "減価償却費及び償却費",
    "Depreciation": "減価償却費",
    "Other Non Cash Items": "その他非現金項目",
    "Change In Working Capital": "運転資本の増減",
    "Change In Other Current Assets": "その他流動資産の増減",
    "Change In Other Current Liabilities": "その他流動負債の増減",
    "Change In Payable": "買掛金・未払金の増減",
    "Change In Prepaid Assets": "前払資産の増減",
    "Change In Receivables": "債権の増減",
    "Taxes Refund Paid": "税金の支払・還付",
    "Capital Expenditure": "設備投資額",
    "Purchase Of PPE": "有形固定資産の取得支出",
    "Net PPE Purchase And Sale": "有形固定資産の取得・売却純額",
    "Purchase Of Investment": "投資有価証券の取得支出",
    "Sale Of Investment": "投資有価証券の売却収入",
    "Net Investment Purchase And Sale": "投資有価証券の取得・売却純額",
    "Purchase Of Business": "事業取得支出",
    "Sale Of Business": "事業売却収入",
    "Net Business Purchase And Sale": "事業取得・売却純額",
    "Purchase Of Intangibles": "無形資産の取得支出",
    "Net Intangibles Purchase And Sale": "無形資産の取得・売却純額",
    "Net Other Investing Changes": "投資活動その他の増減",
    "Repayment Of Debt": "借入金の返済額",
    "Issuance Of Debt": "借入金の調達額",
    "Issuance Of Capital Stock": "株式発行による収入",
    "Cash Dividends Paid": "配当金の支払額",
    "Common Stock Dividend Paid": "普通株式配当の支払額",
    "Interest Paid Cff": "財務活動の支払利息",
    "Net Common Stock Issuance": "普通株式の発行・取得純額",
    "Common Stock Issuance": "普通株式の発行収入",
    "Net Issuance Payments Of Debt": "借入金の発行・返済純額",
    "Net Short Term Debt Issuance": "短期借入金の調達・返済純額",
    "Net Long Term Debt Issuance": "長期借入金の調達・返済純額",
    "Long Term Debt Payments": "長期借入金の返済額",
    "Long Term Debt Issuance": "長期借入金の調達額",
    "Net Other Financing Charges": "財務活動その他の支出・収入",
    "Beginning Cash Position": "期首現金残高",
    "End Cash Position": "期末現金残高",
    "Changes In Cash": "現金及び現金同等物の増減",
    "Effect Of Exchange Rate Changes": "為替変動による現金への影響",
}
SUMMARY_ITEMS = [
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Free Cash Flow",
    "Changes In Cash",
    "Beginning Cash Position",
    "End Cash Position",
]
GROUPS = [
    (
        "営業活動",
        [
            "Operating Cash Flow", "Net Income From Continuing Operations",
            "Depreciation And Amortization", "Depreciation", "Other Non Cash Items",
            "Change In Working Capital", "Change In Other Current Assets",
            "Change In Other Current Liabilities", "Change In Payable",
            "Change In Prepaid Assets", "Change In Receivables", "Taxes Refund Paid",
        ],
    ),
    (
        "投資活動",
        [
            "Investing Cash Flow", "Capital Expenditure", "Purchase Of PPE",
            "Net PPE Purchase And Sale", "Purchase Of Investment", "Sale Of Investment",
            "Net Investment Purchase And Sale", "Purchase Of Business", "Sale Of Business",
            "Net Business Purchase And Sale", "Purchase Of Intangibles",
            "Net Intangibles Purchase And Sale", "Net Other Investing Changes",
        ],
    ),
    (
        "財務活動",
        [
            "Financing Cash Flow", "Repayment Of Debt", "Issuance Of Debt",
            "Issuance Of Capital Stock", "Cash Dividends Paid", "Common Stock Dividend Paid",
            "Interest Paid Cff", "Net Common Stock Issuance", "Common Stock Issuance",
            "Net Issuance Payments Of Debt", "Net Short Term Debt Issuance",
            "Net Long Term Debt Issuance", "Long Term Debt Payments", "Long Term Debt Issuance",
            "Net Other Financing Charges",
        ],
    ),
    (
        "現金残高・その他",
        [
            "Beginning Cash Position", "End Cash Position", "Changes In Cash",
            "Effect Of Exchange Rate Changes", "Free Cash Flow",
        ],
    ),
]


def get_database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL がありません。.env を確認してください")
    prefix = "postgresql+psycopg://"
    if url.startswith(prefix):
        url = "postgresql://" + url.removeprefix(prefix)
    return url


def fetch_tasks(
    conn: psycopg.Connection[Any], ticker: str, run_id: int | None
) -> list[tuple[Any, ...]]:
    return conn.execute(
        """SELECT task.run_id, run.status, run.started_at, task.dataset,
                  task.status, task.artifact_path, task.row_count, task.last_error
           FROM yf_fetch_task AS task
           JOIN yf_fetch_run AS run USING (run_id)
           WHERE task.ticker = %s
             AND task.dataset = ANY(%s)
             AND (%s::bigint IS NULL OR task.run_id = %s)
           ORDER BY task.run_id, task.dataset""",
        (ticker, list(DATASETS), run_id, run_id),
    ).fetchall()


def fetch_company_name(
    conn: psycopg.Connection[Any], ticker: str, run_id: int | None
) -> str | None:
    row = conn.execute(
        """SELECT NULLIF(btrim(company_name), '')
           FROM yf_asset_universe_member
           WHERE ticker = %s AND (%s::bigint IS NULL OR run_id = %s)
           ORDER BY run_id DESC
           LIMIT 1""",
        (ticker, run_id, run_id),
    ).fetchone()
    return row[0] if row else None


def period_label(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return str(value) if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    if not number.is_finite():
        return "—"
    return f"{number:,.10f}".rstrip("0").rstrip(".")


def collect_data(
    tasks: list[tuple[Any, ...]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]], dict[str, list[str]], list[str]]:
    values: dict[str, dict[str, dict[str, Any]]] = {
        dataset: defaultdict(dict) for dataset in DATASETS
    }
    period_sources: dict[str, dict[str, int]] = {dataset: {} for dataset in DATASETS}
    item_order: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    errors: list[str] = []

    for run_id, _run_status, _started_at, dataset, status, artifact_path, _row_count, last_error in tasks:
        if status != "succeeded" or not artifact_path:
            message = f"run {run_id} / {dataset}: {status}"
            if last_error:
                message += f" ({last_error})"
            errors.append(message)
            continue

        path = (PROJECT_ROOT / artifact_path).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            errors.append(f"run {run_id} / {dataset}: artifact path is outside the project")
            continue
        if not path.is_file():
            errors.append(f"run {run_id} / {dataset}: file not found: {artifact_path}")
            continue

        try:
            frame = pd.read_parquet(path)
        except Exception as error:
            errors.append(f"run {run_id} / {dataset}: {type(error).__name__}: {error}")
            continue
        if frame.empty:
            errors.append(f"run {run_id} / {dataset}: cashflow statement has no rows")
            continue

        for original_period in frame.columns:
            period = period_label(original_period)
            period_sources[dataset][period] = run_id
            for item, value in frame[original_period].items():
                item_name = str(item)
                if item_name not in item_order[dataset]:
                    item_order[dataset].append(item_name)
                values[dataset][item_name][period] = value

    return values, period_sources, item_order, errors


def append_table(
    report: list[str],
    values: dict[str, dict[str, Any]],
    periods: list[str],
    items: list[str],
) -> set[str]:
    report.extend([
        "| 項目 | Yahoo Finance項目名 | " + " | ".join(periods) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in periods) + " |",
    ])
    included: set[str] = set()
    for item in items:
        if item not in values:
            continue
        label = ITEM_LABELS.get(item, "未分類項目")
        amounts = [format_value(values[item].get(period)) for period in periods]
        report.append(f"| {label} | `{item}` | " + " | ".join(amounts) + " |")
        included.add(item)
    report.append("")
    return included


def build_report(
    ticker: str,
    company_name: str | None,
    tasks: list[tuple[Any, ...]],
    values: dict[str, dict[str, dict[str, Any]]],
    period_sources: dict[str, dict[str, int]],
    item_order: dict[str, list[str]],
    errors: list[str],
) -> str:
    runs = sorted({row[0] for row in tasks})
    title = f"{ticker} {company_name}" if company_name else ticker
    report = [
        f"# {title} キャッシュフロー計算書（Yahoo Finance取得値）",
        "",
        f"- レポート作成日時 (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- 対象run: {', '.join(map(str, runs))}",
        "- 同じ対象期間が複数runにある場合は、run番号が大きい取得値を採用しています。",
        "- 数値はYahoo Financeの取得値をそのまま表示します。通貨・単位は自動換算していません。",
        "- Yahoo Financeが返していない項目や期間は補完せず `—` と表示します。",
        "",
    ]

    for dataset, title in DATASETS.items():
        dataset_values = values[dataset]
        sources = period_sources[dataset]
        periods = sorted(sources, reverse=True)
        report.extend([f"## {title}", ""])
        if not periods:
            report.append("保存済みのデータはありません。")
            report.append("")
            continue

        report.extend([
            f"対象期間: {', '.join(periods)}",
            "",
            "### 主要項目",
            "",
        ])
        append_table(report, dataset_values, periods, SUMMARY_ITEMS)

        included: set[str] = set(SUMMARY_ITEMS)
        for group_title, items in GROUPS:
            available = [item for item in items if item in dataset_values and item not in included]
            if available:
                report.extend([f"### {group_title}", ""])
                included.update(append_table(report, dataset_values, periods, available))

        other_items = [item for item in item_order[dataset] if item not in included]
        if other_items:
            report.extend(["### その他取得項目", ""])
            append_table(report, dataset_values, periods, other_items)

        report.extend(["### 期間ごとの採用run", "", "| 対象期間 | Run |", "| --- | ---: |"])
        for period in periods:
            report.append(f"| {period} | {sources[period]} |")
        report.append("")

    if errors:
        report.extend(["## 未取得・確認事項", ""])
        report.extend(f"- {error}" for error in errors)
        report.append("")
    report.extend([
        "## 読み方",
        "",
        "- キャッシュフローは期間中の資金の流入・流出です。年次と四半期は別セクションにし、期間を混在させません。",
        "- 0は取得値の0、`—` は値が未提供または未取得です。",
        "- 項目名は日本語の説明とYahoo Financeの原項目名を併記しています。",
        "- 表示項目はYahoo Financeの分類・金額をそのまま掲載し、会計上の再分類や指標の再計算はしていません。",
        "",
    ])
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a readable Japanese cash-flow report from saved Yahoo Finance artifacts."
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. 4755.T")
    parser.add_argument("--run-id", type=int, help="Use only this run; default includes all runs")
    parser.add_argument("--output", help="Output Markdown path")
    args = parser.parse_args()
    if args.run_id is not None and args.run_id < 1:
        parser.error("run-id must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        with psycopg.connect(get_database_url(), connect_timeout=10) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            tasks = fetch_tasks(conn, args.ticker, args.run_id)
            company_name = fetch_company_name(conn, args.ticker, args.run_id)
        if not tasks:
            print(f"キャッシュフローの取得履歴がありません: {args.ticker}", file=sys.stderr)
            return 1

        values, period_sources, item_order, errors = collect_data(tasks)
        if not any(period_sources.values()):
            print("保存済みのキャッシュフローデータがありません。", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        report = build_report(
            args.ticker, company_name, tasks, values, period_sources, item_order, errors
        )
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", args.ticker)
        suffix = f"run-{args.run_id:06d}" if args.run_id is not None else "all-runs"
        output = Path(args.output).expanduser() if args.output else (
            PROJECT_ROOT / "data" / "reports" / "yfinance" / f"{safe_ticker}-cashflow-{suffix}.md"
        )
        if not output.is_absolute():
            output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(output)
        print(f"キャッシュフローレポートを作成しました: {output}")
        for dataset, periods in period_sources.items():
            print(f"{DATASETS[dataset]}: {len(periods)}期間")
        return 0
    except (OSError, RuntimeError, psycopg.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
