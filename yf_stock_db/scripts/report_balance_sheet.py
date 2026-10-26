from __future__ import annotations

import argparse
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

ITEM_LABELS = {
    "Total Assets": "総資産",
    "Total Liabilities Net Minority Interest": "負債合計（非支配持分調整後）",
    "Stockholders Equity": "株主資本",
    "Current Assets": "流動資産",
    "Other Current Assets": "その他流動資産",
    "Restricted Cash": "拘束性預金",
    "Prepaid Assets": "前払資産",
    "Inventory": "棚卸資産",
    "Other Inventories": "その他棚卸資産",
    "Finished Goods": "製品・商品",
    "Other Receivables": "その他債権",
    "Accounts Receivable": "売掛金・受取手形",
    "Gross Accounts Receivable": "売掛金・受取手形（総額）",
    "Cash Cash Equivalents And Short Term Investments": "現金・現金同等物・短期投資",
    "Other Short Term Investments": "その他短期投資",
    "Cash And Cash Equivalents": "現金及び現金同等物",
    "Total Non Current Assets": "固定資産合計",
    "Other Non Current Assets": "その他固定資産",
    "Non Current Deferred Taxes Assets": "固定繰延税金資産",
    "Investmentin Financial Assets": "金融資産投資",
    "Held To Maturity Securities": "満期保有目的有価証券",
    "Available For Sale Securities": "売却可能有価証券",
    "Goodwill And Other Intangible Assets": "のれん・無形資産",
    "Other Intangible Assets": "その他無形資産",
    "Goodwill": "のれん",
    "Net PPE": "有形固定資産純額",
    "Gross PPE": "有形固定資産総額",
    "Other Properties": "その他不動産",
    "Machinery Furniture Equipment": "機械・備品・設備",
    "Buildings And Improvements": "建物・改良施設",
    "Land And Improvements": "土地・土地改良",
    "Properties": "不動産",
    "Current Liabilities": "流動負債",
    "Other Current Liabilities": "その他流動負債",
    "Current Debt And Capital Lease Obligation": "短期借入金・リース債務",
    "Current Capital Lease Obligation": "短期リース債務",
    "Current Debt": "短期借入金",
    "Pensionand Other Post Retirement Benefit Plans Current": "流動退職給付債務",
    "Current Provisions": "流動引当金",
    "Payables": "未払金等",
    "Total Tax Payable": "未払税金合計",
    "Accounts Payable": "買掛金・支払手形",
    "Total Non Current Liabilities Net Minority Interest": "固定負債合計（非支配持分調整後）",
    "Other Non Current Liabilities": "その他固定負債",
    "Non Current Pension And Other Postretirement Benefit Plans": "固定退職給付債務",
    "Non Current Deferred Taxes Liabilities": "固定繰延税金負債",
    "Long Term Debt And Capital Lease Obligation": "長期借入金・リース債務",
    "Long Term Capital Lease Obligation": "長期リース債務",
    "Long Term Debt": "長期借入金",
    "Long Term Provisions": "固定引当金",
    "Total Equity Gross Minority Interest": "純資産合計（非支配持分を含む）",
    "Minority Interest": "非支配持分",
    "Common Stock Equity": "普通株主資本",
    "Other Equity Interest": "その他持分",
    "Fixed Assets Revaluation Reserve": "固定資産再評価差額金",
    "Treasury Stock": "自己株式",
    "Retained Earnings": "利益剰余金",
    "Additional Paid In Capital": "資本剰余金",
    "Capital Stock": "資本金",
    "Common Stock": "普通株式資本",
    "Treasury Shares Number": "自己株式数",
    "Ordinary Shares Number": "普通株式数",
    "Share Issued": "発行済株式数",
    "Total Debt": "有利子負債合計",
    "Tangible Book Value": "有形純資産",
    "Invested Capital": "投下資本",
    "Working Capital": "運転資本",
    "Net Tangible Assets": "有形資産純額",
    "Capital Lease Obligations": "リース債務",
    "Total Capitalization": "資本総額",
}

SUMMARY_ITEMS = [
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Stockholders Equity",
]

GROUPS = [
    (
        "流動資産",
        [
            "Current Assets", "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents", "Restricted Cash", "Other Short Term Investments",
            "Accounts Receivable", "Gross Accounts Receivable", "Other Receivables",
            "Inventory", "Other Inventories", "Finished Goods", "Prepaid Assets",
            "Other Current Assets",
        ],
    ),
    (
        "固定資産",
        [
            "Total Non Current Assets", "Net PPE", "Gross PPE", "Properties",
            "Land And Improvements", "Buildings And Improvements", "Machinery Furniture Equipment",
            "Other Properties", "Investmentin Financial Assets", "Held To Maturity Securities",
            "Available For Sale Securities", "Goodwill And Other Intangible Assets", "Goodwill",
            "Other Intangible Assets", "Non Current Deferred Taxes Assets", "Other Non Current Assets",
        ],
    ),
    (
        "流動負債",
        [
            "Current Liabilities", "Current Debt And Capital Lease Obligation", "Current Debt",
            "Current Capital Lease Obligation", "Accounts Payable", "Payables", "Total Tax Payable",
            "Current Provisions", "Pensionand Other Post Retirement Benefit Plans Current",
            "Other Current Liabilities",
        ],
    ),
    (
        "固定負債",
        [
            "Total Non Current Liabilities Net Minority Interest", "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt", "Long Term Capital Lease Obligation", "Long Term Provisions",
            "Non Current Pension And Other Postretirement Benefit Plans",
            "Non Current Deferred Taxes Liabilities", "Other Non Current Liabilities",
        ],
    ),
    (
        "純資産",
        [
            "Total Equity Gross Minority Interest", "Common Stock Equity", "Common Stock",
            "Capital Stock", "Additional Paid In Capital", "Retained Earnings", "Treasury Stock",
            "Fixed Assets Revaluation Reserve", "Other Equity Interest", "Minority Interest",
        ],
    ),
]

SUPPLEMENTAL_ITEMS = [
    "Total Debt", "Working Capital", "Tangible Book Value", "Invested Capital",
    "Net Tangible Assets", "Capital Lease Obligations", "Total Capitalization",
    "Treasury Shares Number", "Ordinary Shares Number", "Share Issued",
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


def fetch_balance_sheet_tasks(
    conn: psycopg.Connection[Any], ticker: str, run_id: int | None
) -> list[tuple[Any, ...]]:
    return conn.execute(
        """SELECT task.run_id, run.status, run.started_at, task.status,
                  task.artifact_path, task.row_count, task.last_error
           FROM yf_fetch_task AS task
           JOIN yf_fetch_run AS run USING (run_id)
           WHERE task.ticker = %s
             AND task.dataset = 'balance_sheet'
             AND (%s::bigint IS NULL OR task.run_id = %s)
           ORDER BY task.run_id""",
        (ticker, run_id, run_id),
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
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def format_amount(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    if not amount.is_finite():
        return "—"
    rendered = f"{amount:,.10f}".rstrip("0").rstrip(".")
    return rendered


def collect_periods(
    tasks: list[tuple[Any, ...]], ticker: str
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[str], list[str]]:
    values: dict[str, dict[str, Any]] = defaultdict(dict)
    period_sources: dict[str, int] = {}
    item_order: list[str] = []
    errors: list[str] = []

    for run_id, run_status, started_at, task_status, artifact_path, row_count, last_error in tasks:
        if task_status != "succeeded" or not artifact_path:
            details = f"run {run_id}: {task_status}"
            if last_error:
                details += f"（{last_error}）"
            errors.append(details)
            continue

        path = (PROJECT_ROOT / artifact_path).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            errors.append(f"run {run_id}: 保存先がプロジェクト外のため除外しました")
            continue
        if not path.is_file():
            errors.append(f"run {run_id}: ファイルが見つかりません: {artifact_path}")
            continue

        frame = pd.read_parquet(path)
        if frame.empty:
            errors.append(f"run {run_id}: 貸借対照表は0行でした")
            continue

        normalized_periods = [(period_label(column), column) for column in frame.columns]
        for period, original_column in normalized_periods:
            period_sources[period] = run_id
            for item, value in frame[original_column].items():
                item_name = str(item)
                if item_name not in item_order:
                    item_order.append(item_name)
                values[item_name][period] = value

    periods = sorted(period_sources, reverse=True)
    if not periods:
        return {}, {}, [], errors
    return values, period_sources, item_order, errors


def render_group(
    title: str,
    items: list[str],
    values: dict[str, dict[str, Any]],
    periods: list[str],
) -> list[str]:
    lines = [f"### {title}", "", "| 項目 | Yahoo Finance項目名 | " + " | ".join(periods) + " |"]
    lines.append("| --- | --- | " + " | ".join("---:" for _ in periods) + " |")
    for item in items:
        if item not in values:
            continue
        label = ITEM_LABELS.get(item, "未分類項目")
        amounts = [format_amount(values[item].get(period)) for period in periods]
        lines.append(f"| {label} | `{item}` | " + " | ".join(amounts) + " |")
    lines.append("")
    return lines


def decimal_value(value: Any) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def bar_and_ratio(value: Decimal | None, total: Decimal | None) -> str:
    if value is None or total is None or total == 0:
        return "—"
    ratio = value / total * Decimal(100)
    blocks = max(0, min(20, round(float(ratio / Decimal(5)))))
    bar = "█" * blocks + "░" * (20 - blocks)
    return f"{ratio:.1f}% `{bar}`"


def render_balance_chart(
    values: dict[str, dict[str, Any]], periods: list[str]
) -> list[str]:
    lines = [
        "## 資産と負債・純資産の左右比較",
        "",
        "各期の資産構成を左側、負債・純資産構成を右側に示します。バーは総資産に対する構成比（最大20ブロック）です。",
        "",
    ]
    sides = (
        (
            "資産の部",
            [("流動資産", "Current Assets"), ("固定資産", "Total Non Current Assets")],
        ),
        (
            "負債・純資産（資本）の部",
            [
                ("流動負債", "Current Liabilities"),
                ("固定負債", "Total Non Current Liabilities Net Minority Interest"),
                ("純資産", "Total Equity Gross Minority Interest"),
            ],
        ),
    )

    for period in periods:
        asset_total = decimal_value(values.get("Total Assets", {}).get(period))
        lines.extend([
            f"### {period}",
            "",
            f"| 資産の部 | 金額 | 構成比・バー | 負債・純資産（資本）の部 | 金額 | 構成比・バー |",
            "| --- | ---: | --- | --- | ---: | --- |",
        ])

        left_rows = sides[0][1]
        right_rows = sides[1][1]
        for row_index in range(max(len(left_rows), len(right_rows))):
            cells: list[str] = []
            for side_rows in (left_rows, right_rows):
                if row_index >= len(side_rows):
                    cells.extend(["—", "—", "—"])
                    continue
                label, item = side_rows[row_index]
                amount = decimal_value(values.get(item, {}).get(period))
                cells.extend([
                    f"{label} (`{item}`)",
                    format_amount(amount),
                    bar_and_ratio(amount, asset_total),
                ])
            lines.append("| " + " | ".join(cells) + " |")

        right_items = [item for _, item in right_rows]
        right_values = [decimal_value(values.get(item, {}).get(period)) for item in right_items]
        right_total = (
            sum((amount for amount in right_values if amount is not None), Decimal(0))
            if all(amount is not None for amount in right_values)
            else None
        )
        if asset_total is None or right_total is None:
            balance_note = "左右合計の照合不可（必要な合計項目が未取得）"
        else:
            difference = asset_total - right_total
            balance_note = (
                "左右一致"
                if difference == 0
                else f"左右差: {format_amount(difference)}"
            )
        lines.extend([
            f"| **総資産** | **{format_amount(asset_total)}** | **{bar_and_ratio(asset_total, asset_total)}** | **負債・純資産合計** | **{format_amount(right_total)}** | **{bar_and_ratio(right_total, asset_total)}** |",
            "",
            f"- 合計照合: {balance_note}",
            "",
        ])
    return lines


def build_report(
    ticker: str,
    company_name: str | None,
    tasks: list[tuple[Any, ...]],
    values: dict[str, dict[str, Any]],
    period_sources: dict[str, int],
    item_order: list[str],
    errors: list[str],
) -> str:
    periods = sorted(period_sources, reverse=True)
    title = f"{ticker} {company_name}" if company_name else ticker
    report = [
        f"# {title} 貸借対照表（Yahoo Finance取得値）",
        "",
        f"- レポート作成日時 (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- 対象run: {', '.join(str(row[0]) for row in tasks)}",
        f"- 対象期末日: {', '.join(periods)}",
        "- 同一期末日が複数runにある場合は、run番号が大きい取得値を採用しています。",
        "- 数値はYahoo Financeの値をそのまま表示します。通貨・単位は自動換算していません。",
        "- 以下の区分は読みやすさのための表示グループであり、日本の開示書類上の正式分類ではありません。",
        "",
    ]
    report.extend(render_balance_chart(values, periods))
    report.extend([
        "## 主要合計",
        "",
        "| 項目 | Yahoo Finance項目名 | " + " | ".join(periods) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in periods) + " |",
    ])
    for item in SUMMARY_ITEMS:
        if item in values:
            amounts = [format_amount(values[item].get(period)) for period in periods]
            report.append(f"| {ITEM_LABELS[item]} | `{item}` | " + " | ".join(amounts) + " |")
    report.append("")

    included = set(SUMMARY_ITEMS)
    for title, items in GROUPS:
        report.extend(render_group(title, items, values, periods))
        included.update(items)

    supplemental = [item for item in SUPPLEMENTAL_ITEMS if item in values and item not in included]
    report.extend(render_group("補助指標・株式数", supplemental, values, periods))
    included.update(supplemental)

    other_items = [item for item in item_order if item in values and item not in included]
    if other_items:
        report.extend(render_group("その他取得項目", other_items, values, periods))

    report.extend(["## 期末日ごとの採用run", "", "| 期末日 | 採用run |", "| --- | ---: |"])
    for period in periods:
        report.append(f"| {period} | {period_sources[period]} |")

    if errors:
        report.extend(["", "## 未取得・確認事項", ""])
        report.extend(f"- {error}" for error in errors)
    report.extend([
        "",
        "## データの読み方",
        "",
        "- 金額行の単位・通貨はYahoo Financeから取得した原本に従います。銘柄の通貨情報を確認せずに円などと断定していません。",
        "- 取得元の項目が欠けている期は `—` と表示します。0とは区別してください。",
        "- 合計値と内訳はYahoo Financeが返した値です。このレポートで再計算・会計上の組替えは行っていません。",
        "",
    ])
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a readable Japanese balance-sheet report from saved Yahoo Finance artifacts."
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. 1878.T")
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
            tasks = fetch_balance_sheet_tasks(conn, args.ticker, args.run_id)
            company_name = fetch_company_name(conn, args.ticker, args.run_id)
        if not tasks:
            print(f"貸借対照表の取得記録がありません: {args.ticker}", file=sys.stderr)
            return 1

        values, period_sources, item_order, errors = collect_periods(tasks, args.ticker)
        if not period_sources:
            print("保存済みの貸借対照表データがありません。", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        report = build_report(
            args.ticker, company_name, tasks, values, period_sources, item_order, errors
        )
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", args.ticker)
        suffix = f"run-{args.run_id:06d}" if args.run_id is not None else "all-runs"
        output = Path(args.output).expanduser() if args.output else (
            PROJECT_ROOT / "data" / "reports" / "yfinance" / f"{safe_ticker}-balance-sheet-{suffix}.md"
        )
        if not output.is_absolute():
            output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(output)
        print(f"貸借対照表レポートを作成しました: {output}")
        print(f"期末日数: {len(period_sources)}; 項目数: {len(item_order)}")
        return 0
    except (OSError, RuntimeError, psycopg.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
