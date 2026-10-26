from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_LABELS = {
    "info": "企業概要・株価指標",
    "fast_info": "主要株価・時価情報",
    "history": "株価・出来高・配当・分割の時系列",
    "income_stmt": "年次損益計算書",
    "quarterly_income_stmt": "四半期損益計算書",
    "balance_sheet": "年次貸借対照表",
    "quarterly_balance_sheet": "四半期貸借対照表",
    "cashflow": "年次キャッシュフロー計算書",
    "quarterly_cashflow": "四半期キャッシュフロー計算書",
    "dividends": "配当履歴",
    "splits": "株式分割履歴",
    "actions": "配当・株式分割イベント",
    "recommendations": "アナリスト評価の履歴",
    "calendar": "決算・イベント予定",
    "analyst_price_targets": "アナリスト目標株価",
    "major_holders": "主要株主区分",
    "institutional_holders": "機関投資家保有情報",
    "mutualfund_holders": "投資信託保有情報",
    "insider_transactions": "インサイダー取引履歴",
    "insider_purchases": "インサイダー売買集計",
    "insider_roster_holders": "役員・関係者保有情報",
    "news": "ニュース",
}

FIELD_LABELS = {
    "address1": "住所",
    "address2": "住所（続き）",
    "zip": "郵便番号",
    "phone": "電話番号",
    "fax": "FAX番号",
    "industryKey": "業種コード",
    "sectorKey": "セクターコード",
    "companyOfficers": "役員情報",
    "executiveTeam": "経営陣情報",
    "irWebsite": "IRサイト",
    "marketState": "市場の取引状態",
    "exchangeTimezoneName": "取引所タイムゾーン",
    "exchangeTimezoneShortName": "取引所タイムゾーン略称",
    "quoteSourceName": "株価情報源",
    "priceHint": "価格表示精度",
    "nonDilutedMarketCap": "希薄化前時価総額",
    "averageVolume10days": "10日平均出来高",
    "averageDailyVolume10Day": "10日平均日次出来高",
    "averageDailyVolume3Month": "3か月平均日次出来高",
    "fiftyTwoWeekRange": "52週株価レンジ",
    "allTimeHigh": "上場来高値",
    "allTimeLow": "上場来安値",
    "trailingAnnualDividendRate": "過去1年間の1株当たり配当",
    "trailingAnnualDividendYield": "過去1年間の配当利回り",
    "fiveYearAvgDividendYield": "過去5年平均配当利回り",
    "lastDividendValue": "直近配当額",
    "lastDividendDate": "直近配当日",
    "exDividendDate": "権利落ち日",
    "targetMedianPrice": "アナリスト目標株価中央値",
    "profitMargins": "純利益率",
    "grossMargins": "売上総利益率",
    "operatingMargins": "営業利益率",
    "ebitdaMargins": "EBITDAマージン",
    "returnOnAssets": "総資産利益率（ROA）",
    "returnOnEquity": "自己資本利益率（ROE）",
    "debtToEquity": "負債資本倍率",
    "quickRatio": "当座比率",
    "currentRatio": "流動比率",
    "totalCash": "現金及び現金同等物合計",
    "totalCashPerShare": "1株当たり現金",
    "totalRevenue": "売上高合計",
    "grossProfits": "売上総利益",
    "operatingCashflow": "営業キャッシュフロー",
    "freeCashflow": "フリーキャッシュフロー",
    "earningsGrowth": "利益成長率",
    "revenueGrowth": "売上高成長率",
    "earningsQuarterlyGrowth": "四半期利益成長率",
    "revenuePerShare": "1株当たり売上高",
    "priceToSalesTrailing12Months": "過去12か月株価売上高倍率",
    "pegRatio": "PEGレシオ",
    "trailingPegRatio": "実績PEGレシオ",
    "enterpriseToRevenue": "企業価値売上高倍率",
    "enterpriseToEbitda": "企業価値EBITDA倍率",
    "heldPercentInsiders": "内部関係者保有割合",
    "heldPercentInstitutions": "機関投資家保有割合",
    "bid": "買気配値",
    "ask": "売気配値",
    "bidSize": "買気配数量",
    "askSize": "売気配数量",
    "lastSplitFactor": "直近株式分割比率",
    "lastSplitDate": "直近株式分割日",
    "tradeable": "取引可能フラグ",
    "regularMarketTime": "通常取引の時刻",
    "regularMarketOpen": "通常取引の始値",
    "regularMarketDayLow": "通常取引の当日安値",
    "regularMarketDayHigh": "通常取引の当日高値",
    "regularMarketVolume": "通常取引の出来高",
    "regularMarketChange": "通常取引の前日比",
    "regularMarketChangePercent": "通常取引の前日比率",
    "52WeekChange": "52週株価変化率",
    "SandP52WeekChange": "S&P 500の52週変化率",
    "language": "言語",
    "region": "地域",
    "typeDisp": "証券種別表示名",
    "exchangeDataDelayedBy": "相場データ遅延時間（分）",
    "symbol": "Yahoo Finance ticker",
    "shortName": "会社名",
    "longName": "正式社名",
    "displayName": "表示名",
    "quoteType": "証券種別",
    "exchange": "取引所",
    "fullExchangeName": "取引所名",
    "market": "市場",
    "currency": "通貨",
    "financialCurrency": "財務情報の通貨",
    "country": "国",
    "city": "都市",
    "industry": "業種",
    "industryDisp": "業種表示名",
    "sector": "セクター",
    "sectorDisp": "セクター表示名",
    "website": "企業Webサイト",
    "longBusinessSummary": "事業概要",
    "fullTimeEmployees": "従業員数",
    "marketCap": "時価総額",
    "enterpriseValue": "企業価値",
    "currentPrice": "現在株価",
    "regularMarketPrice": "通常取引の株価",
    "lastPrice": "直近株価",
    "previousClose": "前日終値",
    "regularMarketPreviousClose": "通常取引の前日終値",
    "open": "始値",
    "dayLow": "当日安値",
    "dayHigh": "当日高値",
    "lastVolume": "直近出来高",
    "volume": "出来高",
    "averageVolume": "平均出来高",
    "tenDayAverageVolume": "10日平均出来高",
    "threeMonthAverageVolume": "3か月平均出来高",
    "fiftyDayAverage": "50日移動平均株価",
    "twoHundredDayAverage": "200日移動平均株価",
    "fiftyTwoWeekLow": "52週安値",
    "fiftyTwoWeekHigh": "52週高値",
    "yearLow": "年初来安値",
    "yearHigh": "年初来高値",
    "yearChange": "前年比変化率",
    "shares": "発行済株式数",
    "sharesOutstanding": "発行済株式数",
    "floatShares": "浮動株数",
    "bookValue": "1株当たり純資産",
    "priceToBook": "株価純資産倍率（PBR）",
    "trailingPE": "実績株価収益率（PER）",
    "forwardPE": "予想株価収益率（PER）",
    "trailingEps": "実績1株当たり利益（EPS）",
    "forwardEps": "予想1株当たり利益（EPS）",
    "dividendRate": "年間配当予想",
    "dividendYield": "配当利回り",
    "payoutRatio": "配当性向",
    "beta": "ベータ値",
    "targetMeanPrice": "アナリスト目標株価平均",
    "targetHighPrice": "アナリスト目標株価上限",
    "targetLowPrice": "アナリスト目標株価下限",
    "recommendationKey": "アナリスト評価",
    "recommendationMean": "アナリスト評価平均",
    "numberOfAnalystOpinions": "アナリスト評価人数",
    "date": "日付",
    "Date": "日付",
    "Open": "始値",
    "High": "高値",
    "Low": "安値",
    "Close": "終値",
    "Adj Close": "調整後終値",
    "Volume": "出来高",
    "Dividends": "配当金",
    "Stock Splits": "株式分割比率",
    "period": "対象期間",
    "strongBuy": "強い買い評価数",
    "buy": "買い評価数",
    "hold": "中立評価数",
    "sell": "売り評価数",
    "strongSell": "強い売り評価数",
    "Tax Effect Of Unusual Items": "特殊要因の税効果",
    "Tax Rate For Calcs": "計算用税率",
    "Normalized EBITDA": "調整後EBITDA",
    "Total Revenue": "売上高",
    "Cost Of Revenue": "売上原価",
    "Gross Profit": "売上総利益",
    "Operating Expense": "営業費用",
    "Operating Income": "営業利益",
    "EBIT": "利払前税引前利益",
    "EBITDA": "利払前税引前減価償却前利益",
    "Interest Expense": "支払利息",
    "Pretax Income": "税引前利益",
    "Tax Provision": "法人税等",
    "Net Income": "当期純利益",
    "Basic EPS": "基本的1株当たり利益",
    "Diluted EPS": "希薄化後1株当たり利益",
    "Basic Average Shares": "基本的加重平均株式数",
    "Diluted Average Shares": "希薄化後加重平均株式数",
    "Total Assets": "総資産",
    "Current Assets": "流動資産",
    "Cash And Cash Equivalents": "現金及び現金同等物",
    "Inventory": "棚卸資産",
    "Accounts Receivable": "売掛金・受取手形",
    "Total Liabilities Net Minority Interest": "負債合計",
    "Current Liabilities": "流動負債",
    "Total Debt": "有利子負債合計",
    "Stockholders Equity": "株主資本",
    "Working Capital": "運転資本",
    "Retained Earnings": "利益剰余金",
    "Ordinary Shares Number": "普通株式数",
    "Operating Cash Flow": "営業キャッシュフロー",
    "Investing Cash Flow": "投資キャッシュフロー",
    "Financing Cash Flow": "財務キャッシュフロー",
    "Free Cash Flow": "フリーキャッシュフロー",
    "Capital Expenditure": "設備投資",
    "Depreciation And Amortization": "減価償却費",
}


def database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL がありません。.env を確認してください")
    prefix = "postgresql+psycopg://"
    if url.startswith(prefix):
        url = "postgresql://" + url.removeprefix(prefix)
    return url


def field_label(value: Any) -> str:
    if isinstance(value, tuple):
        raw = " / ".join(str(part) for part in value if part is not None)
    else:
        raw = str(value)
    label = FIELD_LABELS.get(raw)
    return f"{label} (`{raw}`)" if label else f"Yahoo項目 `{raw}`"


def format_value(value: Any) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return "—"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return "—"
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def index_label(name: Any, position: int) -> str:
    if name is None:
        return "行ラベル" if position == 0 else f"行インデックス{position + 1}"
    return field_label(name)


def render_dataframe(frame: pd.DataFrame) -> list[str]:
    index_frame = frame.index.to_frame(index=False)
    index_headers = [index_label(name, index) for index, name in enumerate(frame.index.names)]
    column_headers = [field_label(column) for column in frame.columns]
    headers = index_headers + column_headers
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]

    if frame.empty:
        lines.append("| " + " | ".join("データなし" for _ in headers) + " |")
        return lines

    for index_values, row_values in zip(index_frame.itertuples(index=False, name=None), frame.itertuples(index=False, name=None)):
        cells = [format_value(value) for value in (*index_values, *row_values)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_json(value: Any, level: int = 0, label: str | None = None) -> list[str]:
    indent = "  " * level
    lines: list[str] = []
    if isinstance(value, dict):
        if label is not None:
            lines.append(f"{indent}- **{label}**")
            indent += "  "
        if not value:
            lines.append(f"{indent}- （項目なし）")
        for key, child in value.items():
            display = field_label(key)
            if isinstance(child, (dict, list)):
                lines.extend(render_json(child, level + (1 if label is not None else 0), display))
            else:
                lines.append(f"{indent}- **{display}:** {format_value(child)}")
        return lines

    if isinstance(value, list):
        if label is not None:
            lines.append(f"{indent}- **{label}**")
        if not value:
            lines.append(f"{indent}  - （0件）")
            return lines
        for position, child in enumerate(value, start=1):
            if isinstance(child, (dict, list)):
                lines.append(f"{indent}  - **{position}件目**")
                lines.extend(render_json(child, level + 2))
            else:
                lines.append(f"{indent}  - {format_value(child)}")
        return lines

    prefix = f"**{label}:** " if label else ""
    lines.append(f"{indent}- {prefix}{format_value(value)}")
    return lines


def load_artifact(relative_path: str) -> tuple[Path, Any]:
    path = (PROJECT_ROOT / relative_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"artifact path is outside the project: {relative_path}")
    if path.suffix == ".parquet":
        return path, pd.read_parquet(path)
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return path, json.load(source)
    raise ValueError(f"Unsupported artifact format: {relative_path}")


def fetch_tasks(
    conn: psycopg.Connection[Any], ticker: str, run_id: int | None
) -> list[tuple[Any, ...]]:
    return conn.execute(
        """SELECT task.run_id, run.status, run.started_at, task.dataset,
                  task.status, task.artifact_path, task.row_count, task.attempts,
                  task.last_error
           FROM yf_fetch_task AS task
           JOIN yf_fetch_run AS run USING (run_id)
           WHERE task.ticker = %s AND (%s::bigint IS NULL OR task.run_id = %s)
           ORDER BY task.run_id, task.dataset""",
        (ticker, run_id, run_id),
    ).fetchall()


def default_output_path(ticker: str, run_id: int | None) -> Path:
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    suffix = f"-run-{run_id:06d}" if run_id is not None else "-all-runs"
    return PROJECT_ROOT / "data" / "reports" / "yfinance" / f"{safe_ticker}{suffix}.md"


def build_report(ticker: str, rows: list[tuple[Any, ...]]) -> str:
    report: list[str] = [
        f"# Yahoo Finance取得データレポート: {ticker}",
        "",
        f"- レポート作成時刻 (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- 対象run: {', '.join(str(run_id) for run_id in sorted({row[0] for row in rows}))}",
        "- 読み方: 日本語の項目説明の後ろに元のYahoo項目名を併記しています。",
        "- 数値は取得時の値をそのまま表示し、通貨・単位が不明な値は換算していません。",
        "- このレポートは取得済みファイルの全行・全項目を掲載します。",
        "",
        "## 取得状況",
        "",
        "| Run | Run状態 | データセット | 取得状態 | 行数 | 試行回数 |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]

    for run_id, run_status, _started_at, dataset, status, _artifact, row_count, attempts, _error in rows:
        count = str(row_count) if row_count is not None else "—"
        report.append(f"| {run_id} | {run_status} | `{dataset}` | {status} | {count} | {attempts} |")

    by_dataset: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for row in rows:
        by_dataset[row[3]].append(row)

    report.extend(["", "## データ詳細", ""])
    for dataset in sorted(by_dataset):
        tasks = by_dataset[dataset]
        report.extend([
            f"### {DATASET_LABELS.get(dataset, '取得データ')} (`{dataset}`)",
            "",
        ])
        successful = [row for row in tasks if row[4] == "succeeded" and row[5]]
        if not successful:
            report.append("取得済みのデータファイルはありません。")
            for row in tasks:
                if row[8]:
                    report.append(f"- run {row[0]} のエラー: {format_value(row[8])}")
            report.append("")
            continue

        for row in successful:
            run_id, run_status, started_at, _name, status, artifact_path, row_count, attempts, _error = row
            report.extend([
                f"#### Run {run_id}（開始: {started_at}, run状態: {run_status}）",
                "",
                f"- 取得状態: {status}; 試行回数: {attempts}; 行数: {row_count if row_count is not None else '—'}",
                f"- 保存ファイル: `{artifact_path}`",
                "",
            ])
            try:
                path, value = load_artifact(artifact_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if isinstance(value, pd.DataFrame):
                    report.append(f"表のサイズ: {len(value)} 行 × {len(value.columns)} 列")
                    report.append("")
                    report.extend(render_dataframe(value))
                else:
                    report.extend(render_json(value))
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                report.append(f"データの読み込みに失敗しました: {format_value(str(error))}")
            report.append("")

    report.extend([
        "## 注意",
        "",
        "Yahoo Financeが返したデータのみを表示しています。値が空、表が0行、または情報が未提供でも、それ自体は取得エラーを意味しません。取得中の項目はこのレポートには含まれないため、取得の進行後に再実行すると追加分が反映されます。",
        "",
    ])
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand all saved Yahoo Finance artifacts for a ticker into a readable Markdown report."
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. 1878.T")
    parser.add_argument("--run-id", type=int, help="Include only one fetch run; default is all runs")
    parser.add_argument("--output", help="Output Markdown path; default is under data/reports/yfinance")
    args = parser.parse_args()
    if args.run_id is not None and args.run_id < 1:
        parser.error("run-id must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        with psycopg.connect(database_url(), connect_timeout=10) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            rows = fetch_tasks(conn, args.ticker, args.run_id)
        if not rows:
            print(f"取得履歴がありません: ticker={args.ticker}", file=sys.stderr)
            return 1

        report = build_report(args.ticker, rows)
        output = Path(args.output).expanduser() if args.output else default_output_path(args.ticker, args.run_id)
        if not output.is_absolute():
            output = Path.cwd() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(output)

        succeeded_files = sum(1 for row in rows if row[4] == "succeeded" and row[5])
        other_tasks = len(rows) - succeeded_files
        print(f"レポートを作成しました: {output}")
        print(f"保存済みデータセット: {succeeded_files}; 取得中・失敗・保存先なし: {other_tasks}")
        return 0
    except (OSError, RuntimeError, psycopg.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())