from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_FILES = (
    "tse_growth_assets.csv",
    "tse_prime_assets.csv",
    "tse_standard_assets.csv",
)
REPORTS = (
    ("balance_sheet", "report_balance_sheet.py"),
    ("cashflow", "report_cashflow.py"),
    ("company_profile", "report_company_profile.py"),
    ("income_statement", "report_income_statement.py"),
)


def database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL がありません。.env を確認してください")
    prefix = "postgresql+psycopg://"
    return "postgresql://" + url.removeprefix(prefix) if url.startswith(prefix) else url


def load_assets() -> dict[str, str]:
    assets: dict[str, str] = {}
    for file_name in ASSET_FILES:
        path = PROJECT_ROOT / "data" / file_name
        with path.open(encoding="utf-8-sig", newline="") as asset_file:
            for row in csv.DictReader(asset_file):
                ticker = (row.get("ticker") or "").strip()
                if ticker:
                    assets.setdefault(ticker, (row.get("company") or "").strip())
    return assets


def safe_component(value: str) -> str:
    component = re.sub(r"[\x00-\x1f/\\:]+", "_", value).strip(" .")
    return component or "unknown"


def fetch_latest_runs(
    conn: psycopg.Connection[Any], tickers: list[str]
) -> dict[str, int]:
    rows = conn.execute(
        """SELECT ticker, MAX(run_id)
           FROM yf_fetch_task
           WHERE ticker = ANY(%s)
           GROUP BY ticker""",
        (tickers,),
    ).fetchall()
    return {ticker: run_id for ticker, run_id in rows}


def create_reports(
    ticker: str,
    company_name: str,
    run_id: int,
    output_root: Path,
) -> list[str]:
    asset_dir = f"{safe_component(ticker)}_{safe_component(company_name or ticker)}"
    destination = output_root / asset_dir
    failures: list[str] = []
    for report_kind, script_name in REPORTS:
        output = destination / (
            f"{safe_component(ticker)}_{safe_component(company_name or ticker)}_"
            f"{run_id:06d}_{report_kind}.md"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / script_name),
                ticker,
                "--run-id",
                str(run_id),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            failures.append(f"{report_kind}: {detail or f'exit {result.returncode}'}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create four Yahoo Finance reports for every ticker in the asset CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "report" / "yfinance",
        help="Report root directory (default: reports/yfinance)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        assets = load_assets()
        if not assets:
            print("銘柄一覧にtickerがありません。", file=sys.stderr)
            return 1
        with psycopg.connect(database_url(), connect_timeout=10) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            latest_runs = fetch_latest_runs(conn, list(assets))

        skipped = 0
        failed = 0
        for ticker, company_name in assets.items():
            run_id = latest_runs.get(ticker)
            if run_id is None:
                print(f"SKIP {ticker}: 取得runがありません")
                skipped += 1
                continue
            errors = create_reports(ticker, company_name, run_id, args.output_dir)
            if errors:
                failed += len(errors)
                print(f"FAIL {ticker} run={run_id:06d}: " + "; ".join(errors))
            else:
                print(f"OK {ticker} run={run_id:06d}")

        print(
            f"完了: 銘柄 {len(assets)}、取得履歴なし {skipped}、"
            f"レポート失敗 {failed}件、出力先 {args.output_dir}"
        )
        return 1 if failed else 0
    except (OSError, RuntimeError, psycopg.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())