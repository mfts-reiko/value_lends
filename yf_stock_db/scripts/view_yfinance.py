from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing; create .env from .env.example")
    prefix = "postgresql+psycopg://"
    if url.startswith(prefix):
        url = "postgresql://" + url.removeprefix(prefix)
    return url


def fetch_tasks(
    conn: psycopg.Connection[Any], run_id: int, ticker: str
) -> list[tuple[Any, ...]]:
    return conn.execute(
        """SELECT dataset, status, artifact_path, row_count, attempts, last_error
           FROM yf_fetch_task
           WHERE run_id = %s AND ticker = %s
           ORDER BY dataset""",
        (run_id, ticker),
    ).fetchall()


def print_task_list(run_id: int, ticker: str, tasks: list[tuple[Any, ...]]) -> None:
    print(f"run={run_id} ticker={ticker}")
    if not tasks:
        print("No fetch tasks found for this run and ticker.")
        return

    print(f"{'DATASET':32} {'STATUS':10} {'ROWS':>10} {'ATTEMPTS':>8} ARTIFACT")
    for dataset, status, artifact_path, row_count, attempts, error in tasks:
        rows = str(row_count) if row_count is not None else "-"
        path = artifact_path or "-"
        print(f"{dataset:32.32} {status:10.10} {rows:>10} {attempts:>8} {path}")
        if error:
            print(f"  last error: {str(error).replace(chr(10), ' ')[:240]}")


def read_artifact(
    artifact_path: str, limit: int, max_chars: int
) -> None:
    path = (PROJECT_ROOT / artifact_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("Artifact path resolves outside the project directory")
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file not found: {path}")

    print(f"artifact: {path.relative_to(PROJECT_ROOT)}")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        print(f"shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
        if frame.empty:
            print("(empty table)")
        else:
            print(frame.head(limit).to_string(max_cols=30, max_colwidth=80))
        return

    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n... (output truncated; increase --max-chars)"
    print(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List or display data fetched for a ticker in a run."
    )
    parser.add_argument("run_id", type=int, help="Fetch run number")
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. 1301.T")
    parser.add_argument("--dataset", help="Display one dataset artifact")
    parser.add_argument("--limit", type=int, default=20, help="Parquet rows to display")
    parser.add_argument(
        "--max-chars", type=int, default=12000, help="Maximum JSON characters to display"
    )
    args = parser.parse_args()
    if args.run_id < 1 or args.limit < 1 or args.max_chars < 1:
        parser.error("run_id, limit, and max-chars must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        with psycopg.connect(database_url(), connect_timeout=10) as conn:
            tasks = fetch_tasks(conn, args.run_id, args.ticker)
            if args.dataset is None:
                print_task_list(args.run_id, args.ticker, tasks)
                return 0 if tasks else 1

            selected = next((task for task in tasks if task[0] == args.dataset), None)
            if selected is None:
                print(f"No dataset '{args.dataset}' found for run={args.run_id}, ticker={args.ticker}.")
                if tasks:
                    print("Available datasets:", ", ".join(task[0] for task in tasks))
                return 1

            dataset, status, artifact_path, row_count, attempts, error = selected
            print(
                f"run={args.run_id} ticker={args.ticker} dataset={dataset} "
                f"status={status} attempts={attempts} rows={row_count}"
            )
            if error:
                print(f"last error: {error}")
            if not artifact_path:
                print("No artifact has been saved for this dataset.")
                return 1
            read_artifact(artifact_path, args.limit, args.max_chars)
            return 0
    except (OSError, RuntimeError, psycopg.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
