from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import psycopg
import yfinance as yf
from dotenv import load_dotenv
from psycopg.types.json import Jsonb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ADVISORY_LOCK = (731904, 18)
ASSET_FILES = (
    ("tse_growth_assets.csv", "growth"),
    ("tse_prime_assets.csv", "prime"),
    ("tse_standard_assets.csv", "standard"),
    ("tse_growth2_assets.csv", "growth"),
    ("tse_prime2_assets.csv", "prime"),
    ("tse_standard2_assets.csv", "standard"),
)
SUPPLEMENTAL_ASSET_FILES = (
    ("tse_growth2_assets.csv", "growth"),
    ("tse_prime2_assets.csv", "prime"),
    ("tse_standard2_assets.csv", "standard"),
)


def _attribute(name: str) -> Callable[[yf.Ticker], Any]:
    return lambda ticker: getattr(ticker, name)


def dataset_getters(timeout: float) -> dict[str, Callable[[yf.Ticker], Any]]:
    return {
        "info": _attribute("info"),
        "fast_info": lambda ticker: dict(ticker.fast_info),
        "history": lambda ticker: ticker.history(
            period="max", auto_adjust=False, actions=True, timeout=timeout
        ),
        "income_stmt": _attribute("income_stmt"),
        "quarterly_income_stmt": _attribute("quarterly_income_stmt"),
        "balance_sheet": _attribute("balance_sheet"),
        "quarterly_balance_sheet": _attribute("quarterly_balance_sheet"),
        "cashflow": _attribute("cashflow"),
        "quarterly_cashflow": _attribute("quarterly_cashflow"),
        "dividends": _attribute("dividends"),
        "splits": _attribute("splits"),
        "actions": _attribute("actions"),
        "recommendations": _attribute("recommendations"),
        "calendar": _attribute("calendar"),
        "analyst_price_targets": _attribute("analyst_price_targets"),
        "major_holders": _attribute("major_holders"),
        "institutional_holders": _attribute("institutional_holders"),
        "mutualfund_holders": _attribute("mutualfund_holders"),
        "insider_transactions": _attribute("insider_transactions"),
        "insider_purchases": _attribute("insider_purchases"),
        "insider_roster_holders": _attribute("insider_roster_holders"),
        "news": _attribute("news"),
    }


def supported_datasets(
    datasets: list[str],
    getters: dict[str, Callable[[yf.Ticker], Any]],
) -> list[str]:
    return [dataset for dataset in datasets if dataset in getters]


def acquire_advisory_lock(conn: psycopg.Connection[Any]) -> None:
    wait_started = time.monotonic()
    next_log_at = wait_started
    while True:
        acquired = conn.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            ADVISORY_LOCK,
        ).fetchone()[0]
        conn.commit()
        if acquired:
            logging.info(
                "acquired PostgreSQL advisory lock after %.1f seconds",
                time.monotonic() - wait_started,
            )
            return

        now = time.monotonic()
        if now >= next_log_at:
            logging.info(
                "waiting for PostgreSQL advisory lock; waited %.0f seconds",
                now - wait_started,
            )
            next_log_at = now + 30
        time.sleep(1)


def create_tables(conn: psycopg.Connection[Any]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yf_fetch_run (
            run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'failed', 'succeeded')),
            datasets TEXT[] NOT NULL,
            csv_manifest JSONB NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yf_asset_universe_member (
            run_id BIGINT NOT NULL REFERENCES yf_fetch_run(run_id),
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL DEFAULT '',
            market_segment TEXT NOT NULL CHECK (market_segment IN ('growth', 'prime', 'standard')),
            PRIMARY KEY (run_id, ticker)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yf_run_source_file (
            run_id BIGINT NOT NULL REFERENCES yf_fetch_run(run_id),
            file_name TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            artifact_path TEXT NOT NULL,
            PRIMARY KEY (run_id, file_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yf_fetch_task (
            run_id BIGINT NOT NULL,
            ticker TEXT NOT NULL,
            dataset TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'failed', 'succeeded')),
            attempts INTEGER NOT NULL DEFAULT 0,
            artifact_path TEXT,
            row_count BIGINT,
            last_error TEXT,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            PRIMARY KEY (run_id, ticker, dataset),
            FOREIGN KEY (run_id, ticker)
                REFERENCES yf_asset_universe_member(run_id, ticker)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS yf_fetch_task_resume_idx "
        "ON yf_fetch_task (run_id, status, ticker)"
    )
    conn.commit()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_asset_files(
    asset_files: tuple[tuple[str, str], ...] = ASSET_FILES,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    members: list[dict[str, str]] = []
    manifest: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()

    for file_name, market_segment in asset_files:
        path = DATA_DIR / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Required asset CSV is missing: {path}")

        file_members: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or {name.strip() for name in reader.fieldnames} != {
                "ticker",
                "company",
            }:
                raise ValueError(f"Expected ticker,company columns in {path.name}")

            for line_number, row in enumerate(reader, start=2):
                ticker = (row.get("ticker") or "").strip()
                company = (row.get("company") or "").strip()
                if not ticker:
                    raise ValueError(f"Empty ticker in {path.name}:{line_number}")
                if ticker in seen_tickers:
                    raise ValueError(f"Duplicate ticker {ticker!r} in asset CSV files")
                seen_tickers.add(ticker)
                file_members.append(
                    {
                        "ticker": ticker,
                        "company_name": company,
                        "market_segment": market_segment,
                    }
                )

        if not file_members:
            raise ValueError(f"No asset rows found in {path.name}")
        members.extend(file_members)
        manifest.append(
            {
                "file_name": file_name,
                "sha256": sha256_file(path),
                "row_count": len(file_members),
                "source_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )

    return members, manifest


def archive_asset_files(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_key = json.dumps(
        [(item["file_name"], item["sha256"]) for item in manifest],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    archive_id = hashlib.sha256(manifest_key.encode("utf-8")).hexdigest()
    archive_dir = RAW_DIR / "asset-universe" / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived: list[dict[str, Any]] = []
    for item in manifest:
        source_path = PROJECT_ROOT / item["source_path"]
        destination = archive_dir / item["file_name"]
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, destination)
        archived.append(
            {
                **item,
                "artifact_path": str(destination.relative_to(PROJECT_ROOT)),
            }
        )
    return archived


def create_run(
    conn: psycopg.Connection[Any],
    members: list[dict[str, str]],
    archived_files: list[dict[str, Any]],
    datasets: list[str],
) -> int:
    with conn.transaction():
        row = conn.execute(
            """INSERT INTO yf_fetch_run (status, datasets, csv_manifest)
               VALUES ('running', %s, %s) RETURNING run_id""",
            (datasets, Jsonb(archived_files)),
        ).fetchone()
        run_id = int(row[0])

        with conn.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO yf_asset_universe_member
                   (run_id, ticker, company_name, market_segment)
                   VALUES (%s, %s, %s, %s)""",
                [
                    (run_id, item["ticker"], item["company_name"], item["market_segment"])
                    for item in members
                ],
            )
            cursor.executemany(
                """INSERT INTO yf_run_source_file
                   (run_id, file_name, sha256, row_count, artifact_path)
                   VALUES (%s, %s, %s, %s, %s)""",
                [
                    (
                        run_id,
                        item["file_name"],
                        item["sha256"],
                        item["row_count"],
                        item["artifact_path"],
                    )
                    for item in archived_files
                ],
            )
            cursor.executemany(
                """INSERT INTO yf_fetch_task (run_id, ticker, dataset)
                   VALUES (%s, %s, %s)""",
                [
                    (run_id, item["ticker"], dataset)
                    for item in members
                    for dataset in datasets
                ],
            )
    return run_id


def resumable_run(conn: psycopg.Connection[Any], force_new: bool) -> tuple[int, list[str]] | None:
    conn.execute(
        """UPDATE yf_fetch_task
           SET status = 'failed', last_error = 'Previous process stopped before task completed',
               finished_at = now()
           WHERE status = 'running'"""
    )
    conn.execute(
        """UPDATE yf_fetch_run
           SET status = 'paused', updated_at = now(), finished_at = now()
           WHERE status = 'running'"""
    )
    conn.commit()

    if force_new:
        return None
    row = conn.execute(
        """SELECT run_id, datasets FROM yf_fetch_run
           WHERE status IN ('paused', 'failed')
           ORDER BY run_id DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        return None

    remaining = conn.execute(
        """SELECT count(*) FROM yf_fetch_task
           WHERE run_id = %s AND status <> 'succeeded'""",
        (row[0],),
    ).fetchone()[0]
    if remaining == 0:
        conn.execute(
            """UPDATE yf_fetch_run
               SET status = 'succeeded', updated_at = now(), finished_at = now()
               WHERE run_id = %s""",
            (row[0],),
        )
        conn.commit()
        return None

    conn.execute(
        """UPDATE yf_fetch_run
           SET status = 'running', updated_at = now(), finished_at = NULL
           WHERE run_id = %s""",
        (row[0],),
    )
    conn.commit()
    return int(row[0]), list(row[1])


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def artifact_path_for(run_id: int, ticker: str, dataset: str, result: Any) -> Path:
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    suffix = ".parquet" if isinstance(result, (pd.DataFrame, pd.Series)) else ".json.gz"
    return RAW_DIR / "yfinance" / f"run-{run_id:06d}" / safe_ticker / f"{dataset}{suffix}"


def save_artifact(path: Path, result: Any) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        if isinstance(result, pd.Series):
            frame = result.to_frame(name=result.name or "value")
            frame.to_parquet(temporary, index=True)
            row_count: int | None = len(frame)
        elif isinstance(result, pd.DataFrame):
            result.to_parquet(temporary, index=True)
            row_count = len(result)
        else:
            with gzip.open(temporary, "wt", encoding="utf-8") as target:
                json.dump(json_safe(result), target, ensure_ascii=False, allow_nan=False)
            row_count = None
        os.replace(temporary, path)
        return row_count
    finally:
        if temporary.exists():
            temporary.unlink()


def update_task(
    conn: psycopg.Connection[Any],
    run_id: int,
    ticker: str,
    dataset: str,
    status: str,
    *,
    artifact_path: str | None = None,
    row_count: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """UPDATE yf_fetch_task
           SET status = %s, artifact_path = %s, row_count = %s,
               last_error = %s, finished_at = now()
           WHERE run_id = %s AND ticker = %s AND dataset = %s""",
        (status, artifact_path, row_count, error, run_id, ticker, dataset),
    )
    conn.commit()


def finish_run(conn: psycopg.Connection[Any], run_id: int, paused: bool) -> dict[str, int]:
    counts = {
        row[0]: row[1]
        for row in conn.execute(
            """SELECT status, count(*) FROM yf_fetch_task
               WHERE run_id = %s GROUP BY status""",
            (run_id,),
        ).fetchall()
    }
    if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0 and counts.get("failed", 0) == 0:
        status = "succeeded"
    elif paused:
        status = "paused"
    else:
        status = "failed"
    conn.execute(
        """UPDATE yf_fetch_run
           SET status = %s, updated_at = now(), finished_at = now()
           WHERE run_id = %s""",
        (status, run_id),
    )
    conn.commit()
    return counts


def parse_args() -> argparse.Namespace:
    available = list(dataset_getters(30.0))
    parser = argparse.ArgumentParser(
        description="Fetch yfinance datasets in small, resumable batches."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=available,
        help="Datasets for a new run; defaults to all supported datasets.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum ticker/dataset requests per invocation; 0 means no limit (default: 100).",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Delay between requests (default: 1 second).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a new full run instead of resuming the latest incomplete run.",
    )
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Use only the three supplemental *2 asset CSV files for a new run.",
    )
    args = parser.parse_args()
    if args.max_tasks < 0 or args.pause_seconds < 0 or args.timeout_seconds <= 0:
        parser.error("limits must be non-negative and timeout must be positive")
    return args


def run_fetch(conn: psycopg.Connection[Any], run_id: int, datasets: list[str], args: argparse.Namespace) -> int:
    getters = dataset_getters(args.timeout_seconds)
    fetchable_datasets = supported_datasets(datasets, getters)
    query = """SELECT task.ticker, task.dataset, member.market_segment, member.company_name
               FROM yf_fetch_task AS task
               JOIN yf_asset_universe_member AS member USING (run_id, ticker)
               WHERE task.run_id = %s
                 AND task.dataset = ANY(%s::text[])
                 AND task.status IN ('pending', 'failed')
               ORDER BY task.ticker, array_position(%s::text[], task.dataset)"""
    tasks = conn.execute(
        query,
        (run_id, fetchable_datasets, fetchable_datasets),
    ).fetchall()
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    logging.info("run=%s; pending attempts this invocation=%s", run_id, len(tasks))
    current_ticker: str | None = None
    ticker_object: yf.Ticker | None = None
    completed_attempts = 0
    active_task: tuple[str, str] | None = None

    try:
        for ticker_symbol, dataset, _market_segment, _company_name in tasks:
            active_task = (ticker_symbol, dataset)
            task_started = time.monotonic()
            logging.info(
                "starting run=%s ticker=%s dataset=%s",
                run_id,
                ticker_symbol,
                dataset,
            )
            row = conn.execute(
                """UPDATE yf_fetch_task
                   SET status = 'running', attempts = attempts + 1, started_at = now(),
                       finished_at = NULL, last_error = NULL
                   WHERE run_id = %s AND ticker = %s AND dataset = %s
                   RETURNING attempts""",
                (run_id, ticker_symbol, dataset),
            ).fetchone()
            conn.commit()
            attempt = row[0]

            try:
                if ticker_symbol != current_ticker:
                    current_ticker = ticker_symbol
                    ticker_object = yf.Ticker(ticker_symbol)
                result = getters[dataset](ticker_object)
                path = artifact_path_for(run_id, ticker_symbol, dataset, result)
                row_count = save_artifact(path, result)
                update_task(
                    conn,
                    run_id,
                    ticker_symbol,
                    dataset,
                    "succeeded",
                    artifact_path=str(path.relative_to(PROJECT_ROOT)),
                    row_count=row_count,
                )
                logging.info(
                    "saved run=%s ticker=%s dataset=%s attempt=%s rows=%s elapsed=%.1fs",
                    run_id,
                    ticker_symbol,
                    dataset,
                    attempt,
                    row_count if row_count is not None else "n/a",
                    time.monotonic() - task_started,
                )
            except Exception as error:
                current_ticker = None
                ticker_object = None
                message = f"{type(error).__name__}: {error}"[:4000]
                update_task(
                    conn, run_id, ticker_symbol, dataset, "failed", error=message
                )
                logging.warning(
                    "failed run=%s ticker=%s dataset=%s attempt=%s elapsed=%.1fs error=%s",
                    run_id,
                    ticker_symbol,
                    dataset,
                    attempt,
                    time.monotonic() - task_started,
                    message,
                )
            completed_attempts += 1
            active_task = None
            if args.pause_seconds:
                time.sleep(args.pause_seconds)
    except KeyboardInterrupt:
        if active_task is not None:
            update_task(
                conn,
                run_id,
                active_task[0],
                active_task[1],
                "failed",
                error="Interrupted by user; task will be retried next run",
            )
        finish_run(conn, run_id, paused=True)
        logging.warning("interrupted; run %s is saved and can be resumed", run_id)
        return 130

    counts = finish_run(conn, run_id, paused=completed_attempts < len(tasks) or bool(tasks))
    logging.info("run=%s status summary=%s", run_id, counts)
    return 0 if counts.get("failed", 0) == 0 and counts.get("pending", 0) == 0 else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logging.info(
        "starting yfinance fetch max_tasks=%s pause_seconds=%.1f",
        args.max_tasks,
        args.pause_seconds,
    )
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logging.error("DATABASE_URL is missing; create .env from .env.example")
        return 2

    # Accept the SQLAlchemy-style URL already used by older .env files.
    if database_url.startswith("postgresql+psycopg://"):
        database_url = "postgresql://" + database_url.removeprefix("postgresql+psycopg://")

    logging.info("connecting to PostgreSQL")
    conn = psycopg.connect(database_url, connect_timeout=10)
    try:
        logging.info("connected to PostgreSQL; waiting for fetch lock if occupied")
        acquire_advisory_lock(conn)
        create_tables(conn)

        selected_run = resumable_run(conn, args.new_run)
        if selected_run is None:
            asset_files = SUPPLEMENTAL_ASSET_FILES if args.supplement_only else ASSET_FILES
            members, manifest = load_asset_files(asset_files)
            archived_files = archive_asset_files(manifest)
            datasets = args.datasets or list(dataset_getters(args.timeout_seconds))
            run_id = create_run(conn, members, archived_files, datasets)
            logging.info(
                "created run=%s with %s tickers and %s datasets",
                run_id,
                len(members),
                len(datasets),
            )
        else:
            run_id, datasets = selected_run
            logging.info("resuming incomplete run=%s", run_id)

        return run_fetch(conn, run_id, datasets, args)
    finally:
        try:
            conn.execute("SELECT pg_advisory_unlock(%s, %s)", ADVISORY_LOCK)
            conn.commit()
            logging.info("released PostgreSQL advisory lock")
        finally:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())