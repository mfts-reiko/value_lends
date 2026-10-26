from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import psycopg
import yfinance as yf
from dotenv import load_dotenv

if __package__:
    from .fetch_yfinance import (
        ADVISORY_LOCK,
        PROJECT_ROOT,
        acquire_advisory_lock,
        artifact_path_for,
        json_safe,
        save_artifact,
        update_task,
    )
else:
    from fetch_yfinance import (
        ADVISORY_LOCK,
        PROJECT_ROOT,
        acquire_advisory_lock,
        artifact_path_for,
        json_safe,
        save_artifact,
        update_task,
    )


DEFAULT_RUN_ID = 3
OPTIONAL_DATASETS = (
    "earnings_estimate",
    "revenue_estimate",
    "eps_trend",
    "eps_revisions",
    "growth_estimates",
    "earnings_history",
    "earnings_dates",
    "quarterly_earnings",
    "upgrades_downgrades",
    "recommendations_summary",
    "options",
    "option_chain",
    "sustainability",
)


def dataset_getters() -> dict[str, Callable[[yf.Ticker], Any]]:
    getters = {
        name: (lambda ticker, attribute=name: getattr(ticker, attribute))
        for name in OPTIONAL_DATASETS
        if name != "option_chain"
    }
    getters["option_chain"] = get_option_chain
    return getters


def get_option_chain(ticker: yf.Ticker) -> dict[str, Any]:
    expirations = list(ticker.options)
    chains: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for expiration in expirations:
        chain = ticker.option_chain(expiration)
        chains[expiration] = {
            "calls": chain.calls.to_dict(orient="records"),
            "puts": chain.puts.to_dict(orient="records"),
        }
    return json_safe({"expirations": expirations, "chains": chains})


def add_optional_tasks(conn: psycopg.Connection[Any], run_id: int) -> int:
    run = conn.execute(
        "SELECT datasets FROM yf_fetch_run WHERE run_id = %s FOR UPDATE",
        (run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"Fetch run {run_id} does not exist")

    datasets = list(run[0])
    for dataset in OPTIONAL_DATASETS:
        if dataset not in datasets:
            datasets.append(dataset)

    with conn.transaction():
        conn.execute(
            "UPDATE yf_fetch_run SET datasets = %s, updated_at = now() WHERE run_id = %s",
            (datasets, run_id),
        )
        result = conn.execute(
            """INSERT INTO yf_fetch_task (run_id, ticker, dataset)
               SELECT member.run_id, member.ticker, optional.dataset
               FROM yf_asset_universe_member AS member
               CROSS JOIN unnest(%s::text[]) AS optional(dataset)
               WHERE member.run_id = %s
               ON CONFLICT (run_id, ticker, dataset) DO NOTHING""",
            (list(OPTIONAL_DATASETS), run_id),
        )
    return result.rowcount


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add and fetch optional yfinance datasets for an existing run."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=DEFAULT_RUN_ID,
        help=f"Existing fetch run to extend (default: {DEFAULT_RUN_ID}).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Maximum ticker/dataset requests this invocation; 0 means no limit.",
    )
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.run_id < 1 or args.max_tasks < 0 or args.pause_seconds < 0:
        parser.error("run id must be positive and limits non-negative")
    return args


def run_fetch(
    conn: psycopg.Connection[Any],
    run_id: int,
    args: argparse.Namespace,
) -> int:
    getters = dataset_getters()
    query = """SELECT ticker, dataset
               FROM yf_fetch_task
               WHERE run_id = %s
                 AND dataset = ANY(%s::text[])
                 AND status IN ('pending', 'failed')
               ORDER BY ticker, array_position(%s::text[], dataset)"""
    tasks = conn.execute(
        query,
        (run_id, list(OPTIONAL_DATASETS), list(OPTIONAL_DATASETS)),
    ).fetchall()
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    logging.info(
        "run=%s; optional pending/failed tasks selected=%s",
        run_id,
        len(tasks),
    )
    current_ticker: str | None = None
    ticker_object: yf.Ticker | None = None
    failures = 0
    active_task: tuple[str, str] | None = None

    try:
        for index, (ticker_symbol, dataset) in enumerate(tasks, start=1):
            active_task = (ticker_symbol, dataset)
            task_started = time.monotonic()
            logging.info(
                "starting optional task %s/%s run=%s ticker=%s dataset=%s",
                index,
                len(tasks),
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
                if ticker_object is None:
                    raise RuntimeError(f"Unable to initialize yfinance ticker {ticker_symbol}")
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
                    conn,
                    run_id,
                    ticker_symbol,
                    dataset,
                    "failed",
                    error=message,
                )
                failures += 1
                logging.warning(
                    "failed run=%s ticker=%s dataset=%s attempt=%s elapsed=%.1fs error=%s",
                    run_id,
                    ticker_symbol,
                    dataset,
                    attempt,
                    time.monotonic() - task_started,
                    message,
                )
            active_task = None

            if args.pause_seconds and index < len(tasks):
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
        logging.warning("interrupted; run %s can be resumed", run_id)
        return 130

    counts = conn.execute(
        """SELECT status, count(*) FROM yf_fetch_task
           WHERE run_id = %s AND dataset = ANY(%s::text[])
           GROUP BY status ORDER BY status""",
        (run_id, list(OPTIONAL_DATASETS)),
    ).fetchall()
    status_counts = dict(counts)
    logging.info("run=%s optional task status summary=%s", run_id, status_counts)
    remaining = status_counts.get("pending", 0) + status_counts.get("failed", 0)
    if remaining:
        logging.info(
            "batch finished; remaining optional tasks=%s (pending=%s failed=%s); "
            "rerun this command to continue",
            remaining,
            status_counts.get("pending", 0),
            status_counts.get("failed", 0),
        )
    else:
        logging.info("all optional tasks for run=%s are complete", run_id)
    return 1 if failures else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logging.info(
        "starting optional yfinance fetch run=%s max_tasks=%s pause_seconds=%.1f",
        args.run_id,
        args.max_tasks,
        args.pause_seconds,
    )
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logging.error("DATABASE_URL is missing; create .env from .env.example")
        return 2
    if database_url.startswith("postgresql+psycopg://"):
        database_url = "postgresql://" + database_url.removeprefix("postgresql+psycopg://")

    logging.info("connecting to PostgreSQL")
    with psycopg.connect(database_url, connect_timeout=10) as conn:
        logging.info("connected to PostgreSQL; waiting for fetch lock if occupied")
        acquire_advisory_lock(conn)
        try:
            added = add_optional_tasks(conn, args.run_id)
            logging.info(
                "run=%s; registered optional datasets=%s; new tasks=%s",
                args.run_id,
                len(OPTIONAL_DATASETS),
                added,
            )
            conn.commit()
            return run_fetch(conn, args.run_id, args)
        finally:
            conn.rollback()
            conn.execute("SELECT pg_advisory_unlock(%s, %s)", ADVISORY_LOCK)
            conn.commit()
            logging.info("released PostgreSQL advisory lock")


if __name__ == "__main__":
    sys.exit(main())
