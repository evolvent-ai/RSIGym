"""CLI entry point: `rsiwatch [jobs-dir] [--port N]`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from rsiwatch.scanner import STALL_AFTER_SECONDS
from rsiwatch.server import create_app

# src/rsiwatch/__main__.py → parents[3] is the repo root (tools/src/rsiwatch → …
# → tools → RSIPlatform). Only meaningful when running from a source checkout;
# an installed copy elsewhere resolves somewhere unrelated, which is why a
# missing directory is a plain argparse error rather than a traceback.
DEFAULT_JOBS_DIR = Path(__file__).resolve().parents[3] / "rsi_task" / "jobs"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rsiwatch",
        description="Live dashboard for RSI experiments (Harbor jobs + Claude Code trajectories).",
    )
    parser.add_argument("jobs_dir", nargs="?", type=Path, default=DEFAULT_JOBS_DIR,
                        help=f"Harbor jobs directory (default: {DEFAULT_JOBS_DIR})")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between disk polls (default: 1.0)")
    parser.add_argument("--stall-after", type=float, default=STALL_AFTER_SECONDS,
                        help="Flag a trial as stalled after N idle seconds "
                             f"(default: {int(STALL_AFTER_SECONDS)})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    jobs_dir = args.jobs_dir.expanduser().resolve()
    if not jobs_dir.is_dir():
        parser.error(f"jobs directory not found: {jobs_dir}")

    print(f"rsiwatch → http://{args.host}:{args.port}   watching {jobs_dir}")
    app = create_app(jobs_dir, interval=args.interval, stall_after=args.stall_after)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
