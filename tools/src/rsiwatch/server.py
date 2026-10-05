"""FastAPI app: job overview, trial trajectories, and an SSE change feed.

The scanner is polled on a single background task rather than per connection, so
N open browsers cost one set of file reads. Clients subscribe to an asyncio
queue and receive a bump whenever a poll saw new bytes; they then re-fetch the
slice they care about. Events carry no payload — a dashboard that reloads its
own view stays correct even if it missed a tick.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from rsiwatch.scanner import STALL_AFTER_SECONDS, Scanner

STATIC_DIR = Path(__file__).parent / "static"

_SPINE_CODES = {"thinking": "t", "text": "x", "tool_use": "u",
                "tool_result": "r", "user": "p"}


class Hub:
    """Fan-out of "something changed" ticks to connected clients."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[int]] = set()
        self._tick = 0

    def subscribe(self) -> asyncio.Queue[int]:
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=8)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[int]) -> None:
        self._subscribers.discard(queue)

    def publish(self) -> None:
        self._tick += 1
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(self._tick)
            except asyncio.QueueFull:
                # A client too slow to drain doesn't need every tick — it
                # re-fetches full state on the next one it does receive.
                pass


async def _poll_loop(app: FastAPI) -> None:
    scanner: Scanner = app.state.scanner
    hub: Hub = app.state.hub
    interval: float = app.state.interval
    while True:
        try:
            changed = await asyncio.to_thread(scanner.refresh)
            if changed:
                hub.publish()
        except Exception:  # a poll failure must not kill the watcher
            app.state.logger.exception("scan failed")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    task = asyncio.create_task(_poll_loop(app))
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app(jobs_dir: Path, interval: float = 1.0,
               stall_after: float = STALL_AFTER_SECONDS) -> FastAPI:
    import logging

    app = FastAPI(title="rsiwatch", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.scanner = Scanner(jobs_dir, stall_after=stall_after)
    app.state.hub = Hub()
    app.state.interval = interval
    app.state.logger = logging.getLogger("rsiwatch")
    app.state.scanner.refresh()

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return app.state.scanner.overview()

    @app.get("/api/jobs/{job_name}/trials/{trial_name}")
    def trial_detail(
        job_name: str,
        trial_name: str,
        since: int = Query(0, ge=0, description="Return only events with seq > since"),
        until: int | None = Query(None, ge=1, description="Return only events with seq <= until"),
        limit: int = Query(400, ge=1, le=5000),
    ) -> dict[str, Any]:
        trial = app.state.scanner.trial(job_name, trial_name)
        if trial is None:
            raise HTTPException(status_code=404, detail="trial not found")

        events: list[dict[str, Any]] = []
        truncated = False
        spine = ""
        if trial.reader is not None:
            selected = [e for e in trial.reader.events
                        if e.seq > since and (until is None or e.seq <= until)]
            if len(selected) > limit:
                # Keep the newest window: when catching up on a long run, the
                # tail is what matters.
                selected = selected[-limit:]
                truncated = True
            events = [e.as_dict() for e in selected]
            # One char per event, in seq order (seq is contiguous from 1), so
            # the client can draw and navigate the whole run without its bodies.
            spine = "".join(
                "R" if e.is_error else _SPINE_CODES.get(e.kind, "?")
                for e in trial.reader.events
            )

        return {
            "trial": trial.as_summary(),
            "summary": trial.reader.summary() if trial.reader else {},
            "session_path": str(trial.session_path) if trial.session_path else None,
            "events": events,
            "truncated": truncated,
            "spine": spine,
            "latest_seq": trial.reader.events[-1].seq
            if trial.reader and trial.reader.events else 0,
        }

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        hub: Hub = app.state.hub

        async def stream() -> AsyncGenerator[str]:
            queue = hub.subscribe()
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        tick = await asyncio.wait_for(queue.get(), timeout=10.0)
                    except TimeoutError:
                        # Comment frame on an idle stream, well inside the 30-60s
                        # idle timeout of common proxies and HTTP clients.
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps({'tick': tick})}\n\n"
            finally:
                hub.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
