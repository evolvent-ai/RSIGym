"""API surface tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rsiwatch.server import create_app
from tests.unit.test_scanner import ASSISTANT, make_trial

TOOL_USE = json.dumps({
    "type": "assistant", "isSidechain": False, "timestamp": "2026-08-15T08:20:11.000Z",
    "message": {"id": "m2", "model": "claude-opus-5", "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                             "input": {"command": "ls"}}],
                "usage": {"input_tokens": 10, "output_tokens": 5}},
})


def client_for(tmp_path):
    make_trial(tmp_path, "job1", "t1", session_lines=[ASSISTANT, TOOL_USE])
    # TestClient runs lifespan, which starts the poll loop; the constructor
    # already did one synchronous refresh so state is ready immediately.
    return TestClient(create_app(tmp_path, interval=0.05))


def test_overview(tmp_path):
    with client_for(tmp_path) as client:
        body = client.get("/api/overview").json()
        assert body["jobs"][0]["name"] == "job1"
        trial = body["jobs"][0]["trials"][0]
        assert trial["status"] == "running"
        assert trial["n_tool_calls"] == 1


def test_trial_detail_and_incremental_since(tmp_path):
    with client_for(tmp_path) as client:
        body = client.get("/api/jobs/job1/trials/t1").json()
        assert [e["kind"] for e in body["events"]] == ["text", "tool_use"]
        assert body["latest_seq"] == 2

        # `since` is how the browser fetches only what it has not rendered.
        tail = client.get("/api/jobs/job1/trials/t1?since=1").json()
        assert [e["seq"] for e in tail["events"]] == [2]

        assert client.get("/api/jobs/job1/trials/t1?since=2").json()["events"] == []


def test_limit_keeps_newest_window(tmp_path):
    with client_for(tmp_path) as client:
        body = client.get("/api/jobs/job1/trials/t1?limit=1").json()
        assert body["truncated"] is True
        assert [e["seq"] for e in body["events"]] == [2]


def test_until_pages_backwards(tmp_path):
    with client_for(tmp_path) as client:
        body = client.get("/api/jobs/job1/trials/t1?until=1").json()
        assert [e["seq"] for e in body["events"]] == [1]


def test_spine_covers_all_events_regardless_of_paging(tmp_path):
    with client_for(tmp_path) as client:
        body = client.get("/api/jobs/job1/trials/t1?limit=1").json()
        assert body["spine"] == "xu"


def test_unknown_trial_404s(tmp_path):
    with client_for(tmp_path) as client:
        assert client.get("/api/jobs/job1/trials/nope").status_code == 404
        assert client.get("/api/jobs/nope/trials/t1").status_code == 404


def test_index_served(tmp_path):
    with client_for(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "rsiwatch" in response.text
