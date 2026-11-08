"""Terminus rewrites ATIF during execution rather than appending JSONL."""
import json

from rsiwatch.scanner import Trial
from rsiwatch.trajectory import AtifReader


def test_live_snapshots(tmp_path):
    path = tmp_path / "trajectory.json"
    step = {"step_id": 1, "source": "agent", "model_name": "flint",
            "message": "Inspect files", "reasoning_content": "Need source",
            "tool_calls": [{"tool_call_id": "c1", "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls\n"}}],
            "observation": {"results": [{"source_call_id": "c1", "content": "solve.py"}]},
            "metrics": {"prompt_tokens": 20, "completion_tokens": 5}}
    data = {"schema_version": "1.0", "session_id": "test", "steps": [step]}
    path.write_text(json.dumps(data))
    reader = AtifReader(path)
    assert reader.poll()
    assert [e.kind for e in reader.events] == ["thinking", "text", "tool_use", "tool_result"]
    assert reader.usage.total == 25
    assert reader.summary()["usage"]["cost_usd"] == 0.0
    assert not reader.poll()
    path.write_text('{"schema_version":')
    assert not reader.poll()
    assert reader.usage.total == 25
    step["observation"]["results"][0]["content"] = "updated output"
    path.write_text(json.dumps(data))
    assert reader.poll()
    assert reader.events[-1].body == "updated output"
    assert reader.usage.total == 25
    assert reader.tool_counts == {"bash_command": 1}
    data["steps"] = []
    path.write_text(json.dumps(data))
    assert reader.poll()
    assert not reader.events


def test_terminus_atif_does_not_signal_verification(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "trajectory.json").write_text(json.dumps(
        {"schema_version": "1.0", "steps": []}))
    (tmp_path / "config.json").write_text(json.dumps(
        {"agent": {"name": "terminus-2", "model_name": "openai/flint"}}))
    trial = Trial(name="trial", job_name="job", path=tmp_path)
    trial.refresh()
    assert trial.status() == "running"
    assert isinstance(trial.reader, AtifReader)
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/test-stdout.txt").touch()
    assert trial.status() == "verifying"
