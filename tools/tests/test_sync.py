import json
import subprocess
from pathlib import Path

from rsiwatch.sync import sync


def test_bridge_copies_complete_snapshots_and_preserves_last_good(tmp_path, monkeypatch):
    trial = tmp_path/'job'/'task__ABC'
    trial.mkdir(parents=True)
    (trial/'config.json').write_text(json.dumps({'agent': {'name': 'mini-swe-agent'}}))
    payload = '{"messages": []}'
    calls = []

    def docker(command, **kwargs):
        calls.append(command)
        if 'ps' in command:
            return subprocess.CompletedProcess(command, 0, 'task__abc__env-main-1\n', '')
        Path(command[-1]).write_text(payload if command[-1].endswith('.json') else 'output')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', docker)
    sync(tmp_path, 'colima')
    destination = trial/'agent/mini-swe-agent.trajectory.json'
    assert destination.read_text() == payload
    stamp = destination.stat().st_mtime_ns
    sync(tmp_path, 'colima')
    assert destination.stat().st_mtime_ns == stamp
    payload = '{'
    sync(tmp_path, 'colima')
    assert destination.read_text() == '{"messages": []}'
    assert all(call[:3] == ['docker', '--context', 'colima'] for call in calls)
    (trial/'result.json').write_text('{}')
    calls.clear()
    sync(tmp_path)
    assert len(calls) == 1  # Finished trials are not copied.
