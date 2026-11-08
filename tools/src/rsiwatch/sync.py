"""Copy running mini-swe Docker logs to Harbor's host trial directory for RSIWatch.

Read-only toward containers; snapshots are atomically replaced on the host.
Run alongside RSIWatch when Harbor uses containers without a log bind mount.
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

def sync(jobs_dir: Path, docker_context: str | None = None):
    docker = ['docker'] + (['--context', docker_context] if docker_context else [])
    containers = subprocess.run(docker + ['ps', '--format', '{{.Names}}'],
                                capture_output=True, text=True, timeout=15, check=True)
    running = set(containers.stdout.splitlines())
    for config in jobs_dir.glob('*/*/config.json'):
        trial = config.parent
        if (trial/'result.json').exists():
            continue
        data = json.loads(config.read_text())
        if (data.get('agent') or {}).get('name') != 'mini-swe-agent':
            continue
        name = trial.name.lower()+'__env-main-1'
        if name not in running:
            continue
        agent = trial/'agent'
        agent.mkdir(exist_ok=True)
        for filename in ('mini-swe-agent.trajectory.json', 'mini-swe-agent.txt'):
            with tempfile.TemporaryDirectory(dir=agent) as temp:
                target = Path(temp)/filename
                result = subprocess.run(docker + ['cp',
                                         f'{name}:/logs/agent/{filename}',str(target)],
                                        capture_output=True,timeout=20)
                if result.returncode == 0:
                    if filename.endswith('.json'):
                        try:
                            json.loads(target.read_text())
                        except ValueError:
                            continue
                    destination = agent/filename
                    if not destination.exists() or target.read_bytes() != destination.read_bytes():
                        target.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs_dir', type=Path)
    parser.add_argument('--docker-context', help='Docker context; otherwise use Docker defaults')
    parser.add_argument('--interval', type=float, default=3.)
    parser.add_argument('--once', action='store_true', help='Copy one snapshot and exit')
    args = parser.parse_args()
    if not args.jobs_dir.is_dir():
        parser.error('jobs_dir must be an existing Harbor jobs directory')
    if args.interval <= 0:
        parser.error('--interval must be positive')
    while True:
        try:
            sync(args.jobs_dir, args.docker_context)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            if args.once:
                raise SystemExit(str(error)) from error
            print(error, flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
