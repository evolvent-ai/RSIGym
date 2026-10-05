#!/bin/sh
# usage: sh audit.sh <trial-dir>
set -eu
trial=$1

task=$(jq -r .task.path "$trial/config.json")
[ -d "$task/audit" ] || { echo "task $task defines no audit" >&2; exit 2; }
exec uv run "$task/audit/run.py" "$trial"
