#!/bin/sh
# Copy the skills a task's family needs into its environment/skills/ so they land in
# the Docker build context. Run before building a task.
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)

family_skills() {
    case "$1" in
        data | algo | joint) echo "tinker-training tinker-inference benchmark-server budget-management e2b-sandbox" ;;
        harness) echo "benchmark-server budget-management e2b-sandbox" ;;
        *) echo "unknown task family: $1" >&2; exit 1 ;;
    esac
}

find "$repo/rsi_task" -name task.toml | while read -r toml; do
    rel=${toml#"$repo/rsi_task/"}
    dest=$(dirname "$toml")/environment/skills
    rm -rf "$dest"
    mkdir -p "$dest"
    for skill in $(family_skills "${rel%%/*}"); do
        cp -R "$repo/skills/$skill" "$dest/$skill"
    done
done
