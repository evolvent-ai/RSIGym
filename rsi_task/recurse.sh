#!/bin/sh
# usage: sh recurse.sh <name> <rounds> <hours-per-round> <budget-usd-per-round>
# Runs joint/recursive-pi-skillsbench-qwen3-8-27b round after round: each round's agent is the
# previous round's submitted harness and checkpoint (round 1: pi 0.84.2 on Qwen/Qwen3.8-27B).
set -eu
cd "$(dirname "$0")"
name=$1 rounds=$2 hours=$3 budget=$4
. ./.env

template=$PWD/joint/recursive-pi-skillsbench-qwen3-8-27b
root=$PWD/jobs/recurse/$name
# The harbor on PATH, as for every task, runs benchmark_server's CustomAgent from source.
agent_src=$(cd .. && pwd)/benchmark_server/src
seconds=$(awk "BEGIN { print $hours * 3600 }")

fail() { echo "round $r: $*" >&2; exit 1; }

mkdir -p "$(dirname "$root")"
mkdir "$root" "$root/seed" "$root/memory"
curl -fsSL https://github.com/earendil-works/pi/releases/download/v0.84.2/pi-0.84.2-source.tar.gz \
    | tar -xz -C "$root/seed"
cp "$template/seed/install.sh" "$template/seed/run.sh" "$root/seed/pi-0.84.2/"
tar -czf "$root/seed-agent.tar.gz" -C "$root/seed/pi-0.84.2" .
agent=$root/seed-agent.tar.gz model=Qwen/Qwen3.8-27B state=
touch "$root/rounds.jsonl" "$root/memory/rounds.jsonl"

r=1
while [ "$r" -le "$rounds" ]; do
    dir=$root/round-$r
    mkdir "$dir"
    cp -R "$template" "$dir/task"
    cp "$agent" "$dir/task/environment/seed-agent.tar.gz"
    cp -R "$root/memory" "$dir/task/environment/memory"
    jq --argjson budget "$budget" --arg state "$state" \
        '.budget_usd = $budget
         | if $state != "" then .train_server.lock.init_checkpoint = $state else . end' \
        "$template/key.json" > "$dir/task/key.json"
    sed "s/^timeout_sec = 604800.0$/timeout_sec = $seconds/" "$template/task.toml" > "$dir/task/task.toml"
    grep -q "^timeout_sec = $seconds$" "$dir/task/task.toml" || fail "could not set the agent timeout"
    sed "s/{hours}/$hours/; s/{budget}/$budget/" "$template/instruction.md" > "$dir/task/instruction.md"
    ! grep -q "{hours}\|{budget}" "$dir/task/instruction.md" || fail "could not fill in the instruction"

    # One malformed reply makes the agent exit mid-round; rerun it, at most 3 times, until it
    # has written its final files. Only this round's run restarts; the official eval does not.
    PYTHONPATH=$agent_src sh run.sh "$dir/task" -a benchmark_server.agents.custom_agent:CustomAgent \
        --ak archive_path="$agent" --ak install_cmd="bash /agent/install.sh" \
        --ak 'run_cmd={ instruction=$(cat); for try in 0 1 2 3; do
            [ $try = 0 ] || echo "recurse: no final files, restart $try of 3"
            printf "%s" "$instruction" | OPENAI_API_BASE=$MODEL_SERVER_API_BASE OPENAI_API_KEY=$MODEL_SERVER_API_KEY bash /agent/run.sh
            [ -s /workspace/final-agent.tar.gz ] && [ -s /workspace/final-checkpoint.txt ] && [ -s /workspace/final-data.jsonl ] && break
            [ $try = 0 ] && instruction="$instruction

Your previous session in this round ended before writing the final files. Its log is under /logs/agent/pi/sessions and your work so far is still in /workspace: continue from there."
        done; }' \
        --ae OPENAI_MODEL="$model" --agent-setup-timeout-multiplier 5 \
        --jobs-dir "$dir" --job-name job

    set -- "$dir"/job/*/artifacts/workspace
    out=$1 trial=${1%/artifacts/workspace}
    for f in final-agent.tar.gz final-checkpoint.txt final-data.jsonl; do
        [ -s "$out/$f" ] || fail "no $f in $out"
    done
    [ -d "$out/memory" ] || fail "no memory in $out"

    # The state to continue from is the one train_server saved for the submitted
    # checkpoint's own run, never a path the agent reports.
    checkpoint=$(tr -d '[:space:]' < "$out/final-checkpoint.txt")
    run=$(curl -fsS -H "Authorization: Bearer $TRAIN_SERVER_VERIFIER_API_KEY" \
        "$TRAIN_SERVER_URL/v1/runs/${checkpoint##*/}") || fail "no train_server run for $checkpoint"
    [ "$(printf '%s' "$run" | jq -r .checkpoint)" = "$checkpoint" ] \
        || fail "$checkpoint is not the checkpoint of train_server run ${checkpoint##*/}"
    state=$(printf '%s' "$run" | jq -r '.state_checkpoint // empty')
    [ -n "$state" ] || fail "the run behind $checkpoint saved no state"

    # Audit on demand, as for every task: sh audit.sh <trial>.
    reward=$(cat "$trial/verifier/reward.txt" 2>/dev/null || echo null)
    jq -cn --argjson round "$r" --arg checkpoint "$checkpoint" --arg state "$state" \
        --arg reward "$reward" \
        '{round: $round, checkpoint: $checkpoint, state_checkpoint: $state,
          reward: ($reward | tonumber? // null)}' >> "$root/rounds.jsonl"
    rm -rf "$root/memory"
    cp -R "$out/memory" "$root/memory"
    cp "$root/rounds.jsonl" "$root/memory/rounds.jsonl"  # the record is ours, not the agent's
    agent=$out/final-agent.tar.gz model=$checkpoint
    r=$((r + 1))
done
