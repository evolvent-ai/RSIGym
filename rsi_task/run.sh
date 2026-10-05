#!/bin/sh
# usage: sh run.sh <task-dir> [harbor run args...]
set -aeu
task=$1; shift
. ./.env

response=$(curl -fsS "$AUTH_SERVER_URL/admin/keys" \
    -H "X-Admin-Key: $AUTH_SERVER_ADMIN_KEY" -H 'Content-Type: application/json' \
    -d @"$task/key.json")
key=$(printf '%s' "$response" | jq -r .key)
printf 'issued key %.10s... for %s\n' "$key" "$task"

TRAIN_SERVER_API_KEY=$key MODEL_SERVER_API_KEY=$key
BENCHMARK_SERVER_API_KEY=$key ROLLOUT_SERVER_API_KEY=$key E2B_API_KEY=$key
exec harbor run -y -p "$task" "$@"
