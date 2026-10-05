#!/bin/sh
set -eu

instruction=$(cat)

exec /agent/.venv/bin/python /agent/main.py --instruction "$instruction"
