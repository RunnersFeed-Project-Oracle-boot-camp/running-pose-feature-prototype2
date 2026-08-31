#!/usr/bin/env bash
set -e

echo "Agent 진입"

FEATURES_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$(cd "$FEATURES_DIR/.." && pwd)"
POC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$POC_DIR/.." && pwd)"

# `scripts.*` 패키지를 프로젝트 기준으로 불러온다.
export PYTHONPATH="$POC_DIR"

cd "$POC_DIR"

RUN_FOLDER="$1"
RUN_DIR="$POC_DIR/run/$RUN_FOLDER" 

"$POC_DIR/.venv/bin/python" \
  "$FEATURES_DIR/Running_coach.py" \
  "$PROJECT_DIR" \
  "$RUN_FOLDER"
