#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_FOLDER="${1:?사용법: ./scripts/main.sh RUN_NAME [--agent true|false] [HPE 옵션]}"
shift

RUN_AGENT=true

if [[ "${1:-}" == "--agent" ]]; then
    RUN_AGENT="${2:-true}"
    shift 2
fi

"$SCRIPT_DIR/hpe/hpe.sh" "$RUN_FOLDER" "$@"
"$SCRIPT_DIR/features/features.sh" "$RUN_FOLDER"

if [[ "$RUN_AGENT" == "true" ]]; then
    "$SCRIPT_DIR/Agent/agent.sh" "$RUN_FOLDER"
else
    echo "Agent 실행 생략"
fi
