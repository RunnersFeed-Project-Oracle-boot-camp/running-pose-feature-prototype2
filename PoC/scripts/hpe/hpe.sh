#!/usr/bin/env bash
set -e

echo "HPE 진입"

HPE_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$(cd "$HPE_DIR/.." && pwd)"
POC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "PoC 진행 위치 : $POC_DIR"

# `scripts.*` 패키지를 어느 위치에서 실행해도 찾을 수 있게 한다.
export PYTHONPATH="$POC_DIR"

cd "$POC_DIR"

RUN_FOLDER="$1"
RUN_DIR="$POC_DIR/run/$RUN_FOLDER"

INPUT_DIR="$RUN_DIR/inputs"
OUTPUT_DIR="$RUN_DIR/outputs"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

shift


if [[ "${1:-}" == "--extract" ]]; then
    VIDEO_PATH=$(find "$RUN_DIR" \
        -maxdepth 1 \
        -type f \
        -iname "*.mp4" \
        -print \
        -quit)

    if [[ -z "$VIDEO_PATH" ]]; then
        echo "MP4 파일을 찾지 못했습니다: $RUN_DIR"
        exit 1
    fi

    echo "영상 발견: $VIDEO_PATH"

    "$POC_DIR/.venv/bin/python" \
        "$HPE_DIR/extract_frames.py" \
        "$VIDEO_PATH" \
        "$RUN_DIR/inputs"

    shift
fi

printf '\nHPE 추론\n'
"$POC_DIR/.venv/bin/python" \
  "$HPE_DIR/hpe_model.py" \
  "$POC_DIR" \
  "$RUN_FOLDER" \
  "$@"

printf '\n렌더링\n'
"$POC_DIR/.venv/bin/python" \
  "$HPE_DIR/render.py" \
  "$RUN_DIR/inputs" \
  "$RUN_DIR/outputs"

printf '\n이미지 합성\n'
"$POC_DIR/.venv/bin/python" \
  "$HPE_DIR/compose_video.py" \
  "$RUN_DIR/outputs/details.json" \
  "$RUN_DIR/outputs/rendered" \
  "$RUN_DIR/outputs/_rendered.mp4"

ffmpeg \
  -y \
  -hide_banner \
  -loglevel error \
  -stats \
  -i "$RUN_DIR/outputs/_rendered.mp4" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$RUN_DIR/outputs/rendered.mp4"
