#!/usr/bin/env bash
set -euo pipefail

IMAGE_PATH="${1:-sample.png}"
CONFIG_PATH="${2:-config.example.yaml}"

PYTHONPATH="src:${PYTHONPATH:-}" python -m archai_ocr.cli --image "${IMAGE_PATH}" --config "${CONFIG_PATH}"
