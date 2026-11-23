#!/usr/bin/env bash
set -euo pipefail

# Resolved from this script, like the sibling scripts, so the demo runs from any
# directory: src/ and config.example.yaml used to be found only from the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="${1:-sample.png}"
CONFIG_PATH="${2:-${REPO_ROOT}/config.example.yaml}"

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python -m archai_ocr.cli --image "${IMAGE_PATH}" --config "${CONFIG_PATH}"
