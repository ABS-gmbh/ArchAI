#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/weights/kraken_models"
DRY_RUN=0

DOI_MCCATMUS="10.5281/zenodo.13788177"
DOI_CATMUS_MEDIEVAL="10.5281/zenodo.12743230"
DOI_CATMUS_PRINT="10.5281/zenodo.10592716"
DOI_SEGMENTATION="10.5281/zenodo.14602569"

usage() {
  cat <<'EOF'
Usage: fetch_kraken_models.sh [--dry-run]

Downloads multiple Kraken recognition models and prints stable wiring commands.

Options:
  --dry-run   Print planned actions but do not download.
  --help      Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

KRAKEN_BIN="${KRAKEN_BIN:-}"
if [[ -z "${KRAKEN_BIN}" ]]; then
  if command -v kraken >/dev/null 2>&1; then
    KRAKEN_BIN="$(command -v kraken)"
  elif [[ -x "${REPO_ROOT}/archai/vendor/layout/.venv/bin/kraken" ]]; then
    KRAKEN_BIN="${REPO_ROOT}/archai/vendor/layout/.venv/bin/kraken"
  else
    echo "Error: kraken CLI not found. Install kraken or set KRAKEN_BIN." >&2
    exit 1
  fi
fi

mkdir -p "${TARGET_DIR}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

resolve_downloaded_model_path() {
  local log_file="$1"
  local model_dir model_files first_model
  model_dir="$(sed -n 's/^Model dir: \(.*\) (model files: .*$/\1/p' "${log_file}" | tail -n1)"
  model_files="$(sed -n 's/^Model dir: .* (model files: \(.*\))$/\1/p' "${log_file}" | tail -n1)"
  first_model="$(printf '%s' "${model_files}" | awk -F',' '{gsub(/^ +| +$/, "", $1); print $1}')"
  if [[ -z "${model_dir}" || -z "${first_model}" ]]; then
    return 1
  fi
  printf '%s/%s' "${model_dir}" "${first_model}"
}

download_one() {
  local doi="$1"
  local key="$2"
  local log_file="${WORK_DIR}/${key}.log"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "<download skipped: ${doi}>"
    return 0
  fi
  "${KRAKEN_BIN}" get "${doi}" | tee "${log_file}" >/dev/null
  resolve_downloaded_model_path "${log_file}"
}

echo "Planned recognizers:"
echo "  mccatmus        ${DOI_MCCATMUS}"
echo "  catmus_medieval ${DOI_CATMUS_MEDIEVAL}"
echo "  catmus_print    ${DOI_CATMUS_PRINT}"
echo "Optional segmentation:"
echo "  kraken_segmentation ${DOI_SEGMENTATION}"
echo

MCCATMUS_PATH="$(download_one "${DOI_MCCATMUS}" "mccatmus")"
MEDIEVAL_PATH="$(download_one "${DOI_CATMUS_MEDIEVAL}" "catmus_medieval")"
PRINT_PATH="$(download_one "${DOI_CATMUS_PRINT}" "catmus_print")"
SEG_PATH="$(download_one "${DOI_SEGMENTATION}" "segmentation" || true)"

cat <<EOF

Download complete.

Model discovery tips:
  ${KRAKEN_BIN} show --recognition
  ${KRAKEN_BIN} show --segmentation
  ${KRAKEN_BIN} show --keyword "medieval"
  ${KRAKEN_BIN} show --language lat

Place/symlink models into:
  ${TARGET_DIR}

Stable filenames:
  ln -sfn "${MCCATMUS_PATH}" "${TARGET_DIR}/mccatmus.mlmodel"
  ln -sfn "${MEDIEVAL_PATH}" "${TARGET_DIR}/catmus_medieval.mlmodel"
  ln -sfn "${PRINT_PATH}" "${TARGET_DIR}/catmus_print.mlmodel"
EOF

if [[ -n "${SEG_PATH}" ]]; then
  cat <<EOF
  ln -sfn "${SEG_PATH}" "${REPO_ROOT}/weights/kraken_segmentation.mlmodel"
EOF
fi

cat <<EOF

Backward-compatible defaults:
  ARCHAI_KRAKEN_REC_WEIGHTS=weights/kraken_recognition.mlmodel
  ARCHAI_KRAKEN_MODELS_DIR=weights/kraken_models
EOF
