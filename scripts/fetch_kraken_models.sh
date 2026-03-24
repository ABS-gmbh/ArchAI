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
URL_CREMMA_MEDIEVAL="https://zenodo.org/records/6669508/files/cremmamedievalBicerin_1.1.0.mlmodel?download=1"
URL_CREMMA_LAT_ARCHIVE="https://zenodo.org/records/7014157/files/HTR-United/CREMMA-Medieval-LAT-0.0.1a.zip?download=1"

usage() {
  cat <<'EOF'
Usage: fetch_kraken_models.sh [--dry-run]

Downloads multiple Kraken recognition models and prints stable wiring commands.
CREMMA Medieval is installed directly as a published Kraken model file.
CREMMA-Medieval-LAT is exposed publicly as an archive, not a standalone .mlmodel,
so this script reports its source but does not install it automatically.

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

download_direct() {
  local url="$1"
  local target="$2"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "<download skipped: ${url}>"
    return 0
  fi
  curl -L "${url}" -o "${target}" >/dev/null
  printf '%s' "${target}"
}

echo "Planned recognizers:"
echo "  mccatmus        ${DOI_MCCATMUS}"
echo "  catmus_medieval ${DOI_CATMUS_MEDIEVAL}"
echo "  catmus_print    ${DOI_CATMUS_PRINT}"
echo "  cremma_medieval ${URL_CREMMA_MEDIEVAL}"
echo "Reference only:"
echo "  cremma_medieval_lat archive ${URL_CREMMA_LAT_ARCHIVE}"
echo "Optional segmentation:"
echo "  kraken_segmentation ${DOI_SEGMENTATION}"
echo

MCCATMUS_PATH="$(download_one "${DOI_MCCATMUS}" "mccatmus")"
MEDIEVAL_PATH="$(download_one "${DOI_CATMUS_MEDIEVAL}" "catmus_medieval")"
PRINT_PATH="$(download_one "${DOI_CATMUS_PRINT}" "catmus_print")"
CREMMA_MEDIEVAL_PATH="$(download_direct "${URL_CREMMA_MEDIEVAL}" "${TARGET_DIR}/cremma_medieval.mlmodel")"
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
  ln -sfn "${CREMMA_MEDIEVAL_PATH}" "${TARGET_DIR}/cremma_medieval.mlmodel"
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

Manual LAT model wiring:
  ARCHAI_KRAKEN_CREMMA_LAT_MODEL_PATH=weights/kraken_models/cremma_medieval_lat.mlmodel
  Source archive: ${URL_CREMMA_LAT_ARCHIVE}
EOF
