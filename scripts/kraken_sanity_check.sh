#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMG=""
OUTDIR=""
MODELS_CSV=""

usage() {
  cat <<'EOF'
Usage: kraken_sanity_check.sh --img <line.png> [--outdir <dir>] [--models <m1,m2,...>]

Options:
  --img      Required path to single-line image crop.
  --outdir   Output directory (default: outputs/sanity_check/<timestamp>/).
  --models   Comma-separated recognition model paths.
             Default: weights/kraken_models/*.mlmodel (if any), else weights/kraken_recognition.mlmodel.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --img)
      IMG="${2:-}"
      shift 2
      ;;
    --outdir)
      OUTDIR="${2:-}"
      shift 2
      ;;
    --models)
      MODELS_CSV="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${IMG}" ]]; then
  echo "Error: --img is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${IMG}" ]]; then
  echo "Error: image not found: ${IMG}" >&2
  exit 2
fi

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

if [[ -z "${OUTDIR}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="${REPO_ROOT}/outputs/sanity_check/${ts}"
fi
mkdir -p "${OUTDIR}"

declare -a MODELS=()
if [[ -n "${MODELS_CSV}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
else
  shopt -s nullglob
  for f in "${REPO_ROOT}"/weights/kraken_models/*.mlmodel; do
    MODELS+=("${f}")
  done
  shopt -u nullglob
  if [[ "${#MODELS[@]}" -eq 0 && -f "${REPO_ROOT}/weights/kraken_recognition.mlmodel" ]]; then
    MODELS+=("${REPO_ROOT}/weights/kraken_recognition.mlmodel")
  fi
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "Error: no models found. Pass --models or add weights to weights/kraken_models/." >&2
  exit 2
fi

fail_count=0
success_count=0

for model in "${MODELS[@]}"; do
  model="$(echo "${model}" | xargs)"
  if [[ ! -f "${model}" ]]; then
    echo "Model missing: ${model}" >&2
    fail_count=$((fail_count + 1))
    continue
  fi

  model_name="$(basename "${model}" .mlmodel)"
  out_txt="${OUTDIR}/${model_name}.txt"
  log_txt="${OUTDIR}/${model_name}.log"

  rm -f "${out_txt}" "${log_txt}"

  ok=0
  if "${KRAKEN_BIN}" -i "${IMG}" "${out_txt}" segment -bl ocr -m "${model}" >"${log_txt}" 2>&1; then
    if [[ -s "${out_txt}" ]]; then
      ok=1
    fi
  fi

  if [[ "${ok}" -eq 0 ]]; then
    if "${KRAKEN_BIN}" -i "${IMG}" "${out_txt}" ocr -m "${model}" >>"${log_txt}" 2>&1; then
      if [[ -s "${out_txt}" ]]; then
        ok=1
      fi
    fi
  fi

  sha="$(shasum -a 256 "${model}" | awk '{print $1}')"
  preview="$(tr '\n' ' ' < "${out_txt}" 2>/dev/null | cut -c1-200)"

  echo
  echo "Model: ${model}"
  echo "SHA256: ${sha}"
  echo "Preview (200 chars): ${preview}"
  echo "Output: ${out_txt}"

  if [[ "${ok}" -eq 1 ]]; then
    success_count=$((success_count + 1))
  else
    fail_count=$((fail_count + 1))
    echo "Status: FAILED (see ${log_txt})" >&2
  fi
done

echo
echo "Sanity check output dir: ${OUTDIR}"
echo "Success: ${success_count}, Failed: ${fail_count}"

if [[ "${success_count}" -eq 0 ]]; then
  exit 1
fi

exit 0
