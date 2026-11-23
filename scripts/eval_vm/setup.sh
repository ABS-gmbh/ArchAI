#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 VM (CPU only) to run ArchAI's benchmarks.
#
# Usage:
#   bash setup.sh <repo-source> <ref> [checkout-dir]
#
#   <repo-source>  a git bundle file copied to the VM (recommended: the repo is
#                  private and the VM holds no credentials), e.g.
#                    on a machine with access:  git bundle create archai.bundle --all
#                    then:                      scp archai.bundle vm:~
#                  or any git URL the VM can read without credentials.
#   <ref>          commit, tag or branch to check out (detached).
#   [checkout-dir] default: ~/ArchAI
#
# Idempotent: re-running reuses the venv and the checkout (fetching <ref> again).
# Installs nothing that needs a secret, and writes none.
set -euo pipefail

REPO_SOURCE="${1:?usage: setup.sh <repo-source> <ref> [checkout-dir]}"
REF="${2:?usage: setup.sh <repo-source> <ref> [checkout-dir]}"
CHECKOUT="${3:-$HOME/ArchAI}"
VENV="${ARCHAI_VENV:-$HOME/archai-venv}"
PYTHON_VERSION="3.11"

# Versions of the environment the existing benchmarks were measured in.
TORCH_VERSION="2.9.0"
TORCHVISION_VERSION="0.24.0"
KRAKEN_VERSION="6.0.3"
ULTRALYTICS_VERSION="8.4.13"
NUMPY_VERSION="2.0.2"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

log() { printf '\n== %s\n' "$*"; }

log "System packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git curl ca-certificates build-essential pkg-config \
  libgl1 libglib2.0-0t64 libjpeg-dev libpng-dev libtiff-dev \
  libspatialindex-dev software-properties-common

if ! command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
  log "Python ${PYTHON_VERSION} (Ubuntu 24.04 ships 3.12; kraken ${KRAKEN_VERSION} is used with 3.11 here)"
  # deadsnakes is the usual source of other CPython versions for Ubuntu.
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
  sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
fi

log "Checkout ${REF} into ${CHECKOUT}"
if [[ ! -d "${CHECKOUT}/.git" ]]; then
  git clone --no-checkout "${REPO_SOURCE}" "${CHECKOUT}"
fi
git -C "${CHECKOUT}" fetch --tags "${REPO_SOURCE}" "${REF}" 2>/dev/null || git -C "${CHECKOUT}" fetch --all --tags
git -C "${CHECKOUT}" checkout --detach "${REF}" 2>/dev/null || git -C "${CHECKOUT}" checkout --detach FETCH_HEAD
git -C "${CHECKOUT}" log --oneline -1

log "Virtual environment ${VENV}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  "python${PYTHON_VERSION}" -m venv "${VENV}"
fi
PIP="${VENV}/bin/python -m pip"
${PIP} install --upgrade pip wheel setuptools

# torch from the CPU wheel index first, then a constraints file so that no later
# package (kraken, ultralytics, sentence-transformers) can replace it with a CUDA
# build from PyPI.
CONSTRAINTS="${VENV}/archai-constraints.txt"
cat > "${CONSTRAINTS}" <<CONS
torch==${TORCH_VERSION}+cpu
torchvision==${TORCHVISION_VERSION}+cpu
numpy==${NUMPY_VERSION}
CONS

log "torch ${TORCH_VERSION} (CPU)"
${PIP} install --index-url "${TORCH_CPU_INDEX}" "torch==${TORCH_VERSION}+cpu" "torchvision==${TORCHVISION_VERSION}+cpu"

log "Backend dependencies (the CI list), the inference stack and the test runner"
${PIP} install -c "${CONSTRAINTS}" --extra-index-url "${TORCH_CPU_INDEX}" \
  "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "python-multipart>=0.0.18" \
  "python-jose[cryptography]>=3.3.0" "pydantic-settings>=2.0.0" "numpy==${NUMPY_VERSION}" \
  "pillow>=10.0.0" "matplotlib>=3.6.0" "shapely>=2.0.0" "rtree>=1.0.0" \
  "langdetect>=1.0.9" "openai>=1.51.0" "chromadb>=0.5.0" "requests>=2.31.0" \
  "opencv-python-headless>=4.8,<5" "pytest>=7.0.0" "httpx>=0.24.0" \
  "kraken==${KRAKEN_VERSION}" "ultralytics==${ULTRALYTICS_VERSION}" "sentence-transformers"

log "Check: torch is the CPU build, and the backend imports"
"${VENV}/bin/python" - <<'PY'
import torch, kraken, ultralytics, sentence_transformers  # noqa: F401
assert "+cpu" in torch.__version__ or not torch.cuda.is_available(), torch.__version__
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
PY
(cd "${CHECKOUT}/archai/backend" && "${VENV}/bin/python" -c "import app.main" && echo "backend imports")

log "Backend test suite (offline, as CI runs it)"
(cd "${CHECKOUT}/archai/backend" && "${VENV}/bin/python" -m pytest -q -p no:cacheprovider -m "not requires_models" | tail -1)

cat <<DONE

Setup complete.
  checkout: ${CHECKOUT} @ $(git -C "${CHECKOUT}" rev-parse --short HEAD)
  venv:     ${VENV}
Next: bash ${CHECKOUT}/scripts/eval_vm/download_models.sh ${CHECKOUT}
DONE
