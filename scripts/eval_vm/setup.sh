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
# The inference stack's versions (torch, torchvision, kraken, ultralytics, NumPy)
# are pinned once, in archai/backend/pyproject.toml's `models` extra: the set the
# benchmarks were measured with. PyTorch's CPU index supplies the CPU build of
# torch, which an extra cannot name.
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

log() { printf '\n== %s\n' "$*"; }
# Run a network step up to 3 times: the deadsnakes PPA answered 503 on the first VM run.
retry() { local n; for n in 1 2 3; do "$@" && return 0; [[ $n -lt 3 ]] && { echo "failed (attempt $n): $*; retrying in 30 s"; sleep 30; }; done; return 1; }

log "System packages"
retry sudo apt-get update -y
retry sudo apt-get install -y --no-install-recommends \
  git curl ca-certificates build-essential pkg-config \
  libgl1 libglib2.0-0t64 libjpeg-dev libpng-dev libtiff-dev \
  libspatialindex-dev software-properties-common

if ! command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
  log "Python ${PYTHON_VERSION} (Ubuntu 24.04 ships 3.12; the pinned kraken is used with 3.11 here)"
  # deadsnakes is the usual source of other CPython versions for Ubuntu.
  retry sudo add-apt-repository -y ppa:deadsnakes/ppa
  retry sudo apt-get update -y
  retry sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
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

# One install, so the resolver sees every pin at once. With the CPU index as an
# extra index, torch==2.9.0 resolves to 2.9.0+cpu (the local version sorts
# above PyPI's CUDA build), and nothing installed later can replace it.
log "Backend with its models, dev and headless extras (archai/backend/pyproject.toml), and sentence-transformers"
${PIP} install --extra-index-url "${TORCH_CPU_INDEX}" \
  -e "${CHECKOUT}/archai/backend[models,dev,headless]" "sentence-transformers"

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
