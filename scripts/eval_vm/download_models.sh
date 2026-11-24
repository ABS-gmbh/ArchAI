#!/usr/bin/env bash
# Download the models ArchAI's benchmarks use, from the sources the repo names.
#
# Usage: bash download_models.sh [checkout-dir]     (default ~/ArchAI; run setup.sh first)
#
# Sources (all public, no credentials):
#   Kraken recognisers and segmentation: scripts/fetch_kraken_models.sh (Zenodo DOIs
#     via `kraken get`, and the CREMMA Medieval Zenodo file), as weights/README.md
#     and docs/model_setup.md document.
#   multilingual-e5-large-instruct: Hugging Face, the id scripts/benchmark_retrieval.py
#     uses (E5_LOCAL_MODEL = intfloat/multilingual-e5-large-instruct).
#   all-MiniLM-L6-v2 (ONNX): ChromaDB's own default embedding function, fetched on
#     first use; pre-fetched here so benchmarks do not download mid-run.
# Not downloaded (no public source in the repo): the layout YOLO weights
# (best_catmus.pt, best_emanuskript_segmentation.pt, best_zone_detection.pt),
# and Ollama / glm-ocr:latest. See scripts/eval_vm/README.md.
set -euo pipefail

CHECKOUT="${1:-$HOME/ArchAI}"
VENV="${ARCHAI_VENV:-$HOME/archai-venv}"
PY="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"

echo "== Kraken models (scripts/fetch_kraken_models.sh)"
# The script downloads, then prints the ln -sfn commands that give each model the
# file name the backend's settings expect; run exactly those.
# Zenodo answers 500 now and then (seen on the VM); a second try usually works.
for attempt in 1 2 3; do
  if KRAKEN_BIN="${VENV}/bin/kraken" bash "${CHECKOUT}/scripts/fetch_kraken_models.sh" > /tmp/archai_fetch_kraken.log 2>&1; then
    cat /tmp/archai_fetch_kraken.log; break
  fi
  tail -3 /tmp/archai_fetch_kraken.log
  [[ ${attempt} -eq 3 ]] && { echo "Kraken download failed 3 times" >&2; exit 1; }
  echo "Kraken download failed (attempt ${attempt}); retrying in 30 s"; sleep 30
done
# Skip a line that links a file to itself (CREMMA Medieval is downloaded in place).
grep -E '^  ln -sfn ' /tmp/archai_fetch_kraken.log | sed 's/^  //' | awk -F'"' '$2 != $4' | bash
# The script fills <repo>/weights; the backend's settings read weights/... relative
# to archai/backend, where that directory is gitignored. Link it, as on the dev machine.
if [[ ! -e "${CHECKOUT}/archai/backend/weights" ]]; then
  ln -s "${CHECKOUT}/weights" "${CHECKOUT}/archai/backend/weights"
fi
ls -l "${CHECKOUT}/archai/backend/weights/kraken_models/"
# The layout YOLO weights go in archai/backend/models (settings.model_dir); they are
# copied there by hand (README.md, "Layout weights"). Report whether they are present.
mkdir -p "${CHECKOUT}/archai/backend/models"
for f in best_catmus.pt best_emanuskript_segmentation.pt best_zone_detection.pt; do
  if [[ -s "${CHECKOUT}/archai/backend/models/${f}" ]]; then echo "layout: ${f} present"; else echo "layout: ${f} MISSING"; fi
done

echo "== multilingual-e5-large-instruct (Hugging Face)"
"${PY}" - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
print("e5 ready, dim", model.get_sentence_embedding_dimension())
PY

echo "== ChromaDB default embedder (all-MiniLM-L6-v2, ONNX)"
"${PY}" - <<'PY'
from chromadb.utils import embedding_functions
print("MiniLM ready, dim", len(embedding_functions.DefaultEmbeddingFunction()(["probe"])[0]))
PY

echo
echo "Done. Disk use:"
du -sh "${CHECKOUT}/weights" ~/.cache/huggingface ~/.cache/chroma 2>/dev/null || true
