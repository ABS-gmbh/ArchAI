# Evaluation VM runbook

Runs ArchAI's offline benchmarks on a CPU-only GWDG VM: Ubuntu 24.04, 8 vCPU,
16 GB RAM, 80 GB disk. The VM gets no secrets. Nothing here calls a chat or
embedding provider, and every step that would need one is marked as blocked.

```bash
bash setup.sh ~/archai.bundle <ref>          # system packages, Python 3.11 venv, deps, checkout, offline suite
bash ~/ArchAI/scripts/eval_vm/download_models.sh ~/ArchAI
```

Both scripts are idempotent. All the commands below run from
`~/ArchAI/archai/backend` with the venv on `PATH`:

```bash
cd ~/ArchAI/archai/backend && source ~/archai-venv/bin/activate
```

## 1. Setup (`setup.sh`)

- **Repository.** It is private, and the VM holds no credentials, so it is
  copied as a git bundle. On a machine with access, run
  `git bundle create archai.bundle --all` and then `scp archai.bundle vm:~`.
  A credential-free URL works too.
- **System packages.** git, a build toolchain, libgl1, libglib2.0-0t64, the
  jpeg/png/tiff headers and libspatialindex (for rtree). Python 3.11 comes from
  the deadsnakes PPA, because Ubuntu 24.04 ships 3.12.
- **Python packages.** torch 2.9.0+cpu and torchvision 0.24.0+cpu come from
  `https://download.pytorch.org/whl/cpu`. A constraints file then keeps any later
  package from replacing them with a CUDA build. After that it installs the
  backend dependency list CI uses, plus kraken 6.0.3, ultralytics 8.4.13 and
  sentence-transformers. The versions are those of the dev environment the
  existing numbers came from.
- **Checks.** torch is the CPU build, `app.main` imports, and the offline suite
  passes (`pytest -m "not requires_models"`; about 1,220 tests in 30 s on the
  dev machine).

Disk use is roughly 6 GB (estimate) for the venv and 1.5 GB for the models below.

## 2. Models (`download_models.sh`)

| Model | Used by | Size | Source (as the repo names it) | On the VM |
|---|---|---|---|---|
| CATMuS Medieval (Kraken) | segmented route, default recogniser | 16.3 MB | DOI 10.5281/zenodo.12743230, `scripts/fetch_kraken_models.sh` | downloaded |
| McCATMuS (Kraken) | segmented route | 16.2 MB | DOI 10.5281/zenodo.13788177, same script | downloaded |
| CATMuS Print (Kraken) | not routed to | 22.9 MB | DOI 10.5281/zenodo.10592716, same script | downloaded |
| CREMMA Medieval (Kraken) | segmented route | 16.1 MB | Zenodo record 6669508, `cremmamedievalBicerin_1.1.0.mlmodel`, same script | downloaded |
| Kraken segmentation (blla) | not used by the backend | 5.0 MB | DOI 10.5281/zenodo.14602569, same script | downloaded |
| CREMMA Medieval LAT | optional | – | the script names an archive only, with no installable model | not installed |
| multilingual-e5-large-instruct | retrieval with `--embedder e5-local` | about 1.1 GB | Hugging Face `intfloat/multilingual-e5-large-instruct` (`benchmark_retrieval.py`, `E5_LOCAL_MODEL`) | downloaded |
| all-MiniLM-L6-v2 (ONNX) | ChromaDB's default embedder, the `local` space | about 170 MB in cache | fetched by chromadb itself on first use | pre-fetched |
| Layout YOLO: `best_catmus.pt`, `best_emanuskript_segmentation.pt`, `best_zone_detection.pt` | segmented route (line and zone detection) | 365 + 122 + 83 MB | **no public source in the repo** (audit HYG-13); the only copies are in the dev machine's iCloud-offloaded Desktop folder, where reading them hangs | **blocked** until that folder is set to "Keep Downloaded" |
| GLM-OCR `glm-ocr:latest` (Ollama) | `glmocr` engine | about 2.2 GB | the Ollama tag named by `settings.glmocr_ollama_model`; Ollama itself from its official install script (section 3f) | installed after 3a and 3b |

The Kraken script fills `<repo>/weights`. The backend reads `weights/...`
relative to `archai/backend`, so the download script links
`archai/backend/weights` to it, as on the dev machine. Layout weights go into
`archai/backend/models/` (`settings.model_dir`). The script reports which of the
three are there.

## 3. Measurements

The run times are **estimates for 8 vCPU, not measured**: the first run is the
measurement. For scale, the MiniLM retrieval benchmark below takes 9 s on the
dev laptop. e5-large is about 25 times larger (560M parameters against 22M).

### 3a. Does e5 close the cross-lingual retrieval gap?

```bash
# before: the local space (ChromaDB's MiniLM), the published numbers
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --questions eval/retrieval/questions.json
# after: e5, with the pipeline's query instruction
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --questions eval/retrieval/questions.json --embedder e5-local
# control: e5 without the instruction (as the pipeline embedded queries before the instruction)
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --questions eval/retrieval/questions.json --embedder e5-local --no-query-instruction
# passage benchmark (no questions), both spaces
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --embedder e5-local
# relevance floor: does e5 separate answerable from not-in-sources lookups? (MiniLM: AUC 0.44-0.51)
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --questions eval/retrieval/questions.json --embedder e5-local --separation
```

- **Output.** The MRR / Hit@1 / Hit@5 table for dense, BM25 and hybrid (RRF)
  over 270 question lookups, with 95% intervals resampled over readings, and
  the change against dense.
- **Baseline (MiniLM).** Hybrid MRR 0.475 [0.415, 0.533], Hit@5 0.770.
- **Time.** About 10 s for MiniLM. For e5, 1 to 2 min to load the model plus
  roughly 3 to 10 min of encoding, per run.
- **Needs.** Nothing the VM lacks.

### 3b. Does e5 put the answer into the chat evidence?

```bash
python scripts/evaluate_chat_answers.py                                   # before: MiniLM, top_k 8 (default): 56/62
python scripts/evaluate_chat_answers.py --embedder e5-local               # after
RAG_TOP_K=5 python scripts/evaluate_chat_answers.py                       # before at top_k 5: 53/62
RAG_TOP_K=5 python scripts/evaluate_chat_answers.py --embedder e5-local   # after at top_k 5
```

- **Output.** `evidence_holds_answer` as k/62 per page, the retrieval status
  for all 79 items, and whether the "not in the sources" instruction is
  present.
- **The question.** Do the 6 answers still missing at top_k 8 (9 at top_k 5)
  come in? All of them are English questions that share no word with the Old
  French or Latin passage.
- **Time.** About 2 to 5 min per e5 run.
- **Needs.** Nothing the VM lacks. `--embedder e5-local` is new in this
  change.
- **Blocked.** `--live`, which scores the answers themselves, needs the chat
  provider key. Run it elsewhere.

### 3c. Recogniser benchmark

```bash
python scripts/benchmark_recognizers.py --images ~/pages --engines segmented segmented:kraken_cremma_medieval segmented:kraken_mccatmus --json > recognizers.json
```

- **Output.** Per page and engine: CER (diplomatic policy), WER and search-key
  CER.
- **Time.** About 2 to 5 min per page per engine for Kraken on CPU. For 5 pages
  and 3 engines, 30 to 75 min.
- **Needs page images.** `eval/recognizers/manifest.jsonl` matches them by
  SHA-256, so file names don't matter:
  - `old-french` and `latin-abaton` are the thesis demo pages `french.JPEG`
    and `Latin.png`, which exist only on the dev machine and must be copied;
  - `latin-apocalypse` is e-codices `sbe-0027` f. 1v, `middle-french-antitus`
    is e-codices `acv-P-Antitus` f. 1r, and `old-french-verse` is e-codices
    `fmb-cb-0001` f. 1r. The README says only that they are "the full-size
    downloads (about 6130 x 8175 px)". **No URL is recorded.** Any copy whose
    SHA-256 differs from the manifest is rejected.
- **Needs the layout YOLO weights.** Without them the segmented route has no
  lines.
- **Blocked.** The images and the layout weights are on the iCloud step (Decisions 2-3). `glmocr` also needs Ollama (Decision 5).

### 3d. Quality gate on fresh readings

```bash
python scripts/calibrate_quality_gate.py      # the committed set: pins the verdicts in the README table
```

- **Time.** Seconds.
- **Needs nothing** on the committed `eval/quality_gate/readings.json`.
- **Fresh readings.** These are new recogniser output, so they need 3c's page
  images and layout weights. `benchmark_recognizers.py --save-readings
  fresh.json` writes the texts read in the `readings.json` format, and
  `calibrate_quality_gate.py --readings fresh.json` checks the gate on them.
- **Alternative.** `--db <pipeline sqlite>` runs the held-out check on any
  pipeline database. There is none on the VM, and the live
  `app/archai.sqlite` must not be copied there without the user's say.

### 3e. The OCR-quality findings: what a before/after needs

These scores are not changed on reasoning alone, only with a benchmark showing
the gain.

| Finding | Before / after measurement | On this VM |
|---|---|---|
| SVC-09: Kraken confidence is a mean pooled over all characters | Run 3c with the current pooling and with a per-line mean (or length-weighted median). Correlate each page's confidence with its CER (Spearman over pages × engines), and each line's confidence with its line CER. Report how often the gate's verdict changes. | Needs 3c's images and layout weights. With 5 pages × 3 engines the page-level correlation rests on 15 points; the line level is the useful one. |
| SVC-12: multi-view ranking always picks the binarized variant | On the SAIA VLM path: read the same crops with each variant forced first. Compare CER and how often each variant wins. | **Blocked.** It needs the SAIA provider key, which the VM won't get. The ranking itself (binarized always first) can be shown offline, but that shows nothing about OCR quality. |
| SVC-13: seam_fragment_ratio never gets the seam geometry | On the tiled SAIA path: count seam-triggered retries and check the RISKY labels against the lines that touch a seam. Compare page-wide against seam-only, with CER after the retry. | **Blocked**, for the same reason (tiled SAIA reads). |
| SVC-16: trailing_fragment_ratio uses the Latin vowel set for every script | Needs pages in a non-Latin script (Greek, Cyrillic, Hebrew, Arabic, CJK) with references, run through `calibrate_quality_gate.py` before and after. | **No data.** Every reference page is in Latin script, so the change is a no-op on the whole set by construction. |

## 4. Summary: what the VM lacks

| Measurement | Page images | Layout weights | Provider key | Ollama + GLM-OCR |
|---|---|---|---|---|
| 3a retrieval, MiniLM and e5 | – | – | – | – |
| 3b chat evidence, offline | – | – | – | – |
| 3b chat answers (`--live`) | – | – | **needed** | – |
| 3c recognisers, Kraken | **needed** | **needed** | – | – |
| 3c recognisers, GLM-OCR | **needed** | – | – | **needed** |
| 3d quality gate, committed set | – | – | – | – |
| 3d quality gate, fresh readings | **needed** | **needed** | – | – |
| SVC-09 | **needed** | **needed** | – | – |
| SVC-12, SVC-13 | **needed** | – | **needed** | – |
| SVC-16 | non-Latin pages with references (none exist) | – | – | – |

## Decisions (coordinator, 2026-11-23)

1. **Repository:** a git bundle copied by scp. No deploy key and no credentials of any kind on the VM.
2. **Layout YOLO weights:** blocked. The only copies are in the dev machine's iCloud-offloaded Desktop folder, where reading them hangs. Everything that needs them (3c, fresh 3d readings, SVC-09) stays blocked until the user sets that folder to "Keep Downloaded".
3. **Page images:** none of the five reference images is tracked in the repo or its history (checked by SHA-256 against the manifest), and none was found outside the offloaded Desktop folder. 3c and fresh 3d readings are blocked on the same iCloud step. They are not re-downloaded from e-codices: a copy at a guessed size would not match the manifest's SHA-256, and its CERs would not be comparable.
4. **Demo pages** `french.JPEG` and `Latin.png`: may be copied once readable (test with a short read first); not found outside the offloaded folder.
5. **Ollama:** after 3a and 3b, install the current stable release with Ollama's official install script, record the exact version (`ollama --version`) here and in every result, and pull `glm-ocr:latest` (the name in `settings.glmocr_ollama_model`). Its GLM numbers are a first look: no committed baseline exists for that version.
6. **Saving the text read:** `benchmark_recognizers.py --save-readings PATH` (3d).

## Order on the VM

setup.sh; download_models.sh (e5 first); 3a; 3b. Every command and its raw output goes into `~/results/` on the VM, which is copied back. Report MRR and Hit@k with their intervals and lookup counts, and answers in evidence as k/n.
