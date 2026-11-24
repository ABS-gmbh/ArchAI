# Evaluation VM runbook

Runs ArchAI's offline benchmarks on a CPU-only GWDG VM: Ubuntu 24.04, 8 vCPU,
15 GB RAM, 80 GB disk. The VM gets no secrets. Nothing here calls a chat or
embedding provider, and every step that would need one is marked as blocked.
With sentence-transformers 6, e5 is loaded as float32. Its float16 checkpoint
encodes about 5 times slower on a CPU.

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
- **Python packages.** One `pip install` of the backend with its `models`,
  `dev` and `headless` extras, plus sentence-transformers.
  `https://download.pytorch.org/whl/cpu` is passed as an extra index, so torch
  2.9.0 resolves to the CPU build. The versions (torch 2.9.0, torchvision
  0.24.0, kraken 6.0.3, ultralytics 8.4.13, NumPy 2.0.2) are pinned once, in
  `archai/backend/pyproject.toml`'s `models` extra. They are those of the
  environment the existing numbers came from.
- **Checks.** torch is the CPU build, `app.main` imports, and the offline suite
  passes (`pytest -m "not requires_models"`; 1,545 tests in 180 s on the VM).

Disk use is roughly 6 GB (estimate) for the venv; the models below take
about 1.35 GB, 1.9 GB with the layout weights.

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
| Layout YOLO: `best_catmus.pt`, `best_emanuskript_segmentation.pt`, `best_zone_detection.pt` | segmented route (line and zone detection) | 365 + 122 + 83 MB | **no public source in the repo** (audit HYG-13); the only copies are in the dev machine | copied by hand to `~/layout_weights` and linked into `archai/backend/models` (2026-11-24) |
| GLM-OCR `glm-ocr:latest` (Ollama) | `glmocr` engine | about 2.2 GB | the Ollama tag named by `settings.glmocr_ollama_model`; Ollama itself from its official install script | **not installed on the VM**: the install (a remote script run with sudo) needs the user's approval. Measured on the dev machine instead (Ollama 0.40.6, `eval/recognizers/README.md`) |

The Kraken script fills `<repo>/weights`. The backend reads `weights/...`
relative to `archai/backend`, so the download script links
`archai/backend/weights` to it, as on the dev machine. Layout weights go into
`archai/backend/models/` (`settings.model_dir`). The script reports which of the
three are there.

## 3. Measurements

The run times below were measured on the VM on 2026-11-24 (`run_3ab.log`,
3c's command log), sharing the CPU with a live backend. For scale, the MiniLM
retrieval benchmark takes 9 s on the dev laptop; e5-large is about 25 times
larger (560M parameters against 22M).

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
- **Time.** 15 to 27 s per MiniLM run, 42 to 78 s per e5 run, loading
  included.
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
- **Time.** 39 to 48 s per run, MiniLM or e5.
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
- **Time.** 595 s for 5 pages and 3 engines (about 40 s an engine-page);
  the later runs without and with the initial join took 324 s and 268 s.
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
- **Run on 2026-11-24** with `--save-readings` (595 s for 15 engine-pages,
  sharing the CPU with a live backend). Micro CER over the five pages:
  CATMuS Medieval (the `segmented` default) 7.6%, CREMMA Medieval 15.1%,
  McCATMuS 36.5%, with versal initials kept apart (see the next item; with
  the join, now the default, 6.7%, 14.1% and 36.2%). CATMuS is best on every page. 13 of the 15 texts are
  byte-identical to the committed `eval/quality_gate/readings.json`, which was
  read on the dev machine. The other two are McCATMuS, at CER 34.0% against
  34.3% and an identical CER.
- **Versal initials** (2026-11-24): joined by default since 75181719. Micro CER
  for CATMuS goes from 7.6% to 6.7% (latin-abaton 6.6 -> 5.5,
  old-french-verse 7.2 -> 5.4; see `eval/recognizers/README.md`).
- **Memory.** By default each run starts the layout process pool (three
  workers). Next to a running backend with its own pool, the VM (15 GB, no
  swap) ran out of memory. Run benchmarks one at a time with
  `ARCHAI_DISABLE_PROCESS_POOL=1`, which keeps the layout models in process.
- **GLM-OCR.** Measured on the dev machine, not the VM (Ollama 0.40.6,
  `glm-ocr:latest` 6effedd0dc8a; `eval/recognizers/README.md`, "GLM-OCR").
  The VM install is still pending (Decision 5).
- **Chat key.** Before 7bd8c99, the OCR agent needed a chat key even for local
  OCR. The run used `CHAT_AI_API_KEY=offline-placeholder-not-a-key` in the
  command's environment, never in a file.

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
- **Run on 2026-11-24** on 3c's fresh readings. Every verdict matches the
  committed set: CATMuS 5/5 allowed, CREMMA 4/5 (latin-apocalypse at 46.9% is
  blocked), McCATMuS 3/5 (the two Latin pages are blocked and detected as
  French), synthetic samples 13/35 allowed.
- **Alternative.** `--db <pipeline sqlite>` runs the held-out check on any
  pipeline database. There is none on the VM, and the live
  `app/archai.sqlite` must not be copied there without the user's say.

### 3e. The OCR-quality findings: what a before/after needs

These scores are not changed on reasoning alone, only with a benchmark showing
the gain.

| Finding | Before / after measurement | On this VM |
|---|---|---|
| SVC-09: Kraken confidence is a mean pooled over all characters | Run 3c with every Kraken record's per-character confidences kept (`scripts/eval_vm/svc09_measure.py`). Compare the current pooling, a per-line mean and an all-characters mean against CER at page level (Spearman over engines × pages) and line level. | **Measured 2026-11-24.** The segmented route reads one crop per line, and none of the 549 crops gave more than one Kraken record, so the current pooling and a per-line mean are identical (largest difference 0.00000). The change would do nothing on this route. Confidence against page CER: rho -0.70 (current and per-line) and -0.78 (all characters), over 15 engine-pages. Line level: rho -0.23 over 549 lines (CATMuS -0.16, CREMMA -0.35, McCATMuS -0.08). Confidence stays between 0.959 and 0.981 for CERs from 4% to 47%: McCATMuS at 41% CER scores 0.976 on a page where CATMuS at 18% scores 0.970. It cannot rank models or gate a reading; the lexical gate does that. No change. |
| SVC-12: multi-view ranking always picks the binarized variant | On the SAIA VLM path: read the same crops with each variant forced first. Compare CER and how often each variant wins. | **Blocked.** It needs the SAIA provider key, which the VM won't get. The ranking itself (binarized always first) can be shown offline, but that shows nothing about OCR quality. |
| SVC-13: seam_fragment_ratio never gets the seam geometry | On the tiled SAIA path: count seam-triggered retries and check the RISKY labels against the lines that touch a seam. Compare page-wide against seam-only, with CER after the retry. | **Blocked**, for the same reason (tiled SAIA reads). |
| SVC-16: trailing_fragment_ratio uses the Latin vowel set for every script | Needs pages in a non-Latin script (Greek, Cyrillic, Hebrew, Arabic, CJK) with references, run through `calibrate_quality_gate.py` before and after. | **No data.** Every reference page is in Latin script, so the change is a no-op on the whole set by construction. |

## 4. Summary: what the VM lacks (2026-11-24)

| Measurement | Status |
|---|---|
| 3a retrieval, MiniLM and e5 | run (`eval/retrieval/README.md`) |
| 3b chat evidence, offline | run (`eval/chat/README.md`) |
| 3b chat answers (`--live`) | blocked: needs the chat provider key |
| 3c recognisers, Kraken | run |
| 3c recognisers, GLM-OCR | measured on the dev machine, not the VM (Ollama 0.40.6); VM install pending (Decision 5) |
| 3d quality gate, committed set and fresh readings | run |
| SVC-09 | run: no effect on the segmented route |
| SVC-12, SVC-13 | blocked: need the SAIA provider key |
| SVC-16 | no data: no non-Latin pages with references |

## Decisions (coordinator, 2026-11-23, updated 2026-11-24)

1. **Repository:** a git bundle copied by scp. No deploy key and no credentials of any kind on the VM.
2. **Layout YOLO weights:** copied from the dev machine on 2026-11-24, once iCloud had them local again, to `~/layout_weights`, and linked into `archai/backend/models`.
3. **Page images:** none is tracked in the repo. The five were copied from the dev machine to `~/pages` on 2026-11-24, and all five SHA-256 values match `eval/recognizers/manifest.jsonl`. They are not re-downloaded from e-codices: a copy at another size would not match, and its CERs would not be comparable.
4. **Demo pages** `french.JPEG` and `Latin.png`: copied with the others (3).
5. **Ollama:** install the current stable release with Ollama's official install script, record the exact version (`ollama --version`) here and in every result, and pull `glm-ocr:latest` (the name in `settings.glmocr_ollama_model`). Its GLM numbers are a first look: no committed baseline exists for that version. **Not done yet on the VM:** the install runs a remote script with sudo and waits for the user's approval. GLM-OCR was measured on the dev machine instead (Ollama 0.40.6, 2026-11-24; `eval/recognizers/README.md`).
6. **Saving the text read:** `benchmark_recognizers.py --save-readings PATH` (3d).

## Order on the VM

setup.sh; download_models.sh (e5 first); 3a; 3b; 3c with `--save-readings`;
3d; SVC-09. Every command and its raw output goes into `~/results/` on the VM, which is copied back. Report MRR and Hit@k with their intervals and lookup counts, and answers in evidence as k/n.
