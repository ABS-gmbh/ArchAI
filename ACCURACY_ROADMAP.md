# ArchAI accuracy roadmap

72 findings on OCR and knowledge-retrieval accuracy, **70 of them confirmed by executing code** against real manuscript pages and the production SQLite database. Ordered by expected gain, then by effort.

Every item marked MEASURED carries a number someone obtained by running the code, not by reading it.

## Area headlines

**Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)**

> AL-01 is confirmed by execution: the composite scorer's realistic maximum is 0.718–0.794 against AUTO_SELECT thresholds of 0.80/0.85/0.90, so no candidate can ever be auto-linked by scoring — in the real production DB only 38 of 10,299 candidates ever reached "linked", every one of them via the hard-coded 0.92 canonical clamp, and all for a single entity (Lancelot Q215681).

**Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation**

> Text is indexed one OCR line per chunk (median 25 chars / 5 words), so top_k=5 hands the LLM ~136 characters — about 11% of one page — and on a real page switching to 6-line windows with 2-line overlap raised span coverage@5 from 39.1% to 100.0% (0/23 → 23/23 complete spans) with the same embedder.

**OCR output selection and combination (backend/pass choice, voting, proofreader guards)**

> There is no voting or consensus anywhere in the pipeline — selection is strictly first-wins — and the score it would need to select with, `_region_quality_value`, is effectively a constant (pure noise 0.789 vs clean Latin 0.796 on 60 measured real region/model pairs), so which model's text becomes the transcription is decided entirely by a hardcoded, unvalidated language→model table that picks a 5.3x-worse model on the real Old English page I tested.

**Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI**

> The vendor backend hands Kraken a line boundary 1.9x taller than the actual text line — because crop_agent silently pads the crop by 45% per side and then throws the crop rectangle away — which measured CER 0.777 instead of 0.359 on real line regions; binarisation with a scale-blind 31px window that erases 65% of stroke ink at ocr_crop_upscale=2 is the second-largest loss, and the root CLI (grayscale + Kraken's own blla line segmentation) already gets conf 0.969 and readable text on the same page where the vendor path gets 0.860 and gibberish.

**layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)**

> Zone-level recall is not the bottleneck — MainZone is detected on 16/16 real manuscript pages at conf 0.95-0.97 — the real accuracy cap is that one unmapped class name (DigitizationArtefactZone) aborts the entire region pipeline for a page, and that every text-bearing zone except MainZone is deliberately filtered out before OCR.

**Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)**

> There is no abbreviation-expansion layer anywhere in the repo and no normalised search form stored alongside the diplomatic text, so the very characters the Kraken models are trained to emit (macron, ꝑ, ꝓ, ꝗ, ꝯ, ꝰ, ⁊) are either deleted or indexed verbatim — 0/21 medieval-vs-modern spelling pairs collapse to the same search key, and single-line chunking additionally loses 12–26% of content words to un-rejoined line breaks.

**measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)**

> There is no ground truth and no CER/WER/recall@k code anywhere in the repo, so no accuracy claim — including the thesis's own 4.1% CER and mAP@50 0.89 — is verifiable; the single file named "_gold.txt" is a second machine run that disagrees with its pair at 48.4% CER / 95.8% WER, and the pipeline's only internal quality proxy scores real Vulgate Latin and that gibberish identically at 0.0000/HIGH.

---

## Findings


# LARGE EXPECTED GAIN (27)

### AL-01 CONFIRMED: composite score ceiling (measured 0.7936 best case, 0.7179 with real Wikidata descriptions) is below every AUTO_SELECT_THRESHOLD, so scoring alone can never produce a link

- **Where:** `vendor:app/services/entity_scoring.py:118`
- **Gain / effort:** large / low — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
QUALITY_THRESHOLDS = HIGH 0.80 / MEDIUM 0.85 / LOW 0.90 (entity_scoring.py:116-132). I built the best possible candidate offline (surface==label=="Lancelot", canonical_norm="lancelot", P31=Q3658341 literary character so type_ok=True, exact alias hit, domain_bonus saturated at 0.22, co-occurrence and chronology bonuses both firing) and called the real _score_candidate + disambiguate:
  BEST-CASE _score_candidate = 0.7936
  {"label_similarity":1.0,"alias_match_quality":1.0,"document_context_compatibility":0.0143,"source_confidence":1.0,"segmentation_label_prior":0.02,"cooccurrence_bonus":0.05,"chronology_bonus":0.03,"domain_bonus":0.22,"base_score":0.7536,"final_score":0.7936}
  HIGH threshold=0.8 clears=False / MEDIUM 0.85 False / LOW 0.90 False
  disambiguate(HIGH) -> unresolved  "best score (0.794) < threshold (0.8)"
With the REAL Wikidata description for Q215681 taken from the on-disk cache ("Arthurian character", 2 tokens) the max drops to 0.7179, because domain_bonus can only reach 0.07 (1 keyword) instead of 0.22. Full end-to-end offline run of run_authority_linking() against the real cache gave the scorer 0.5979 for Q215681; the persisted score of 0.9200 came entirely from the rescore clamp. Production DB app/archai.sqlite: 10,299 entity_candidates, max scorer output 0.55 at p99, only 38 rows ever is_selected+linked (24 at exactly 1.0 for Q116 "King"/Q355567 "Count" written by an older code path, 14 at exactly 0.9200 = the clamp constant, all Q215681 Lancelot). The newer mention_links table: 41 rows, link_status = 36 unresolved_low_quality / 4 skipped / 1 unresolved, ZERO linked. Sweep of the only free variable (context/description Jaccard) shows you need J>=0.26 to clear 0.80, >=0.49 for 0.85, >=0.83 for 0.90; measured real J = 0.0143-0.0286.
```

- **What it costs today:** Knowledge-retrieval recall for entity linking is ~0 for everything except the 23 hard-coded surfaces in _CANONICAL_ENTITIES. Of 14,476 real mentions in the production DB, only 5 distinct surfaces ('leantolot','leantlote','leantilote','graat','Artus') ever match canonically — 3,554 of them the single OCR variant 'leantolot'. Every place (6,468 mentions, 45%), every org, every work, every non-Arthurian person is structurally unresolvable. Downstream RAG/authority enrichment therefore has essentially no linked entities to retrieve over.

- **Recommendation:** Recalibrate rather than re-engineer: the ranking already works (measured correct-vs-best-wrong margins of 0.3445 for Lancelot and 0.3198 for Lausanne, both far above MIN_MARGIN=0.15) — only the absolute calibration is broken. Either (a) drop AUTO_SELECT_THRESHOLD to 0.62/0.68/0.74 (HIGH/MEDIUM/LOW), which admits the measured 0.68-0.79 correct candidates while still excluding the 0.25-0.36 wrong ones, or (b) renormalise _score_candidate so its weights sum to 1.0 at full signal (they currently sum to 0.68+0.12+0.08+0.06=0.94 plus up to 0.10 of bonuses, and base_score itself is capped at 0.75 when the context term is 0). Note authority_linking.py:12 still documents "AUTO_SELECT_THRESHOLD = 0.75 and MIN_MARGIN = 0.10" — the thresholds were raised in v3 without re-calibrating the scorer, and that stale docstring is the fingerprint of the regression. Ship a calibration test that asserts the best-case candidate clears the HIGH threshold.

### alias_sim at entity_scoring.py:237 is wired to context_similarity (Jaccard of a 200-400 char Old-French context window against a 2-10 token English Wikidata description), stranding 25% of the score weight at ~0.01 — and the same signal is then counted a second time

- **Where:** `vendor:app/services/entity_scoring.py:237`
- **Gain / effort:** large / low — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
compute_score computes `alias_sim = context_similarity(context_text, candidate_description)` and weights it _W_ALIAS = 0.25 (entity_scoring.py:202). context_similarity (entity_scoring.py:181-193) is a token-set Jaccard: len(ctx & desc)/len(ctx | desc). Measured with a real medieval-French page and the real Q215681 description:
  with chunk_text    ctx_len= 624 ctx_tokens= 61 alias_sim=0.0145  compute_score=0.7536
  base window only   ctx_len= 217 ctx_tokens= 36 alias_sim=0.0000  compute_score=0.7500
  best case where EVERY description token appears in the context: alias_sim=0.0769, compute_score=0.7692
So compute_score's structural cap with alias_sim≈0 is 0.55*1 + 0.15*1 + 0.05*1 = 0.75. The identical quantity is then computed AGAIN in _score_candidate as `document_context_compatibility = context_similarity(context, description_text)` (authority_linking.py:352) and added with weight 0.08 (authority_linking.py:396), so context similarity contributes 0.25*0.68 + 0.08 = 0.25 of the total while measuring at 0.014. Meanwhile the genuine alias signal IS already computed one line above — `alias_match_quality` at authority_linking.py:339 measured 1.0 for an exact alias hit — but it only gets weight 0.12.
```

- **What it costs today:** A quarter of the composite score is permanently dead weight: the Jaccard cannot exceed ~0.22 even if the Wikidata description were a strict subset of the context window, and measures 0.00-0.03 on real Old French vs English descriptions. This is the single mechanism that puts the ceiling at 0.75 in compute_score and 0.72-0.79 end-to-end. It also means the score carries no usable semantic-context evidence at all, so type-plausible-but-wrong candidates are separated only by string similarity.

- **Recommendation:** Pass the already-computed alias_match_quality into compute_score as alias_sim (rename the parameter honestly), and move the context/description signal to a single term. Replace token Jaccard with either (a) asymmetric coverage — |ctx ∩ desc| / |desc| — which is scale-invariant to the context window length and reaches 1.0 when the description is fully supported, or (b) embedding cosine, which the docstring at entity_scoring.py:184-185 already anticipates. Measured: with alias_sim wired to alias_match_quality the best case rises from 0.75 to ~0.94 in compute_score, comfortably above every threshold, without touching MIN_MARGIN.

### Duplicate chunks accumulate across re-runs of the same page and crowd out top-k: 87.9% of the live index is exact-duplicate text, and top-5 returns 1.27 distinct texts

- **Where:** `vendor:app/db/pipeline_db.py:489`
- **Gain / effort:** large / low — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
create_run mints `run_id = str(uuid.uuid4())` per OCR call with no lookup on asset_sha256; insert_chunks (pipeline_db.py:630) mints `chunk_id = str(row.get("chunk_id") or uuid.uuid4())`. Chroma ids are those chunk_ids (rag_store.py:363 `ids.append(str(ch["chunk_id"]))`), so `collection.upsert` (rag_store.py:158-162) can never overwrite a previous run's copy — it only ever adds. Nothing prunes: rag_store.delete_run (line 486) is only reachable from the DELETE /index/{run_id} endpoint (routers/index.py:38). MEASURED on the live store: archai_chunks__multilingual_e5_large_instruct holds 1055 docs of which 927 rows (87.9%) share their exact text with another row — 164 distinct texts duplicated, top offenders at x8 copies ('o arcadigns and a uint', 'Quer arreze tellement', ...); asset 'e-codices_fmb-cb-0001_001r_max.jpg' is indexed under 4 separate run_ids, 'old-english' 4, 'old-french' 3. archai_chunks: 191/214 rows (89.3%) duplicated. MEASURED crowding, 40 probes using real stored vectors as queries: top-5 contains mean 1.27 distinct texts out of 5, and mean 4.45 of the 5 slots (max 5/5) are literal copies of the query chunk itself. Unfiltered global search is the normal case whenever the page has not been OCR'd in this session: DocumentChatWorkspace.tsx:1271 sends `ocr_run_id: currentOcrRunId || undefined`, and chat_ai.py:1080-1086 then calls retrieve_chunks with run_ids=None, so no where-filter is applied (rag_store.py:606-620).
```

- **What it costs today:** Nearly four of the five evidence slots are wasted on byte-identical repeats, cutting the already tiny 136-char evidence budget to ~31 chars of unique text. With no run filter the store also mixes pages from unrelated manuscripts (15 distinct asset_refs), so a question about one folio can be answered with lines from another and cited with a wrong asset_ref/run_id — a citation that looks valid and is not.

- **Recommendation:** Three cheap independent fixes: (1) make chunk ids content-addressed and page-scoped — e.g. sha256(asset_sha256 + chunk_idx + text) — so re-OCRing a page overwrites rather than duplicates; (2) call rag_store.delete_run for prior runs of the same asset_sha256 before indexing a new one, in _auto_index_run (ocr.py:136); (3) dedupe in retrieve_chunks after the query by normalised text, keeping the best-scoring copy, and over-fetch (n_results = 4*k) so dedupe still leaves k results. Also make the frontend always send ocr_run_id, and reject an unfiltered global search from the chat path.

### select_backend_plan routes every Germanic language hint to McCATMuS first, which is 5.3x worse than CATMuS on the real Old English page

- **Where:** `vendor:app/services/ocr_backends.py:155`
- **Gain / effort:** large / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`elif hint in _GERMAN_LANGUAGE_HINTS or hint in _DUTCH_LANGUAGE_HINTS or hint in _ENGLISH_LANGUAGE_HINTS: attempt_backends = ("kraken_mccatmus", "kraken_catmus")`. MEASURED by executing `select_backend_plan`: hint old_english/english/middle_english/german -> primary=kraken_mccatmus; hint unknown -> primary=kraken_catmus. MEASURED on 20 real region crops of the Old English page: Old-English-lexicon hit rate McCATMuS 0.043 (6/140 tokens) vs CATMuS 0.229 (36/157). Sample region 3 — McCATMuS: 'Mome eoore poey purgcapoes hlayoys vee p...' vs CATMuS: 'tide eode þaer ƿingeardes hlapord iit ⁊g...' (the latter is recognisable Old English, 'þær wingeardes hlaford ut'). Combined with the early break (item 2), the better model is never even scored.
```

- **What it costs today:** Any page whose language hint resolves to English/German/Dutch gets the weaker recogniser on every region with no possibility of recovery. On this page that is the difference between a mostly-readable transcription and unusable garbage — the single largest measured accuracy delta in my area.

- **Recommendation:** Do not hand-maintain this table. Either (a) make it a configurable prior that only sets the *order* for a sample-based election that actually measures (see item 2 step 2), or (b) at minimum re-derive the table by benchmarking each installed model per language on held-out pages; note that CATMuS Medieval covers Old English in its training data while McCATMuS's advantage is elsewhere. Also add 'old_english'/'middle_english' handling explicitly rather than folding them into _ENGLISH_LANGUAGE_HINTS.

### The proofreader hallucination guard is applied at 1 of 4 call sites, and the guard used on the router path actively rewards hallucination

- **Where:** `vendor:app/routers/ocr.py:4367`
- **Gain / effort:** large / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
Four call sites of `OcrProofreaderAgent.proofread` (which itself applies no delta check): saia_ocr_agent.py:2063 guarded by `check_proofread_delta` at 2078 (the real guard: char_edit<=0.40, line_drift<=0.30, uncertainty_retention>=0.50, token_churn<=0.50); routers/ocr.py:4353 guarded by the WEAKER `proofreading_quality_guard`; ocr_agent.py:1408 and ocr_agent.py:1730 guarded by NOTHING (`final_text = proofreader.proofread(...)` then only `if raw_ocr.text and not final_text: final_text = raw_ocr.text`). MEASURED with real McCATMuS page output (923 chars, 20 lines) as raw and a fully invented fluent 20-line Latin rewrite as proofread output: `check_proofread_delta` -> accepted=False, 'PROOFREAD_REJECTED:char_edit_ratio=0.77>0.4', token_churn=1.000; `proofreading_quality_guard` -> accepted=True, 'proofreading accepted', because the label went RISKY -> HIGH and gibberish only moved 0.0050 -> 0.0169. The guard returned the fully invented text verbatim. `proofreading_quality_guard` (pipeline_hardening.py:96-148) has no edit-distance and no token-churn test at all — its only tests are label degradation >1 level, gibberish +0.10, and uncertainty-marker removal, every one of which a fluent invention passes.
```

- **What it costs today:** The worst failure mode of an LLM proofreader on a recognisable pericope — recognising the passage and writing it out from memory — is not merely undetected on the router path, it is *preferred*, because invented fluent text scores HIGH while true garbled OCR scores RISKY. On ocr_agent.py:1730 (reached by the live /ocr/extract, /agents/ocr and /agents/ocr/extract endpoints whenever no regions are supplied, with apply_proofread defaulting to True per schemas/agents_ocr.py:78) there is no check whatsoever. This can turn a mediocre transcription into a confidently wrong one, which is worse for retrieval than no transcription.

- **Recommendation:** Call `check_proofread_delta` at all four sites (it is already imported in saia_ocr_agent). Then either delete `proofreading_quality_guard` or invert its logic: a *large* label improvement combined with high char_edit/token_churn is evidence of invention, not of correction, and must be rejected. Keep the label check only as a secondary tiebreak after the delta check passes.

### OCR-002 confirmed: best-attempt selection ignores whether quality gates passed, so a gate-passing attempt is discarded for a gate-failing one

- **Where:** `vendor:app/routers/ocr.py:4116`
- **Gain / effort:** large / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`is_better = (best_quality_report is None or _quality_rank(hardened_quality_label) < _quality_rank(best_quality_report.quality_label) or (rank equal and ocr_quality_report.gibberish_score < best_quality_report.gibberish_score))` — `all_gates_passed`, computed 20 lines earlier at 4053, appears nowhere in the expression, and the `if all_gates_passed: break` at 4132 runs *after* `best_*` is already assigned at 4124-4130. REPRODUCED by executing the real `compute_quality_report`, `enforce_quality_gates` and this exact ranking on two real transcriptions of the same page (attempt0 with detected_language='latin', attempt1 a near-identical re-OCR with detected_language='unknown', 9/10 lines identical so cross-pass stability is 0.936 and not a confound): attempt0 label=HIGH gib=0.0000 all_gates_passed=False failed=['LEXICAL_PLAUSIBILITY']; attempt1 label=HIGH gib=0.0064 all_gates_passed=True failed=[]. Result: best_attempt_idx=0, chosen_gates_passed=False, gates_ever_passed=True. The gate-passing attempt was thrown away. Running the same two attempts in the reverse order yields a different final transcription, so the outcome is order-dependent. The trigger is realistic because the LEXICAL_PLAUSIBILITY gate is only constructed when `_lex_lang != 'unknown'` (routers/ocr.py:4045), and detected_language flips between attempts — the shipped run manifest for this page records 'Detected language: Unknown (low confidence)'.
```

- **What it costs today:** Two compounding harms. (1) The transcription handed downstream can be the one the gates rejected, while a validated alternative was computed and discarded. (2) `gates_ever_passed=True` was set by the *discarded* attempt, which suppresses the FAILED_QUALITY salvage branch at routers/ocr.py:4294 (`if not gates_ever_passed and has_blocking`), so the run is reported as clean. I also measured that the surviving `best_gate_decisions` then reports `downstream_mode='token_based'` while `blocked_stages=['token_ner','token_search']` — the two fields contradict each other, so token-based NER and ligature search run on text the gates blocked.

- **Recommendation:** Make gate status the primary sort key: `is_better` should compare `(not all_gates_passed, _quality_rank(label), gibberish_score)` so any gate-passing attempt beats any gate-failing one. Also make `enforce_quality_gates` derive `downstream_mode` from `blocked_stages` as well as the label, so a blocked page cannot be marked token_based.

### adaptiveThreshold blockSize is hardcoded to 31px while ocr_crop_upscale scales the strokes — at upscale=2 it erases 65% of stroke ink, and the Kraken models want grayscale, not binary

- **Where:** `vendor:app/services/ocr_backends.py:257`
- **Gain / effort:** large / low — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
ocr_backends.py:257-263 `cv2.adaptiveThreshold(arr, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 15)` — blockSize=31 is a literal. The upscale runs FIRST, at crop_agent.py:102 `crop.resize((w*upscale_factor, h*upscale_factor), Image.Resampling.LANCZOS)`, so the window is scale-blind with respect to ocr_crop_upscale=2.

MEASURED on a real line crop from e-codices_fmb-cb-0001_001r_max, comparing adaptive-31 against Otsu as the ink reference:
  upscale=1: crop 2219x456, median stroke ~8.8px, adaptive-31 marks 27.8% of the Otsu ink as WHITE
  upscale=2: crop 4438x912, median stroke ~16.8px, adaptive-31 marks 65.0% of the Otsu ink as WHITE (stroke interiors hollowed out)

MEASURED model expectation (loaded all three .mlmodel files): one_channel_mode == 'L' (grayscale) for mccatmus, catmus_medieval and cremma_medieval — none is a binary-input model.

MEASURED recognition effect, 8 real line regions of page fmb-cb-0002, full-crop boundary, same model:
  pipeline_as_is (denoise+deskew+adaptiveThreshold): mean_conf 0.8802, 209 chars, CER 0.8607
  grayscale only:                                    mean_conf 0.8963, 299 chars, CER 0.8297
  gray+autocontrast:                                 mean_conf 0.8945, 294 chars, CER 0.8297
Binarising loses 30% of the recognised characters (209 vs 299) on identical pixels. Example line 6: binarised 'Largam Dd Wataz ---.' vs grayscale 'pile orHabe 88. HS LATOADMRDMRDORCE.'

Upscale itself is near-worthless: with the tight boundary, upscale=1 CER 0.3591 vs upscale=2 CER 0.3684 — no gain for 4x the pixels, because Kraken rescales to height 120 regardless. Crops reach 11856x1556px at upscale=2, within 2% of the ocr_max_long_edge=12000 guard.
```

- **What it costs today:** Two compounding costs. (a) Domain mismatch: hard-binarised input is fed to models whose one_channel_mode is 'L'; grayscale alone recovers 30% more characters and +0.016 mean confidence on the same crops. (b) The binariser is actively destructive at the configured upscale: 65% of stroke ink is flipped to white at upscale=2 versus 27.8% at upscale=1, so minims, hairlines and abbreviation marks are hollowed into disconnected fragments before the recogniser ever sees them. The raw_metadata string at ocr_backends.py:633 advertises this as 'grayscale+denoise+deskew+binarize', which reads as a feature and hides the loss.

- **Recommendation:** Stop binarising on the Kraken path: pass ImageOps.autocontrast(grayscale) (optionally CLAHE, which services/multiview.py already implements) and let Kraken do its own normalisation. If a binarised variant is wanted at all, make it a scored alternative view rather than the only input, use Sauvola from multiply.py:147 rather than a fixed-31 Gaussian window, derive the window from the measured stroke width (or at minimum multiply blockSize by upscale_factor), and run it BEFORE the upscale, not after. Separately, drop ocr_crop_upscale to 1 for the Kraken path — it measured no CER benefit, quadruples memory, and is what makes the fixed blockSize catastrophic.

### DigitizationArtefactZone is emitted by the shipped zone model but has no COCO mapping, so a KeyError aborts region-based layout for the whole page and OCR silently falls back to whole-page

- **Where:** `vendor:app/core/image_batch_classes.py:59`
- **Gain / effort:** large / low — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
`cls_int = coco_class_mapping[cls_string]` (image_batch_classes.py:59) with `cls_string = catmus_zones_mapping.get(self.name, self.name)`. `catmus_zones_mapping` (constants.py:4-17) has entries for 12 names but NOT DigitizationArtefactZone, and `MODEL_CLASSES["zone"] = None  # all classes` (constants.py:54) lets the zone model emit all 11 of its classes, which include class 0 = DigitizationArtefactZone. MEASURED: running the real `run_single_segmentation(..., confidence=0.25, iou=0.3, selected_classes=FINAL_CLASSES)` on .tasks/0a876bacebaa/'Latin script.jpg' raises `KeyError: 'DigitizationArtefactZone'` from image_batch_classes.py:59. That page's zone detections at production settings are {'DigitizationArtefactZone': 1 @conf 0.96, 'MainZone': 1, 'DropCapitalZone': 1, 'MarginTextZone': 2}. Fires on 1/22 unique real images in .tasks (the artefact is the photographic ruler beside the folio, box (2835,0)-(3359,4000)). With ONE in-memory mapping entry added, the same call succeeds and `_extract_segmented_regions` returns 73 line-level OCR regions ('Main script black' 57 + 'Main script coloured' 16) instead of 0. The exception is swallowed at ocr.py:3818-3820 and ocr.py:3830-3832 into the warning strings SEGMENTATION_FAILED / SEGMENTATION_REGIONS_FAILED, after which `structured_regions` is empty and control falls through to the unsegmented whole-page OCR loop.
```

- **What it costs today:** On an affected page the pipeline loses 100% of its layout: 73 line regions -> 0, so there is no per-line crop, no reading order, no column clustering, and the recognizer gets one whole-page image. The failure is invisible in the output text; it only appears as a warning string. Any folio photographed with a ruler, colour target or edge artefact is exposed.

- **Recommendation:** Add DigitizationArtefactZone (and any other zone-model class absent from coco_class_mapping) to catmus_zones_mapping in app/core/constants.py, or make Annotation.to_coco_format skip unmapped classes instead of raising. Belt-and-braces: set MODEL_CLASSES["zone"] to the explicit list of class ids the mapping covers so an unmapped class can never reach coco assembly, and promote SEGMENTATION_FAILED from a warning string to a hard error so a total layout loss cannot pass unnoticed.

### Layout bboxes are computed on a downscaled copy of the page and returned to the OCR stage without being rescaled, so every region crop is taken from the wrong pixels

- **Where:** `vendor:app/routers/predict.py:112`
- **Gain / effort:** large / low — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
`_prepare_image_for_segmentation` resizes when the page exceeds SEGMENTATION_MAX_PIXELS=85000000 or SEGMENTATION_MAX_LONG_EDGE=12000 (predict.py:30-31): `working = working.resize((new_w, new_h), ...)` then `working.save(normalized_path)` (predict.py:112-123). ocr.py:2453-2461 feeds that `prepared_path` to `run_single_segmentation`, and `_extract_segmented_regions(coco)` returns `bbox_xyxy=[x, y, x + w, y + h]` (ocr.py:2382) straight from the downscaled COCO with no inverse scale, while the crops downstream come from the ORIGINAL upload (`base.crop(crop_box)` at ocr.py:2782, `image.crop((x1, y1, x2, y2))` at saia_ocr_agent.py:1119). MEASURED end-to-end on a real page upscaled to 9000x12000 (108 MP): prepared image = 7984x10646 (`changed = True`, scale 0.8872), coco image entry = (7984, 10646), 20 regions returned with max x2 = 7521.6 and max y2 = 8754.6 — i.e. expressed in the downscaled frame. Vertical displacement at the page foot is ~1354 px, 11.3% of page height, roughly four text lines on that 20-line page. Two 11000x11000 (121 MP) test images already sit in .tasks/ (huge_test_seg.tiff, gray_11k_seg_test.tiff), so this path has been exercised.
```

- **What it costs today:** For any page over 85 MP or 12000 px on the long edge every region crop is offset and shrunk, so crops straddle line boundaries and the transcription is garbage while the pipeline reports success. e-codices max scans are 50 MP today, but a 600 dpi or stitched capture crosses the threshold, and the failure is silent and total rather than degraded.

- **Recommendation:** Have _prepare_image_for_segmentation return the applied scale factor and multiply every bbox/segmentation coordinate by 1/scale before the COCO leaves run_single_segmentation (or run the crops against the prepared image instead of the original). Add a regression test that segments a >85 MP page and asserts max(bbox x2, y2) is within the ORIGINAL page dimensions.

### No normalised/searchable form is stored alongside the diplomatic transcription — the RAG index receives raw recognizer output and the normalized_text column is a no-op copy

- **Where:** `vendor:app/routers/ocr.py:1727`
- **Gain / effort:** large / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
ocr.py:1727 writes `"normalized_text": str(chunk.get("text") or "").strip()` — the raw text stripped, not normalize_for_search(). rag_store.py:353-360 in _index_chunks_for_run does `text = str(ch["text"] or "").strip(); ... documents.append(text)` with no normalisation, and the `chunks` table schema has only a `text` column (no normalised sibling). retrieve_chunks (rag_store.py:576-637) passes the user query through unchanged (`query_texts=[query]`) with no lexical/BM25 fallback. MEASURED on the live DB (app/archai.sqlite): `select sum(case when trim(raw_text)=normalized_text then 1 else 0 end), count(*) from evidence_spans` returns **2184 / 2188 = 99.8%** of evidence spans whose 'normalized' text is byte-identical to the raw text. MEASURED round-trip: an NFD line 'clémence de Sainte Marie ⁊ ꝑsona ꝯdit eiꝰ' is indexed verbatim (is_NFC=False), while normalize_for_search would have produced 'clemence de sainte marie ꝑsona ꝯdit eiꝯ'; a user's NFC query 'clémence' does not substring-match the stored NFD text (measured False) but does match after normalize_for_search on both sides (measured True). By contrast authority_linking.py:228 and :1393 DO call normalize_for_search for their evidence rows, so the two paths disagree.
```

- **What it costs today:** Retrieval is dense-only over un-normalised NFD text, so there is no path at all by which a normalised query token can meet a normalised index token. Any lexical or hybrid retrieval added later has nothing to match against, and joins on evidence_spans.normalized_text silently compare raw strings. The good news, worth stating: nothing in the pipeline mutates the stored transcription, so the diplomatic reading a thesis needs IS preserved — the defect is purely the missing second field, which is the cheap half of the fix.

- **Recommendation:** Add a `text_search` column to `chunks` (and actually populate `evidence_spans.normalized_text` with normalize_for_search output at ocr.py:1727). Index both: the diplomatic text as the displayed/cited document and the normalised+expanded text as the embedded/searchable field, and normalise the query with the same function in retrieve_chunks. This keeps the diplomatic reading authoritative while making retrieval possible.

### One chunk per manuscript line with no line-break word rejoining loses 12-26% of content words from retrieval

- **Where:** `vendor:app/routers/ocr.py:427`
- **Gain / effort:** large / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
_build_line_chunks() iterates `for line in value.splitlines()` and emits one chunk per line with no joining, no overlap and no neighbour window; it is the only chunker in the repo (called at ocr.py:1713 and :2023). `grep -rE 'hyphen|dehyph|rejoin|00ad|¬' --include='*.py' app/ src/` finds no rejoining anywhere — only ocr_quality.py:359/379/481 which *detect* a continuation hyphen and never act on it. MEASURED on the live DB: mean distinct chunk length is **24.1 chars** (min 8, max 1108, n=1285), i.e. 4-5 words per retrieval unit. MEASURED recall, running the real _build_line_chunks on a real Latin prose passage (Jerome's Genesis prologue) and a real Old French prose passage (Villehardouin) laid out at manuscript line widths, asking whether each content word (>=4 chars) survives intact inside any chunk: Latin 34/46=73.9% at width 24, 38/46=82.6% at width 34, 40/46=87.0% at width 44; Old French 27/34=79.4%, 30/34=88.2%, 31/34=91.2%. Applying line-break rejoining before chunking gives **100.0% in all six cases**. Words measurably lost at width 34: ['auribus','desiderii','epistolas','hebreo','nova','quasi','septuaginta','translatum'] and ['entre','folques',"l'incarnation",'saint'].
```

- **What it costs today:** At the chunk length actually present in the database (24 chars) roughly a quarter of content words exist only as two fragments in two different Chroma documents, so no query — dense or lexical — can retrieve them; the loss falls hardest on long, distinctive words, which are exactly the high-information query terms ('septuaginta', 'Folques'). Separately, embedding a 4-5-word fragment gives a very weak dense signal and there is no neighbour expansion to restore context around a hit.

- **Recommendation:** Before chunking, rejoin words across line breaks (consume a trailing hyphen/¬/soft hyphen where present; for pre-1300 hands that break without any mark, join when the trailing fragment plus the leading fragment forms a token the lexicon accepts and neither fragment does). Then chunk on sentence or N-line windows (~300-600 chars) with overlap rather than one line each, keeping per-line offsets in metadata so citations still point at the exact manuscript line.

### gibberish_score cannot distinguish real Latin from the pipeline's own gibberish — both score 0.0000 / HIGH

- **Where:** `vendor:app/services/ocr_quality.py:676`
- **Gain / effort:** large / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`def gibberish_score(text, script='latin', language=None) -> float:` with the docstring "Composite gibberish detector (0 = clean, 1 = total gibberish)". MEASURED by executing compute_quality_report in the vendor venv on three controls: real Vulgate Latin (Genesis 1, 5 lines) → gibberish=0.0000, quality_label=HIGH, non_wordlike_frac=0.000, rare_bigram_ratio=0.000, char_entropy=3.965; the pipeline's actual output for fmb-cb-0002 (5 lines, e.g. 'turiau prutan icoiu pinrayou') → gibberish=0.0000, quality_label=HIGH, non_wordlike_frac=0.000, rare_bigram_ratio=0.000, char_entropy=4.172; uniform random letters → gibberish=0.1630, quality_label=RISKY, non_wordlike_frac=0.433. Worse, on the two full real transcriptions the labels invert against quality: the pipeline-B run ('gold', median_conf 0.969) → gibberish=0.002, quality_label=RISKY, while the pipeline-A run (median_conf 0.887, the more degraded text) → gibberish=0.0033, quality_label=OK.
```

- **What it costs today:** The gate is blind at exactly the operating point that matters. Medieval HTR failure is not random noise — it is pronounceable pseudo-text with normal entropy, normal bigram load and normal token lengths, which is why the character-level signals (char_entropy:126, rare_bigram_ratio:230, non_wordlike_score:250) all read clean. So the only in-pipeline proxy for OCR accuracy returns 0.000 on 100%-wrong output and additionally labels the worse of two real runs OK while labelling the better one RISKY. Any decision keyed off quality_label — seam retry, downstream_mode, gate enforcement at routers/ocr.py:2969, 3261, 3410 — is being made on a signal with no measured discriminative power. The previously-fixed defect (gibberish_score could not detect scrambled/repeated text) fixed the repetition/transposition case; it did not fix the pronounceable-nonsense case, which is the dominant real failure mode.

- **Recommendation:** Do not try to fix gibberish_score in isolation — it is a proxy, and a proxy can only be trusted once it has been correlated against real CER. Once the gold set from item 1 exists, compute Pearson/Spearman correlation between each component signal (gibberish_score, non_wordlike_frac, rare_bigram_ratio, char_entropy, cross_pass_stability, lexical_implausibility) and per-line CER, and publish that table in eval/results/. Keep only the signals that correlate, and re-weight the composite by measured correlation instead of hand-picked constants. The lexical signal (item 3) is the one that already shows separation and should carry most of the weight.

### The one discriminative quality signal is silently disabled on 78% of real runs, and always for per-line scoring

- **Where:** `vendor:app/services/ocr_quality.py:657`
- **Gain / effort:** large / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`if language not in _TRIGRAM_PROFILES or language == "unknown": return None` — lexical_implausibility returns None whenever no trigram profile exists, and gibberish_score then collapses to the character-only path. MEASURED: with language='latin', lexical_implausibility separates cleanly — real Vulgate Latin 0.5577 vs pipeline gibberish 0.8309, driving gibberish_score to 0.0881 vs 0.3116, a 3.5x separation (versus 0.0000 vs 0.0000 with language=None). But `_TRIGRAM_PROFILES` (services/lexicon_trust.py) holds only 11 keys — latin, old_french, middle_french, french, anglo_norman, occitan, italian, spanish, portuguese, catalan, unknown(size 0) — while routers/ocr.py `_normalize_detected_language` happily returns profile-less values I verified by execution: 'Old English'→'old_english', 'Middle High German'→'middle_high_german', 'German'→'german', 'enm'→'middle_english'. All four fall through line 657 to None. Separately, ocr_quality.py:942 `line_gs = gibberish_score(line, report.script_family)` omits the language argument entirely, so per-line gibberish scoring is permanently on the blind character-only path. And in practice the language is usually absent: across the 9 real page reports in outputs/, `grep -h '^Detected language:'` gives 7× 'Unknown (low confidence)', 1× 'fr', 1× 'ca' — 7/9 = 78% of real runs pass language=None.
```

- **What it costs today:** The one signal with demonstrated 3.5x separation between real text and pipeline gibberish is off on 78% of real pages and off on 100% of per-line scoring. Two of the three canonical showcase pages are affected structurally rather than incidentally: the Old English page can never get a lexical score because no old_english profile exists, and the Latin Abaton page is misdetected as Catalan (documented in artifacts/thesis_showcase/figure26/limitations_summary.json, case lim_la_langid) so it is scored against the wrong profile. This means the quality gate's accuracy is itself unmeasured and, where measurable, near-zero.

- **Recommendation:** Three separate changes, all verifiable against the item-1 gold set: (a) pass `language` through at ocr_quality.py:942 so per-line scoring uses the lexical path; (b) add trigram profiles for old_english, middle_english, german/middle_high_german, or make _normalize_detected_language refuse to return a value with no profile so the miss is visible rather than silent; (c) treat language-id accuracy as a measured quantity — add a `language` column to eval/gold/manifest.json and report language-id accuracy per page alongside CER, since 78% 'Unknown' is currently an unmeasured upstream failure that disables downstream quality scoring.

### The eval substrate is entirely gitignored — a fresh clone contains zero manuscript pages, zero gold text, and zero pinned models

- **Where:** `.gitignore:30`
- **Gain / effort:** large / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
Line 27 `weights/*`, line 30 `outputs/`, line 39 `fixes/`, line 60 `archai/vendor/layout/backend/.tasks/`. MEASURED: `git ls-files outputs/ | wc -l` = 0, so the sole gold-named file is untracked and machine-local. `git ls-files | grep -iE '\.(jpg|jpeg|png|tif|tiff)$'` = 18 tracked images, every one of them a rendered figure or UI screenshot under artifacts/thesis_showcase/ except three region crops (figure24/crops/ex01_old-english_r21.png etc.) which the bundle's own README states have no gold text. All 458 task directories with the 14 distinct real manuscript pages are under the ignored .tasks/. The models are not pinned either: of the four entries in weights/kraken_models/, three are symlinks into `/Users/mobasuony/Library/Application Support/htrmopo/...` (catmus_medieval, catmus_print, mccatmus → McCATMuS_nfd_nofix_V1.mlmodel) and only cremma_medieval.mlmodel is a real 16 MB file — and weights/* is ignored regardless. The one tracked ground truth in the entire repo, archai/vendor/layout/compare/data/ground_truth_coco.json, references 10 images (e-codices_bbb-0219_001v_max.jpg, _003v, _007v, _023v, _041v, _044r...) and `find . -name 'e-codices_bbb-0219_001v_max.jpg'` finds none of them; its expected data directory `Aleyna 1 (2024)` (compare.py:507) does not exist.
```

- **What it costs today:** Reproducibility is a precondition for accuracy, not a nicety. Today, no accuracy number can be recomputed by anyone else, on any other machine, or by this machine after a `git clean` — the images, the gold text, the model binaries and the thesis chapter that cites the numbers are all outside version control. It also means the 48.4% CER I measured in item 1 is the only accuracy datapoint that exists, and it would be destroyed by a checkout.

- **Recommendation:** Do not un-ignore outputs/ or weights/ — instead create a small, deliberately tracked eval island: commit `eval/gold/manifest.json` with an `image_sha256` per page (the pipeline already computes and prints this, e.g. 1080cd11a13a3560... in the report header, and artifacts/thesis_showcase/manifest.json already records `asset_sha256` per source page — reuse that convention), commit the ~10 downscaled eval page images (or a fetch script plus checksums if licensing forbids), commit the gold line files, and commit a `weights/MANIFEST.json` pinning each recogniser by SHA256 (the report header already emits `Recognizer SHA256: dfb911ba25fd...`) so a run can assert it used the model the numbers were measured with. Add `!eval/` and the specific paths ahead of the ignore rules.

### The real annotation corpus lives on a private CVAT server with redacted credentials and no snapshot in the repo

- **Where:** `archai/vendor/layout/CVAT_download/download.py:10`
- **Gain / effort:** large / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`HOST = "http://134.76.21.30:8080"`, line 11 `USERNAME = "XXXXXX"`, line 12 `PASSWORD = "XXXXXXX"`, line 13 `PROJECT_ID = 7` (the commented org slug at line 26 is `"eManusKript"`). MEASURED: the export target `cvat_project_7_export` does not exist anywhere on disk (`find . -maxdepth 4 -name 'cvat_project*'` → nothing; CVAT_download/ contains only download.py and unzip.py); `cvat_sdk` is not installed in either venv (ModuleNotFoundError). The named expert datasets from the recorded metrics — 'Aleyna 1 (2024)', 'Annika 2 (2024)', 'Luise 1 (2024)', 'Luise 2 (2024)', 'Nuray 1 (2024)', 'Nuray 2 (2024)', 'SampleBatch2/3/4' — are absent (`find . -iname '*Aleyna*'` → nothing), yet expert_datasets_model_comparison_summary.json records their GT annotation counts (44/136/0/99/64/25) and sample_batches (1437/909/1160), totalling 3,874 annotations that once existed. compare.py:507 expects them at `os.path.join(SCRIPT_DIR, "Aleyna 1 (2024)")`, and compare/data/README.md documents the intended run from a hardcoded Linux path `/home/hasan/layout/compare/data`.
```

- **What it costs today:** This is the shortest path out of the measurement hole and it is currently blocked. The gold data for layout — 3,874 expert annotations over what the thesis calls 220 pages — is not lost, it is on an institutional CVAT instance that this repo has a downloader for; but the credentials are stripped, the SDK is uninstalled, and no export snapshot was ever committed. Until that export exists, items 4 and 5 cannot be finished and the layout stage stays unmeasured, which in turn caps OCR accuracy because bad crops are unrecoverable downstream.

- **Recommendation:** Export CVAT project 7 once, then commit the derived artifacts rather than the raw dump: a COCO ground-truth file per dataset, plus `eval/layout/manifest.json` listing image filename + SHA256 + dataset + split. Put credentials in .env (already gitignored at .gitignore:19-21, with a tracked *.example convention) and never back into download.py. Note the redacted literals are placeholders, not leaked secrets — but the file being tracked with credential-shaped constants is what the repo's own secret-scan CI job (.github/workflows/ci.yml, 'no committed secrets') exists to prevent, so move them regardless. If the server is unreachable, fall back to publicly available annotated sets so the harness has real input.

### AL-02 CONFIRMED: every VIAF and GeoNames candidate is born with instance_of_qids=[] and is therefore forced type_compatible=False and hard-gated out — and because the gate inspects only the TOP-ranked candidate, one such candidate can poison an otherwise resolvable mention

- **Where:** `vendor:app/services/authority_sources.py:102`
- **Gain / effort:** large / medium — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
search_viaf hard-codes `"instance_of_qids": []` (authority_sources.py:102) and so does search_geonames (authority_sources.py:164). Neither source is ever enriched, because the enrich branch requires `source_name == "wikidata" and candidate.get("qid")` (authority_linking.py:1768). is_type_compatible then hits wikidata_client.py:566-570: `if not instance_of_qids: if ent_lower != "person": return False`. Measured directly:
  is_type_compatible('person', [], description='VIAF authority record') = False
  is_type_compatible('work',   [], description='VIAF authority record') = False
  is_type_compatible('place',  [], description='capital of a political entity') = False
  is_type_compatible('place',  [], description='seat of a first-order administrative division | Bourgogne > France') = False
  is_type_compatible('person', [], description='Personal | Chretien de Troyes') = False
End-to-end offline run of run_authority_linking (network hard-blocked, VIAF/GeoNames served from fixtures shaped exactly like the real responses):
  MENTION Paris place -> unresolved  "best candidate type_incompatible (score=0.397, qid=)"
    0.3970 src=geonames 2988507 Paris tc=False
  MENTION Artus person -> unresolved "best candidate type_incompatible (score=0.208, qid=)"
    0.2082 src=viaf 96993265 Lancelot tc=False
Run summary reported type_mismatch_count: 8 of 9 candidates. The poisoning is structural: disambiguate checks `if not best.get("type_compatible", True)` on ranked[0] only (entity_scoring.py:371-381) and returns unresolved for the whole mention rather than falling through to the next compatible candidate.
```

- **What it costs today:** VIAF and GeoNames contribute zero possible links despite being queried on every person/work and place mention (the offline run logged api_calls_viaf=4, api_calls_geonames=1 for 3 mentions — pure latency and rate-limit cost for candidates that cannot be selected). Combined with the absence of any place in _CANONICAL_ENTITIES, this makes the 6,468 place mentions (45% of the production corpus) permanently unlinkable. The top-only gate additionally converts mentions that DO have a good Wikidata candidate into 'unresolved' whenever a type-blind VIAF/GeoNames row happens to outrank it.

- **Recommendation:** Two independent fixes. (1) Give the non-Wikidata sources a type: map GeoNames fcode to a compatible QID (PPL*/PPLC -> Q486972/Q515, ADM* -> Q56061, RGN -> Q82794) and VIAF nameType to Q5/Q7725634, or add a `source_typed` escape in is_type_compatible so a source that natively asserts a type is not judged on absent P31. (2) Make disambiguate's type gate a filter, not a veto: drop type-incompatible candidates from `ranked` and then apply the score/margin gates to the surviving list, so one untyped row can no longer sink the mention. Keep the incompatible rows in result['all'] for the audit trail.

### Chunking is one OCR line per chunk, no merging and no overlap — the context handed to the LLM is starved

- **Where:** `vendor:app/routers/ocr.py:427`
- **Gain / effort:** large / medium — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
_build_line_chunks: `for line in value.splitlines(): ... chunks.append({"idx": idx, "start_offset": start, "end_offset": end, "text": line})` — one chunk per physical OCR line, no window, no overlap, no region/zone grouping. Called at ocr.py:1713 and ocr.py:2023, and those rows are what rag_store._index_chunks_for_run embeds verbatim (rag_store.py:352-378 reads pipeline_db.list_chunks and upserts ch['text'] unchanged). MEASURED on the live Chroma store (backend/.data/chroma): archai_chunks__multilingual_e5_large_instruct = 1055 docs, chars min=7 median=25 mean=28.4 p90=32; words median=5; 1052/1055 (99.7%) of chunks are <=60 chars. archai_chunks = 214 docs, median 24 chars. MEASURED evidence budget: sampling 40 real stored vectors as queries against the real collection, top-5 delivers mean 136 chars (median 135) = ~11.4% of the median indexed page (1193 chars). Running the real retrieve_chunks() against the real store for three natural-language questions returned 199, 94 and 79 total characters of evidence. MEASURED A/B on a real page (outputs/e-codices_fmb-cb-0001_001r_max, 71 lines, 1796 chars), same MiniLM embedder, same k=5, 23 three-line target spans, query = the middle line: per-line chunking (71 chunks, median 24 chars) -> span coverage@5 = 39.1%, evidence chars@5 = 126, complete spans 0/23; 6-line windows with 2-line overlap (18 chunks, median 150 chars) -> span coverage@5 = 100.0%, evidence chars@5 = 751, complete spans 23/23.
```

- **What it costs today:** Any question that needs more than one manuscript line to answer cannot be answered from retrieved evidence: the retriever locates the right line but never returns its neighbours, so the model sees ~30 words of a page and either refuses or fills the gap from parametric knowledge. It also makes distances meaningless — a 5-word chunk has almost no lexical surface to match a question against, which is why the real top-5 distances were 0.69-0.85 (cosine) i.e. near-orthogonal.

- **Recommendation:** Chunk by SegmOnto region first (the layout stage already emits MainZone / MarginTextZone / etc.), then pack lines into ~400-800 character windows with ~1-2 lines of overlap, keeping start_offset/end_offset spanning the window so citations and evidence_spans still resolve. Keep the per-line rows in SQLite for offset/citation anchoring, and index the windows. Store region label and line range in the chunk metadata so retrieval can drop DropCapitalZone/marginalia noise or boost MainZone. Do not simply raise rag_top_k: with 8x duplicates in the store (see next finding) that mostly buys more copies of the same line.

### The index is split across two incompatible embedding spaces, and any provider hiccup silently moves the query to the wrong one

- **Where:** `vendor:app/services/rag_store.py:145`
- **Gain / effort:** large / medium — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
_provider_embed swallows every failure: `except Exception as exc: log.warning("Provider embedding failed (%s), falling back to local: %s", ...); return None, _LOCAL_EMBED_BACKEND` (rag_store.py:145-147), including a missing `openai` package (line 133) and a missing API key. The backend key then selects a *different collection* via _collection_name_for_backend (rag_store.py:66-70), and retrieve_chunks re-resolves the backend per query (rag_store.py:603-605), so a transient provider failure sends the query to a collection that does not contain the documents. MEASURED on the live store: both spaces are populated — archai_chunks__multilingual_e5_large_instruct = 1055 docs at dim 1024, archai_chunks = 214 docs at dim 384 (chromadb DefaultEmbeddingFunction, confirmed = all-MiniLM-L6-v2 at onnx_mini_lm_l6_v2.py:38). Runs: 24 in the e5 collection, 10 in the local one, only 4 in BOTH — so 20 runs are unreachable whenever the provider embedder fails and 6 are unreachable whenever it works. Assets reachable only in the e5 space include 'e-codices_fmb-cb-0001_001r_max.jpg', 'thesis-page-001r', 'latin-abaton', 'old-english'. Executed against a copy of the real store with RAG_EMBEDDING_MODEL='': retrieve_chunks resolved backend 'local_default', collection 'archai_chunks', count 214 — the 1055 e5 chunks were invisible with no error surfaced to the caller.
```

- **What it costs today:** Recall silently drops to zero for most of the corpus on any embedding-provider blip, and the only signal is a log.warning. Two thirds of the runs in this store are already stranded in a space the other path cannot search. The user sees a confident answer with no evidence rather than an error.

- **Recommendation:** Pick one embedding space and make it authoritative. Prefer a locally-hosted multilingual model so there is no network in the retrieval path; if the provider stays, then on embedding failure raise instead of degrading, and surface it to the chat turn ('retrieval unavailable') rather than returning zero hits. Record the embedding model and dimension in the collection metadata and refuse to query a collection whose model does not match the configured one. Then reindex the 6 stranded local-only runs into the canonical space and drop the other collection.

### The backend-selection score is non-discriminative: it cannot tell noise from correct text, so it never selects anything

- **Where:** `vendor:app/agents/ocr_agent.py:1012`
- **Gain / effort:** large / medium — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`combined = (0.25 * confidence_value) + (0.45 * text_quality) + (0.30 * lexical_score)`. MEASURED by executing `_region_quality_value` on the 3 Kraken models x 20 real region crops of e-codices_fmb-cb-0002_001r_max: all 60 values fall in [0.7883, 0.8216]; 0/60 below the default quality_floor=0.60. MEASURED dynamic range: empty=0.2750, single char 'a'=0.6923, pure noise 'qjxvv bbxzz cccnnn kkkppp'=0.7890, degenerate repeat 'abc abc abc...'=0.7980, real CATMuS line=0.8146, clean Latin 'Simile est regnum caelorum...'=0.7957. Noise outscores clean Latin by nothing (0.7890 vs 0.7957) and a degenerate repeat outscores clean Latin (0.7980 > 0.7957). Cause: `score_text_quality` (ocr_agent.py:592) only measures letter/printable/short-line/token-count ratios, and `lexical_plausibility(text,'unknown')` returns a hardcoded 0.50 (lexicon_trust.py:113) because the default `language_hint` is 'unknown' (schemas/agents_ocr.py:76) — so 0.30 of the weight is a constant 0.15 for every candidate, and Kraken region confidences vary only 0.9391-0.9544 across models.
```

- **What it costs today:** This is the only score used to pick between OCR backends per region. Because every non-empty candidate scores ~0.81 and the loop breaks at the first candidate >= quality_floor (0.60), the multi-backend attempt chain degenerates to 'always use attempt[0]' — MEASURED: 20/20 regions selected attempt #0, and backend #2 was never invoked. The pipeline therefore has no mechanism at all for choosing the better of two available transcriptions, only for rejecting empty ones.

- **Recommendation:** Replace the selector score with something monotone in correctness before adding any voting: (a) drop the constant lexical term when language is unknown and renormalise; (b) score against a real word list / character n-gram LM per script rather than 'is it letters'; (c) use the character-confidence *distribution* (min, 10th percentile, fraction below 0.5) instead of the mean, which Kraken pins near 0.94 regardless of correctness. Validate the new score by checking it separates the noise/degenerate/clean fixtures above by more than the 0.007 it does today.

### Line boundary handed to Kraken is 1.9x too tall: crop_agent pads the crop but discards the crop rectangle, so _local_boundary_from_metadata assumes crop == bbox

- **Where:** `vendor:app/services/ocr_backends.py:403`
- **Gain / effort:** large / medium — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
ocr_backends.py:402-403 `scale_x = crop_width / bbox_width` / `scale_y = crop_height / bbox_height`, then :409-419 the no-polygon branch emits the FULL crop rectangle as the line boundary. But the crop it is describing was expanded by crop_agent.py:54 `pad_y = max(10, int(round(region_h * 0.45)))`, and crop_agent.py:104 `return region_id, encode_png_base64(crop)` returns only the PNG — the expanded rectangle is never returned, so ocr_backends cannot know about the padding. ocr_agent.py:1055-1056 passes the ORIGINAL page-coordinate bbox/polygon as metadata while passing the EXPANDED crop as crop_b64.

MEASURED, page e-codices_fmb-cb-0002_001r_max (6129x8174), 20 real line regions from its own outputs/.../layout_coco.json, McCATMuS, kraken pad=16:
  _local_boundary_from_metadata returns boundary y-range 0..633 for a line whose true bbox height is 334px = 1.89x too tall (identical 1.90x for the polygon branch — the polygon gets stretched to fill the padded crop).
  _baseline_from_boundary (:440 `y = top + 0.78*(bottom-top)`) then puts the baseline at y=494 when the true baseline is at y~411: 83px low on a 334px line.

Boundary-height sweep, grayscale, upscale=1, CER vs the repo's own gold file, 8 lines:
  0.85x bbox -> CER 0.3777  conf 0.9792  326 chars
  1.00x bbox -> CER 0.3591  conf 0.9656  324 chars
  1.15x bbox -> CER 0.4025  conf 0.9646  317 chars
  1.30x bbox -> CER 0.4520  conf 0.9461  318 chars
  1.60x bbox -> CER 0.5851  conf 0.9293  312 chars
  1.898x bbox (WHAT THE CODE DOES) -> CER 0.8762  conf 0.5602  177 chars

Through the real production entry point KrakenBackend.recognize() (upscale=2, full preprocessing): CER=0.7771, mean_conf=0.8802, 209 chars.

Replicated on a second page, e-codices_fmb-cb-0001_001r_max (6132x8176), 10 line regions: boundary 1.000x -> conf 0.9777 / 260 chars; 1.898x (today) -> conf 0.4245 / 90 chars, i.e. 65% of characters lost.
```

- **What it costs today:** Kraken's baseline line extractor dewarps and rescales the boundary band to the model's trained input height of 120px (measured: nn.input == (1,1,120,0) for mccatmus, catmus_medieval and cremma_medieval). A band 1.9x too tall means the glyphs occupy only ~63 of the 120 rows the model was trained for, and the baseline is placed 83px below the real one, so the band is also vertically off-centre. This is the single largest accuracy loss in the pre-recognition path: measured CER 0.7771 on the production path versus 0.3591 with a bbox-tight boundary — a 54% relative CER reduction available from one geometry fix. It also cascades into retrieval: every downstream stage (quality gate, proofreader, authority linking, RAG chunks) indexes this text, so a 0.78-CER transcription cannot be retrieved against.

- **Recommendation:** Make crop_region return the crop rectangle it actually used (it already computes x1,y1,x2,y2 at crop_agent.py:99) and thread it into OCRRecognitionMetadata as a crop_offset/crop_box. Then _local_boundary_from_metadata must map page coords into crop coords with that offset — `(px - crop_x1) * upscale` — instead of `(px - bbox_x1) * crop_width/bbox_width`. The boundary should be the real bbox/polygon band (the sweep bottoms out at 0.85-1.00x bbox height), with the surrounding pad present in the pixels but NOT inside the boundary. Do not shrink crop_agent's padding to fix this: Kraken needs the context pixels around the polygon, it just must not be told they are part of the line.

### Root CLI is materially healthier than the vendor backend: crop_padding=5 does not clip, grayscale conversion is exactly what the models want, and delegating line segmentation to Kraken's blla beats the vendor path by 0.11 confidence on the same page

- **Where:** `src/archai_ocr/pipeline/htr_kraken.py:110`
- **Gain / effort:** large / medium — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
HEAD-TO-HEAD, EXECUTED on the same MainZone of e-codices_fmb-cb-0001_001r_max with the same McCATMuS model:
  A root-CLI path (crop -> im.convert('L') at htr_kraken.py:80, then blla.segment at htr_kraken.py:110, rpred default pad):
     35 lines, 927 chars, mean_conf 0.9688, readable Old French: 'fr es gent yl los seignoz moroient' / 'I sou huitai ten fuivoient' / 'ler r li vilain p ovoit ne cent'
  B vendor backend path (crop_agent upscale=2 + _preprocess_kraken_crop_with_metadata + inflated boundary):
     35 lines, 722 chars, mean_conf 0.8598, gibberish: 'df des sen preslegnn mersten' / 'Ey SonEuan CnEauent' / 'Lad ERenByrNtet'

crop_padding=5 does NOT clip ascenders — EXECUTED padding sweep with real blla+rpred on the MainZone crop:
  padding=5   -> crop 2067x6359, 35 lines, 927 chars, conf 0.9688
  padding=20  -> 36 lines, 923 chars, conf 0.9699
  padding=60  -> 35 lines, 925 chars, conf 0.9730
  padding=120 -> crop 2297x6589, 35 lines, 920 chars, conf 0.9683
No meaningful difference (spread 0.0047 conf, 7 chars), because blla re-segments and dewarps each baseline inside the crop.

Grayscale is correct, not lossy: MEASURED one_channel_mode == 'L' on all three models, so crop_regions.py's PNG + htr_kraken.py:80 `im.convert('L')` matches the trained input. crop_regions.py:36 also applies EXIF orientation, which the vendor path does not.
Absence of global deskew in the root CLI is likewise not a cost here: blla emits per-line baselines and Kraken dewarps along them.

For contrast, EXECUTED: feeding a whole 5212x6204 MainZone to KrakenBackend.recognize() returns 1 character ('5', conf 0.987) because the inflated-boundary code treats the entire zone as one text line.
```

- **What it costs today:** This is the counterfactual that sizes the other findings: the same model on the same pixels yields conf 0.9688 and readable text under the root CLI's approach versus 0.8598 and gibberish under the vendor backend's. It says the accuracy is available today and is being destroyed by the vendor backend's crop/preprocess/boundary layer, not by the models, the weights, or the scans. It also rules out three suspected root-CLI defects (crop padding clipping, lossy grayscale, missing deskew) so effort is not spent there.

- **Recommendation:** Adopt the root CLI's shape inside the vendor backend: crop the layout ZONE (not per-line boxes), convert to grayscale, and let blla.segment produce the line boundaries and baselines, replacing _local_boundary_from_metadata / _baseline_from_boundary entirely. That removes findings 1, 2 and 4 at once instead of patching each. Keep the layout regions for reading order and zone typing, not for line geometry. Also guard KrakenBackend.recognize against zone-sized single-line boundaries (a boundary whose height exceeds ~2x the median line height should be rejected or re-segmented, not squashed to 120px).

### Every text-bearing zone except MainZone is filtered out before OCR — marginalia, glosses, running titles, catchwords, quire marks and foliation are never transcribed by either pipeline

- **Where:** `vendor:app/routers/ocr.py:211`
- **Gain / effort:** large / medium — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
`_TEXT_LABEL_EXCLUDE_TOKENS = ("border", "column", "table", "diagram", "illustration", "graphic", "music", "zone", "gloss", "header", "catchword", "page number", "quire")` (ocr.py:211-224) is checked first in `_is_relevant_text_label` (ocr.py:2165-2171) and gates both `_extract_location_suggestions` (ocr.py:2212) and `_extract_segmented_regions` (ocr.py:2376). MarginTextZone maps to "Gloss" (constants.py:11), so it is excluded twice over — by "gloss" and by "zone". The root CLI excludes the same content by configuration: `main_text_class: MainZone` (config.example.yaml:16, config.py:24) with a case-folded exact-match filter at layout_yolo.py:116/125. MEASURED with the real pipeline: on 'Latin script.jpg' the classes rejected by _is_relevant_text_label are {'Border': 1, 'Column': 1, 'Gloss': 2, 'Table': 1, 'Page Number': 1} of 83 annotations. Zone-model MarginTextZone fires on 4/20 unique real images at conf 0.25, and on .tasks/*/352386106_MS0001_0008_3000x2250.jpg it returns 6 MarginTextZone against only 3 MainZone. Recall cost measured with full-page Kraken blla (blla.mlmodel) baselines vs the MainZone boxes: 'Latin script.jpg' 58 of 101 lines (57%) fall outside every kept zone, 29 of them on the facing page of the spread; e8a6872... 9/36 (25%); 'Old English.jpg' 4/26 (15%). Marginalia recall is also the one thing confidence_threshold controls: MarginTextZone confidences on 'Latin script.jpg' are 0.642, 0.578, 0.193, 0.052 -> 2 boxes at 0.25, 3 at 0.10, 4 at 0.05, whereas MainZone sits at 0.953 with the next candidate at 0.008, so MainZone is unchanged across conf 0.05-0.40 on all 7 pages tested.
```

- **What it costs today:** The scholarly apparatus — the interlinear and marginal glosses, running titles, catchwords, quire signatures and foliation that carry most of the retrievable metadata about a codex — is structurally unreachable. For knowledge retrieval this is not a CER penalty but missing documents: a query about a gloss or a quire mark can never be answered because that text was never in the index. Up to 57% of a page's detected text lines on a spread.

- **Recommendation:** Split the label filter into a main-text lane and a secondary-text lane rather than one allow/deny list: keep MainZone-derived regions as the primary reading-order stream, and OCR MarginTextZone/Gloss, RunningTitleZone, NumberingZone, QuireMarksZone and Catchword into separately tagged fields so they reach the index without interleaving into the body text. In the CLI, set layout.main_text_class to [MainZone, MarginTextZone] (the list form and case-folding already work, layout_yolo.py:116) and emit the marginal regions to their own output section. Lower confidence_threshold to ~0.10 only for the marginal classes — it buys nothing for MainZone.

### No abbreviation expansion exists; normalize_for_search deletes the abbreviation mark and passes MUFI abbreviation letters through verbatim

- **Where:** `vendor:app/services/text_normalization.py:244`
- **Gain / effort:** large / medium — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
normalize_for_search() step 3 is `nfkd = unicodedata.normalize("NFKD", text); text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))` (lines 256-257). Executed against the recognizers' own alphabets. MEASURED, loading the real .mlmodel codecs via kraken.lib.models.load_any: mccatmus alphabet=116 symbols / 11 combining marks, catmus_medieval=224 / 32, cremma_medieval=98 / 16; **all three contain 0 precomposed letters**, and catmus_medieval can emit 14/19 tested scribal abbreviation signs (U+0304 macron, U+0303 tilde, U+0363/0365/0367/036C superscript a/i/o/r, U+017F long s, U+A751 ꝑ per/par, U+A753 ꝓ pro, U+A757 ꝗ qui, U+A759 ꝙ, U+A76F ꝯ con, U+A770 ꝰ us, U+204A ⁊ Tironian et, U+00E6 æ). MEASURED output of normalize_for_search: 'd̄n̄s'->'dns', 'ōīm'->'oim', 'q̄'->'q', 'i̅'->'i' (mark silently deleted, no expansion); 'ꝑsona'->'ꝑsona', 'ꝯdit'->'ꝯdit', 'noꝛmam'->'noꝛmam', 'ꝓsit'->'ꝓsit' (passed through, no expansion). MEASURED corruption: NFKC maps U+A770 (MODIFIER LETTER US) to U+A76F (LATIN SMALL LETTER CON), so 'eiꝰ' (eius) becomes 'eiꝯ' (ei+con) in every search key. MEASURED spelling folding: over 21 diplomatic-vs-modern pairs (Iohannes/Johannes, seruus/servus, uita/vita, ciuitas/civitas, michi/mihi, clemencia/clementia, presbiteri/presbyteri, ecclesie/ecclesiae, Iherusalem/Jerusalem ...) normalize_for_search collapses **0/21 = 0%** to the same key. _OCR_CONFUSION_MAP (line 86) contains exactly three letter-alternation rules in total: 'vn'->'un', 'vne'->'une', 'ieh'->'jeh'. `grep -rE 'abbrev|expand_abbr|macron|rotunda|0304|0305|017f' --include='*.py' app/` returns only LLM prompt strings, never an expansion function.
```

- **What it costs today:** Every abbreviated token — which in medieval Latin is 15-30% of all tokens and includes the most frequent words (dominus, omnium, per, pro, qui, con-, -us, et) — enters the index and the authority-linking key in a form no query token can ever equal. The macron case is worse than no handling: the mark is deleted, so 'd̄n̄s' and 'dns' and a genuinely-read 'dns' become indistinguishable, and 'ꝰ'→'ꝯ' actively converts the us-abbreviation into the con-abbreviation. Lexicon matching, Wikidata surface search (wikidata_client.py:303-305 calls normalize_unicode+ocr_confusion_fixes on the query) and retrieval recall all fail on exactly the tokens a paleographer would search for.

- **Recommendation:** Add an expansion layer between recognition and indexing that (a) maps the MUFI/Unicode abbreviation letters to their expansions with a marker (ꝑ->per|par, ꝓ->pro, ꝗ->qui, ꝯ->con, ꝰ->us, ꝛ->r, ⁊->et), (b) resolves combining macron/tilde by context (V+U+0304 -> Vn/Vm, q̄->quod/que, consonant+bar -> the standard suspension set) instead of deleting it, (c) folds the medieval orthographic alternations that actually block matching: u/v, i/j, i/y, ci/ti, e/ae/oe, ch/h, single/double consonant. Emit BOTH keys per token (diplomatic + expanded) rather than replacing. Remove the NFKC step, or special-case U+A770 before it, so the us-abbreviation is not rewritten to con.

### The only "gold" file is a second machine run, not ground truth — measured 48.4% CER / 95.8% WER between the two

- **Where:** `outputs/e-codices_fmb-cb-0002_001r_max/e-codices_fmb-cb-0002_001r_max_gold.txt:9`
- **Gain / effort:** large / medium — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
The file carries the pipeline's own report header: line 7 `Recognizer: McCATMuS (...McCATMuS_nfd_nofix_V1.mlmodel)`, line 9 `Preprocess: pipeline=B (grayscale+nlbin+conditional_upscale)`, line 12 `Quality score: 0.9578`. Its pair `e-codices_fmb-cb-0002_001r_max.txt` is the same page, same model, same image SHA256 (1080cd11...), differing only at line 9 `pipeline=A (grayscale+autocontrast+conditional_upscale+mild_sharpen)`. The last line ends with the literal two-character escape `\n\n[manual correction marker]` rather than a real newline — a hand-appended placeholder, and no code in the repo writes that string (`grep -rn 'manual correction marker' --include=*.py` → 0 hits; `grep -rn 'ArchAI HTR Extraction' --include=*.py` → 0 hits, so even the report writer's source is gone). MEASURED with my own Levenshtein harness after stripping the report header and zone labels: aligned 20-line 'Main script black' zone → 428 char edits / 885 ref chars = CER 48.36%, 137 word edits / 143 ref words = WER 95.80%; whole body (30 vs 20 lines) → CER 63.96%, WER 97.09%; only 10 of 198 distinct reference words appear verbatim in the hypothesis. Per-line CER ranges 31.71%–68.75%.
```

- **What it costs today:** Two runs of the same page with the same recognizer, differing only in the preprocessing pipeline, agree on 52% of characters and 4% of words. Nobody can tell which is better, because neither is ground truth. Every other improvement in this project — model choice, preprocessing A vs B, reading order, seam retry, proofreading — is currently unfalsifiable: there is no reference against which a change can be shown to help rather than hurt. `grep -rnwiE 'cer|wer|char_error|word_error' --include=*.py` over the whole repo returns exactly four hits, all of them disclaimer strings in scripts/build_thesis_showcase_payloads.py (lines 555, 574, 626) and its copy — zero CER/WER computation. `grep -rniE 'recall@|mrr|ndcg|precision@|hit@'` over .py/.md/.json/.csv/.tex returns zero real hits (only three base64 substrings inside package-lock.json). Levenshtein does exist five times in the codebase but exclusively as internal fuzzy string matching, never as an evaluation metric: routers/ocr.py:727 `_levenshtein`, services/text_normalization.py:332 `normalized_edit_distance` (token-vs-lexicon matching at ocr.py:1585-1652), services/authority_linking.py:907 `_levenshtein_quick` (authority alias matching), agents/ocr_proofreader_agent.py:213 `_normalised_levenshtein` (proofreader diff guard), services/ocr_quality.py:530 `normalized_levenshtein_similarity` (cross-pass stability). None of these ever sees a human reference.

- **Recommendation:** Build the minimal OCR harness. Smallest credible gold set: 10 pages × ~20 lines = ~200 lines, drawn from the 14 distinct real manuscript pages already on disk under archai/vendor/layout/backend/.tasks/*/ (e-codices_fmb-cb-0001_001r_max, fmb-cb-0002_001r_max, sbe-0027_001v_max, acv-P-Antitus_001r_max, plus the three canonical showcase pages Latin.png / Old English.jpg / french.JPEG). Transcribing them is cheap because the per-region crops are already rendered: outputs/<page>/debug/region_NNN_preproc.png (21 files for fmb-cb-0002), and outputs/<page>/*_summary.json already contains `region_bbox_map` (region_id → [x1,y1,x2,y2]) plus `sections['Main script black']['columns']` (ordered line text), while outputs/<page>/region_cache/<sha>.json holds per-line `{text, confidence}`. Concretely add: (1) `eval/gold/manifest.json` — one row per page: {page_id, image_path, image_sha256, language, script, zone_labels, gold_path, split: train|heldout}; (2) `eval/gold/<page_id>.lines.txt` — one diplomatic line per row, in reading order, aligned to region_bbox_map ids; (3) `eval/metrics.py` — pure-Python `cer(ref, hyp)` / `wer(ref, hyp)` (no new dependency needed; jiwer/rapidfuzz/editdistance/pycocotools are all absent from pyproject.toml) with an explicit, documented normalisation spec: NFC vs NFD (McCATMuS is an `nfd_nofix` model, so this choice alone moves CER), case folding on/off, punctuation, and whether abbreviation marks are expanded — pin one and record it in the manifest, because an unspecified normalisation makes reported CER unreproducible; (4) `eval/run_ocr_eval.py` — reads the manifest, runs the pipeline per page, writes `eval/results/<git-sha>.json` with per-page and per-line CER/WER plus the recognizer SHA256 the report header already emits; (5) a CI job that fails if held-out mean CER regresses by more than a stated margin. Report the baseline as a range, not a point, until at least 10 pages are in.

### The layout evaluation harness cannot be imported at all, and every metric it has ever recorded is 0.0 with an error string — across 3,874 ground-truth annotations

- **Where:** `archai/vendor/layout/compare/data/old_models.py:11`
- **Gain / effort:** large / medium — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`import pycocotools.mask as mask_util` — an unguarded top-level import. compare.py:15-20 carefully guards its own pycocotools import (`except ImportError: HAS_PYCOCOTOOLS = False`), but compare.py:34 then does `from old_models import process_dataset as process_old_models`, which re-enters the unguarded import. MEASURED by executing `sys.path.insert(0,'.'); import compare` in the vendor venv: `ModuleNotFoundError: No module named 'pycocotools'` at old_models.py:11 — the guard is dead code and the whole harness is un-runnable. pycocotools is absent from both venvs and from pyproject.toml. Worse, the recorded outputs show it never worked even when it ran: expert_datasets_model_comparison_summary.json has 6 rows and sample_batches_model_comparison_summary.json has 3, and all 9 report `mAP@50: 0.0, mAP@[.50:.95]: 0.0, Precision: 0.0, Recall: 0.0, F1: 0.0` with an error — 'segm error: 'score', bbox error: 'score'' (new models), 'No valid prediction annotations' (old models), 'list index out of range' (sample batches), 'No valid GT annotations' (Luise 1). Summing gt_annotations across those rows: 44+136+0+99+64+25+1437+909+1160 = 3,874 human annotations, and mAP was computed as 0.0 for every one of them.
```

- **What it costs today:** The layout model gates everything downstream — crops feed the recogniser, so a missed or merged region is an unrecoverable OCR loss, and the previously-fixed 'default layout class matched nothing' defect shows how expensive a silent layout failure is. Yet the layout model has never been scored: the only harness in the repo is un-importable, and its only recorded results are nine zeros with error messages. There is no mAP, no IoU, no region recall number in this project that came from running code. This is why the 'multi-column reading order' and 'default layout class' fixes could only be validated structurally, not quantitatively.

- **Recommendation:** Guard the import at old_models.py:11 the same way compare.py:15 does, add pycocotools to the dependency set, and make calculate_metrics raise rather than return a 0.0-with-error dict — a metric of 0.0 that means 'the harness broke' is worse than a crash, because it silently entered a summary file and looks like a result. Then fix the actual root cause in item 5. The good news, which I verified: the label vocabularies already line up — ground_truth_coco.json has 22 categories and the pipeline's own outputs/*/layout_coco.json emits 25, sharing 21 of the 22 (only 'Ignore' is GT-only; 'Column', 'GraphicZone', 'MusicLine', 'MusicZone' are pipeline-only), so mAP is computable with no category-mapping work once scores and images exist.

### Retrieval accuracy is measured only against a fake substring-matching store with a corpus of one document

- **Where:** `archai/vendor/layout/backend/tests/test_rag_entity_index.py:57`
- **Gain / effort:** large / medium — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
The test's `_FakeCollection.query` scores by literal substring containment: `score = 0.0 if query and query in text.lower() else 0.9`, then `hits.sort(key=lambda item: item[0])`. The two retrieval tests assert only presence and cardinality: line 181 `assert entity_hits`, line 182 `assert entity_hits[0]['entity_id'] == 'wikidata:Q45720'`, line 215 `assert hits`, line 217 `assert chunk_collections['provider_1024'].count() == 1`. MEASURED: corpus size is 1, so recall@k is trivially 1.0 and no ranking can be wrong. The real ranking surface exists and is well shaped for evaluation — services/rag_store.py:576 `retrieve_chunks(query, *, top_k, run_ids, asset_ref)`, :641 `retrieve_entities`, :750 `retrieve_debug` which already returns `'score': round(1.0 - float(h.get('distance', 0)), 4)` at line 769 and 785 — but no test ever calls it with a real embedding backend, and `grep -rn 'retrieve_chunks'` in tests/ shows the only other uses are in test_chat_api.py (lines 183, 242, 345) where it is monkeypatched to return `[]`. `grep -rniE 'recall@|mrr|ndcg|precision@|hit@'` across the repo: zero real hits.
```

- **What it costs today:** Knowledge-retrieval accuracy is completely unmeasured. There is no query set, no expected-span annotations, no recall@k, no MRR, and the only test substitutes a substring matcher for the embedding model — so the actual embedding provider, chunk size, top_k default (settings.rag_top_k) and entity-vs-chunk blend have never been compared against each other on any task. A regression that silently drops the correct chunk from the top-k would pass CI, because CI's assertion is `assert hits`.

- **Recommendation:** Add the smallest credible retrieval gold set: 20–30 queries over the same 10 eval pages, each row `{query_id, query, page_id, expected_chunk_ids: [...], expected_span: {start, end}, kind: factual|interpretive}`. Seed it from real artifacts rather than inventing it — artifacts/thesis_showcase/figure25/comparison.json already contains one selected question with its evidence spans and chunk_ids, and artifacts/thesis_showcase/figure30/examples.json has 6 mention-handling examples; both were produced by the live pipeline, so the chunk_ids are real. Concretely add: `eval/retrieval/queries.json` (the rows above), `eval/retrieval/run_retrieval_eval.py` calling rag_store.retrieve_chunks at fixed top_k ∈ {1,3,5,10} and reporting recall@k, MRR and mean reciprocal position per query kind, and results into eval/results/. Report recall@5 and MRR as the two headline retrieval numbers. Critically: run it against the real embedding backend, not the fake — a corpus of at least a few hundred chunks (10 pages × ~25 lines chunked) is the minimum at which ranking errors become observable.

### No consensus or voting exists: selection is first-wins with an early break, and comparison backends are telemetry-only

- **Where:** `vendor:app/agents/ocr_agent.py:1212`
- **Gain / effort:** large / high — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`if backend_result.text.strip() and current_quality >= payload.options.quality_floor: break` — the attempt loop (1165-1219) exits on the first candidate over the floor. Every `OCRComparisonResult` built from `plan.comparison_backends` is constructed with `selected=False` hardcoded (lines 1233, 1247, 1264, 1278, 1360, 1374) and never feeds back into the chosen text. `compare_backends` defaults to `[]` (schemas/agents_ocr.py:80), so by default no comparison backend runs at all. Repo-wide grep for vot(e|ing)/consensus/rover/ensembl/majority in the backend returns only a script-hint majority vote in saia_ocr_agent.py:2456-2472 and an unrelated test-fixture class — nothing on OCR text. `agreement_score(texts)` — the exact multi-hypothesis agreement primitive needed — exists at services/lexicon_trust.py:153 and has ZERO callers. MEASURED inter-model diversity on the same 20 real crops (character edit ratio): catmus vs mccatmus 0.540, catmus vs cremma_medieval 0.426, mccatmus vs cremma_medieval 0.582 — i.e. the hypotheses disagree on roughly half of all characters, which is where voting headroom lives.
```

- **What it costs today:** Three Kraken models plus Calamari and VLM backends are installed and loadable, they disagree on ~50% of characters, and the pipeline consumes exactly one of them chosen without evidence. MEASURED page-level truth proxy (Old English function-word hit rate, this page is Aelfric's Matthew 20 vineyard homily): CATMuS 0.229, cremma_medieval 0.057, McCATMuS 0.043 — a 5.3x spread that the pipeline is structurally unable to see.

- **Recommendation:** Sequence matters, and I measured that naive voting BACKFIRES with today's signals: per-region max-_region_quality_value scored 0.165, max-confidence 0.184, max-agreement-to-others 0.168, unweighted 3-way character ROVER 0.133 — all WORSE than simply always using CATMuS (0.229), and the per-region oracle also equals 0.229. So: (1) fix the selector signal (item 1); (2) then do page-level model election — run all attempt backends on a 5-6 region sample and elect one model for the page (the `_reorder_attempt_backends_from_samples` scaffolding at ocr_agent.py:1022 already does the sampling, it just ranks with the constant score); (3) only after (1) lands, add confidence-weighted per-line ROVER, weighting each system by its measured page-level score rather than equally.


# MODERATE EXPECTED GAIN (27)

### Candidate recall: _MAX_ENRICH_PER_MENTION=3 means only the first 3 Wikidata hits ever get P31, so any correct entity at search rank 4+ is forced type_compatible=False — and the relevance sort that was supposed to prevent this is computed and then thrown away

- **Where:** `vendor:app/services/authority_linking.py:744`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
`_MAX_ENRICH_PER_MENTION = 3  # max wbgetentities calls per mention` (authority_linking.py:744), gating the only source of instance_of_qids (authority_linking.py:1768). Real cached Wikidata results show the correct entity routinely sits at rank 4-5: query 'holy grail' -> Q162808 Holy Grail at rank 5; query 'tint' -> Q470697 Tinténiac (commune in France) at rank 4; query 'lancelot' -> Q215681 'Arthurian character' at rank 3 (only just inside the cap, behind 'male given name' and 'family name'). Executed offline with a 5-hit fixture where the correct place Q807 Lausanne is at rank 5:
  wbgetentities was called for: ['Q669678','Q309456','Q658975'] (3 of 5)
  status: unresolved | best candidate type_incompatible (score=0.321, qid=Q669678)
    0.3214 Q669678 Lausanne railway station tc=False P31=['Q55488']
    0.2534 Q807    Lausanne                 tc=False P31=[]      <- correct entity, unenriched
Lifting ONLY the cap (al._MAX_ENRICH_PER_MENTION = 99), same fixture, same code:
  wbgetentities called for all 5; status: unresolved | "best score (0.679) < threshold (0.8)"
    0.6794 Q807 Lausanne tc=True P31=['Q515','Q1549591']   <- now rank 1 and type-compatible
Separately, _prefilter_candidates (authority_linking.py:1175) sorts by relevance and its docstring at :1183-1185 promises "so the enrichment cap processes the best candidates first", but authority_linking.py:1753-1758 keeps only `allowed_keys` as a SET and rebuilds `candidates` in dict-insertion order. Measured on the same fixture:
  _prefilter order        : ['Q807','Q669678','Q309456','Q658975']   <- correct entity first
  order actually enriched : ['Q669678','Q309456','Q658975']
And _prefilter_candidates is only called for ent_type in {person, work} (authority_linking.py:1747), so place/org/event mentions get neither the reordering nor the modern-occupation noise filter.
```

- **What it costs today:** Wikidata's wbsearchentities ranks 'male given name' and 'family name' items above the real character (measured on the real cache for 'lancelot'), so the 3-call budget is systematically spent on exactly the rows that _NAME_ENTITY_QIDS then rejects. The correct entity is retrieved by search but never type-resolved, so it is hard-gated even when its label matches exactly. This is a pure recall loss that no threshold change can recover.

- **Recommendation:** Two-line fix with a measured effect: use the ORDER returned by _prefilter_candidates (assign the filtered+sorted list back into `candidates` instead of deriving a set of allowed_keys), and call it for every ent_type, not just person/work. Then raise _MAX_ENRICH_PER_MENTION to top_k (5) — wbgetentities accepts up to 50 ids in one request via `ids=Q1|Q2|...`, so enriching all candidates costs ONE extra HTTP call per mention, not five. Note the demo above shows this fix alone still leaves Q807 at 0.6794 < 0.80, so it must ship together with the AL-01 recalibration to convert into an actual link.

### AL-06 CONFIRMED: rescore_with_canonical clamps to the fixed constants 0.92/0.90/0.88, destroying a real 0.177 margin and turning a linkable mention into 'ambiguous'

- **Where:** `vendor:app/services/entity_scoring.py:297`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
entity_scoring.py:295-301 sets `cand["score"] = max(cand.get("score",0.0), boost)` where boost is 0.92 (domain hit) / 0.90 / 0.88 — constants, not functions of the candidate. Executed with two distinct plausible Arthurian candidates, both label_sim>=0.95 vs the canonical form and both with a medieval-domain description, whose scorer outputs differed by a healthy margin:
  after rescore: [('Q215681', 0.92), ('Q3825066', 0.92)]
  disambiguate(HIGH) -> ambiguous | "margin too small (0.000 < 0.15) between top two candidates (scores: 0.920, 0.920)"
  pre-clamp scores were 0.5979 vs 0.4211 (margin 0.1768); post-clamp margin = 0.0000
The pre-clamp margin of 0.1768 EXCEEDS MIN_MARGIN=0.15, so absent the clamp this mention would have resolved cleanly to Q215681. Because the clamp is max(), a genuinely better candidate cannot pull ahead either:
  [('QGOOD', 0.92), ('QBAD', 0.92)] -> ambiguous   (pre-clamp 0.79 vs 0.10)
The clamp is also the ONLY thing producing links in production: all 14 'linked' scored rows in app/archai.sqlite sit at exactly 0.9200.
```

- **What it costs today:** The one code path that can clear AUTO_SELECT_THRESHOLD simultaneously erases the evidence disambiguate needs to pick between candidates. Any canonical surface with two or more medieval-domain namesakes (the normal case on Wikidata — 'lancelot' alone returns a given name, a family name, an Arthurian character, a TV character and a video game) collapses to a zero margin and is reported ambiguous, requiring human adjudication for a decision the scorer had already made correctly.

- **Recommendation:** Make the canonical signal additive and order-preserving instead of a clamp: add a bounded bonus (e.g. `cand["score"] = min(1.0, cand["score"] + 0.25 if label_sim>=0.95 and domain_hit else +0.20 ...)`), or set score = max(score, boost) but break the resulting tie on the pre-clamp score by storing it (`cand["score_preclamp"]`) and having disambiguate compute the margin on the pre-clamp values. Once AL-01 is recalibrated the clamp should be deleted entirely — it exists only to punch through a threshold the scorer cannot reach.

### AL-04 CONFIRMED: a failed wbgetentities call is cached permanently as an empty enrichment, blacklisting that QID forever — the "empty results are never cached" guard is defeated by wrapping the empty dict in a list

- **Where:** `vendor:app/services/wikidata_client.py:467`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
_http_get returns {} on HTTPError or any exception (wikidata_client.py:243-250). enrich_wikidata_item then builds an all-empty result and unconditionally calls `cache_put("wikidata_enrich", cache_key_str, [result])` (wikidata_client.py:467). cache_put's guard is `if not results: return` (wikidata_client.py:215) — but `results` is `[{...}]`, truthy, so the poison row is written. The read at wikidata_client.py:404 is `cache_get("wikidata_enrich", cache_key_str)` with NO max_age_hours, so the stale-negative expiry at wikidata_client.py:145-155 never applies. Executed against a fresh cache file:
  1st call (network failed) -> {"viaf_id":"","geonames_id":"","instance_of_qids":[],"canonical_label":"","description":"","aliases":[],...}  http calls: 1
  2nd call (network HEALTHY) -> identical empty dict;  http calls total: 1  <-- never hit the network
  cache row persisted: [{"instance_of_qids": [], "aliases": [], ...}]
  consequence: is_type_compatible('place'|'person'|'org'|'work', [], '') = False for all four
This is not hypothetical — it is already degrading the live cache. All 50 wikidata_enrich rows in app/.data/wikidata_cache.sqlite have canonical_label=None and description='' and no aliases (they predate the current enrichment shape and can never be refreshed). Measured effect in the end-to-end run against that real cache: Q215681 scored 0.5979 with alias_match_quality=0.0, versus 0.7179 for the identical candidate with its aliases present — a 0.12 score loss caused purely by unrefreshable cache rows.
```

- **What it costs today:** One transient 429/500/timeout from Wikidata permanently converts a QID into an untyped, alias-less, description-less candidate that is hard-gated for every entity type. Because Wikidata rate-limits and the pipeline fires up to 3 wbgetentities per mention with only a 0.25s delay (_REQUEST_DELAY_S), partial failures are expected, and each one silently and irreversibly removes a correct entity from the reachable set. Two independent hard gates then fire on it (type_compatible=False, alias_match_quality=0), so the damage is invisible in the logs and looks like a scoring miss.

- **Recommendation:** Do not cache a failed fetch. Detect failure explicitly — `entity = data.get("entities", {}).get(qid, {})`; if `not entity` (or the response carries an 'error' key), return {} WITHOUT calling cache_put, and let the caller retry. Additionally pass `max_age_hours` on the enrich read at wikidata_client.py:404 so any already-poisoned row expires, add a schema/version stamp to cached enrichments so rows written by an older shape are refetched, and one-time purge the 50 degraded rows (`DELETE FROM wikidata_cache WHERE source='wikidata_enrich' AND result_json LIKE '%"aliases": []%'`). Tighten cache_put's guard to reject a list whose only element is an empty-valued dict.

### The fallback embedder is all-MiniLM-L6-v2, an English-only model, and it measurably ranks medieval Latin and Old French by language rather than by meaning

- **Where:** `vendor:app/services/rag_store.py:35`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
`_LOCAL_EMBED_BACKEND = "local_default"` and the local path passes `query_texts=[query]` (rag_store.py:629-635) so Chroma's DefaultEmbeddingFunction embeds both documents and queries. Confirmed identity by execution: DefaultEmbeddingFunction -> dim 384, L2 norm 1.0; site-packages/chromadb/utils/embedding_functions/onnx_mini_lm_l6_v2.py:38 `MODEL_NAME = "all-MiniLM-L6-v2"`. This is the space holding 214 of the 1269 indexed chunks in the live store. MEASURED probe with that exact embedder over four hand-written passages (Latin land-grant, Latin theft record, Old French land-grant, Old French love line): English query 'which king granted land to an abbey?' ranked LAT_theft 0.171 ABOVE LAT_king 0.090 (wrong); English query 'who stole cattle?' ranked LAT_king 0.242 ABOVE LAT_theft 0.185 (wrong) — 2/2 cross-lingual queries put the wrong passage first. cos(LAT_king, LAT_theft) = 0.391 > cos(LAT_king, OF_king) = 0.381: two topically unrelated Latin passages are more similar to each other than a Latin passage is to its own Old French equivalent. cos(OF_king, OF_love) = 0.520 > cos(OF_king, LAT_king) = 0.382. Separately, the configured provider model 'multilingual-e5-large-instruct' (config.py:85) is used with NO E5 prefixes anywhere — grep for 'query:', 'passage:', 'Instruct:' across rag_store.py and chat_ai.py returns nothing; _provider_embed sends raw text for both documents (rag_store.py:429/144) and queries (rag_store.py:603). E5-family models are trained with asymmetric 'query: ' / 'passage: ' prefixes and the -instruct variant expects an instruction-prefixed query; that part is inferred from reading plus the published model contract, not measured here (measuring it needs the GWDG endpoint).
```

- **What it costs today:** For the 214 chunks in the local space, and for every query during a provider outage, the retriever is scoring 'how Latin-shaped is this' instead of 'does this answer the question' — the two English probes both retrieved the wrong passage. On the provider path, dropping the E5 prefixes costs the asymmetric query/passage calibration the model was trained for, which typically shifts retrieval quality by several points on the model's own benchmarks.

- **Recommendation:** Never fall back to an English-only model for this corpus. Ship a local multilingual encoder (multilingual-e5-small/base run locally, or LaBSE) as the fallback so the space is at least language-appropriate, and add the E5 prefixes: 'passage: ' on documents and 'query: ' (or the -instruct 'Instruct: <task>\nQuery: ' form) on queries, applied in _provider_embed so index and query paths cannot diverge. Include the prefix scheme in the backend slug so a prefix change forces a reindex rather than silently mixing calibrations.

### The new OCR quality gate is bypassed at retrieval time: retrieve_chunks silently indexes runs the pipeline refused to index

- **Where:** `vendor:app/services/rag_store.py:551`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
_ensure_chunk_runs_indexed: `if not run_ids or not settings.rag_auto_index: return` then `for run_id in run_ids: if _collection_has_run(...): continue; run = pipeline_db.get_run(...); _index_chunks_for_run(...)` — no consultation of the quality report or of token_search_allowed. Same hole in _ensure_entity_runs_indexed (rag_store.py:564). It runs on every retrieval (rag_store.py:604 and :651), and chat_ai.py:1086 calls retrieve_chunks(query, run_ids=run_ids) on every chat turn that carries an ocr_run_id. The pipeline itself gates correctly — ocr.py:3107-3110, ocr.py:3532-3535 and ocr.py:4602-4607 all log 'RAG auto-index SKIPPED: token_search_allowed=False' — so the gate exists and is simply routed around. EXECUTED end to end in an isolated DB + Chroma dir: garbage OCR text -> compute_quality_report gives quality_label RISKY, enforce_quality_gates gives token_search_allowed=False, ner_allowed=False; 15 chunks written to SQLite; pipeline skips indexing so 'chroma count BEFORE any retrieval: 0'; one call to rag_store.retrieve_chunks('what does this page say?', run_ids=[blocked_run]) -> 'chroma count AFTER: 15' and 5 hits returned, e.g. retrieval_score 0.1735 'lapoploi ly popepraps relilarops Tpapphes' (3 of the 5 identical). The live entity store shows this has already happened in production: all 27 entity docs are record_type 'unresolved_mention' and carry 'reason_unresolved: token_search_allowed=False quality=RISKY'.
```

- **What it costs today:** Every transcription the freshly-fixed gate classifies as RISKY or UNRELIABLE still reaches the LLM as citable [OCR_CHUNK_EVIDENCE] with a run_id, chunk_id and offsets, the moment a user asks anything about that page. The gate's guarantee — that unreliable text does not become searchable knowledge — does not hold, so the gibberish work does not translate into retrieval accuracy.

- **Recommendation:** Put the gate inside the store, not only in the routers: in _ensure_chunk_runs_indexed / _ensure_entity_runs_indexed (and in _index_chunks_for_run / index_entity_run so the manual POST /index/{run_id} path is covered too), read the persisted quality report for the run and return early with status 'blocked_by_quality_gate' when token_search_allowed is false. Add a regression test that asserts count()==0 after retrieve_chunks(run_ids=[risky_run]) — the executed reproduction above is the test.

### No relevance threshold: near-orthogonal chunks are always injected as citable evidence, and the prompt then licenses answering from general knowledge

- **Where:** `vendor:app/services/rag_store.py:800`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
format_evidence_blocks emits every hit unconditionally, stamping `f"retrieval_score: {round(1.0 - float(ch.get('distance', 1.0)), 4)}"` (rag_store.py:817) with no floor; grep of rag_store.py finds no min-score, threshold or cutoff anywhere, and retrieve_chunks/retrieve_entities return whatever Chroma's n_results gives. chat_ai.py:1330-1333 appends it to the system prompt whenever it is non-empty. MEASURED by running the real retrieve_chunks against the real store: 'What year or date appears on this page?' returned distances 0.8280/0.8280/0.8484/0.8508/0.8508 — retrieval_score 0.172 down to 0.149 — presented as evidence; 'list the place names' returned 0.7754-0.7927 (score 0.21-0.22). In the executed gate-bypass reproduction the top hit was score 0.1735 on pure gibberish. Compounding it, _RAG_INSTRUCTION (chat_ai.py:372-383) ends 'If none of the evidence is relevant, say so explicitly and answer from general knowledge.'
```

- **What it costs today:** The model is handed five near-random 25-character strings labelled as manuscript evidence with a numeric score it has no calibration for, plus an explicit instruction that it may answer from general knowledge — the exact recipe for a fluent, well-cited claim about a manuscript that the manuscript does not support. There is no way for a downstream reader to tell a 0.15 citation from a 0.85 one.

- **Recommendation:** Drop hits below a calibrated similarity floor (calibrate per embedding model — for cosine on e5, start around 0.35-0.4 similarity, i.e. distance <= 0.6-0.65) and, when nothing clears it, emit no evidence block and tell the chat layer retrieval found nothing so the answer can say so. Replace the bare retrieval_score number with a coarse band (high/medium/low) plus an instruction to refuse rather than fall back to general knowledge for manuscript-content questions.

### Entity documents embed uuids and field labels — 62% boilerplate, mean pairwise cosine 0.804, so the entity collection cannot discriminate between records

- **Where:** `vendor:app/services/rag_store.py:309`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
The unresolved-mention document is built as `doc = "\n".join([f"mention_id: {mention_id}", f"surface: {surface}", f"probable_type: ...", f"link_status: ...", f"reason_unresolved: {reason}", f"evidence_text: {evidence_text}", f"run_id: {run_id}", f"asset_ref: {asset_ref}"])` (rag_store.py:309-320) — two raw uuids and a machine reason string go into the embedded text. The linked-entity document (rag_store.py:262-278) likewise embeds 'entity_id: ...', 'wikidata_qid: ...', 'viaf_id: ...', 'geonames_id: ...', 'confidence_max: 0.9134'. MEASURED on the live store's 27 entity docs: uuid + field-label characters are 62.2% of the embedded text (median 60.2%); only 9 distinct 'surface' values across 27 docs; mean pairwise cosine between entity docs 0.804 (min 0.659) under the same embedder that scores a query against a document at only 0.15-0.30. All 27 are record_type 'unresolved_mention' — the linked-entity branch has produced nothing in this store. Sample embedded doc: 'mention_id: 6a57a401-d74f-45cd-a859-ab27e6aad960 / surface: vilanie / probable_type: place / link_status: unresolved_low_quality / reason_unresolved: token_search_allowed=False quality=RISKY / evidence_text: vilanie / run_id: 0a7e8b87-... / asset_ref: page-1-1a9e9bc3-...'.
```

- **What it costs today:** Documents 0.80 similar to each other, queried by a vector only 0.15-0.30 similar to any of them, means rag_entity_top_k=4 returns an essentially arbitrary 4 records — the ordering is dominated by shared boilerplate, not by which entity the question is about. Authority-linked identity, the part of the system that should be most precise, is retrieved at chance.

- **Recommendation:** Split identity from prose: embed only the natural-language surface (canonical label, aliases, description, entity type, place hierarchy, mention surfaces, evidence snippets) as a plain sentence, and move mention_id, run_id, asset_ref, wikidata_qid, viaf_id, geonames_id and confidence into metadata only — they are already there (rag_store.py:281-299, 322-338) and are duplicated into the embedded text for no retrieval benefit. Drop 'reason_unresolved: token_search_allowed=False quality=RISKY' from the embedded text entirely; it is machine state, not content.

### The full-page transcript — the largest evidence channel by far — is hard-truncated at 1500 characters mid-word

- **Where:** `vendor:app/services/chat_ai.py:338`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
`lines.append(str(transcript)[:1500])` in _context_to_system_message, with no sentence/line boundary and no notice to the model that it was cut. The frontend always sends the whole page here (DocumentChatWorkspace.tsx:1269 `transcript: currentExtractedText || undefined`, set from the merged OCR result at line 1759). MEASURED against the real outputs: transcription body length is 1452 chars for outputs/e-codices_fmb-cb-0002_001r_max/..._gold.txt (97% of the 1500 budget, i.e. one denser folio away from truncation) and 1994 chars for outputs/e-codices_fmb-cb-0001_001r_max/... — 25% of that page is dropped. The median indexed page in the live Chroma store is 1193 chars, so pages routinely sit right at the limit. For comparison, RAG contributes only ~136 chars of evidence per turn (finding 1), so this channel is roughly 10x the retrieval channel.
```

- **What it costs today:** On a two-column or densely written folio the model is answering about the page while the last quarter of the transcription is missing, silently — it cannot know it was truncated, so 'the page does not mention X' is produced with full confidence about text that was cut. Because this path carries far more of the page than retrieval does, it is the dominant accuracy channel today and the truncation is a real ceiling.

- **Recommendation:** Raise the cap to something matched to the chat model's real context (qwen3-30b-a3b-instruct has ample room for a full folio), truncate on a line boundary rather than mid-word, and when truncation does occur append an explicit marker such as '[transcript truncated: N of M characters shown]' so the model can qualify its answer. Same treatment for authority_report[:2000] on line 340 and document_notes[:1000] on line 333.

### kraken_cremma_lat is the primary backend for language_hint=latin but its model file does not exist

- **Where:** `vendor:app/services/ocr_backends.py:152`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`if hint in _LATIN_LANGUAGE_HINTS: attempt_backends = ("kraken_cremma_lat", "kraken_catmus", "kraken_mccatmus")` with `kraken_cremma_lat_model_path: str = "weights/kraken_models/cremma_medieval_lat.mlmodel"` (app/config.py:112). MEASURED: `_resolve_model_path` resolves it to '/Users/mobasuony/Desktop/Thesis project/weights/kraken_models/cremma_medieval_lat.mlmodel', exists=False (the directory contains catmus_medieval, catmus_print, cremma_medieval, mccatmus only). MEASURED over 20 real region crops: 20/20 raised `OCRBackendError: Kraken model for backend kraken_cremma_lat not found`. All other four configured paths resolve to files that exist.
```

- **What it costs today:** Every Latin page burns its first attempt on a guaranteed OCRBackendError, records a spurious BACKEND_ERROR fallback for every region, and silently degrades to CATMuS. Because the error is caught and turned into a `continue`, this is invisible in the response — the run looks like a normal CATMuS run. It also means the language routing table has demonstrably never been validated end-to-end.

- **Recommendation:** Either ship the CREMMA-Medieval-LAT model to weights/kraken_models/, or drop kraken_cremma_lat from the Latin attempt chain. Separately: `build_backend_runtime` should validate `model_path.exists()` at construction time and fail loudly (or prune the backend from the plan) instead of deferring to a per-region exception.

### The live best-attempt ranking has a truncation bias: a shorter transcription of the same page outranks the complete one

- **Where:** `vendor:app/routers/ocr.py:4118`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`_quality_rank(hardened_quality_label)` is the primary key and the tiebreak is `gibberish_score` — no term for token count, line count, or character count. MEASURED with real outputs of the same page: the 414-char transcription (outputs/'e-codices_fmb-cb-0002_001r_max (1)') scores label=HIGH, lead_frag=0.0000, gib=0.0000 and passes every gate, while the 838-char transcription of the same page (outputs/e-codices_fmb-cb-0002_001r_max) scores label=OK, gib=0.0033, and the 960-char full CATMuS output scores label=RISKY, lead_frag=0.1000. Truncating that 960-char text to its first 5 lines gives label=RISKY lead_frag=0.2000; to 1 line gives label=HIGH lead_frag=0.0000; to 2 words gives label=HIGH.
```

- **What it costs today:** An attempt that dropped regions or died early produces short, clean-looking text that outranks a complete but messier transcription, so the retry loop can converge on the least complete pass. Every downstream stage (NER, authority linking, RAG indexing) then indexes a fraction of the page. This is the live path, distinct from the dead `select_best_pass` in item 8.

- **Recommendation:** Add a coverage term to the ranking: require the candidate's non-empty region count / token count to be within some fraction of the best seen so far before its label can win, or make the key `(not all_gates_passed, -coverage_bucket, quality_rank, gibberish)`. The `ocr_lines_total` / `dropped_regions_*` counters already recorded in the run manifest are the raw material.

### No trigram profile exists for any Germanic language, so the LEXICAL_PLAUSIBILITY gate and the lexical term in the selector are constants on those manuscripts

- **Where:** `vendor:app/services/lexicon_trust.py:102`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`profile = _TRIGRAM_PROFILES.get(detected_language); if profile is None or not profile: return 0.50`. MEASURED: `sorted(_TRIGRAM_PROFILES)` == ['anglo_norman','catalan','french','italian','latin','middle_french','occitan','old_french','portuguese','spanish','unknown'] — no old_english, middle_english, english, german, old_high_german, middle_high_german or dutch, and the 'unknown' entry is empty so it also returns 0.50. `lexical_plausibility('tha eode se hlaford ut','old_english')` -> 0.5000; `lexical_plausibility('anything at all','unknown')` -> 0.5000. Yet `select_backend_plan` (ocr_backends.py:155) has explicit routing for exactly those hints, and `_normalize_language_hint` (ocr_backends.py:100-130) has aliases for ang/enm/de/deu/goh/gmh/nl/nld/dut. Also MEASURED: `lexical_plausibility` of both real transcriptions of this Old English page against the 'latin' profile is 0.0877 and 0.0990, far below the 0.20 gate limit — so mislabelling a Germanic page as Latin fails the gate on correct text.
```

- **What it costs today:** On Germanic manuscripts the LEXICAL_PLAUSIBILITY gate is never even constructed (routers/ocr.py:4045 skips it when the language is unknown) or is neutral at 0.50, and 0.30 of `_region_quality_value`'s weight is a constant — which is a direct cause of item 1. It also makes the gate a source of false failures when the language is misdetected as Latin (measured 0.088 on correct Old English text vs a 0.20 threshold).

- **Recommendation:** Add trigram profiles for old_english / middle_english / german-family / dutch (cheap: derive from any public corpus and commit the ~120 most frequent trigrams, same shape as the existing entries), and make `lexical_plausibility` return None rather than 0.50 for an unprofiled language so callers can drop the term and renormalise instead of silently averaging in a constant.

### Deskew is a no-op for half of all skew directions and never fires on real pages: minAreaRect returns 0-90 on OpenCV 4.x, so the abs(angle)<=8 gate rejects one whole tilt direction

- **Where:** `vendor:app/services/ocr_backends.py:246`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
ocr_backends.py:243-253:
  `coords = np.column_stack(np.where(probe < 250))`   # (row,col) i.e. (y,x) — transposed
  `angle = cv2.minAreaRect(coords.astype("float32"))[-1]`
  `angle = 90 + angle if angle < -45 else angle`
  `angle = -angle`
  `if abs(angle) <= 8.0:`  ... only then is deskew_angle set and the rotation applied

MEASURED (cv2 4.11.0), synthetic text line rotated by a known skew, running the exact code above:
  true skew -6 deg: minAreaRect angle=83.991 -> negated -83.991 -> abs<=8 FALSE -> applied +0.000
  true skew -3 deg: minAreaRect angle=87.023 -> negated -87.023 -> abs<=8 FALSE -> applied +0.000
  true skew  0 deg: minAreaRect angle=90.000 -> negated -90.000 -> abs<=8 FALSE -> applied +0.000
  true skew +3 deg: minAreaRect angle= 3.002 -> negated  -3.002 -> abs<=8 TRUE  -> applied -3.002 (correct)
  true skew +6 deg: minAreaRect angle= 6.019 -> negated  -6.019 -> abs<=8 TRUE  -> applied -6.019 (correct)
The `angle < -45` normalisation branch at :246 is dead — OpenCV 4.x minAreaRect returns (0, 90].
Same result on a 6-line block crop: -4 and -2 deg -> +0.000 applied; +2 and +4 deg -> corrected.

MEASURED on real data: deskew_angle == 0.000 for all 8 line crops of e-codices_fmb-cb-0002_001r_max through the production KrakenBackend.recognize().
```

- **What it costs today:** Deskew exists but corrects only one of the two tilt directions, and measured 0/8 real line crops on this page got any correction. Any page or crop tilted the other way is recognised skewed. The cost is bounded on these particular scans because they are near-horizontal, so the measured CER impact here is nil — but it means the pipeline has no working global deskew for a tilted intake, and it silently reports success (raw_metadata says 'deskew' ran). Ranked below the two above because it is latent on the sample pages rather than active.

- **Recommendation:** Normalise the OpenCV 4.x angle properly before gating: `if angle > 45: angle -= 90` and only then reject on abs(angle) > 8, so both tilt directions reach the gate. Better, replace minAreaRect-over-all-ink (which measures the ink bounding rectangle, not the text orientation) with a projection-profile or Hough-based estimator; minAreaRect on a multi-line block returns the block's rectangle angle, not the baseline angle. Add a regression test that feeds a +/-4 deg synthetic line and asserts the estimate has the right sign and magnitude in BOTH directions.

### _rotate_points rotates the boundary and baseline the wrong way, so any deskew that does fire displaces the line polygon by twice the angle

- **Where:** `vendor:app/services/ocr_backends.py:377`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
The image is deskewed at ocr_backends.py:250-254 with `pil_work.rotate(deskew_angle, ...)` (PIL rotates counter-clockwise). The boundary and baseline are then mapped with ocr_backends.py:454-455 `_rotate_points(boundary, deskew_angle, ...)`, whose matrix (:388-393) `x' = cx + dx*cos+(-dy*sin); y' = cy + dx*sin + dy*cos` is a rotation by +angle in y-down coords — the opposite direction to PIL's forward point map.

MEASURED: single black marker at (700,100) in an 800x400 image.
  angle +5.0: marker actually lands at (690.0, 74.0); _rotate_points says (707.5, 126.6) -> 55.4px error; the correct forward map gives (690.2, 74.2) -> 0.3px error.
  angle -5.0: marker actually lands at (707.0, 127.0); _rotate_points says (690.2, 74.2) -> 55.4px error; correct map 0.7px error.
The error is a rotation by 2x the deskew angle. On a real 5900px-wide line crop a 2 deg deskew would displace the boundary ends by ~2 * 5900/2 * sin(2 deg) = 206px vertically, against a line height of ~390px.
```

- **What it costs today:** Whenever deskew does fire, the polygon handed to Kraken is rotated away from the deskewed pixels instead of onto them, so the extracted band is displaced by twice the angle and clips or misses the line at its ends. Today this is masked by the finding above — deskew fires on 0/8 real crops — so it is latent, not currently costing CER on these pages. It becomes an active large regression the moment the angle gate is fixed, so it must be fixed in the same change.

- **Recommendation:** Negate the angle in the two call sites at ocr_backends.py:454-455 (or flip the sign inside _rotate_points and audit its other callers), then assert with the marker test above that a rotated point lands within ~1px of where PIL actually put the pixel. Fix this together with the angle-gate fix, never separately.

### _extract_trigrams strips every abbreviation character before scoring, so a faithful diplomatic transcription is measurably penalised relative to an expanded one

- **Where:** `vendor:app/services/lexicon_trust.py:201`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
`cleaned = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text.lower())` keeps only ASCII letters and U+00C0-U+00FF. MEASURED deletions: U+0304 macron, U+0303 tilde, U+0363/U+036C superscript a/r (all combining marks are outside À-ÿ), U+0101 ā, U+0113 ē, U+012B ī, U+014D ō, U+016B ū, U+0119 ę, U+017F ſ, U+A751 ꝑ, U+A753 ꝓ, U+A757 ꝗ, U+A76F ꝯ, U+A770 ꝰ, U+A75B ꝛ, U+204A ⁊ — every one returns ''. MEASURED effect on the SAME charter text: expanded form 145 chars -> 80 trigrams, plausibility(latin) 0.432; diplomatic abbreviated form 121 chars -> **46 trigrams (-42.5%)**, plausibility **0.356**, which is below the 0.40 LEXICAL_WEAK line and so multiplies confidence by 0.75. Tokens also vanish entirely: 'd̄n̄s' -> 'dns' survives but 'scē' -> 'sc' (len<3, dropped by line 205), and after the strip 'ꝑsentibus' -> 'sentibus'.
```

- **What it costs today:** Transcribing abbreviations faithfully — the whole point of a diplomatic edition — lowers the pipeline's own confidence in the page and can push it across the LEXICAL_WEAK/RISKY boundary, which then blocks token_search and NER on correct text. It also silently deletes ~40% of the evidence the lexical gate is supposed to reason over, so the gate is scoring a mutilated version of the text.

- **Recommendation:** Run the abbreviation-expansion layer (item 1) before trigram extraction, and widen the keep-set to Latin Extended-A/B, Latin Extended Additional and Latin Extended-D plus the combining diacritical block, so nothing is deleted merely for being outside Latin-1.

### The recognizers emit NFD only, but the CLI forces NFC and the backend never normalises — the same abbreviation bar ends up stored two different ways inside one file

- **Where:** `src/archai_ocr/pipeline/assemble_text.py:31`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
assemble_text() defaults `normalize: NormalizationForm | None = "NFC"` and applies `payload = unicodedata.normalize(normalize, payload)` (line 33). The backend has no equivalent: `grep -rn 'unicodedata.normalize' --include='*.py' app/` returns only text_normalization.py:65/67/256 and wikidata_client.py:44, all inside search-key functions, never on stored or indexed text. MEASURED from the real model files: all three Kraken alphabets contain 0 precomposed letters, and the model in use is named McCATMuS_nfd_nofix_V1.mlmodel — i.e. NFD output by construction. MEASURED consequence of forcing NFC on it: base+U+0304 composes for a,e,i,o,u,y,g ('ā' U+0101, 'ē' U+0113, 'ī' U+012B, 'ō' U+014D, 'ū' U+016B, 'ȳ' U+0233, 'ḡ' U+1E21) but stays decomposed for n,m,q,c,p,r,s,t,d,b (no precomposed codepoint exists), so within one transcription the identical scribal bar is stored one-codepoint over vowels and two-codepoint over consonants. MEASURED encoding-only edit distance, with zero recognition errors: CER(NFD vs NFC) = **0.075** on an abbreviated Latin charter and **0.185** on accented Old French. MEASURED on disk: outputs/e-codices_fmb-cb-0001_001r_max.txt is_NFC=True / is_NFD=False, while backend-indexed chunk text is not NFC.
```

- **What it costs today:** String equality, the trigram extractor, dict/lexicon lookups and any diff or CER against CATMuS-style NFD ground truth all treat 'n̄' and 'ō' as belonging to different alphabets, and treat CLI output and backend output as different transcriptions of the same page. Up to 18.5 CER points of pure encoding noise sit on top of any accuracy number, and there is currently no CER harness in the repo at all (no cer/char_error function exists) to expose it.

- **Recommendation:** Pick NFD as the single canonical stored form (it matches the models and the CATMuS ground truth, and keeps the abbreviation bar as one addressable codepoint), apply it once at the recognition boundary in both pipelines, and normalise the same way in every comparison. Change the assemble_text default from NFC to NFD, or make it explicit and shared with the backend rather than a CLI-only default.

### token_quality_score structurally penalises abbreviated tokens, so the most frequent words in a Latin manuscript can never become entity mentions

- **Where:** `vendor:app/services/text_normalization.py:137`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
`_VOWELS = set("aeiouyàâäéèêëïîôùûüœæ")` contains no macroned vowel (ā ē ī ō ū), no combining mark and no MUFI letter; _max_consonant_run (line 149) treats every combining mark as a consonant because U+0304 fails `c in _VOWELS or not c.isalpha()`... it is category Mn so isalpha() is False, and _is_alpha_token (line 162) therefore returns False for any macroned token, costing -0.15. MEASURED on the SAME charter: expanded form mean token quality **0.900** with 0/22 tokens below 0.30; diplomatic abbreviated form mean **0.770** with 2/22 below 0.30. MEASURED per token: 'd̄n̄s' = **0.25** (vowel_ratio 0.00, all_isalpha False), 'scē' = 0.45, 'qꝺ' = 0.15, 'eccl̄ie' = 0.65, 'Guill̄s' = 0.70, 'dec̄as' = 0.70. authority_linking.py:1119 and :1514 gate a mention on `len(surface_low) >= 4 and token_quality_score(surface_low) >= 0.45`; 'd̄n̄s' scores 0.25 and, after normalize_for_search, becomes 'dns' of length 3, so it fails both halves.
```

- **What it costs today:** The nomina sacra and standard suspensions — dominus, sanctus, Iesus, Christus, omnium, and the whole -bus/-que/-rum family — are the highest-frequency tokens on a Latin manuscript page and are exactly the tokens this scorer rates as OCR garbage. They are excluded from entity mentions by the 0.45 floor and drag the page-level text_quality_label down, which in turn gates translation, NER and entity claims. The scorer rewards editorial expansion and punishes diplomatic fidelity.

- **Recommendation:** Score the expanded form, not the diplomatic form: run abbreviation expansion first, then compute quality. Failing that, add macroned vowels and the MUFI vowel letters to _VOWELS, skip combining marks in _max_consonant_run and _is_alpha_token instead of counting them as non-alpha consonants, and treat a token carrying a recognised abbreviation mark as a positive signal rather than a penalty.

### The proofreader guard's thresholds sit inside the OCR engine's own page-to-page noise band, and it accepts half a page of wholly invented lines

- **Where:** `vendor:app/agents/ocr_proofreader_agent.py:196`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
Thresholds are `_MAX_CHAR_EDIT_RATIO = 0.40`, `_MAX_LINE_COUNT_DRIFT = 0.30`, `_MIN_UNCERTAINTY_RETENTION = 0.50`, `_MAX_TOKEN_CHURN = 0.50` (lines 196-199). MEASURED by running check_proofread_delta over 28 pairs drawn from 20 real independent OCR outputs of the same 70-line page in app/archai.sqlite: char_edit_ratio mean **0.191**, max **0.499**; token_churn mean **0.390**, max **0.868**; the guard would classify **21/28** pairs of unrelated recognizer outputs as 'safe corrections'. MEASURED substitution sweep replacing manuscript lines one at a time with fluent invented Old French: 5/10 lines (50%) wholly invented is ACCEPTED (char_edit 0.272, token_churn 0.436); rejection first occurs at 6/10 (char_edit 0.330, churn 0.525). A 10/10 full rewrite is rejected (char_edit 0.500).
```

- **What it costs today:** The guard is calibrated at or below the recognizer's own variance, so it cannot distinguish an LLM hallucination from ordinary OCR disagreement. Half a page of fluent invention survives into the stored transcription and into the RAG index, where it reads as high-quality text and will be retrieved and cited in preference to the correct-but-noisy original — the worst possible failure mode for a thesis that must be able to defend every reading.

- **Recommendation:** Tighten the thresholds well below the measured OCR variance (char_edit ~0.08-0.10, token_churn ~0.15) and enforce them per line rather than per page, so one rewritten line cannot hide inside an otherwise-unchanged page. Add an alignment check that every accepted edit is within small edit distance of the token it replaces, and reject any edit that introduces a token absent from both the raw OCR and the language's lexicon.

### COMMON_WORD_BLACKLIST rejects the head words of common French toponyms, so multi-word place names lose their identifying token

- **Where:** `vendor:app/services/text_normalization.py:25`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
The set includes 'main','port','pont','tour','mont','fort','terre','dame','dieu','lieu','voie','cors','char','cour','cuer','camp','mort' (lines 41-45); 'cors' appears twice and the single letter 'd' is a member (line 34). is_blacklisted_token is used to veto mentions at ocr.py:775, :872 (per-token filter over multi-word names), :926 and :1606. MEASURED: is_blacklisted_token returns True for Mont, Pont, Port, Tour, Terre, Dame, Dieu, Camp, Cuer, Mort, Main, Fort, Vers, Cors, Char, Cour, Amen, Post, Inter, Super, Ante, Voie, Lieu. MEASURED per-token filtering as ocr.py:872 does it: 'Mont Saint Michel' -> kept ['saint','michel']; 'La Tour Landry' -> kept ['landry']; 'Terre Sainte' -> kept ['sainte']; "Pont de l'Arche" -> kept ['l','arche']; 'Notre Dame de Paris' -> kept ['notre','paris'].
```

- **What it costs today:** Mont-, Pont-, Port-, Tour- and Terre- are among the most productive toponym heads in medieval French, and Dame/Dieu in hagionyms and church dedications. Stripping them before authority lookup turns 'Mont Saint Michel' into 'saint michel' and 'La Tour Landry' into 'landry', which either fails to resolve or resolves to the wrong entity. The list was clearly built to suppress false positives (its own comment cites 'run 81ae066e'), and it does so at the cost of place-name recall.

- **Recommendation:** Split the list into function words (safe to veto anywhere) and content words that are only noise in isolation. For the latter, veto a single-token mention but keep the token when it is part of a capitalised multi-token span, and let the candidate scorer decide. Drop the duplicate 'cors' and the bare 'd'.

### Layout predictions carry no `score` field, so COCOeval will fail on today's output exactly as it did in the recorded runs

- **Where:** `archai/vendor/layout/compare/data/new_models.py:121`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`merged["annotations"].append(new_ann)` — and `grep -n 'score' new_models.py` returns zero hits, so predictions are emitted with no confidence. By contrast old_models.py:144 does emit `"score": conf`. pycocotools' COCOeval requires `score` on every detection, which is precisely the recorded failure: 'segm error: 'score', bbox error: 'score'' on all five new-model rows in expert_datasets_model_comparison_summary.json. MEASURED that the defect is still live in the production path today: `json.load('outputs/e-codices_fmb-cb-0001_001r_max/layout_coco.json')` annotation keys are ['area','attributes','bbox','category_id','id','image_id','iscrowd','segmentation'] — `'score' in ann` is False for 81 annotations across the 6 emitted classes (Main script black 70, Plain initial- coloured 3, Plain initial - Black 3, Embellished 2, Column 2, Page Number 1).
```

- **What it costs today:** This is the specific reason the layout mAP numbers are zero, and it is not confined to the archived compare/ scripts — the production pipeline's own COCO export has the same omission, so piping today's output into any COCO evaluator reproduces the same KeyError. It also means there is no per-region confidence available for triage: you cannot rank regions by detector confidence to find the ones worth re-segmenting, and you cannot compute a precision/recall curve at all, only a single hard-threshold count.

- **Recommendation:** Emit the YOLO detection confidence as `score` on every annotation in both compare/data/new_models.py and the production layout_coco.json writer. Then the existing compare.py mAP@50 / mAP@[.50:.95] / mAR_100 path works unchanged against ground_truth_coco.json, and layout accuracy becomes a number for the first time.

### Kraken line confidence is well-ranked but grossly miscalibrated: mean confidence 0.885 against 49.6% measured line CER

- **Where:** `vendor:app/services/ocr_quality.py:1162`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`def build_effective_quality(report, gate_decisions, confidence=...)` folds the recogniser's confidence into an effective quality figure consumed at routers/ocr.py:2972 and 3266. MEASURED by joining the 60 per-line confidences in outputs/e-codices_fmb-cb-0002_001r_max/region_cache/*.json (each `{lines: [{text, confidence}], confidence, warnings}`) to the per-line CER between the two runs of that page: 20 lines joined; mean line confidence 0.8853; mean line CER 49.59%; Pearson r(confidence, CER) = -0.5731. The extremes: conf 0.9538 → line CER 37.50%; conf 0.7424 → line CER 65.85%. The page-level summary tells the same story — run A `median_conf: 0.8874`, run B `median_conf: 0.9687`, and the two disagree at 48.4% CER.
```

- **What it costs today:** Confidence has genuine signal (r = -0.57, monotone at the extremes) so it is usable to *rank* lines for review or retry, but it is off by roughly a factor of four as an accuracy estimate: it claims ~89% correctness on lines that differ from a second pass at ~50% CER. Any threshold expressed in absolute confidence terms — `low_quality_line_pct: 0.0` in the run summary, the effective-quality blend, review triage — is therefore calibrated against nothing. Note the 49.6% here is run-to-run disagreement, which is a lower bound on true CER, not true CER; the real figure can only be worse.

- **Recommendation:** Once the item-1 gold set exists, fit and commit a calibration curve: bin lines by confidence, plot measured CER per bin, and store the mapping in eval/results/ so confidence can be reported as an expected-CER band rather than a raw score. Until then, use confidence only for relative ranking and stop treating it as a probability of correctness. The r = -0.57 I measured also gives a concrete target: any replacement quality signal should beat it.

### thesis_real_page_smoke.py is the only end-to-end check and it asserts liveness only, against an input that lives outside the repo

- **Where:** `archai/vendor/layout/backend/scripts/thesis_real_page_smoke.py:14`
- **Gain / effort:** moderate / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`DEFAULT_IMAGE_PATH = Path("/Users/mobasuony/Downloads/e-codices_fmb-cb-0001_001r_max.jpg")` — an absolute path in one user's Downloads folder (it does exist on this machine, 8.9 MB, but nowhere in the repo), and line 15 `DEFAULT_BASE_URL = "http://127.0.0.1:8000/api"`, so the script needs a live backend plus an LLM. Its checks are presence-shaped, not accuracy-shaped: line 300 `bool(transcript and run_id and int(extraction.get("chunks_count") or 0) > 0)`; line 394 `authority_ok = all(token not in authority_lower ... for token in ("enfant [place]", "vilanie [place]"))` — a two-string blocklist; lines 24-29 BAD_TRANSLATION_PHRASES, a four-phrase blocklist; line 468 page_number_ok, which passes as long as the response is non-empty and does not contain the prompt text ('strict diplomatic transcription task', 'preserve reading order exactly'); lines 33-40 VERIFICATION_BLOCK_RE, a regex on the shape of the verification block. Line 506 prints `Overall: PASS/FAIL` from `all(stage['passed'])`. The companion thesis_smoke_check.py (6.8 KB) is the same shape. Neither file contains any reference transcription or metric. There are 21 `def test_` in the root tests/test_cli_and_logging.py, 19 in test_config.py, 16 in test_layout_ordering.py, 20 in test_pipeline_io.py — all structural I/O and ordering, no accuracy.
```

- **What it costs today:** The one script that exercises the whole pipeline on a real page can pass while every transcribed character is wrong — the fmb-cb-0002 output that scores CER 48% against its own second run would satisfy every one of these assertions, since it is non-empty, produced chunks, and contains none of the six blocklisted strings. It also cannot run on another machine or in CI (absolute Downloads path, live backend, live LLM), so it is not a reproducible gate. Its four- and two-item blocklists are memorised past failures rather than a metric, so they detect only recurrences of those exact strings.

- **Recommendation:** Keep it as a liveness smoke test but stop treating it as evaluation, and rename the 'evaluation' vocabulary in it (`_evaluation_row`) so it is not mistaken for one. Point DEFAULT_IMAGE_PATH at a repo-relative eval page from the item-1 manifest so it is runnable elsewhere, and add the real gate next to it: eval/run_ocr_eval.py asserting per-page CER against gold. Replace the string blocklists with assertions expressed against gold spans, so a new failure of the same kind is caught rather than only the exact remembered string.

### There is no retrieval evaluation of any kind — confirmed, nothing to regress against

- **Where:** `vendor:app/services/rag_debug.py:27`
- **Gain / effort:** moderate / medium — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
Confirmed the grep: recall@ / ndcg / mrr / precision@ / map@ appear nowhere in any .py or .md in the repo. The only @k metrics are layout-detection mAP@50 values in archai/vendor/layout/compare/data/expert_datasets_model_comparison_summary.json — object detection, not retrieval. What exists instead is observability only: rag_debug.py is 80 lines of ARCHAI_DEBUG_RAG-gated JSON-line logging (is_rag_debug_enabled / log_retrieve_debug / log_chat_evidence), and rag_store.retrieve_debug (rag_store.py:751-798) returns scored previews for a debug endpoint. Neither has a notion of a correct answer: there is no query set, no relevance judgement, no ground-truth chunk mapping, and no assertion anywhere that retrieval quality has not regressed. Every number in the eight findings above had to be measured from scratch for this investigation.
```

- **What it costs today:** Nothing above can be verified as fixed, and any of the fixes could silently make retrieval worse. The chunking change in finding 1 in particular trades chunk precision for recall; without a metric that is a guess. It also means the 87.9% duplicate rate and the split index sat in production undetected.

- **Recommendation:** A minimal harness needs four things and can be built in a day from what is already on disk. (1) A gold query set: 20-30 questions over the pages in outputs/ (the e-codices folios plus the gold transcription at outputs/e-codices_fmb-cb-0002_001r_max/..._gold.txt), each labelled with the character span in the page transcription that answers it — the chunks table already stores start_offset/end_offset, so relevance can be judged by span overlap rather than by chunk id, which makes the judgements survive a re-chunking. (2) A fixed corpus snapshot: index those pages into a throwaway chroma_persist_dir (the CHROMA_PERSIST_DIR + ARCHAI_DB_PATH env override used in this investigation works cleanly) so the eval never reads the polluted live store. (3) Metrics: recall@k and nDCG@k over span overlap, plus two that would have caught the findings above directly — distinct-texts@k (measured 1.27/5 today) and evidence-chars@k (measured 136 today). (4) A CI assertion pinning each metric with a tolerance, and an offline embedding backend so the harness never needs the network.

### No per-line confidence or per-line unit exists on the Kraken path: one forced line per region, per-character confidences flattened to one mean

- **Where:** `vendor:app/services/ocr_backends.py:622`
- **Gain / effort:** moderate / medium — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`confidences = [float(score) for record in records for score in list(getattr(record, 'confidences', []) or [])]` then `confidence = _mean(confidences)` — the per-character confidence vectors of every line in the region are concatenated and averaged into a single float; line-level and character-level confidence is discarded before it leaves the backend. `_build_segmentation_for_crop` (ocr_backends.py:449-501) always constructs `lines=[containers.BaselineLine(id=metadata.region_id, ...)]` — exactly one line per region — and grep for `blla|pageseg|segment(` across the entire backend `app/` returns zero hits, so the vendor backend never performs line segmentation inside a region (the root CLI does: src/archai_ocr/pipeline/htr_kraken.py:116 calls `blla.segment`). Note app/core/constants.py:7 maps SegmOnto `MainZone` -> `Column`, i.e. multi-line zones are expected. MEASURED on the real page: single-line region 3 -> 1 line, conf 0.975, 'tide eode þaer ƿingeardes hlapord iit ⁊gentecoe'; the union of 4 consecutive line regions (height 1309px, 4 lines of text) -> 1 line, conf 0.351, text 't a' — four lines of manuscript reduced to three characters.
```

- **What it costs today:** Two effects. (1) Selection and gating can only ever operate at region granularity, so a region with one bad line among three cannot be repaired or voted on — there is no per-line hypothesis to compare. (2) Any multi-line zone reaching KrakenBackend is destroyed outright (measured: 4 lines -> 3 chars). The one usable signal, the collapsed confidence, did drop to 0.351 on that case, so the failure is detectable but nothing acts on it.

- **Recommendation:** Call `kraken.blla.segment` on the region crop (or accept baselines from the layout stage) so `rpred` returns one record per text line, keep each record's own `confidences` list in `raw_metadata`, and expose per-line text+confidence on `OCRBackendResult`. That is also the prerequisite for the per-line ROVER in item 2 step 3. As an immediate cheap guard, reject a region whose collapsed confidence is far below the page median (0.351 vs ~0.95 here) rather than merging it into the page text.

### Full-page Kraken blla would NOT beat zone-crop-then-segment for these layouts: it matches inside the zone and injects decoration as false-positive lines

- **Where:** `src/archai_ocr/pipeline/htr_kraken.py:108`
- **Gain / effort:** moderate / medium — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
The CLI segments per crop: `_segment_crop` calls `blla.segment(image, model=seg_model)` on each region PNG (htr_kraken.py:108-114). MEASURED both strategies with the real blla.mlmodel. fmb-cb-0001 (2 columns): full page = 87 lines, of which 71 have their baseline centre inside a MainZone; zone crops = 36 + 35 = 71. Exact parity inside the zone. The 16 extra full-page lines are decoration, not text — I cropped three of the reported baselines (x 154-349, cy 435 / 2520 / 3090) and they are pen-flourish curls and the red/blue vertical penwork bar, so full-page blla would add 16 hallucinated lines (18% spurious). fmb-cb-0002: full page = 21 lines but the MainZone crop yields 27 (+29%), because blla rescales its input internally and a 5212x6204 crop is effectively higher resolution than a 6129x8174 page — the vendor line detector independently finds 20 'Main script black' regions and the gold manifest records coco_text_regions_total=21 / ocr_lines_total=30. 'Old English.jpg': 22 = 22. 'Latin script.jpg' (a two-page spread): full page 101, 41 inside MainZone, zone crop 38. Net: zone-crop matches or beats full-page inside the zone on 3 of 4 pages. Splitting the spread is not a workaround either — the zone model on the left half alone (470x4000 strip) returns zero detections.
```

- **What it costs today:** Switching to full-page blla would trade a real recall gain on spreads and margins for an 18% false-positive line rate on decorated folios, feeding penwork into the recognizer as text and into the index as gibberish. The measured recall gap is entirely about WHICH zones are kept (item 3), not about the segmentation strategy — so this is the evidence against a full-page rewrite, and it redirects the effort to the zone allow-list.

- **Recommendation:** Keep zone-crop-then-segment. Spend the effort on widening the kept-zone set (item 3) so marginalia and facing-page text get their own crops, then segment each crop with blla as today. If a full-page pass is added at all, use it only as a recall audit — count baselines outside every kept zone per page and warn when the ratio exceeds a threshold (it was 57% on 'Latin script.jpg', 25% and 15% elsewhere, 0% on fmb-cb-0002 and acv-P-Antitus) — rather than as the production segmenter.

### lexicon_trust trigram profiles are ~95 hand-typed entries with duplicate literals and a hard-coded 0.55 scale that does not match reality; the calibration constants are derived from numbers that are backwards

- **Where:** `vendor:app/services/lexicon_trust.py:123`
- **Gain / effort:** moderate / medium — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
The module docstring claims '~120 most frequent character trigrams ... derived from representative medieval corpora'. MEASURED sizes: latin 95, old_french 94, middle_french 103, french 74 unique — because the source lists contain duplicate literals (latin 108 written/95 unique/13 dupes; old_french 108/94/14; middle_french 108/103/5; french 81/74/7). lexical_plausibility returns `min(1.0, ratio / 0.55)` (line 123), i.e. it assumes real text hits the profile 55% of the time. MEASURED trigram-token hit rate on real clean medieval text: **latin 112/416 = 26.9%**, old_french 83/221 = 37.6%. The latin profile covers only 60/296 = 20.3% of clean-Latin trigram TYPES, and 35/95 = 36.8% of its own entries never occur in the clean sample. MEASURED distributions against a deterministic sample of all 1171 distinct real OCR chunks (>=18 chars) in app/archai.sqlite: latin profile — clean mean 0.487 / median 0.448 / min 0.227, garbage mean 0.224, AUC 0.850, and **3/10 clean medieval Latin samples are penalised by lexical_trust_adjustment (2x LEXICAL_WEAK ×0.75, 1x LEXICAL_IMPLAUSIBLE ×0.55)**; old_french profile — clean mean 0.686, garbage mean 0.420, AUC 0.810, and **331/1171 = 28.3% of real garbage OCR chunks score >=0.60**, the module's own 'lexically consistent' line. ocr_quality_config.py:73-78 justifies LEXICON_CLEAN_FLOOR=0.45 with 'clean Latin scores ~0.82 plausible but clean Old French only ~0.57'; MEASURED it is the reverse (Latin 0.487, Old French 0.686), so the floor is calibrated on an inverted premise while the lexical term carries weight 0.45 in GIBBERISH_WEIGHTS_WITH_LANGUAGE. MEASURED alternative: a frequency-ranked top-95 profile built from ~416 trigrams of real corpus text (leave-one-out) reaches **AUC 0.914 vs 0.869** for the hand-built list on the same data — a few hundred words of real corpus already beats it. Aliases at lines 96-101 point italian/spanish/portuguese at the latin profile object (verified `is` identity): MEASURED Dante 0.424/0.428, Cervantes 0.076, Cid 0.251.
```

- **What it costs today:** The dominant component (weight 0.45) of the gibberish score is mis-scaled by roughly 2x, so clean diplomatic Latin lands in the 'uncertain'/'weak' band and gets its confidence multiplied by 0.75 or 0.55, while 28% of genuinely scrambled Old French passes as 'lexically consistent'. Confidence therefore drops on the pages that are right and holds on the pages that are wrong, which propagates into gate decisions (pipeline_hardening.py:477-485 blocks token_search and token_ner below 0.20) and into model/backend selection. For Italian, Spanish, Portuguese and Catalan the signal is not merely noisy but wrong-language.

- **Recommendation:** Replace the hand-typed sets with frequency-weighted profiles counted from a real corpus per language (CATMuS/CREMMA ground-truth transcriptions already on disk, or a public Latin/Old French corpus), store counts not membership, score with log-probability against a held-out set, and calibrate the thresholds and LEXICON_CLEAN_FLOOR from that measured distribution instead of the current 0.55 constant. Add a real lemma/word list on top for the token-level check (a word list would beat trigrams outright on the u/v-i/j-folded forms). Delete the italian/spanish/portuguese/occitan/catalan aliases and return the neutral 0.50 until real profiles exist.

### The thesis's cited accuracy numbers cannot be reproduced from any code or data in the repo, and the project's own showcase bundle contradicts them

- **Where:** `fixes/archai_thesis_source_submission_ready/content/ch6_evaluation.tex:98`
- **Gain / effort:** moderate / medium — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
Line 98 states "Latin: GLM-OCR CER = 4.1\%, Transkribus CER = 4.9\%, and Qwen 3 VL via SAIA CER = 6.6\%; Old French / Anglo-Norman: GLM-OCR CER = 5.0\%... Old English: GLM-OCR CER = 5.4\%...", line 90 a table row "Transkribus benchmark & 5.6 & 13.4 & 68.1 & 0.8 & 28.7", line 18 "a curated multilingual subset of 30 manuscript pages with 1,284 automatically segmented, manually evaluation-aligned text lines", and line 302 "an internal segmentation evaluation over 220 annotated pages, the model produced 207 usable segmentation outputs, corresponding to a 94.1\% usability rate" followed by "mAP@50 = 0.89, region-level recall = 0.86, macro IoU = 0.82". MEASURED against the repo: (a) no CER/WER computation exists (item 1); (b) no 30-page gold set exists — exactly one file is named *_gold.txt and it is a machine run (item 1); (c) no 220-page annotation set exists — I enumerated every COCO-shaped JSON in the repo and the only human ground truth is ground_truth_coco.json at 10 images / 44 annotations; the other 12 COCO files (354 annotations total across 22 image entries) are all pipeline predictions in the gitignored outputs/; (d) the only mAP harness is un-importable and every metric it ever recorded is 0.0 (item 4). And the project's own showcase bundle for the same thesis, artifacts/thesis_showcase/manifest.json, states `"bundle_note": "This is a showcase build, not a gold-standard benchmark. No ground truth, CER/WER, or gold entity links are fabricated."` and lists under unavailable_subsystems: "gold ground truth: no authoritative transcriptions available for these pages", "kraken region OCR: package not installed", "transkribus: no API credentials configured". Its figure29/by_language.csv reports only qualitative labels (high/medium) — zero numbers.
```

- **What it costs today:** Not an accuracy defect in the pipeline, but the decisive credibility risk and the reason measurement must come first: the chapter reports point estimates to one decimal place (4.1%, 94.1%, 0.89) for which the repo contains neither the code, the gold data, nor the 220-page or 30-page corpora, while the artifact bundle generated for the very same figures explicitly disclaims having any ground truth. Nothing in the repo can confirm or refute the numbers, and the one accuracy figure I could actually compute — 48.4% CER between two runs of the same page — is an order of magnitude away from the 4.1–7.5% band cited. That gap may be entirely explained by the cited numbers coming from a different (GLM/Transkribus/SAIA) path against a corpus that is not here, which is exactly the problem: it cannot be checked.

- **Recommendation:** Treat every number in ch6 as unsourced until a script in the repo regenerates it. Practically: either (a) recover the evaluation corpus (item 9) and re-derive the numbers with the item-1 and item-5 harnesses, emitting them into eval/results/<git-sha>.json and having the LaTeX cite that file, or (b) restate the chapter to the evidence that actually exists — the honest qualitative framing already used in artifacts/thesis_showcase/README.md. Do not leave the two co-existing. Also note `fixes/` is gitignored (.gitignore:39), so the chapter making the claims is itself outside version control.

### CI never runs the 28k-LOC backend's tests, and 34 test modules' sources have been deleted while their bytecode remains

- **Where:** `.github/workflows/ci.yml:67`
- **Gain / effort:** moderate / medium — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`run: python -m compileall -q archai/vendor/layout/backend/app` — the backend job checks only that modules compile; the comment at lines 64-66 says a full test run needs weights and the inference stack, so it is expected to be run locally. The pytest job (lines 50-51) covers only the root CLI: `pytest tests -q --cov=archai_ocr`, i.e. 4 files / 76 structural tests. MEASURED: the backend has 20 tracked test files (`git ls-files archai/vendor/layout/backend/tests/ | wc -l` = 20 including __init__.py) but 54 distinct module names in its __pycache__ — 34 modules whose .py no longer exists and which appear in no commit (`git log --all --diff-filter=A` finds none). The missing set is exactly the quality-measuring layer: test_extraction_quality, test_ocr_quality, test_ocr_quality_hallucination, test_gate_enforcement, test_rag_store, test_adaptive_selection, test_seam_retry, test_pipeline_production_ocr, test_saia_extraction, test_audit_compliance, test_grounded_answer_policy, test_ocr_geometry_layout, plus 10 named test_run_<runid>_regression modules (5241c606, 540fad29, 81ae066e, 91b3a986, a49b676c, cfb920bf, d881, d9b050cb, fefcc761) that were clearly pinned to specific real runs. I decompiled the largest orphan, test_pipeline_golden.cpython-312-pytest-9.0.2.pyc: despite the name it contains no gold-text assertions — 8 classes / 41 tests over synthetic images (TestPreprocessing, TestLayoutAnalysis, TestLineSegmentation, TestSchemaExtensions, TestProofreaderDiffGuard, TestContentDrivenTiling, TestOCRPromptLayoutMetadata, TestRegressionProperties, TestPipelineBudget), all structural. A parallel orphan exists in the root CLI: src/archai_ocr/pipeline/__pycache__/preprocessing.cpython-312.pyc with no preprocessing.py.
```

- **What it costs today:** Two compounding losses. First, the pipeline that does the actual OCR has no automated regression protection at all in CI — only compileall — so a change that degrades transcription quality reaches main unopposed; the previously-fixed defects (empty transcriptions from a mismatched default layout class, unreachable Kraken fallback chain, no-op whitespace normaliser) are exactly the class of bug a run-pinned regression test catches and compileall cannot. Second, the deleted modules were the measurement layer: ten regressions pinned to named real runs plus test_extraction_quality/test_ocr_quality/test_gate_enforcement/test_rag_store are precisely the tests that would have encoded 'this run used to produce this output'. Their bytecode is the only surviving record.

- **Recommendation:** Two things. (a) Recover what the orphans asserted — the .pyc files are decompilable and carry the test names, thresholds and string constants; harvest the ten test_run_<runid>_regression modules first, since each pins a real run and is effectively a snapshot gold set already, then either restore them or fold their assertions into the new eval harness. (b) Add a CI job that runs the backend suite with heavy dependencies stubbed (the existing tests already monkeypatch the embedding store and OCR backends, so most of the 20 tracked files should run without weights), and a separate scheduled or manual job that runs the eval harness with real weights and publishes CER/WER/recall@k. Also delete the stale __pycache__ from version-control-adjacent state or add a CI check for .pyc files without a matching .py, so this loss is loud next time.


# SMALL EXPECTED GAIN (17)

### Cosmetic: disambiguate's documented rule 5 (ent_type == "role" never auto-links) is not implemented in the function body

- **Where:** `vendor:app/services/entity_scoring.py:335`
- **Gain / effort:** small / low — MEASURED
- **Area:** Entity → authority linking (entity_scoring.py, authority_linking.py, authority_sources.py, wikidata_client.py)
- **Evidence:**

```
The docstring at entity_scoring.py:335 states rule 5 `` ``ent_type == "role"`` → never auto-link ``, but disambiguate (entity_scoring.py:320-420) never reads ent_type from the candidates or takes it as a parameter — the three gates it applies are type_compatible, AUTO_SELECT_THRESHOLD and MIN_MARGIN. The rule is in fact enforced upstream at authority_linking.py:1410 (`if ent_type == "role":`), which is why the production DB shows 3 mention_links with reason 'ent_type=role (not linkable)'. Consistent with the docstring being aspirational, the 24 highest-scoring 'linked' rows in the production DB are Q116 "King" (royal title) and Q355567 "Count" (noble title) at score 1.0 — exactly the role links rule 5 forbids — written by an older code path that bypassed disambiguate (their score_breakdown is empty, and neither QID appears anywhere in app/).
```

- **What it costs today:** No impact today: role mentions are filtered before disambiguate is reached, and 3,753 of the 14,476 production mentions are ent_type='role', all correctly skipped upstream. The risk is latent — any future caller of disambiguate() that does not pre-filter roles gets no protection despite the docstring promising it, and role links are high-visibility precision errors (linking every occurrence of 'roi' to Q116).

- **Recommendation:** Either implement the gate (accept an `ent_type` argument and return unresolved with reason 'ent_type=role' when it is 'role'), or delete rule 5 from the docstring and point it at authority_linking.py:1410. Prefer implementing it — disambiguate is the precision chokepoint and should not rely on callers. While there, fix the stale module docstring at authority_linking.py:12 which still claims AUTO_SELECT_THRESHOLD = 0.75 and MIN_MARGIN = 0.10.

### SVC-10 confirmed: select_best_pass prefers truncated passes and has zero callers

- **Where:** `vendor:app/services/pipeline_hardening.py:69`
- **Gain / effort:** small / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`return min(reports, key=lambda r: (label_rank.get(r.quality_label, 99), r.gibberish_score, r.leading_fragment_ratio))` — no length or coverage term. MEASURED by executing it on four `compute_quality_report`s derived from the same real 960-char CATMuS page text: full 20 lines -> RISKY, first 5 lines -> RISKY, first 1 line (32 chars) -> HIGH, first 2 words (18 chars) -> HIGH. `select_best_pass` returns the 32-char single-line pass, discarding 97% of the transcription; a 2-way full-vs-1-line call also returns the 1-line pass. It does correctly reject empty text (empty -> UNRELIABLE). Dead code confirmed: `grep -rn select_best_pass` across the whole repo (excluding .venv) returns only this definition and the AUDIT_FINDINGS.md entry; routers/ocr.py imports `decide_downstream_mode, enforce_quality_gates, extract_high_recall_mentions, format_gate_report, proofreading_quality_guard, should_use_shape_based_search` from this module at line 93-100 but not `select_best_pass`, even though the module docstring lists 'Pick best pass' as step 4 of the architecture.
```

- **What it costs today:** Zero today — nothing calls it. It matters only as a trap: it is the obvious-looking function to reach for when wiring up multi-pass selection, and doing so would make truncation the winning strategy. The equivalent bias already exists in the live path (item 7).

- **Recommendation:** Delete it, or fix it to match whatever coverage-aware ranking item 7 lands on and then actually call it from routers/ocr.py so there is one ranking implementation instead of two.

### model_router assigns an OCR model that no OCR path consults (cosmetic)

- **Where:** `vendor:app/services/model_router.py:52`
- **Gain / effort:** small / low — MEASURED
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`ocr_model = str(settings.glmocr_ollama_model or "glm-ocr:latest").strip() or "glm-ocr:latest"` inside `get_task_model_assignments`. Grep for `get_task_model_assignments` outside this module returns only agents/label_analysis_agent.py (437, 445, 755, 1213, 1231, 1243, 1254) and agents/paleography_verification_agent.py (268) — never `ocr_backends.build_backend_runtime`, never `select_backend_plan`, never `ocr_agent`. The `GlmOcrBackend` (ocr_backends.py:822) ignores it entirely and reads `settings.glmocr_device`.
```

- **What it costs today:** None directly. Worth naming only because 'model_router' reads like the place OCR model choice is decided, and it is not — the real decision is the hardcoded table in `select_backend_plan` (item 3). Anyone tuning accuracy via this file will change nothing.

- **Recommendation:** Rename to reflect that it routes chat/vision LLMs, or move `ocr_model` out of it, so the actual OCR model decision has exactly one home.

### SVC-03 confirmed as written but UNREACHABLE: seam_band_crop invents vertical seams inside full-width tiles and re-tiles with zero overlap, but select_retry_strategy never reaches it

- **Where:** `vendor:app/services/seam_strategies.py:122`
- **Gain / effort:** small / low — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
EXECUTED on the real page e-codices_fmb-cb-0002_001r_max (measured 6129x8174) with the natural 3x1 attempt-0 tiling [(0,0,6129,2724),(0,2724,6129,5449),(0,5449,6129,8174)]:
  seam_band_crop returns preproc {'seam_ys': [2724, 5449], 'seam_xs': [3064, 3064], ...} — it INVENTED two vertical seams at x=3064 although all three input tiles span the full page width. Cause: seam_strategies.py:122 `seam_x = (b_left[2] + b_right[0]) // 2` computes (6129+0)//2 = 3064 for two tiles with identical x-spans.
  Resulting 6 tiles: (0,0,3048,2708) (3048,0,6129,2708) (0,2708,3048,5433) (3048,2708,6129,5433) (0,5433,3048,8174) (3048,5433,6129,8174).
  Measured overlaps: 0px between vertically adjacent bands AND 0px horizontally (the plan even self-reports overlap_pct=0.0 at seam_strategies.py:303). Every one of the ~5000px-wide text lines on this page is cut in half at x=3048 with no shared context.

HOWEVER, EXECUTED select_retry_strategy in a loop mirroring routers/ocr.py:3890, attempts 1..7:
  attempt 1..7 -> grid_shift 2x2, grid_shift 3x2, 2x2, 3x2, 2x2, 3x2, 2x2
seam_band_crop is never selected. _STRATEGY_CHAIN (seam_strategies.py:352) puts grid_shift first, and grid_shift (:158) walks a 5-entry _GRID_SEQUENCE returning the first grid that differs from prev — which always exists — so the chain short-circuits on grid_shift every time.
```

- **What it costs today:** None today: the code path cannot execute from the only caller (routers/ocr.py:3890 / :3981). It is a live trap rather than a live cost — the moment grid_shift is changed or a caller invokes seam_band_crop directly, every retry would fragment every text line at a fabricated mid-page seam with zero overlap. Reporting it as cosmetic-today so it is not ranked above the measured losses.

- **Recommendation:** Either delete seam_band_crop and expand_overlap from _STRATEGY_CHAIN and document that retry == grid_shift only, or make the chain actually rotate strategies by attempt index. If seam_band_crop is kept: guard _seam_x_coords / _seam_y_coords to emit a seam only where the two boxes' spans are genuinely adjacent (b_left[2] < b_right[0] + tol) rather than identical, and give the resulting tiles a real overlap instead of the hardcoded overlap_pct=0.0.

### SVC-04 confirmed as written but UNREACHABLE: grid_shift's offset fallback leaves the top 898px (11.0% of page height) uncovered

- **Where:** `vendor:app/services/seam_strategies.py:176`
- **Gain / effort:** small / low — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
EXECUTED `_make_grid(3, 1, 6129, 8174, 0.15, y_offset_frac=0.33)` — the exact call at seam_strategies.py:176 — on the measured real page size:
  tiles: (0,898,6129,4030) (0,3214,6129,6754) (0,5938,6129,8174)
  uncovered rows: 898 = 10.99% of page height, range y=0..897
Cause: seam_strategies.py:214 `y1 = r * tile_h + y_off` shifts the first row down by y_off = int(2724*0.33) = 898 with nothing added above it, while the last row is clipped back to img_h so the bottom loses nothing.

BUT the same reachability execution as SVC-03 shows this fallback is never taken: grid_shift returns at :158-172 from the 5-entry _GRID_SEQUENCE on every attempt (measured attempts 1..7 all returned 2x2 or 3x2, never '3x1_offset'). Reaching :176 requires prev_boxes to be simultaneously identical to all five grids, which is impossible.

For completeness, the grid that IS used has no coverage hole: grid_shift 2x2 on this page returns (0,0,3523,4700) (2605,0,6128,4700) (0,3474,3523,8174) (2605,3474,6128,8174) — 918px x-overlap, 1226px y-overlap, 0 uncovered rows (1px lost at the right edge, x2=6128 vs img_w=6129).
```

- **What it costs today:** None today — dead code, as measured. If it ever became reachable, on this page the missing top band y=0..897 contains the running-title/first-line region (the layout COCO puts a text region at y=471..805), so that text would simply never be read. Reported as speculative-until-reachable, and explicitly NOT ranked above the measured losses.

- **Recommendation:** If the fallback is kept, make it cover the page: either extend the first row up to y=0 (`y1 = max(0, r*tile_h + y_off - (y_off if r == 0 else ov_y))`) or add a leading tile for the offset band, and assert in a test that the union of tile boxes covers [0, img_h) for every grid and offset. Also close the 1px right-edge gap in _clip (:60-67 caps x2 at img_w but the grid never generates it).

### kraken_line_padding=16 is CORRECT and must not be raised — it is horizontal-only padding and exactly matches the value all three models were trained with

- **Where:** `vendor:app/services/ocr_backends.py:580`
- **Gain / effort:** small / low — MEASURED
- **Area:** Image handling before recognition (crop / preprocess / tiling) — archai/vendor/layout/backend + root CLI
- **Evidence:**

```
ocr_backends.py:580 `self.pad = max(0, int(settings.kraken_line_padding))` (config.py:106 default 16), used at ocr_backends.py:616 `rpred.rpred(network, processed, bounds, pad=self.pad)`.
In the installed kraken, rpred.py:137 constructs `ImageInputTransforms(batch, height, width, channels, (pad, 0), valid_norm)` — the tuple is (horizontal, vertical), so pad adds blank columns to the LEFT and RIGHT of the already height-normalised line image and adds nothing vertically. rpred.py:81 documents it as 'Extra blank padding to the left and right of text line'.
MEASURED from the model files themselves: hyper_params['pad'] == 16 for mccatmus (McCATMuS_nfd_nofix_V1), catmus_medieval and cremma_medieval. The runtime value already equals the training value.
```

- **What it costs today:** Zero — this setting is not an accuracy limiter and is already optimal. Recording it so effort is not spent here: ascender/descender coverage is governed entirely by the boundary polygon (finding 1), not by kraken_line_padding. Raising kraken_line_padding would move it away from the trained value and could only hurt.

- **Recommendation:** Leave kraken_line_padding at 16 and add a comment at config.py:106 stating it is horizontal-only and must track the models' hyper_params['pad']. Redirect any ascender/descender concern to _local_boundary_from_metadata.

### iou_threshold is a dead config knob: the shipped zone checkpoint is an end2end (NMS-free) YOLOv10, and ultralytics returns before iou_thres is ever read

- **Where:** `src/archai_ocr/pipeline/layout_yolo.py:103`
- **Gain / effort:** small / low — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
layout_yolo.py:99-105 passes `iou=config.layout.iou_threshold`. MEASURED: `torch.load(weights/layout_yolo.pt)['train_args']` gives `model: 'yolov10b.pt'`, and `YOLO(...).model.end2end` is `True`. ultralytics 8.4.13 `utils/nms.py:66-70`: `if prediction.shape[-1] == 6 or end2end: output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]` then the `classes` filter, then `return output` — `iou_thres` is never referenced on that branch. Confirmed by measurement, not just by reading: detection counts are byte-identical at iou = 0.1 / 0.3 / 0.5 / 0.7 / 0.9 on both 'Latin script.jpg' (21 boxes at conf 0.001) and fmb-cb-0001 (10 boxes); at iou=0.9 a surviving same-class pair has IoU 0.936 on 'Latin script.jpg' and 0.775 on fmb-cb-0001; `agnostic_nms=True` changes nothing (21 -> 21, 10 -> 10). Practical impact at production conf=0.25 across 20 unique real images: exactly 1 page carries a same-class pair above iou_threshold=0.3 that NMS would have merged (acv-P-Antitus, DropCapitalZone pair at IoU 0.84). Note AUDIT_FINDINGS.md:350 marks OCR-006 **FIXED**, but layout_yolo.py still passes iou at line 104 with no end2end detection or warning anywhere in the module.
```

- **What it costs today:** Small on accuracy, large on diagnosability. A researcher tuning iou_threshold to chase duplicate regions gets zero response and will mis-attribute the result. The CLI is protected for main text because layout_yolo.py:153 runs its own `_nms(kept, iou_threshold)` over the class-filtered MainZone set; the vendor pipeline has no equivalent for other classes — `Image.unify_names` (image_batch_classes.py:223-243) only de-duplicates MainZone and MarginTextZone — so duplicate DropCapitalZone/GraphicZone boxes reach the COCO, which is what the one affected page shows.

- **Recommendation:** Detect `getattr(model.model, "end2end", False)` in load_layout_model and either log once that layout.iou_threshold is inert for this checkpoint, or rename the key to dedup_iou_threshold so it plainly describes the only place it still works (the module's own _nms at layout_yolo.py:153). Correct the FIXED marker at AUDIT_FINDINGS.md:350. In the vendor pipeline, extend the unify_names de-duplication beyond MainZone/MarginTextZone, since inference.py:43-52 relies entirely on an iou argument the zone model ignores.

### max_det really does cap detections across all 11 classes before the main-text filter — and OCR-006's proposed fix (passing classes=) provably does not prevent it for this end2end model

- **Where:** `src/archai_ocr/pipeline/layout_yolo.py:103`
- **Gain / effort:** small / low — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
layout_yolo.py:103 `max_det=config.layout.max_regions` (default 50, config.example.yaml:19). ultralytics `utils/nms.py:67` slices `[:max_det]` and only then applies the `classes` filter on line 68-69; `engine/predictor.py:390-394` additionally bakes it into the head for end2end models (`model.set_head_attr(max_det=self.args.max_det, ...)`). MEASURED in fresh processes on 'Latin script.jpg': max_det=1, no class filter -> {'DigitizationArtefactZone': 1}; max_det=1 WITH classes=[3] (MainZone) -> {} — MainZone is starved out even though it was explicitly requested. max_det=2 -> {'DigitizationArtefactZone': 1, 'MainZone': 1}; with classes=[3] -> {'MainZone': 1}. So the mechanism in OCR-006 is real, and the fix it recommends ("Pass classes=[main_text_class_id] to model.predict so ultralytics filters by class before applying max_det") is factually wrong for this checkpoint. MEASURED headroom, however, is comfortable: across 20 unique real images at conf 0.25 the maximum total zone detections on any page is 12 (352386106_MS0001_0008) against a cap of 50 — 4.2x. The vendor path is even safer: inference.py:43-52 passes no max_det at all, so it uses the ultralytics default of 300, and detections at max_det=300 vs 1000 vs 5000 are identical on all four pages tested (emanuskript 28-85, catmus 20-71, zone 1-12).
```

- **What it costs today:** Zero on the pages in this repo — nothing is near the cap. It is a latent cliff: a densely decorated or heavily glossed folio that produces more than 50 zone detections would lose MainZone regions before the class filter ever ran, with no log line. Worth reporting mainly because the recorded fix is wrong and would leave the cliff in place while appearing to close it.

- **Recommendation:** Decouple the two knobs: pass a generous max_det (e.g. 300, the ultralytics default) to model.predict and keep max_regions as the post-filter bound on main-text regions only. Do not rely on `classes=` for this — measurement shows the cap precedes it. Update the fix text at AUDIT_FINDINGS.md:356.

### max_regions truncation is currently unreachable, and where it does bite it drops the rightmost column rather than the bottom of the page — OCR-007's impact description is wrong and the fix was never applied

- **Where:** `src/archai_ocr/pipeline/layout_yolo.py:159`
- **Gain / effort:** small / low — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
layout_yolo.py:159 `ordered = ordered[: config.layout.max_regions]` and layout_yolo.py:168 `logger.info("layout.regions_detected", extra={... "count": len(ordered)})` — the count is taken after the slice, and nothing compares it to len(deduped), so OCR-007's logging claim is confirmed by reading. But line 103 already passes the SAME value as `max_det`, and the class filter (line 125), `_drop_degenerate` (line 149) and `_nms` (line 153) only remove regions, so len(ordered) <= max_regions always holds and the slice can never discard anything as shipped. MEASURED through the real functions: 60 synthetic MainZone regions in 3 columns, `order_regions(regs, mode="column", column_overlap_ratio=0.5)` then `[:50]` discards 10 regions whose bbox x1 is 1900 for all 10 — the entire third column, not the bottom of the page. Only with reading_order="simple" do the discarded regions occupy y1 3300..3900 (the page foot) as OCR-007 describes; "column" is the default (config.py:32). AUDIT_FINDINGS.md:358 marks this FIXED, yet there is no score-based truncation and no warning at layout_yolo.py:159.
```

- **What it costs today:** None today — the slice is dead code given max_det == max_regions. It becomes a silent whole-column loss the moment anyone decouples the two knobs (which is exactly the recommended fix for the max_det item above), which is why it should be corrected in the same change rather than left as a booby trap.

- **Recommendation:** Sort by score descending, slice to max_regions, then sort positionally, and emit `logger.warning` with both len(deduped) and max_regions whenever the slice removes anything. Fix the impact text at AUDIT_FINDINGS.md:363, which assumes the non-default "simple" reading order.

### crop_padding=5 is too small for 6000 px folios: line polygons and drop capitals are clipped at the MainZone crop edge

- **Where:** `src/archai_ocr/pipeline/crop_regions.py:38`
- **Gain / effort:** small / low — MEASURED
- **Area:** layout detection recall (zone detection, region selection, and zone-vs-line segmentation strategy)
- **Evidence:**

```
crop_regions.py:38-42 clamps the crop to `min(x1,x2) - padding` .. `max(x1,x2) + padding` with `padding: int = 5` (config.example.yaml:20, config.py:26). MEASURED against full-page blla line boundaries: on fmb-cb-0001 (6132x8176) zone1 one line polygon of 36 is cut by the padded crop with a worst overhang of 34 px; zone0 one of 35 at 5 px; fmb-cb-0002 one of 21 at 15 px. Separately, DropCapitalZone containment inside MainZone on fmb-cb-0001 measures 0.77 / 0.85 / 0.93 — up to 23% of the illuminated initial lies outside the crop. Visually verified: the DropCapitalZone at (2339,1775,2808,2228) straddles MainZone[1]'s left edge at x=2445, so the left stem and penwork of the illuminated 'R' are 106 px outside a crop padded by 5. 5 px is 0.08% of that page's width.
```

- **What it costs today:** One to three clipped lines per zone lose ascenders/descenders or a leading glyph, and the initial letter of each decorated paragraph reaches the recognizer as a partial glyph. A per-character effect on CER, not a missing-text effect — small next to the items above, and drop capitals are poorly recognised by HTR models regardless.

- **Recommendation:** Make padding relative to page size rather than absolute — e.g. max(crop_padding, round(0.004 * max(page_w, page_h))), about 25-33 px on these folios — or expand each MainZone box to the union of the zones whose area is >50% contained in it before cropping, so a straddling DropCapitalZone is pulled inside. Cheap and safe; the crop already clamps to the page bounds.

### apply_decorated_initial_fix fabricates a fixed word, discarding the initial the recognizer actually read

- **Where:** `vendor:app/agents/ocr_proofreader_agent.py:149`
- **Gain / effort:** small / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
`if len(second) <= 12 and second_norm in DECORATED_INITIAL_WHITELIST: merged_lines = ["Reuerendi", *lines[2:]]` — the literal 'Reuerendi' is emitted regardless of what `first` contained, and `first` is only required to match SINGLE_INITIAL_RE = `^[A-Z]$`. MEASURED: input 'R\neuerendi\n...' -> 'Reuerendi'; input 'K\neuerendi' -> 'Reuerendi'; 'B' -> 'Reuerendi'; 'D' -> 'Reuerendi'; 'P' -> 'Reuerendi'. The whitelist is six document-specific strings ('cuero di','cvero di','euerendi','euerendu','euerendi i','euerendu i'). apply_archai_safe_normalizer runs this on the raw OCR before the LLM sees it (proofread(), line 347) AND on the LLM's output (line 370), so it applies even when the LLM is unavailable or leaves the text alone.
```

- **What it costs today:** A decorated initial that the recognizer read as K, B, D or P is silently replaced by R in the stored diplomatic transcription. It is a small number of characters, but it is an outright fabrication in the one artefact that must be defensible letter by letter, and it is hard-coded from a single document, so it will fire on unrelated manuscripts whose second line happens to start 'euer'.

- **Recommendation:** Prepend the initial that was actually read (`f"{first}{lines[1].lstrip()}"`, as the sibling branch at line 153 already does) instead of emitting the literal 'Reuerendi', and drop the document-specific whitelist. If the merge is only meant to fix a segmentation artefact, it should never change any character — only remove the line break.

### LATIN_SAFE_CORRECTIONS rewrites attested medieval orthography as if it were an OCR error

- **Where:** `vendor:app/agents/ocr_proofreader_agent.py:65`
- **Gain / effort:** small / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
`LATIN_SAFE_CORRECTIONS = {"patru": "patri", "clemencia": "clementia"}`. MEASURED via apply_archai_safe_normalizer(..., 'latin'): 'de clemencia et misericordia dei omnipotentis' -> 'de clementia et misericordia dei omnipotentis'; 'Clemencia' -> 'Clementia'; the rule fires unconditionally on every occurrence, both on the raw OCR before the LLM (line 347) and on its output (line 370). ti->ci assibilation is one of the most common and best-attested medieval Latin orthographic features, not a recognition error. The system prompt at line 40 advertises three examples including 'faunente->fauente'; MEASURED: 'faunente' is not in the dict, so the prompt describes behaviour the deterministic layer does not implement. The whole normaliser is also inert unless _normalize_script_hint returns 'latin' (lines 95-105 map greek/cyrillic/mixed/insular_old_english to 'unknown').
```

- **What it costs today:** A correct medieval spelling is normalised into its classical form in the stored diplomatic transcription, which is precisely the class of change a paleographic edition must not make silently. The blast radius is small today (2 rules) but the mechanism invites growth, and every rule added on this pattern converts an orthographic variant into a fabricated reading.

- **Recommendation:** Remove 'clemencia' (and audit any future entry against a medieval-Latin orthography reference rather than a classical one). Handle ci/ti, y/i, e/ae and ch/h as *matching* equivalences in the search key, never as edits to the transcription. Keep the deterministic rule table and the prompt's advertised examples in sync, or drop the examples from the prompt.

### ocr_confusion_fixes applies unguarded whole-text str.replace, silently rewriting real names in the search key

- **Where:** `vendor:app/services/text_normalization.py:86`
- **Gain / effort:** small / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
_OCR_CONFUSION_MAP is applied as `for src, tgt in _OCR_CONFUSION_MAP: text = text.replace(src, tgt)` (line 128-129) with no word-boundary or position guard. MEASURED: 'Ravna' -> 'Rauna', 'avncle' -> 'auncle', 'vnicorne' -> 'unicorne' (from 'vn'->'un'); 'Noël' -> 'Noel', 'Aïsne' -> 'Aisne', 'Loïs' -> 'Lois' (from the ë->e and ï->i entries at lines 103-104, which fire before the NFKD diacritic strip and are therefore redundant with it). The map is reached from normalize_for_search (line 254) and directly from wikidata_client.py:303-305 on the outbound query surface. _OCR_REGEX_FIXES at line 113 also contains a no-op: `(re.compile(r"(?<=[a-z])ii(?=[a-z])"), "ii")` substitutes 'ii' with 'ii'.
```

- **What it costs today:** Wikidata/authority queries are sent for a mutated surface form, so any place or person name containing the literal bigram 'vn' is looked up under a spelling that does not exist. The effect is narrow (few real medieval names contain 'vn') and the ë/ï entries are harmless because the later NFKD strip would remove the diacritic anyway — so this is a correctness wart with limited measured impact rather than a major recall loss.

- **Recommendation:** Make the alternation rules position-aware (word-initial 'vn'/'vne' only) or fold them into a general u/v equivalence in the matching key instead of a substring edit. Delete the ë->e and ï->i entries (subsumed by the NFKD strip) and the 'ii'->'ii' no-op regex.

### ocr_aware_similarity is dead code whose OCR equivalence classes would merge distinct medieval names if it were ever wired up

- **Where:** `vendor:app/services/text_normalization.py:269`
- **Gain / effort:** small / low — MEASURED
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
`_OCR_CONFUSION_CLASSES = {"i":"I","l":"I","1":"I","u":"U","v":"U","c":"C","e":"C"}` plus a substring 'rn'->'m' (lines 269-292). `grep -rn 'ocr_aware_similarity' --include='*.py'` across the whole repo (excluding venvs) returns exactly one hit: the definition itself at line 295 — it is unused. MEASURED, were it used: over a small list of real medieval place/person names, 6 distinct pairs collapse to similarity 1.000 — Cambrai/Eambrai, Cluny/Eluny, Cologne/Eologne, Corbie/Eorbie, Cort/Eort, Cecile/Cccile; and rn->m gives sim('Bernard','Bemard')=1.000, sim('Vernon','Vemon')=1.000, sim('Cornouaille','Comouaille')=1.000, sim('Milan','Mllan')=1.000, while genuinely useful medieval alternations are NOT collapsed (sim('Cluny','Cluni')=0.750 — i/y is missing from the class table even though the module comment at line 92 claims 'i / j / y interchange').
```

- **What it costs today:** None today — the function is never called, so this is cosmetic. It matters only as a trap: the c/e collapse is far too aggressive for entity matching (it would let 'Eluny' link to Cluny with perfect confidence) while the i/y alternation that actually blocks medieval matching is absent, so wiring it in as-is would trade recall gains for confident false links.

- **Recommendation:** Either delete it, or before using it drop the c/e class, make rn->m position-aware, and add the alternations that actually matter for medieval orthography (i/y, ci/ti, single/double consonant). Note in the docstring that it is matching-only.

### The paleography verification agent never touches the transcription, but its assessment parser fails open to 'partially_supported'

- **Where:** `vendor:app/agents/paleography_verification_agent.py:117`
- **Gain / effort:** small / low — inferred from reading
- **Area:** Post-recognition text handling: normalisation, abbreviation expansion, line-break rejoining, lexical trust, and proofreading (services/text_normalization.py, services/lexicon_trust.py, agents/ocr_proofreader_agent.py, agents/paleography_verification_agent.py, src/archai_ocr/pipeline/assemble_text.py)
- **Evidence:**

```
Read in full. The agent is a downstream answer verifier: its prompt (lines 14-37) reviews a draft answer against evidence and returns assessment/corrected_answer/notes/citations_checked; it contains no normalisation of the OCR text and no write path back into the transcription, so it poses no diplomatic-fidelity risk — a negative result worth recording. The one issue found by reading: `_normalized_assessment` (lines 113-117) accepts only {'supported','partially_supported','unsupported','unavailable'} and `return "partially_supported"` for everything else, including an empty string from a failed parse. Combined with _extract_json_blob (lines 71-96) returning None on unparseable output, a verification that did not actually happen is reported as partial support. NOT measured — this agent needs a live SAIA model call, which I did not make (no network calls per the brief); the control flow is read from source.
```

- **What it costs today:** A failed or unparseable verification is indistinguishable from a genuine 'partially supported' verdict, so an unverified answer carries a verdict that looks like evidence of partial support. This affects reported confidence in retrieval answers rather than OCR characters, and only on the parse-failure path.

- **Recommendation:** Return 'unavailable' (already an accepted value) rather than 'partially_supported' when the assessment string is missing or unrecognised, and surface the parse failure in stage_metadata so a degraded verification is visible instead of silently optimistic.

### kraken is not installed in the root venv, so the CLI pipeline cannot produce transcriptions for evaluation there

- **Where:** `pyproject.toml:23`
- **Gain / effort:** small / low — MEASURED
- **Area:** measurement / evaluation infrastructure (OCR accuracy + retrieval accuracy)
- **Evidence:**

```
`"kraken>=5.3.0",` is declared as a dependency of the root archai_ocr package. MEASURED: with PYTHONPATH=src and YOLO_AUTOINSTALL=false, `./venv/bin/python` imports archai_ocr.cli, archai_ocr.pipeline.layout_yolo, archai_ocr.pipeline.htr_kraken and archai_ocr.pipeline.assemble_text all successfully, and `import ultralytics` succeeds, but `import kraken` → ModuleNotFoundError. The vendor venv at archai/vendor/layout/.venv does have kraken. Related: artifacts/thesis_showcase/manifest.json independently records under unavailable_subsystems "kraken region OCR: package not installed in this environment" and "calamari region OCR: package not installed" — i.e. the showcase figures were built with the HTR engines missing.
```

- **What it costs today:** Minor on its own, but it is a direct obstacle to building the item-1 harness: the root CLI is the cheapest path to a reproducible page→text run (858 LOC, no FastAPI, no LLM), and it cannot recognise anything in its own venv. It also explains why the showcase bundle has no Kraken or Calamari comparison, which is one of the reasons the thesis's multi-engine CER table has no substrate here.

- **Recommendation:** Install kraken into the root venv (or run the harness from archai/vendor/layout/.venv, which has it) and record in eval/results/ which interpreter and which recogniser SHA256 produced each number. Then pin the models: three of the four entries in weights/kraken_models/ are symlinks into ~/Library/Application Support/htrmopo, so add weights/MANIFEST.json with the SHA256s the report header already prints and have the harness assert them before recording a result.

### The only candidate-comparison logic in the repo is a 2-way multiview retry on the VLM path, gated on self-reported confidence and never reachable from the Kraken path

- **Where:** `vendor:app/agents/saia_ocr_agent.py:2298`
- **Gain / effort:** small / medium — inferred from reading
- **Area:** OCR output selection and combination (backend/pass choice, voting, proofreader guards)
- **Evidence:**

```
`_mv_weak = (tile_text and tile_conf < 0.65 and (weird_ratio >= 0.08 or single_char_ratio >= 0.12))`, then `alt_score = alt_conf * (1.0 - alt_sanity.get('weird_ratio', 0.0))` vs `orig_score = tile_conf * (1.0 - tile_sanity.get('weird_ratio', 0.0))` and the higher wins (saia_ocr_agent.py:2310-2336). `generate_variants` can produce 3 variants (multiview.py:44-93) but `pick_retry_variant` (multiview.py:96) returns only the single best non-'enhanced_rgb' variant, so at most 2 hypotheses are ever compared and the third variant is generated and thrown away. `multiview` has exactly one importer (grep: saia_ocr_agent.py:16) and no other caller. `tile_conf` is the VLM's own self-reported `confidence` field parsed out of its JSON reply.
```

- **What it costs today:** This is the closest thing to combination logic that exists, and it is confined to the SAIA/VLM tiled path, triggers only when the model volunteers a confidence below 0.65, and arbitrates using that same self-reported number — LLM self-reported confidence is not calibrated, so the trigger will miss confidently-wrong tiles, which are exactly the ones worth retrying. The Kraken/segmented path (the recommended manuscript path per ocr_backends.py:641) gets no multiview at all.

- **Recommendation:** Drop the self-reported-confidence trigger in favour of the structural signals it already computes (weird_ratio / single_char_ratio / gibberish_score), score all 3 generated variants instead of discarding the third, and extend the same variant loop to the Kraken backends where a real per-character confidence is available to arbitrate with.


# UNKNOWN EXPECTED GAIN (1)

### Retrieval is dense-only — bm25.py / hybrid.py / rerank.py are one-line TODO stubs and nothing equivalent exists in the vendor app, while Chroma's full-text index sits populated and unused

- **Where:** `archai/backend/src/archai_backend/retrieval/bm25.py:1`
- **Gain / effort:** unknown / medium — MEASURED
- **Area:** Knowledge retrieval (RAG): chunking, embeddings, search, index hygiene, evaluation
- **Evidence:**

```
bm25.py, hybrid.py, rerank.py, embeddings.py and vector_index.py are each exactly 1 line: '# TODO: implement'. In the vendor backend, grep -in 'bm25|rerank|cross.encoder|mmr|reciprocal|rrf|hybrid' across app/ returns exactly one hit, an unrelated prompt string in agents/label_analysis_agent.py:36. retrieve_chunks/retrieve_entities issue a single col.query() with no second retriever and no reordering (rag_store.py:622-639, 666-683). Meanwhile Chroma already maintains embedding_fulltext_search tables in the live DB. MEASURED, and this is where I have to contradict the obvious expectation: on the current tiny chunks dense retrieval is already excellent at exact lookup — 12 probes using a verbatim 2-word prefix of a unique chunk returned that chunk at rank 1 in 12/12 cases, and Chroma's unused where_document {'$contains'} FTS also scored 12/12. A cross-line-boundary phrase probe (last 2 words of line i + first 2 of line i+1, a phrase present on the page but in no chunk) still surfaced one of the two source lines in the top-5 in 50/50 cases. So BM25 buys nothing today: a 2-word probe is most of a 5-word chunk. The gain from hybrid/rerank is speculative and only materialises once chunks grow to paragraph size (finding 1), where lexical signal on abbreviated/idiosyncratic manuscript orthography stops being handed to you for free.
```

- **What it costs today:** Not currently a limiter — I measured it and it is not. It becomes one immediately after chunk sizes are fixed: with 400-800 char windows, exact-name and abbreviation lookups ('Ludovicus', 'sancti Dionysii', a roman-numeral date) stop being dominated by the chunk's tiny lexical surface and dense-only recall will start missing them.

- **Recommendation:** Do not build this before finding 1 lands, and delete or clearly mark the stub files so they stop reading as implemented capability. After chunks grow: add a lexical leg using the Chroma full-text index that is already there (where_document $contains, or a small in-process BM25 over the chunk table), fuse with reciprocal-rank fusion, over-fetch ~4x k, then dedupe and truncate to k. Add a cross-encoder reranker only after the eval harness (last finding) exists to prove it helps.

