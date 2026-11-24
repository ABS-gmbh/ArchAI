import logging
import os
import secrets

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings

_EPHEMERAL_JWT_SECRET: str | None = None


class Settings(BaseSettings):
    # CORS
    cors_origins: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000,"
        "http://localhost:3001,"
        "http://127.0.0.1:3001"
    )

    # Logging (app/logging_setup.py): level of the root logger, and "text" or
    # "json" lines on stderr. uvicorn's own access and error logs are unaffected.
    log_level: str = "INFO"
    log_format: str = "text"

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL, got {value!r}")
        return normalized

    @field_validator("log_format")
    @classmethod
    def _known_log_format(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"text", "json"}:
            raise ValueError(f"LOG_FORMAT must be text or json, got {value!r}")
        return normalized

    # Failed analytics logins allowed within the window before 429, per client
    # address and per user name from any address (app/services/login_limiter.py).
    analytics_login_max_failures: int = 5
    analytics_login_max_failures_per_username: int = 10
    analytics_login_window_seconds: int = 900

    @field_validator(
        "analytics_login_max_failures", "analytics_login_max_failures_per_username", "analytics_login_window_seconds"
    )
    @classmethod
    def _positive_login_limit(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"{(info.field_name or '').upper()} must be at least 1, got {value}")
        return value

    # /docs, /redoc and /openapi.json, which describe every route to anyone.
    # Unset, they are on without API_TOKEN and off with it (api_docs_enabled);
    # true or false decides either way. Switched on with a token set, they answer
    # without it: a browser opening /docs cannot send the token.
    enable_api_docs: bool | None = None

    # Bearer token for the whole API except /api/health (app/auth.py). Empty, the
    # default, leaves the API open, which is only safe on localhost.
    api_token: str = ""

    @field_validator("api_token")
    @classmethod
    def _long_enough_token(cls, value: str) -> str:
        value = str(value or "").strip()
        if value and len(value) < 24:
            raise ValueError(f"API_TOKEN must be at least 24 characters, got {len(value)}")
        return value

    # Retention of pipeline runs (app/services/run_retention.py). 0 disables the
    # scheduled sweep, and is the default: nothing is deleted unless an operator
    # sets a window. scripts/prune_runs.py previews a window without deleting.
    run_retention_days: int = 0
    run_retention_max_per_sweep: int = 50

    @field_validator("run_retention_days")
    @classmethod
    def _non_negative_retention(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"RUN_RETENTION_DAYS must be 0 (off) or a number of days, got {value}")
        return value

    @field_validator("run_retention_max_per_sweep")
    @classmethod
    def _positive_sweep_cap(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"RUN_RETENTION_MAX_PER_SWEEP must be at least 1, got {value}")
        return value

    # Model paths
    model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # Every OCR run appends its raw and final text here, with model and decoding
    # (app/services/ocr_evidence.py). It used to be "data/evidence/..." relative
    # to the working directory, so the test suite, run from the backend, appended
    # its fake pages to a tracked file. Generated output: gitignored.
    ocr_evidence_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "evidence", "ocr_evidence.jsonl"
    )

    # Analytics
    analytics_db_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "analytics.db"
    )
    # Analytics auth. These intentionally have NO usable defaults: shipping
    # working credentials and a fixed JWT signing secret in tracked source lets
    # anyone who can read the repository forge a valid analytics token. Both must
    # be supplied via environment/.env, or the analytics endpoints refuse to serve.
    analytics_username: str = ""
    analytics_password: str = ""

    # JWT. The HMAC-SHA256 key of every analytics session: at least
    # JWT_SECRET_MIN_LENGTH characters (256 bits of text, as RFC 7518 asks of an
    # HS256 key). A shorter one is logged as an ERROR at startup, not refused, so
    # an existing installation keeps starting. Unset, a random one is made.
    jwt_secret: str = ""
    jwt_expiry_minutes: int = 60

    # Canned OCR fixtures (app/services/ocr_test_fixtures.py) bypass the real
    # OCR pipeline for four specific images matched by SHA-256. They exist for
    # demos and regression tests and MUST stay off by default: with them enabled,
    # an endpoint can return hand-written transcriptions indistinguishable from
    # genuine model output.
    enable_test_ocr_fixtures: bool = False

    # Processing
    max_pool_workers: int = 3
    task_ttl_minutes: int = 60
    # Where layout tasks (/api/predict, /api/download) keep their uploads and
    # outputs, one directory per task, removed after task_ttl_minutes. Must be
    # a writable directory; it is created if missing (checked at startup).
    task_base_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tasks")
    # Upload limits for /predict: one upload, and everything a batch ZIP may
    # expand to. Uploads were read whole into memory and ZIP members
    # inflated without a bound (audit PRED-002).
    max_upload_megabytes: int = 1024
    max_extracted_megabytes: int = 4096
    # Every request body that is not a multipart upload (app/body_limit.py). The
    # largest legitimate one is a page image as a base64 data URL: an A4 page at
    # 600 dpi as PNG makes a 36 MB body.
    max_json_megabytes: int = 48
    # Most members a batch ZIP may have. Empty members cost no extraction
    # budget, so a 5.6 MB ZIP of 60,000 empty .png members created 60,000
    # files in 3.5 s (security review C6). A batch is read page by page, a few
    # seconds each, so 2,000 pages is already hours of work.
    max_zip_members: int = 2000

    # Chat AI (GWDG OpenAI-compatible)
    chat_ai_api_key: str = ""
    chat_ai_base_url: str = "https://chat-ai.academiccloud.de/v1"
    archai_chat_ai_api_key: str = ""
    archai_chat_ai_base_url: str = "https://chat-ai.academiccloud.de/v1"
    archai_chat_ai_model: str = "qwen3-30b-a3b-instruct-2507"
    chat_rag_model: str = "qwen3-30b-a3b-instruct-2507"
    translation_model: str = "llama-3.3-70b-instruct"
    # Per-request timeout of the chat client (chat, translation, model list).
    # Without one the OpenAI SDK waits 600 s, and retries twice, before a
    # non-streaming completion gives up. A streamed answer only needs each
    # chunk within this time; a non-streamed one needs the whole answer.
    chat_ai_timeout_seconds: float = 180

    @field_validator("chat_ai_timeout_seconds")
    @classmethod
    def _positive_chat_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"CHAT_AI_TIMEOUT_SECONDS must be greater than 0, got {value}")
        return value

    # Longest text a chat message may carry, per message and per text part (an
    # attached page image is bounded by the request body limit instead). A typed
    # question is a few hundred characters; an earlier answer or a pasted
    # passage a few thousand. Longer text was only an input-side cost.
    chat_max_message_chars: int = 20000

    # SAIA OCR Agent configuration
    saia_api_key: str = ""
    saia_base_url: str = "https://chat-ai.academiccloud.de/v1"
    saia_timeout_seconds: int = 120
    saia_models_cache_ttl_seconds: int = 300
    saia_label_analysis_model: str = "qwen3-vl-30b-a3b-instruct"
    label_visual_model: str = "qwen3-vl-30b-a3b-instruct"
    label_visual_fallback_model: str = "internvl3.5-30b-a3b"
    saia_ocr_model_preferences: str = ""
    saia_ocr_model_prefs: str = ""
    saia_ocr_models: str = ""
    saia_ocr_temperature: float = 0.0
    saia_ocr_max_tokens: int = 4096
    ocr_crop_upscale: int = 2
    ocr_max_pixels_per_tile: int = 160000000
    ocr_max_long_edge: int = 12000
    # The largest page the provider path decodes at all (width x height); larger
    # pages are refused before a pixel is read. The default is PIL's own refusal
    # point (twice Image.MAX_IMAGE_PIXELS); a page under it but over
    # ocr_max_pixels_per_tile is decoded and resized down. At 3 bytes a pixel
    # the default is about 540 MB once decoded.
    ocr_max_decode_pixels: int = 178956970
    ocr_image_size_retry_limit: int = 2
    ocr_image_retry_shrink: float = 0.82
    # Tiles of one page read by the VLM at once (ocr_agent.run_ocr_extraction).
    # Each tile is an independent blocking request; one after another, a page of
    # 17 tiles took 17 round trips end to end (audit AG-11). 2 is a cautious
    # default: the provider's rate limits are not known, and the speed-up on it
    # has not been measured. 1 restores the sequential loop.
    ocr_tile_concurrency: int = 2

    # Backward-compatible SAIA key names (legacy config)
    archai_saia_api_key: str = ""

    # RAG / ChromaDB
    chroma_persist_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".data", "chroma"
    )
    rag_collection_name: str = "archai_chunks"
    rag_entity_collection_name: str = "archai_entities"
    rag_embedding_model: str = "multilingual-e5-large-instruct"
    # Retrieval chunk geometry. One line per chunk (window_lines=1) was the
    # original behaviour and starved the context: measured 24.1 chars median, so
    # top_k=5 delivered ~136 chars, about 11% of a page. 6-line windows with 2
    # lines of overlap raise the share of multi-line spans a single chunk can
    # contain from 0% to 100% on a real page, and the text 5 evidence slots carry
    # from 198 to 1304 chars. Those bound what retrieval COULD return; they are
    # not themselves retrieval measurements. Set rag_chunk_window_lines=1 to
    # restore per-line chunks.
    rag_chunk_window_lines: int = 6
    rag_chunk_overlap_lines: int = 2

    # Chunks retrieved for a chat question. Overlapping hits share one evidence
    # block (rag_store.format_evidence_blocks), so 8 costs less prompt than 5
    # did unmerged. On 62 answerable questions (scripts/evaluate_chat_answers.py,
    # local MiniLM, four pages) the answer reached the evidence for 53 at 5
    # and 56 at 8; mean evidence 1,739 characters at 5 unmerged, 1,620 at 8
    # merged. The 9 misses at 5 are English questions sharing no word with the
    # Old French or Latin passage. With multilingual-e5 (--embedder e5-local)
    # it reached the evidence for 55 at 5 and all 62 at 8 (eval/chat/README.md).
    rag_top_k: int = 8
    rag_entity_top_k: int = 4
    rag_auto_index: bool = True
    # Hybrid retrieval: the dense ranking is fused (Reciprocal Rank Fusion) with
    # character 4-gram BM25 over the abbreviation-expanded search key. The right
    # chunk reached the top 5 - what chat passed on before rag_top_k was 8 - for 89.6% of 222 passage
    # lookups over four reference pages dense-only, 95.9% with BM25 alone and
    # 96.8% fused (scripts/benchmark_retrieval.py --readings, ChromaDB's default
    # model; 91.0% / 95.9% / 97.7% with e5). Set rag_lexical_fusion=False for
    # dense only.
    rag_lexical_fusion: bool = True
    rag_rrf_k: int = 60
    # BM25's weight in the fusion; the dense ranking's is 1. At 2, passage
    # lookup improves (Hit@5 96.8% -> 98.2%, MRR 0.863 -> 0.903; with e5 MRR
    # 0.817 -> 0.883), but questions about the same pages (--questions, 151
    # lookups) do not with e5: MRR -0.011 [-0.028, +0.002], Hit@5 -0.007
    # [-0.020, 0.000]. On questions BM25 alone finds the answer in the top 5 for
    # 49% and e5 for 82%. With ChromaDB's default model 2 helps both (+0.049
    # MRR on questions). The default stays 1.
    rag_lexical_weight: float = 1.0
    # The weight used instead when the dense side is ChromaDB's default model
    # (rag_embedding_model empty): an English MiniLM that ranks Old French and
    # Latin worse than BM25 does. Measured against weight 1 with that model
    # (scripts/benchmark_retrieval.py, paired 95% CIs over readings): 270
    # question lookups (--questions, 4 pages) MRR +0.037 [+0.011, +0.063],
    # Hit@1 +0.056 [+0.011, +0.099]; 222 passage lookups (--readings) MRR
    # +0.040 [+0.019, +0.063], Hit@1 +0.068 [+0.029, +0.105]. Weight 3 was no
    # better on passages. The e5 space keeps rag_lexical_weight (1): with e5 on
    # the 270 question lookups, hybrid MRR is 0.601 / 0.611 / 0.598 / 0.591 at
    # weights 0.5 / 1 / 2 / 3, a flat curve (eval/retrieval/README.md). In the
    # local space rag_lexical_weight is not read: set this one
    # (RAG_LEXICAL_WEIGHT_LOCAL) to change the weight.
    rag_lexical_weight_local: float = 2.0
    # Candidates each ranker contributes to the fusion.
    rag_fusion_pool: int = 50
    # The lexical index is built per query; scopes larger than this skip it.
    rag_lexical_max_chunks: int = 5000

    # Authority source expansion
    geonames_base_url: str = "https://secure.geonames.org"
    # Empty disables GeoNames. The shared "demo" account is over its daily
    # quota and answers every search with an error (audit AL-12).
    geonames_username: str = ""
    geonames_timeout_seconds: int = 10

    # Paleography verification
    paleography_verification_enabled: bool = True
    paleography_verification_model: str = "qwen3-235b-a22b"
    paleography_verification_temperature: float = 0.0
    paleography_verification_max_tokens: int = 900
    # Per request; the OpenAI client retries twice, so a stalled verifier holds
    # a chat answer for at most three times this before it degrades.
    paleography_verification_timeout_seconds: int = 60

    # OCR backend routing
    ocr_backend_default: str = "auto"
    # Engine behind /ocr/extract_full_page when the request leaves the backend on
    # "auto". "segmented": layout segmentation, then line-by-line Kraken
    # recognition (CATMuS Medieval first), with GLM-OCR as the fallback when a
    # page yields no text regions or no text. "glmocr": GLM-OCR on the whole
    # page, as before. Over the five reference pages the segmented engine reads
    # at 7.6% CER, GLM-OCR at 27.7% (scripts/benchmark_recognizers.py,
    # eval/recognizers/); at 6.7% with versal initials joined
    # (ocr_join_detached_initials).
    full_page_ocr_engine: str = "segmented"
    # Let a text-quality score reorder the backends of an "auto" plan and replace
    # a backend's non-empty text with another's. The score favours fluent-looking
    # text over faithful diplomatic transcription: switched on, it raises CER over
    # the five reference pages from 7.6% to 8.4%, and on the Latin page from 6.6%
    # to 8.6%. Off, the plan order decides, and a later backend reads a line only
    # when every earlier one fails or reads nothing.
    ocr_backend_election: bool = False
    # Join a versal initial that Kraken wrote apart from its word ("C aldꝰ" ->
    # "Caldꝰ"), as the reference transcriptions do (app/services/detached_initials.py).
    # Over the five reference pages it changes two, the ones with versals: CER
    # on latin-abaton 6.6% -> 5.5%, on old-french-verse 7.2% -> 5.4%; the other
    # three are unchanged. The three thresholds below were set on those two
    # development pages, with one annotator.
    ocr_join_detached_initials: bool = True
    # A column needs this many lines before its initials are judged: protects a
    # short block (a heading, a marginal note) where two detached glyphs would
    # already be a majority.
    ocr_detached_initial_min_column_lines: int = 5
    # Share of a column's lines that must begin with a detached glyph: protects
    # pages that do not set initials apart. On latin-apocalypse, 2 of 21 lines
    # begin that way, both misreads ("R Rian"), and stay as read.
    ocr_detached_initial_min_share: float = 0.5
    # The gap after the glyph, as a share of the line's median word gap, must be
    # at least this: protects a one-glyph word at word spacing (Qͥ, Q̾), which
    # measured 0.47-0.99, below every initial (1.15 and up).
    ocr_detached_initial_min_gap_ratio: float = 1.0
    # Runs reserved with POST /ocr/runs for a client to follow. The route is
    # unauthenticated, so reservations are bounded: one not used within the TTL
    # can no longer be claimed and is deleted by the next reservation, and no
    # more than the cap may wait at once (beyond it the route answers 429).
    ocr_run_reservation_ttl_seconds: int = 3600
    ocr_run_reservation_max_live: int = 100

    # Kraken recognition
    kraken_device: str = "cpu"
    kraken_line_padding: int = 16
    kraken_models_dir: str = "weights/kraken_models"
    kraken_default_recognition_model_path: str = "weights/kraken_recognition.mlmodel"
    kraken_mccatmus_model_path: str = "weights/kraken_models/mccatmus.mlmodel"
    kraken_catmus_model_path: str = "weights/kraken_models/catmus_medieval.mlmodel"
    kraken_cremma_medieval_model_path: str = "weights/kraken_models/cremma_medieval.mlmodel"
    kraken_cremma_lat_model_path: str = "weights/kraken_models/cremma_medieval_lat.mlmodel"

    # Calamari recognition
    calamari_models_dir: str = "archai/backend/weights/calamari_models"
    calamari_default_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/gt4histocr"
    calamari_gt4histocr_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/gt4histocr"
    calamari_historical_french_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/historical_french"
    calamari_antiqua_historical_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/antiqua_historical"
    calamari_fraktur_historical_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/fraktur_historical"
    calamari_gothic_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/gt4histocr"
    calamari_bastard_model_dir: str = "archai/backend/weights/calamari_models/gt4histocr_tmp/deep3_htr-bastard"
    calamari_english_model_dir: str = "archai/backend/weights/calamari_models/calamari_models_v1/uw3-modern-english"

    # GLM-OCR (self-hosted mode)
    glmocr_device: str = "0"
    glmocr_ollama_host: str = "http://localhost:11434"
    glmocr_ollama_model: str = "glm-ocr:latest"
    glmocr_ollama_timeout_seconds: int = 300
    glmocr_ollama_temperature: float = 0.0
    glmocr_ollama_retries_per_variant: int = 2
    glmocr_max_payload_bytes: int = 2000000

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def emanuskript_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_emanuskript_segmentation.pt")

    @property
    def catmus_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_catmus.pt")

    @property
    def zone_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_zone_detection.pt")

    @property
    def analytics_auth_configured(self) -> bool:
        """True only when operator-supplied analytics credentials are present."""
        return bool(self.analytics_username and self.analytics_password)

    def resolve_jwt_secret(self) -> str:
        """Return the configured JWT secret, or a per-process ephemeral one.

        Falling back to a random secret keeps a dev instance usable while making
        the consequence explicit: tokens do not survive a restart, and no secret
        is ever baked into the source tree.
        """
        if self.jwt_secret:
            return self.jwt_secret
        global _EPHEMERAL_JWT_SECRET
        if _EPHEMERAL_JWT_SECRET is None:
            _EPHEMERAL_JWT_SECRET = secrets.token_urlsafe(48)
            logging.getLogger(__name__).warning(
                "JWT_SECRET is not set; generated an ephemeral signing secret. "
                "Analytics tokens will be invalidated on restart. Set JWT_SECRET "
                "to a persistent random value in production."
            )
        return _EPHEMERAL_JWT_SECRET

    model_config = {"env_file": (".env", ".env.local"), "extra": "ignore"}


JWT_SECRET_MIN_LENGTH = 32


def api_docs_enabled(config: Settings) -> bool:
    """ENABLE_API_DOCS if it is set; otherwise on only while the API has no token."""
    if config.enable_api_docs is not None:
        return bool(config.enable_api_docs)
    return not config.api_token


# Read with os.getenv where they are used, not through Settings, which reads
# .env into its own fields only: set in .env, as .env.example shows, they were
# ignored, and ARCHAI_DB_PATH there left the database inside the checkout.
ENV_ONLY_VARIABLES = ("ARCHAI_DB_PATH", "ARCHAI_WIKIDATA_CACHE", "ARCHAI_DEBUG_RAG", "ARCHAI_DISABLE_PROCESS_POOL")


def export_env_only_variables(env_files: tuple[str, ...] = (".env", ".env.local")) -> None:
    """Copy ENV_ONLY_VARIABLES from the .env files into the environment.

    The files are read as Settings reads them: from the working directory, a
    later file overriding an earlier one, and the process environment winning
    over both.
    """
    from dotenv import dotenv_values  # a dependency of pydantic-settings

    found: dict[str, str] = {}
    for env_file in env_files:
        if os.path.isfile(env_file):
            values = dotenv_values(env_file)
            found.update({key: value for key, value in values.items() if key in ENV_ONLY_VARIABLES and value is not None})
    for key, value in found.items():
        os.environ.setdefault(key, value)


export_env_only_variables()
settings = Settings()
