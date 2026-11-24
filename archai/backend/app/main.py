import multiprocessing as mp
import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import ApiTokenGate, require_api_token
from app.body_limit import RequestBodyLimit, upload_paths
from app.error_handlers import UnhandledErrors, install_error_handlers
from app.security_headers import SecurityHeaders
from app.config import JWT_SECRET_MIN_LENGTH, api_docs_enabled, settings
from app.logging_setup import configure_logging
from app.process_guard import guard_for_database
from app.db import pipeline_db
from app.services.data_paths import check_data_paths, log_data_paths
from app.services.model_files import log_missing_models
from app.services.file_manager import check_task_base_dir, cleanup_expired_tasks
from app.services.inference import init_model_pool, shutdown_model_pool
from app.services.run_retention import RetentionSweeper
from app.routers import health, classes, predict, download, analytics, chat, agents_ocr, ocr, evidence, index, rag_debug, authority



_TASK_CLEANUP_INTERVAL_SECONDS = 15 * 60


async def _periodic_task_cleanup(retention: RetentionSweeper) -> None:
    """Sweep expired task directories, and expired runs if retention is on, on a timer."""
    while True:
        await asyncio.sleep(_TASK_CLEANUP_INTERVAL_SECONDS)
        try:
            # Filesystem work, so keep it off the event loop.
            await asyncio.to_thread(cleanup_expired_tasks)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception("Periodic task cleanup failed.")
        await _retention_sweep(retention)


async def _retention_sweep(retention: RetentionSweeper) -> None:
    """Off unless RUN_RETENTION_DAYS > 0; see app/services/run_retention.py."""
    try:
        await asyncio.to_thread(retention.sweep)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.getLogger(__name__).exception("Run retention sweep failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model pool on startup, clean up on shutdown."""
    configure_logging(settings.log_level, settings.log_format)
    if settings.analytics_auth_configured and not settings.jwt_secret:
        # Not a refusal: analytics still works, with sessions that end at the
        # next restart, and a server that starts today keeps starting.
        logging.getLogger(__name__).error(
            "JWT_SECRET is not set while analytics login is configured: analytics tokens are signed "
            "with a random secret made at startup, so every login ends when the process restarts, "
            "and a second worker would reject the first one's tokens. Set JWT_SECRET to a long random value."
        )
    elif settings.jwt_secret and len(settings.jwt_secret) < JWT_SECRET_MIN_LENGTH:
        # Not a refusal either: an installation with a short secret keeps starting.
        logging.getLogger(__name__).error(
            f"JWT_SECRET is shorter than {JWT_SECRET_MIN_LENGTH} characters: it is the key that signs "
            "analytics sessions, and a short one can be guessed offline from any session token. Set "
            f"JWT_SECRET to a random value of at least {JWT_SECRET_MIN_LENGTH} characters (.env.example "
            "shows how); every analytics session ends when it changes."
        )
    if not settings.api_token:
        logging.getLogger(__name__).warning(
            "API_TOKEN is not set: the API is unauthenticated and must only be reachable from "
            "localhost. Set API_TOKEN before exposing it (docs/deployment.md)."
        )
    # A data path this process cannot write stops startup here, naming its
    # setting, instead of failing on first use or, for the OCR evidence log,
    # silently not being written.
    check_data_paths()
    log_data_paths(check_task_base_dir())
    # Missing weights or packages do not stop the server (chat, retrieval and
    # GLM-OCR still work), but the first page read would fail without a word.
    log_missing_models()
    # One worker process is supported (docs/deployment.md); a second one using
    # the same pipeline database is logged as an error, not refused.
    process_guard = guard_for_database(pipeline_db._db_config().path)
    process_guard.acquire()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    init_model_pool()

    # Reclaim expired task directories at startup and then periodically.
    # cleanup_expired_tasks existed but had no callers, so .tasks/ grew without
    # bound - 462 directories totalling 3.0 GB, the oldest 218 days past a
    # 60-minute TTL. Startup is the important one: it is the only moment the
    # orphans left by a previous process are guaranteed to be swept.
    cleanup_expired_tasks()
    # With retention on, the first sweep only logs what the next one deletes.
    retention = RetentionSweeper()
    await _retention_sweep(retention)
    sweeper = asyncio.create_task(_periodic_task_cleanup(retention))
    try:
        yield
    finally:
        sweeper.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper
        shutdown_model_pool()
        process_guard.release()


# /docs, /redoc and /openapi.json: on without API_TOKEN, off with it, unless
# ENABLE_API_DOCS says otherwise (app/config.py).
_docs = api_docs_enabled(settings)
app = FastAPI(
    title="Manuscript Layout Analysis API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
    # A trailing slash was redirected (307) to a URL built from the Host the
    # frontend proxy sends: the backend's own address, passed to the browser.
    # No route ends in a slash and nothing calls one (tests/test_no_slash_redirects.py).
    redirect_slashes=False,
)

# /api/health stays open for monitors and load balancers. Every other router
# requires API_TOKEN when it is set (app/auth.py); add new routers to this list.
API_PREFIX = "/api"
OPEN_ROUTERS = (health.router,)
PROTECTED_ROUTERS = tuple(
    module.router
    for module in (classes, predict, chat, agents_ocr, ocr, evidence, index, rag_debug, authority, download, analytics)
)

install_error_handlers(app)
# Innermost: an exception no route handles is answered in JSON here, so the
# answer passes through every middleware below (app/error_handlers.py).
app.add_middleware(UnhandledErrors)
# Added first so it sits inside CORS: a 413 still carries the CORS headers the
# browser needs to read it. Multipart bodies get the upload limit only on the
# routes that take an upload; every other body gets the JSON limit.
app.add_middleware(
    RequestBodyLimit,
    upload_paths=upload_paths((API_PREFIX, router) for router in OPEN_ROUTERS + PROTECTED_ROUTERS),
)
# Outside the body limit and inside CORS: a request without the API token is
# answered 401 before any of its body is received (app/auth.py).
app.add_middleware(ApiTokenGate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)
# Outermost, so every answer carries it: 413s, CORS preflights, errors.
app.add_middleware(SecurityHeaders)

for _router in OPEN_ROUTERS:
    app.include_router(_router, prefix=API_PREFIX)
_PROTECTED = [Depends(require_api_token)]
for _router in PROTECTED_ROUTERS:
    app.include_router(_router, prefix=API_PREFIX, dependencies=_PROTECTED)
