import multiprocessing as mp
import os
import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logging_setup import configure_logging
from app.services.file_manager import cleanup_expired_tasks
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


app = FastAPI(
    title="Manuscript Layout Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(health.router, prefix="/api")
app.include_router(classes.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(agents_ocr.router, prefix="/api")
app.include_router(ocr.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(index.router, prefix="/api")
app.include_router(rag_debug.router, prefix="/api")
app.include_router(authority.router, prefix="/api")
app.include_router(download.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
