import multiprocessing as mp
import os
import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.file_manager import cleanup_expired_tasks
from app.services.inference import init_model_pool, shutdown_model_pool
from app.routers import health, classes, predict, download, analytics, chat, agents_ocr, ocr, evidence, index, rag_debug, authority



_TASK_CLEANUP_INTERVAL_SECONDS = 15 * 60


async def _periodic_task_cleanup() -> None:
    """Sweep expired task directories on a timer for long-running processes."""
    while True:
        await asyncio.sleep(_TASK_CLEANUP_INTERVAL_SECONDS)
        try:
            # Filesystem work, so keep it off the event loop.
            await asyncio.to_thread(cleanup_expired_tasks)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception("Periodic task cleanup failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model pool on startup, clean up on shutdown."""
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
    sweeper = asyncio.create_task(_periodic_task_cleanup())
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
