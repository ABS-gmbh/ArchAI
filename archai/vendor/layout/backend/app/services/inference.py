"""Model pool management and parallel YOLO inference."""

import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict

from app.config import settings
from app.core.constants import MODEL_CLASSES

# Per-process model cache
_worker_models: dict = {}
_model_pool: ProcessPoolExecutor | None = None


def _worker_init():
    """Initialize worker process — models are loaded lazily on first use."""
    global _worker_models
    _worker_models = {}


def _run_single_model(args: tuple) -> tuple[str, str]:
    """Run a single YOLO model prediction in a worker process.

    Models are cached per-process to avoid reloading on subsequent calls.
    """
    global _worker_models
    model_name, model_path, image_path, output_dir, classes, conf, iou = args

    if model_name not in _worker_models:
        from ultralytics import YOLO

        _worker_models[model_name] = YOLO(model_path)

    model = _worker_models[model_name]

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    predict_kwargs = {
        "device": "cpu",
        "conf": conf,
        "iou": iou,
        "augment": False,
        "stream": False,
    }
    if classes is not None:
        predict_kwargs["classes"] = classes

    results = model.predict(image_path, **predict_kwargs)

    image_id = Path(image_path).stem
    json_path = os.path.join(model_dir, f"{image_id}.json")
    with open(json_path, "w") as f:
        f.write(results[0].to_json())

    return model_name, model_dir


def init_model_pool():
    """Create the shared ProcessPoolExecutor. Called once at app startup."""
    global _model_pool
    if _model_pool is None:
        disable_pool_env = os.getenv("ARCHAI_DISABLE_PROCESS_POOL", "").strip().lower()
        disable_pool = disable_pool_env in {"1", "true", "yes"}
        if not disable_pool and sys.platform == "darwin" and disable_pool_env not in {"0", "false", "no"}:
            disable_pool = True
        if disable_pool:
            _model_pool = None
            reason = "ARCHAI_DISABLE_PROCESS_POOL" if disable_pool_env in {"1", "true", "yes"} else sys.platform
            print(f"Process pool disabled ({reason}). Using sequential inference.", flush=True)
            return
        try:
            _model_pool = ProcessPoolExecutor(
                max_workers=settings.max_pool_workers,
                initializer=_worker_init,
            )
            print("Model worker pool initialized.", flush=True)
        except PermissionError:
            _model_pool = None
            print(
                "Process pools are not permitted in this environment. "
                "Using sequential in-process inference.",
                flush=True,
            )


def _run_models_sequential(model_args: list[tuple]) -> Dict[str, str]:
    """Fallback path when multiprocessing is unavailable or breaks."""
    results: Dict[str, str] = {}
    for args in model_args:
        name, dir_path = _run_single_model(args)
        results[name] = dir_path
    return results


def shutdown_model_pool():
    """Shut down the pool. Called at app shutdown."""
    global _model_pool
    if _model_pool is not None:
        _model_pool.shutdown(wait=False)
        _model_pool = None


def run_models_parallel(
    image_path: str,
    output_dir: str,
    conf: float = 0.25,
    iou: float = 0.3,
) -> Dict[str, str]:
    """Run all 3 YOLO models in parallel using the pre-initialized pool.

    Returns:
        Dict mapping model name to its labels folder path.
    """
    global _model_pool
    if _model_pool is None:
        init_model_pool()

    model_args = [
        (
            "emanuskript",
            settings.emanuskript_model_path,
            image_path,
            output_dir,
            MODEL_CLASSES["emanuskript"],
            conf,
            iou,
        ),
        (
            "catmus",
            settings.catmus_model_path,
            image_path,
            output_dir,
            MODEL_CLASSES["catmus"],
            conf,
            iou,
        ),
        (
            "zone",
            settings.zone_model_path,
            image_path,
            output_dir,
            MODEL_CLASSES["zone"],
            conf,
            iou,
        ),
    ]

    if _model_pool is None:
        return _run_models_sequential(model_args)

    try:
        futures = {_model_pool.submit(_run_single_model, args): args[0] for args in model_args}
        results: Dict[str, str] = {}
        for future in as_completed(futures):
            _model_name = futures[future]
            name, dir_path = future.result(timeout=300)
            results[name] = dir_path
        return results
    except (BrokenProcessPool, TimeoutError, RuntimeError, FileNotFoundError) as exc:
        print(
            f"Model worker pool failed ({exc.__class__.__name__}: {exc}). "
            "Falling back to sequential in-process inference.",
            flush=True,
        )
        shutdown_model_pool()
        return _run_models_sequential(model_args)
