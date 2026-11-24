"""
Prediction endpoints — single image and batch processing.
"""

import asyncio
import json
import logging
import os
import time
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from typing import Any, NamedTuple, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.core.constants import FINAL_CLASSES
from app.agents.segmentation_agent import run_single_segmentation
from app.config import settings
from app.core.model_runner import combine_and_filter_predictions
from app.services.coco_utils import filter_coco_by_classes, merge_coco_list, stats_from_coco
from app.services.image_limits import too_many_pixels
from app.services.public_errors import public_failure
from app.services.file_manager import (
    TaskState,
    create_task,
    discard_task,
    get_task,
    get_task_dir,
    hold_task,
    release_task,
    save_task_state,
)
from app.services.inference import run_models_parallel
from app.services.visualization import draw_coco_on_image

router = APIRouter(tags=["predict"])

# Valid image extensions for batch processing
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
SKIP_PATTERNS = ["._", ".DS_Store", "Thumbs.db", "desktop.ini", "~$"]
SEGMENTATION_MAX_PIXELS = int(os.getenv("SEGMENTATION_MAX_PIXELS", "85000000") or "85000000")
SEGMENTATION_MAX_LONG_EDGE = int(os.getenv("SEGMENTATION_MAX_LONG_EDGE", "12000") or "12000")


def _secure_filename(name: str) -> str:
    """The name the upload is stored under in its task directory: "upload" and a known image suffix.

    The client's name was used as given after basename(): "." and ".." named
    the directory itself and a 300-character name exceeded the file system's
    limit, each a 500; "state.json" and "annotated.jpg" were overwritten by the
    task's own files. Only the suffix is kept, and only a known one.
    """
    suffix = os.path.splitext(os.path.basename(name or ""))[1].lower()
    return f"upload{suffix if suffix in IMAGE_EXTENSIONS else '.img'}"


def _parse_classes(classes: Optional[str]) -> list[str]:
    if not classes:
        return FINAL_CLASSES
    try:
        selected = json.loads(classes)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid 'classes' JSON.")

    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        raise HTTPException(status_code=422, detail="'classes' must be a JSON array of strings.")
    if not selected:
        raise HTTPException(status_code=422, detail="Please select at least one class.")
    return selected


_COPY_CHUNK = 1 << 20


async def _save_upload(upload: UploadFile, path: str, *, empty_detail: str) -> None:
    """Stream an upload to disk, refusing more than max_upload_megabytes with 413."""
    limit = max(1, int(settings.max_upload_megabytes)) * (1 << 20)
    total = 0
    with open(path, "wb") as out:
        while chunk := await upload.read(_COPY_CHUNK):
            total += len(chunk)
            if total > limit:
                break
            out.write(chunk)
    if total > limit:
        os.remove(path)
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_upload_megabytes} MB.")
    if total == 0:
        os.remove(path)
        raise HTTPException(status_code=422, detail=empty_detail)


_MIN_MEMBER_COST = 4096


class BatchArchiveRefused(ValueError):
    """The batch ZIP breaks a limit; the message says which, and is shown to the client."""


# The most central-directory bytes a member may average: its 46-byte record plus
# 978 for name, extra field and comment. Real batch archives average 100-200;
# zipfile parses exactly the directory bytes the end record states, so bounding
# them bounds the objects it builds on open, whatever count the record claims.
_MAX_DIRECTORY_BYTES_PER_MEMBER = 1024


def _zip_directory(zip_path: str) -> tuple[int, int] | None:
    """The member count and central-directory size the end record states, read without parsing the directory.

    zipfile.ZipFile builds an object per member on open, so both are checked
    first: a 1 GB upload of empty members would be about 11 million of them,
    and an end record can state 1 member over a directory of a million.
    None when the record cannot be read; the caller then counts after opening.
    """
    try:
        with open(zip_path, "rb") as handle:
            record = zipfile._EndRecData(handle)  # type: ignore[attr-defined]
        if not record:
            return None
        return int(record[zipfile._ECD_ENTRIES_TOTAL]), int(record[zipfile._ECD_SIZE])  # type: ignore[attr-defined]
    except (OSError, AttributeError, IndexError, TypeError, ValueError):
        return None


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Extract zip safely (prevents Zip Slip path traversal, and stops once the
    extracted bytes pass max_extracted_megabytes, so a ZIP bomb cannot fill
    memory or disk).
    """
    budget = max(1, int(settings.max_extracted_megabytes)) * (1 << 20)
    limit = max(1, int(settings.max_zip_members))
    directory = _zip_directory(zip_path)
    if directory is not None:
        entries, directory_bytes = directory
        if entries > limit:
            raise BatchArchiveRefused(f"ZIP has {entries} members; at most {limit} are accepted (MAX_ZIP_MEMBERS).")
        if directory_bytes > limit * _MAX_DIRECTORY_BYTES_PER_MEMBER:
            raise BatchArchiveRefused(
                f"ZIP directory is {directory_bytes} bytes, more than {limit} members need (MAX_ZIP_MEMBERS)."
            )
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        if len(members) > limit:
            raise BatchArchiveRefused(f"ZIP has {len(members)} members; at most {limit} are accepted (MAX_ZIP_MEMBERS).")
        for member in members:
            member_name = member.filename

            # Skip obvious junk folders/files
            base = os.path.basename(member_name)
            if not base:
                continue
            if any(base.startswith(p) for p in SKIP_PATTERNS) or base.startswith("."):
                continue

            dest_path = os.path.normpath(os.path.join(extract_dir, member_name))
            if not dest_path.startswith(os.path.abspath(extract_dir) + os.sep):
                # Path traversal attempt
                continue

            if member.is_dir():
                continue
            # Only images are read from a batch, so nothing else is written.
            if os.path.splitext(base)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            # A file costs at least a disk block, even an empty one.
            budget -= _MIN_MEMBER_COST
            if budget < 0:
                raise BatchArchiveRefused(f"ZIP expands past {settings.max_extracted_megabytes} MB; extraction stopped.")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                while chunk := src.read(_COPY_CHUNK):
                    budget -= len(chunk)
                    if budget < 0:
                        raise BatchArchiveRefused(
                            f"ZIP expands past {settings.max_extracted_megabytes} MB; extraction stopped."
                        )
                    dst.write(chunk)


class PreparedImage(NamedTuple):
    """Result of preparing a page for YOLO inference.

    ``scale`` is the factor the page was multiplied by. Coordinates produced by
    segmenting ``path`` are in the prepared image's space, so they must be
    divided by ``scale`` before being used against the ORIGINAL upload.
    """

    path: str
    changed: bool
    scale: float


def rescale_coco_to_original(coco: dict[str, Any], scale: float) -> dict[str, Any]:
    """Map COCO geometry from a downscaled page back onto original coordinates.

    Segmentation runs on a bounded copy of the page (SEGMENTATION_MAX_PIXELS /
    SEGMENTATION_MAX_LONG_EDGE), but every crop downstream is taken from the
    original upload. Without this inverse scale the two disagree, and on a page
    large enough to trigger the resize every region crop is cut from the wrong
    pixels while still looking like a plausible box.

    Mutates nothing; returns a new dict.
    """
    if not coco or scale >= 1.0 or scale <= 0.0:
        return coco

    inv = 1.0 / scale
    out = dict(coco)

    images = []
    for image in coco.get("images") or []:
        entry = dict(image)
        if entry.get("width"):
            entry["width"] = int(round(float(entry["width"]) * inv))
        if entry.get("height"):
            entry["height"] = int(round(float(entry["height"]) * inv))
        images.append(entry)
    if images:
        out["images"] = images

    annotations = []
    for annotation in coco.get("annotations") or []:
        entry = dict(annotation)
        bbox = entry.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            entry["bbox"] = [float(v) * inv for v in bbox[:4]]
        if entry.get("area") is not None:
            try:
                # Area is two-dimensional, so it scales by the square.
                entry["area"] = float(entry["area"]) * inv * inv
            except (TypeError, ValueError):
                pass
        segmentation = entry.get("segmentation")
        if isinstance(segmentation, list):
            rescaled_polys = []
            for poly in segmentation:
                if isinstance(poly, (list, tuple)):
                    rescaled_polys.append([float(v) * inv for v in poly])
                else:
                    rescaled_polys.append(poly)
            entry["segmentation"] = rescaled_polys
        annotations.append(entry)
    if annotations:
        out["annotations"] = annotations

    return out


class _TooManyPixels(Exception):
    """The image's header declares more pixels than OCR_MAX_DECODE_PIXELS; the message is for the client."""


def _prepare_image_for_segmentation(image_path: str) -> PreparedImage:
    """
    Ensure image is valid for YOLO inference:
    - decodable image
    - 3-channel RGB
    - bounded dimensions/pixel count for safer processing
    Returns PreparedImage(path, changed, scale). Callers that use the resulting
    coordinates against the original image MUST apply rescale_coco_to_original.
    """
    try:
        with Image.open(image_path) as im:
            # From the header, before load() decodes a pixel.
            refusal = too_many_pixels(im)
            if refusal is not None:
                raise _TooManyPixels(refusal)
            im.load()
            width, height = im.size
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions.")

            working = im.convert("RGB")
            changed = im.mode != "RGB"

            max_pixels = max(1, SEGMENTATION_MAX_PIXELS)
            max_long_edge = max(1, SEGMENTATION_MAX_LONG_EDGE)
            scale = 1.0
            total_pixels = width * height
            if total_pixels > max_pixels:
                scale = min(scale, sqrt(max_pixels / float(total_pixels)))
            long_edge = max(width, height)
            if long_edge > max_long_edge:
                scale = min(scale, max_long_edge / float(long_edge))

            if scale < 1.0:
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                working = working.resize((new_w, new_h), Image.Resampling.BICUBIC)
                changed = True

            if not changed:
                return PreparedImage(image_path, False, 1.0)

            normalized_path = f"{image_path}.seg.jpg"
            working.save(normalized_path, format="JPEG", quality=95)
            return PreparedImage(normalized_path, True, scale)
    except _TooManyPixels as exc:
        raise HTTPException(status_code=413, detail=exc.args[0]) from exc
    except Exception as exc:
        # The exception text names the file's path on the server; the UI shows
        # this detail verbatim, so it says what is wrong, not where.
        _LOGGER.info("Refused an upload that is not a readable image: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not an image the server can read. Use PNG, JPEG or TIFF.",
        ) from exc


def _is_valid_image_file(image_path: str) -> bool:
    """A readable image within OCR_MAX_DECODE_PIXELS; a larger one is skipped from its header, undecoded."""
    try:
        with Image.open(image_path) as im:
            if too_many_pixels(im) is not None:
                _LOGGER.info("Skipped a batch image over OCR_MAX_DECODE_PIXELS: %s", os.path.basename(image_path))
                return False
            im.load()
            width, height = im.size
            return width > 0 and height > 0
    except Exception:
        return False


@router.post("/predict/single")
async def predict_single(
    image: UploadFile = File(...),
    confidence: float = Form(0.25, ge=0.0, le=1.0),
    iou: float = Form(0.3, ge=0.0, le=1.0),
    classes: Optional[str] = Form(None),
):
    """Run 3 YOLO models on a single image and return combined COCO JSON."""
    selected_classes = _parse_classes(classes)

    # Create task and save uploaded image
    task = create_task()
    task_dir = get_task_dir(task.task_id)

    safe_name = _secure_filename(image.filename or "upload.jpg")
    img_path = os.path.join(task_dir, safe_name)

    try:
        await _save_upload(image, img_path, empty_detail="Empty upload.")

        # Validate + normalize to RGB and safe size for model inference. In a
        # worker thread: decoding and resizing a page of up to 85 megapixels took
        # seconds on the event loop, stalling every other request (audit PRED-003).
        prepared = await asyncio.to_thread(_prepare_image_for_segmentation, img_path)
    except HTTPException:
        # Refused before anything was produced: nothing to keep for download.
        discard_task(task.task_id)
        raise
    prepared_img_path, prepared_changed = prepared.path, prepared.changed

    annotated_path = os.path.join(task_dir, "annotated.jpg")
    try:
        filtered_coco, stats = await asyncio.to_thread(
            run_single_segmentation,
            prepared_img_path,
            confidence=confidence,
            iou=iou,
            selected_classes=selected_classes,
            annotated_output_path=annotated_path,
        )
    except Exception as exc:
        _LOGGER.exception("Layout segmentation failed for task %s", task.task_id)
        discard_task(task.task_id)
        raise HTTPException(
            status_code=500,
            detail="Layout segmentation failed on the server. The details are in the server log.",
        ) from exc
    finally:
        if prepared_changed and prepared_img_path.endswith(".seg.jpg"):
            try:
                os.remove(prepared_img_path)
            except OSError:
                pass

    # Segmentation ran on a possibly downscaled copy; return coordinates in the
    # original page's space so callers can crop the upload directly.
    filtered_coco = rescale_coco_to_original(filtered_coco, prepared.scale)
    # The file is stored as upload.<ext>; the annotations name the page as the
    # client did. Data only: it never becomes a path.
    display_name = os.path.basename(image.filename or "")[:255] or safe_name
    for entry in filtered_coco.get("images") or []:
        entry["file_name"] = display_name

    # Store results
    task.status = "completed"
    task.coco_json = filtered_coco
    task.stats = stats
    task.annotated_image_path = annotated_path
    save_task_state(task)

    return {
        "task_id": task.task_id,
        "coco_json": filtered_coco,
        "stats": stats,
        "annotated_image_url": f"/api/download/{task.task_id}/annotated_image",
    }


_LOGGER = logging.getLogger(__name__)


# Batches run in their own small pool, not FastAPI's threadpool: forty of
# them there stalled every synchronous route (security review S3). More wait
# their turn as "pending".
_BATCH_EXECUTOR: ThreadPoolExecutor | None = None
_BATCH_EXECUTOR_LOCK = threading.Lock()


def _batch_executor() -> ThreadPoolExecutor:
    global _BATCH_EXECUTOR
    with _BATCH_EXECUTOR_LOCK:
        if _BATCH_EXECUTOR is None:
            _BATCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=max(1, int(settings.max_concurrent_batches)), thread_name_prefix="archai-batch"
            )
        return _BATCH_EXECUTOR


def _process_batch(task: TaskState, zip_path: str, conf: float, iou: float, selected_classes: list[str]):
    """Background worker for batch processing.

    Any failure ends the task as "error". It used to escape the background
    task with the status still "pending" - a corrupt ZIP was enough - and the
    progress stream, which stops only on completed or error, polled forever
    (audit PRED-001).
    """
    try:
        _run_batch(task, zip_path, conf, iou, selected_classes)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Batch task %s failed", task.task_id)
        task.status = "error"
        # A limit the archive broke, and a file that is not a ZIP, are said as
        # they are; anything else is named by type, its text going to the log
        # (it carried paths and module names; security review C8).
        if isinstance(exc, BatchArchiveRefused):
            task.message = f"Batch processing failed: {exc}"
        elif isinstance(exc, zipfile.BadZipFile):
            task.message = "Batch processing failed: the upload is not a valid ZIP archive."
        else:
            task.message = f"Batch processing failed ({type(exc).__name__}). The details are in the server log."
    # The upload is not read again once the batch has ended; it stayed until
    # the task's TTL (security review S3).
    try:
        os.remove(zip_path)
    except OSError:
        pass
    save_task_state(task)
    release_task(task.task_id)


def _run_batch(task: TaskState, zip_path: str, conf: float, iou: float, selected_classes: list[str]):
    task_dir = get_task_dir(task.task_id)

    # Under the task's own directory, which the task cleanup removes even if
    # the process dies mid-batch; the system temp directory was never swept.
    with tempfile.TemporaryDirectory(dir=task_dir, prefix="extract-") as extract_dir:
        _safe_extract_zip(zip_path, extract_dir)

        # Find valid images
        image_paths = []
        for root, dirs, files in os.walk(extract_dir):
            if "__MACOSX" in root:
                continue
            dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]

            for fn in files:
                if any(fn.startswith(p) for p in SKIP_PATTERNS) or fn.startswith("."):
                    continue
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTENSIONS:
                    continue

                full_path = os.path.join(root, fn)
                if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:
                    continue

                if _is_valid_image_file(full_path):
                    image_paths.append(full_path)

        task.total = len(image_paths)
        task.status = "processing"

        for idx, path in enumerate(sorted(image_paths)):
            fn = os.path.basename(path)
            task.current_image = fn
            task.progress = idx

            try:
                _prep = _prepare_image_for_segmentation(path)
                prepared_path, prepared_changed = _prep.path, _prep.changed
                with tempfile.TemporaryDirectory() as tmp_dir:
                    labels_folders = run_models_parallel(prepared_path, tmp_dir, conf=conf, iou=iou)
                    coco_json = combine_and_filter_predictions(prepared_path, labels_folders)

                filtered = filter_coco_by_classes(coco_json, selected_classes)
                task.batch_coco_list.append(filtered)

                ann_path = os.path.join(task_dir, f"annotated_{idx}.jpg")
                draw_coco_on_image(prepared_path, filtered, selected_classes, output_path=ann_path)
                task.gallery.append({"filename": fn, "annotated_path": ann_path})

                if prepared_changed and prepared_path.endswith(".seg.jpg"):
                    try:
                        os.remove(prepared_path)
                    except OSError:
                        pass

                stats = stats_from_coco(filtered)
                task.stats_per_image.append({"image": fn, "stats": stats})
                for k, v in stats.items():
                    task.stats_summary[k] = task.stats_summary.get(k, 0) + v

            except Exception as e:
                task.errors.append(public_failure(f"Processing {fn}", e, logger=_LOGGER))

        # Save merged COCO JSON before announcing completion, so a client that
        # fetches results on "completed" finds them.
        if task.batch_coco_list:
            merged = merge_coco_list(task.batch_coco_list)
            task.coco_json = merged
            coco_path = os.path.join(task_dir, "annotations.json")
            with open(coco_path, "w") as f:
                json.dump(merged, f, indent=2)

        task.progress = task.total
        task.status = "completed"


@router.post("/predict/batch", status_code=202)
async def predict_batch(
    zip_file: UploadFile = File(...),
    confidence: float = Form(0.25, ge=0.0, le=1.0),
    iou: float = Form(0.3, ge=0.0, le=1.0),
    classes: Optional[str] = Form(None),
):
    """Start batch processing of a ZIP archive."""
    selected_classes = _parse_classes(classes)

    task = create_task()
    # Counted from here until the batch ends, queued or running. Past the
    # limit the upload (already spooled by the form parser) is not kept.
    limit = max(1, int(settings.max_active_batches))
    if not hold_task(task.task_id, limit=limit):
        discard_task(task.task_id)
        raise HTTPException(
            status_code=429,
            detail=f"{limit} batches are already queued or running (MAX_ACTIVE_BATCHES). Try again when one has finished.",
            headers={"Retry-After": "60"},
        )
    task_dir = get_task_dir(task.task_id)

    zip_path = os.path.join(task_dir, "upload.zip")
    try:
        await _save_upload(zip_file, zip_path, empty_detail="Empty ZIP upload.")
        _batch_executor().submit(_process_batch, task, zip_path, confidence, iou, selected_classes)
    except BaseException:
        release_task(task.task_id)
        discard_task(task.task_id)
        raise

    return {
        "task_id": task.task_id,
        "sse_url": f"/api/predict/batch/{task.task_id}/progress",
    }


@router.get("/predict/batch/{task_id}/progress")
async def batch_progress(task_id: str):
    """SSE endpoint streaming batch processing progress."""
    # get_task may read the task's saved state from disk.
    task = await asyncio.to_thread(get_task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    async def event_stream():
        last_progress = -1
        # A task that never finishes must not hold the connection for good.
        deadline = time.monotonic() + max(1, int(settings.task_ttl_minutes)) * 60
        while time.monotonic() < deadline:
            if task.progress != last_progress or task.status in ("completed", "error"):
                data = json.dumps(
                    {
                        "status": task.status,
                        "progress": task.progress,
                        "total": task.total,
                        "current_image": task.current_image,
                        "message": task.message,
                    }
                )
                yield f"data: {data}\n\n"
                last_progress = task.progress

            if task.status in ("completed", "error"):
                break

            await asyncio.sleep(0.5)

    # Not cached or buffered on the way: a buffering reverse proxy (nginx) held
    # the progress events back until the batch had finished.
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/predict/batch/{task_id}/results")
def batch_results(task_id: str):
    """Get final results for a completed batch task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task.status != "completed":
        raise HTTPException(status_code=409, detail=f"Task is still {task.status}.")

    gallery = [
        {
            "filename": g["filename"],
            "annotated_url": f"/api/download/{task_id}/annotated/{idx}",
        }
        for idx, g in enumerate(task.gallery)
    ]

    return {
        "status": task.status,
        "total_processed": task.total - len(task.errors),
        "errors": task.errors,
        "coco_json": task.coco_json,
        "stats_per_image": task.stats_per_image,
        "stats_summary": task.stats_summary,
        "gallery": gallery,
    }
