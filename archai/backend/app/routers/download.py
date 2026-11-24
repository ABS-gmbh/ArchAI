"""File download endpoints."""

import asyncio
import io
import json
import os
import tempfile
import threading
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.services import file_manager
from app.services.file_manager import get_task, get_task_dir

router = APIRouter(tags=["download"])


def _inside_task(task_id: str, path: str | None) -> bool:
    """Whether *path* is a file inside the task's own directory.

    The paths come from the task's state, which is read back from state.json
    after a restart. Before uploads were stored under a fixed name, an upload
    called state.json could replace that file, and the downloads below served
    whatever path it named. Anything outside the task directory is refused.
    """
    if not path:
        return False
    base = os.path.realpath(os.path.join(file_manager.TASK_BASE_DIR, task_id))
    target = os.path.realpath(path)
    return target.startswith(base + os.sep) and os.path.isfile(target)


@router.get("/download/{task_id}/coco_json")
def download_coco_json(task_id: str):
    """Download COCO JSON annotations for a task."""
    task = get_task(task_id)
    if task is None or task.coco_json is None:
        raise HTTPException(status_code=404, detail="Task not found or no results available.")

    content = json.dumps(task.coco_json, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="annotations_{task_id}.json"'},
    )


@router.get("/download/{task_id}/annotated_image")
def download_annotated_image(task_id: str):
    """Download the annotated image for a single-image task."""
    task = get_task(task_id)
    if task is None or task.annotated_image_path is None:
        raise HTTPException(status_code=404, detail="Task not found or no annotated image.")

    if not _inside_task(task_id, task.annotated_image_path):
        raise HTTPException(status_code=404, detail="Annotated image file not found.")

    return FileResponse(
        task.annotated_image_path,
        media_type="image/jpeg",
        filename=f"annotated_{task_id}.jpg",
    )


@router.get("/download/{task_id}/annotated/{index}")
def download_batch_annotated(task_id: str, index: int):
    """Download a specific annotated image from a batch task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    if index < 0 or index >= len(task.gallery):
        raise HTTPException(status_code=404, detail="Image index out of range.")

    path = task.gallery[index]["annotated_path"]
    if not _inside_task(task_id, path):
        raise HTTPException(status_code=404, detail="Annotated image file not found.")

    filename = task.gallery[index]["filename"]
    return FileResponse(path, media_type="image/jpeg", filename=f"annotated_{filename}")


@router.get("/download/{task_id}/results_zip")
async def download_results_zip(task_id: str):
    """Download a ZIP containing all annotated images and the merged COCO JSON."""
    task = get_task(task_id)
    if task is None or not task.gallery:
        raise HTTPException(status_code=404, detail="Task not found or no results.")

    # Built on disk in a worker thread and streamed from there. It was built in
    # memory on the event loop: a large batch held hundreds of megabytes per
    # download and froze every other request while it compressed (audit DL-001).
    path, temporary = await asyncio.to_thread(_results_zip, task, get_task_dir(task_id))
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"results_{task_id}.zip",
        background=BackgroundTask(os.remove, path) if temporary else None,
    )


_RESULTS_ZIP = "results.zip"
_zip_builds: dict[str, threading.Lock] = {}
_zip_builds_guard = threading.Lock()


def _results_zip(task, directory: str) -> tuple[str, bool]:
    """The task's results ZIP and whether it is a temporary copy to remove once sent.

    A finished task's archive is built once, under a lock per task, and kept
    in its directory (the TTL sweep removes it with the task): every request
    built its own, so parallel downloads multiplied the disk and the CPU
    (second security review). A batch still running gets a copy of what it
    has so far, removed after it is sent, since its results still change.
    """
    if task.status != "completed":
        return _write_results_zip(task, directory), True
    final = os.path.join(directory, _RESULTS_ZIP)
    with _zip_builds_guard:
        build = _zip_builds.setdefault(task.task_id, threading.Lock())
    with build:
        try:
            if not os.path.isfile(final):
                # Renamed into place whole: a parallel request never sends half of it.
                os.replace(_write_results_zip(task, directory), final)
        finally:
            with _zip_builds_guard:
                _zip_builds.pop(task.task_id, None)
    return final, False


def _write_results_zip(task, directory: str) -> str:
    """Write the task's annotated images and merged annotations to a new ZIP; return its path."""
    handle, path = tempfile.mkstemp(prefix="results_", suffix=".zip", dir=directory)
    try:
        with os.fdopen(handle, "wb") as out, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for g in task.gallery:
                if _inside_task(task.task_id, g["annotated_path"]):
                    zf.write(g["annotated_path"], f"images/{os.path.basename(g['filename'])}")
            if task.coco_json:
                zf.writestr("annotations.json", json.dumps(task.coco_json, indent=2))
    except BaseException:
        os.remove(path)
        raise
    return path

