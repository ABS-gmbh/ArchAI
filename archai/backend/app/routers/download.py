"""File download endpoints."""

import asyncio
import io
import json
import os
import tempfile
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.services.file_manager import get_task, get_task_dir

router = APIRouter(tags=["download"])


@router.get("/download/{task_id}/coco_json")
async def download_coco_json(task_id: str):
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
async def download_annotated_image(task_id: str):
    """Download the annotated image for a single-image task."""
    task = get_task(task_id)
    if task is None or task.annotated_image_path is None:
        raise HTTPException(status_code=404, detail="Task not found or no annotated image.")

    if not os.path.exists(task.annotated_image_path):
        raise HTTPException(status_code=404, detail="Annotated image file not found.")

    return FileResponse(
        task.annotated_image_path,
        media_type="image/jpeg",
        filename=f"annotated_{task_id}.jpg",
    )


@router.get("/download/{task_id}/annotated/{index}")
async def download_batch_annotated(task_id: str, index: int):
    """Download a specific annotated image from a batch task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    if index < 0 or index >= len(task.gallery):
        raise HTTPException(status_code=404, detail="Image index out of range.")

    path = task.gallery[index]["annotated_path"]
    if not os.path.exists(path):
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
    path = await asyncio.to_thread(_write_results_zip, task, get_task_dir(task_id))
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"results_{task_id}.zip",
        background=BackgroundTask(os.remove, path),
    )


def _write_results_zip(task, directory: str) -> str:
    """Write the task's annotated images and merged annotations to a new ZIP; return its path."""
    handle, path = tempfile.mkstemp(prefix="results_", suffix=".zip", dir=directory)
    try:
        with os.fdopen(handle, "wb") as out, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for g in task.gallery:
                if os.path.exists(g["annotated_path"]):
                    zf.write(g["annotated_path"], f"images/{g['filename']}")
            if task.coco_json:
                zf.writestr("annotations.json", json.dumps(task.coco_json, indent=2))
    except BaseException:
        os.remove(path)
        raise
    return path

