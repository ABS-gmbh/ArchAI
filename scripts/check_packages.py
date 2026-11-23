#!/usr/bin/env python3
"""Check that NumPy, OpenCV and ChromaDB import and work in this environment.

A manual environment check, run as ``python scripts/check_packages.py``; it
exits 1 if any check fails. It is not part of the test suite: it lived at the
repository root as test_packages.py, where pytest would collect it, it wrote to
an absolute path on one developer's Desktop, asserted nothing, and exited 0
after an error (audit OCR-020).

The ChromaDB check embeds with ChromaDB's default model, which it downloads on
first use.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path


def check_numpy() -> None:
    import numpy as np

    total = np.array([1, 2, 3, 4, 5]) + np.array([10, 20, 30, 40, 50])
    assert total.tolist() == [11, 22, 33, 44, 55]
    assert np.mean([1, 2, 3, 4, 5]) == 3.0
    assert np.array([[1, 2, 3], [4, 5, 6]]).T.shape == (3, 2)


def check_opencv() -> None:
    import cv2
    import numpy as np

    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (250, 150), (255, 0, 0), 3)
    cv2.circle(image, (400, 200), 75, (0, 255, 0), -1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "opencv_check.png"
        assert cv2.imwrite(str(path), image), f"cv2.imwrite could not write {path}"
        loaded = cv2.imread(str(path))
    assert loaded is not None and loaded.shape == (400, 600, 3)
    assert cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY).shape == (400, 600)


def check_chromadb() -> None:
    import chromadb

    collection = chromadb.Client().create_collection(name="check_packages")
    documents = [
        "This is a document about Python programming",
        "OpenCV is a computer vision library",
        "ChromaDB is a vector database",
    ]
    collection.add(documents=documents, ids=[f"doc_{i}" for i in range(len(documents))])
    assert collection.count() == len(documents)
    results = collection.query(query_texts=["What is computer vision?"], n_results=1)
    assert results["documents"] and results["documents"][0], "the query returned nothing"


CHECKS: dict[str, Callable[[], None]] = {
    "numpy": check_numpy,
    "opencv": check_opencv,
    "chromadb": check_chromadb,
}


def main() -> int:
    failed = []
    for name, check in CHECKS.items():
        try:
            check()
        except Exception:  # noqa: BLE001 - every failure is reported, then counted
            failed.append(name)
            print(f"FAIL {name}", file=sys.stderr)
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
