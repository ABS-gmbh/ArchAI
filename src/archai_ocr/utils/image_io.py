from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def validate_image_path(path: str | Path) -> Path:
    """Resolve and sanity-check an input page image.

    Opening is deliberately wrapped: importing ultralytics replaces
    PIL.Image.open with a patched version that, on failure, tries to install
    the optional pi-heif package. That turns a corrupt input file into a
    confusing ModuleNotFoundError (or a network call), so any open failure is
    normalized into a ValueError naming the offending file.
    """
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Input image path is not a file: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{image_path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        with Image.open(image_path) as im:
            im.verify()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a readable image file: {image_path}") from exc
    except ImportError as exc:
        # Ultralytics' patched opener only reaches its HEIF import when PIL has
        # already failed to identify the file. A missing pi-heif here therefore
        # means "unreadable image", not "broken environment" — report it as such
        # and keep the real cause chained.
        raise ValueError(f"Not a readable image file: {image_path}") from exc
    except Exception as exc:  # noqa: BLE001 - the patched opener can raise anything
        raise ValueError(f"Could not read image {image_path}: {exc}") from exc

    return image_path


def get_image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size
