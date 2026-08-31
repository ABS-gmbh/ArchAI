from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from archai_ocr import __version__
from archai_ocr.config import (
    VALID_READING_ORDERS,
    AppConfig,
    ConfigError,
    load_config,
    validate_required_weights,
)
from archai_ocr.logging_utils import log_stage, setup_logging
from archai_ocr.utils.image_io import SUPPORTED_EXTENSIONS, validate_image_path

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_PARTIAL_FAILURE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archai",
        description="ArchAI OCR pipeline producing a .txt output per manuscript page image.",
    )
    parser.add_argument(
        "--image",
        required=True,
        nargs="+",
        metavar="PATH",
        help="One or more image paths, or directories to scan for images.",
    )
    parser.add_argument(
        "--config",
        default="config.example.yaml",
        help="Path to YAML config file (default: config.example.yaml)",
    )
    parser.add_argument("--outdir", default=None, help="Override output directory")
    parser.add_argument("--main-class", default=None, help="Override YOLO main text class name")
    parser.add_argument(
        "--reading-order",
        default=None,
        choices=list(VALID_READING_ORDERS),
        help=(
            "Region ordering. 'column' reads each column top-to-bottom (correct for "
            "multi-column pages); 'simple' is a global top-to-bottom sort, which "
            "reproduces pre-0.2.0 output."
        ),
    )
    parser.add_argument("--device", default=None, help="Override Kraken device (cpu, cuda, cuda:0, mps)")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when an --image argument is a directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining images after one fails (exit code 2 if any failed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and inputs, print what would run, then exit without loading models.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _collect_images(raw_paths: Sequence[str], recursive: bool) -> list[Path]:
    """Expand the --image arguments into a deterministic list of image files."""
    collected: list[Path] = []
    seen: set[Path] = set()

    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            pattern = "**/*" if recursive else "*"
            found = sorted(
                p for p in candidate.glob(pattern) if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
            )
            if not found:
                raise FileNotFoundError(
                    f"No supported images found in directory: {candidate} "
                    f"(supported: {sorted(SUPPORTED_EXTENSIONS)})"
                )
            for path in found:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    collected.append(resolved)
        else:
            resolved = validate_image_path(candidate)
            if resolved not in seen:
                seen.add(resolved)
                collected.append(resolved)

    if not collected:
        raise FileNotFoundError("No input images resolved from --image arguments.")
    return collected


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.outdir:
        overrides["runtime.output_dir"] = args.outdir
    if args.main_class:
        overrides["layout.main_text_class"] = args.main_class
    if args.reading_order:
        overrides["layout.reading_order"] = args.reading_order
    if args.device:
        overrides["runtime.kraken_device"] = args.device
    return overrides


def process_image(image_path: Path, config: AppConfig, logger: logging.Logger) -> Path:
    """Run the full pipeline for one page and return the written .txt path."""
    from archai_ocr.pipeline.assemble_text import assemble_text
    from archai_ocr.pipeline.crop_regions import crop_regions
    from archai_ocr.pipeline.htr_kraken import recognize_crops
    from archai_ocr.pipeline.layout_yolo import detect_layout_regions

    run_dir = config.runtime.output_dir / image_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    txt_path = run_dir / f"{image_path.stem}.txt"

    logger.info("pipeline.start", extra={"image": str(image_path), "output": str(run_dir)})

    with log_stage(logger, "layout", image=str(image_path)):
        regions = detect_layout_regions(image_path=image_path, run_dir=run_dir, config=config, logger=logger)

    if not regions:
        logger.warning(
            "layout.no_regions",
            extra={"stage": "layout", "image": str(image_path)},
        )
        assemble_text([], txt_path)
        return txt_path

    with log_stage(logger, "crop", count=len(regions)):
        crop_paths = crop_regions(
            image_path=image_path,
            regions=regions,
            crops_dir=run_dir / "crops",
            padding=config.layout.crop_padding,
            min_size=config.layout.min_region_size,
            logger=logger,
        )

    if not crop_paths:
        logger.warning("crop.no_usable_crops", extra={"stage": "crop", "image": str(image_path)})
        assemble_text([], txt_path)
        return txt_path

    with log_stage(logger, "htr", count=len(crop_paths)):
        region_texts = recognize_crops(crop_paths=crop_paths, config=config, logger=logger)

    with log_stage(logger, "assemble"):
        assemble_text(region_texts, txt_path)

    logger.info("pipeline.done", extra={"image": str(image_path), "output": str(txt_path)})
    return txt_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        logger = setup_logging(args.log_level)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        config = load_config(args.config, overrides=_build_overrides(args))
        images = _collect_images(args.image, recursive=args.recursive)

        if args.dry_run:
            for image_path in images:
                print(f"{image_path} -> {config.runtime.output_dir / image_path.stem}")
            logger.info(
                "pipeline.dry_run",
                extra={
                    "count": len(images),
                    "output": str(config.runtime.output_dir),
                    "reading_order": config.layout.reading_order,
                },
            )
            return EXIT_OK

        validate_required_weights(config)
    except (FileNotFoundError, ConfigError, ValueError) as exc:
        logger.error("pipeline.config_error", extra={"output": str(exc)})
        return EXIT_USER_ERROR

    failures = 0
    for image_path in images:
        try:
            txt_path = process_image(image_path, config, logger)
            print(str(txt_path))
        except Exception:
            failures += 1
            logger.exception("pipeline.failed", extra={"image": str(image_path)})
            if not args.continue_on_error:
                return EXIT_USER_ERROR

    if failures:
        logger.error(
            "pipeline.partial_failure",
            extra={"count": failures, "output": f"{failures} of {len(images)} image(s) failed"},
        )
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
