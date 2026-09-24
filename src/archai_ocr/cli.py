from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from archai_ocr.pipeline.crop_regions import RegionCrop
    from archai_ocr.pipeline.layout_yolo import PageLayout, RegionDetection
    from archai_ocr.pipeline.page import PageTranscription, RecognizedLine
    from archai_ocr.pipeline.zones import Lane

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_PARTIAL_FAILURE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archai",
        description=(
            "ArchAI OCR pipeline: body text, text outside the body, and optionally PAGE XML "
            "for each manuscript page image."
        ),
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
        "--no-secondary-text",
        action="store_true",
        help="Transcribe body text only, skipping marginalia, running titles, foliation and quire marks.",
    )
    parser.add_argument(
        "--page-xml",
        action="store_true",
        help="Also write <stem>.page.xml: every zone, line, baseline and confidence, as PAGE XML.",
    )
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
    if args.no_secondary_text:
        overrides["layout.secondary_text_class"] = []
    if args.page_xml:
        overrides["runtime.write_page_xml"] = True
    return overrides


def process_image(image_path: Path, config: AppConfig, logger: logging.Logger) -> Path:
    """Run the full pipeline for one page and return the written .txt path.

    The ``.txt`` holds the body text, exactly as before. Text found outside the
    body goes to ``<stem>.secondary.txt``, and ``runtime.write_page_xml`` adds
    ``<stem>.page.xml`` with every zone, line, baseline and confidence.
    """
    from archai_ocr.pipeline.assemble_text import assemble_secondary_text, assemble_text
    from archai_ocr.pipeline.crop_regions import crop_region_images
    from archai_ocr.pipeline.htr_kraken import recognize_crop_lines
    from archai_ocr.pipeline.layout_yolo import detect_page_layout
    from archai_ocr.pipeline.page_xml import write_page_xml

    run_dir = config.runtime.output_dir / image_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    txt_path = run_dir / f"{image_path.stem}.txt"

    logger.info("pipeline.start", extra={"image": str(image_path), "output": str(run_dir)})

    with log_stage(logger, "layout", image=str(image_path)):
        layout = detect_page_layout(image_path=image_path, run_dir=run_dir, config=config, logger=logger)

    if not layout.main:
        logger.warning(
            "layout.no_regions",
            extra={"stage": "layout", "image": str(image_path)},
        )

    crops_dir = run_dir / "crops"
    padding, min_size = config.layout.crop_padding, config.layout.min_region_size
    with log_stage(logger, "crop", count=len(layout.main) + len(layout.secondary)):
        main_crops = crop_region_images(
            image_path, layout.main, crops_dir, padding=padding, min_size=min_size, logger=logger
        )
        secondary_crops = crop_region_images(
            image_path,
            layout.secondary,
            crops_dir,
            padding=padding,
            min_size=min_size,
            logger=logger,
            name_prefix="secondary",
        )

    if layout.main and not main_crops:
        logger.warning("crop.no_usable_crops", extra={"stage": "crop", "image": str(image_path)})

    crops = main_crops + secondary_crops
    lines: list[list[RecognizedLine]] = []
    if crops:
        with log_stage(logger, "htr", count=len(crops)):
            lines = recognize_crop_lines(
                crop_paths=[crop.path for crop in main_crops], config=config, logger=logger
            )
            # Secondary zones are too small for Kraken's page segmenter to find
            # lines in; they are cut into lines by their ink profile instead.
            lines += recognize_crop_lines(
                crop_paths=[crop.path for crop in secondary_crops],
                config=config,
                logger=logger,
                segmenter="zone",
            )

    page = _page_transcription(image_path, layout, main_crops, secondary_crops, lines)
    with log_stage(logger, "assemble"):
        assemble_text([region.text for region in page.lane("main")], txt_path)
        secondary_path = assemble_secondary_text(
            page.lane("secondary"), run_dir / f"{image_path.stem}.secondary.txt"
        )
        page_xml_path = None
        if config.runtime.write_page_xml:
            page_xml_path = write_page_xml(page, run_dir / f"{image_path.stem}.page.xml")

    logger.info(
        "pipeline.done",
        extra={
            "image": str(image_path),
            "output": str(txt_path),
            "secondary_output": str(secondary_path) if secondary_path else None,
            "page_xml": str(page_xml_path) if page_xml_path else None,
        },
    )
    return txt_path


def _page_transcription(
    image_path: Path,
    layout: PageLayout,
    main_crops: Sequence[RegionCrop],
    secondary_crops: Sequence[RegionCrop],
    lines: Sequence[Sequence[RecognizedLine]],
) -> PageTranscription:
    """Attach recognised lines to their regions, moved into page coordinates."""
    from archai_ocr.pipeline.crop_regions import oriented_image_size
    from archai_ocr.pipeline.page import PageTranscription, TranscribedRegion

    crop_lines = dict(zip((crop.path for crop in (*main_crops, *secondary_crops)), lines, strict=True))
    regions: list[TranscribedRegion] = []
    # Region ids match the crop file names, so a PAGE region can be traced to
    # the pixels it was recognised from.
    text_lanes: tuple[tuple[str, Lane, Sequence[RegionDetection], Sequence[RegionCrop]], ...] = (
        ("region", "main", layout.main, main_crops),
        ("secondary", "secondary", layout.secondary, secondary_crops),
    )
    for prefix, lane, detections, crops in text_lanes:
        by_index = {crop.index: crop for crop in crops}
        for index, detection in enumerate(detections):
            crop = by_index.get(index)
            found = crop_lines.get(crop.path, ()) if crop is not None else ()
            offset_x, offset_y = (crop.box[0], crop.box[1]) if crop is not None else (0, 0)
            regions.append(
                TranscribedRegion(
                    region_id=f"{prefix}_{index:03d}",
                    detection=detection,
                    lane=lane,
                    lines=tuple(line.translated(offset_x, offset_y) for line in found),
                )
            )
    for index, detection in enumerate(layout.layout):
        regions.append(TranscribedRegion(region_id=f"layout_{index:03d}", detection=detection, lane="layout"))
    return PageTranscription(
        image_name=image_path.name,
        image_size=oriented_image_size(image_path),
        regions=tuple(regions),
    )


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
                    "secondary_text_classes": list(config.layout.secondary_text_classes),
                    "page_xml": config.runtime.write_page_xml,
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
