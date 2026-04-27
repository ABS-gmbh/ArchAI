from __future__ import annotations

import argparse
import logging
from typing import Sequence

from archai_ocr.config import load_config, validate_required_weights
from archai_ocr.logging_utils import log_stage, setup_logging
from archai_ocr.utils.image_io import validate_image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archai",
        description="ArchAI OCR pipeline producing a .txt output per image.",
    )
    parser.add_argument("--image", required=True, help="Path to manuscript page image (png/jpg/tif)")
    parser.add_argument(
        "--config",
        default="config.example.yaml",
        help="Path to YAML config file (default: config.example.yaml)",
    )
    parser.add_argument("--outdir", default=None, help="Override output directory")
    parser.add_argument("--main-class", default=None, help="Override YOLO main text class name")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    logger = logging.getLogger("archai_ocr")
    try:
        overrides: dict[str, str] = {}
        if args.outdir:
            overrides["runtime.output_dir"] = args.outdir
        if args.main_class:
            overrides["layout.main_text_class"] = args.main_class

        config = load_config(args.config, overrides=overrides)
        validate_required_weights(config)

        image_path = validate_image_path(args.image)
        run_dir = config.runtime.output_dir / image_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        from archai_ocr.pipeline.assemble_text import assemble_text
        from archai_ocr.pipeline.crop_regions import crop_regions
        from archai_ocr.pipeline.htr_kraken import recognize_crops
        from archai_ocr.pipeline.layout_yolo import detect_layout_regions

        logger.info(
            "pipeline.start",
            extra={"image": str(image_path), "output": str(run_dir)},
        )

        with log_stage(logger, "layout", image=str(image_path)):
            regions = detect_layout_regions(
                image_path=image_path,
                run_dir=run_dir,
                config=config,
                logger=logger,
            )

        txt_path = run_dir / f"{image_path.stem}.txt"
        if not regions:
            logger.warning("No text regions detected; writing empty output text.", extra={"stage": "layout"})
            assemble_text([], txt_path)
            print(str(txt_path))
            return 0

        with log_stage(logger, "crop", count=len(regions)):
            crop_paths = crop_regions(
                image_path=image_path,
                regions=regions,
                crops_dir=run_dir / "crops",
                padding=config.layout.crop_padding,
            )

        with log_stage(logger, "htr", count=len(crop_paths)):
            region_texts = recognize_crops(
                crop_paths=crop_paths,
                config=config,
                logger=logger,
            )

        with log_stage(logger, "assemble"):
            assemble_text(region_texts, txt_path)

        logger.info("pipeline.done", extra={"output": str(txt_path)})
        print(str(txt_path))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1
    except Exception:
        logger.exception("Unexpected pipeline failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
