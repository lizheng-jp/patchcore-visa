"""Cache SAM foreground masks for VisA images.

Foreground = union of SAM automatic masks that are not background-like
(not larger than --max-area of the image and not touching 3+ image borders),
dilated by --dilate-frac of the image size so defects on object edges are kept.

    .venv/bin/python scripts/sam_masks.py --category candle --splits val,test

Outputs (under --out):
- <image path>.png            foreground mask at original resolution (0/255)
- qa.csv                      per image: foreground fraction, defect coverage, fallback
- preview/<category>_*.jpg    a few overlays for visual inspection

defect_coverage is the share of ground-truth defect pixels inside the foreground.
Values below 1.0 mean masking would hide part of a defect, so check qa.csv before
trusting any masked result.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iad.splits import read_splits  # noqa: E402


def foreground(masks: list[np.ndarray], shape: tuple[int, int], max_area: float, dilate_frac: float):
    h, w = shape
    fg = np.zeros((h, w), dtype=bool)
    kept = 0
    for m in masks:
        m = np.asarray(m, dtype=bool)
        borders = int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())
        if m.mean() > max_area or borders >= 3:
            continue
        fg |= m
        kept += 1
    fallback = kept == 0
    if fallback:
        fg[:] = True
    elif dilate_frac > 0:
        k = max(3, int(round(dilate_frac * max(h, w))) | 1)
        fg = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return fg, kept, fallback


def preview(image: np.ndarray, fg: np.ndarray, gt: np.ndarray | None, path: Path) -> None:
    out = image.copy()
    out[~fg] = (out[~fg] * 0.25).astype(np.uint8)
    contours, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (59, 125, 216), 3)
    if gt is not None and gt.any():
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 45, 85), 3)
    Image.fromarray(out).resize((out.shape[1] // 2, out.shape[0] // 2)).save(path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="all", help="'all' or comma-separated categories")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--root", type=Path, default=Path("data/VisA"))
    parser.add_argument("--split-file", type=Path, default=Path("splits/visa_seed0.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/fg_masks/sam-vit-base"))
    parser.add_argument("--model", default="facebook/sam-vit-base")
    parser.add_argument("--points-per-crop", type=int, default=16)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument(
        "--max-side", type=int, default=512,
        help="downscale before SAM; the pipeline upsamples every candidate mask to input size, which is memory-heavy",
    )
    parser.add_argument("--max-area", type=float, default=0.4)
    parser.add_argument("--dilate-frac", type=float, default=0.02)
    parser.add_argument("--preview", type=int, default=6, help="overlays saved per category")
    parser.add_argument("--limit", type=int, default=0, help="debug: images per category")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    generator = pipeline("mask-generation", model=args.model, device=args.device)

    samples = read_splits(args.split_file)
    splits = set(args.splits.split(","))
    categories = sorted({s.category for s in samples}) if args.category == "all" else args.category.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preview").mkdir(exist_ok=True)

    qa_path = args.out / "qa.csv"
    new_file = not qa_path.exists()
    with open(qa_path, "a", newline="") as qa_file:
        qa = csv.writer(qa_file)
        if new_file:
            qa.writerow(["category", "split", "label", "image", "fg_fraction", "defect_coverage", "n_masks", "fallback", "seconds"])
        for category in categories:
            todo = [s for s in samples if s.category == category and s.split in splits]
            if args.limit:
                todo = todo[: args.limit]
            coverages, previews = [], 0
            for s in todo:
                path = args.out / Path(s.image).with_suffix(".png")
                if args.skip_existing and path.exists():
                    continue
                pil = Image.open(args.root / s.image).convert("RGB")
                image = np.asarray(pil)
                scale = min(1.0, args.max_side / max(pil.size))
                small = pil.resize((round(pil.width * scale), round(pil.height * scale))) if scale < 1 else pil
                t = time.perf_counter()
                result = generator(small, points_per_crop=args.points_per_crop, points_per_batch=args.points_per_batch)
                seconds = time.perf_counter() - t
                fg, kept, fallback = foreground(result["masks"], (small.height, small.width), args.max_area, args.dilate_frac)
                if scale < 1:
                    fg = cv2.resize(fg.astype(np.uint8), (pil.width, pil.height), interpolation=cv2.INTER_NEAREST) > 0

                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(fg.astype(np.uint8) * 255).save(path)

                gt = None
                coverage = ""
                if s.mask:
                    gt = np.asarray(Image.open(args.root / s.mask).convert("L")) > 0
                    if gt.any():
                        coverage = float((gt & fg).sum() / gt.sum())
                        coverages.append(coverage)
                qa.writerow([category, s.split, s.label, s.image, f"{fg.mean():.4f}",
                             f"{coverage:.4f}" if coverage != "" else "", kept, int(fallback), f"{seconds:.2f}"])
                if previews < args.preview and (s.label == 1 or previews < args.preview // 2):
                    preview(image, fg, gt, args.out / "preview" / f"{category}_{previews}.jpg")
                    previews += 1
            qa_file.flush()
            if coverages:
                c = np.array(coverages)
                print(f"{category:12s} {len(todo)} images | defect coverage mean {c.mean():.3f}, "
                      f"min {c.min():.3f}, <0.99 in {int((c < 0.99).sum())}/{len(c)}", flush=True)
            else:
                print(f"{category:12s} {len(todo)} images", flush=True)


if __name__ == "__main__":
    main()
