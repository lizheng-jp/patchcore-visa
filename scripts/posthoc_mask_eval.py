"""E1a: apply SAM foreground masks to saved anomaly maps, without refitting.

For each image the score becomes the maximum of the anomaly map inside the
foreground. For a fair comparison the unmasked score is recomputed the same way
(map maximum), since anomalib's pred_score uses a slightly different reweighting.

    .venv/bin/python scripts/posthoc_mask_eval.py results/E0_baseline/seed0 \
        --fg-masks data/fg_masks/sam-vit-base --out results/E1a_posthoc_mask/seed0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iad.metrics import evaluate, pixel_auroc  # noqa: E402


def load_fg(fg_dir: Path, image: str, size: int) -> np.ndarray:
    fg = np.asarray(Image.open(fg_dir / Path(image).with_suffix(".png")).convert("L")) > 0
    return cv2.resize(fg.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0


def protocol(d, scores: np.ndarray, maps: np.ndarray, method: str) -> dict:
    val, test = d["split"] == "val", d["split"] == "test"
    r = evaluate(d["label"][val], scores[val], d["label"][test], scores[test], method=method)
    r["pixel_auroc_official_test"] = pixel_auroc(d["mask"], maps)
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path, help="directory with <category>/maps.npz")
    parser.add_argument("--fg-masks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-method", default="f1", choices=["f1", "youden"])
    args = parser.parse_args()

    for npz in sorted(args.baseline_dir.glob("*/maps.npz")):
        category = npz.parent.name
        d = np.load(npz)
        size = int(d["image_size"])
        maps = d["map"].astype(np.float32)
        fg = np.stack([load_fg(args.fg_masks, str(p), size) for p in d["image"]])
        masked_maps = np.where(fg, maps, maps.min())

        unmasked = protocol(d, maps.reshape(len(maps), -1).max(1), maps, args.threshold_method)
        masked = protocol(d, masked_maps.reshape(len(maps), -1).max(1), masked_maps, args.threshold_method)
        gt = d["mask"].astype(bool)
        coverage = float((gt & fg).sum() / max(gt.sum(), 1))

        result = {
            "category": category,
            "baseline_dir": str(args.baseline_dir),
            "defect_pixel_coverage": coverage,
            "foreground_fraction": float(fg.mean()),
            "unmasked_map_max": unmasked,
            "masked_map_max": masked,
        }
        out_dir = args.out / category
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        print(
            f"{category:12s} img AUROC {unmasked['image_auroc_official_test']:.3f} -> "
            f"{masked['image_auroc_official_test']:.3f} | test F1 {unmasked['test']['f1']:.3f} -> "
            f"{masked['test']['f1']:.3f} | defect coverage {coverage:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
