"""Tables and figures for the README from a results directory.

    .venv/bin/python scripts/report.py results/runpod --out docs

Writes:
- docs/results/e0_e1a_seed0.csv          per-category E0 metrics (two threshold rules) and E1a masking
- docs/figures/sam_tradeoff.png          AUROC change from SAM masking vs. defect coverage
- docs/figures/examples_<category>.jpg   downscaled heatmap examples
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iad.metrics import evaluate  # noqa: E402

INK, MUTED, UP, DOWN = "#222222", "#8A8A8A", "#2E8B57", "#D1495B"
# Points near (1.0, 0.0) overlap; nudge their labels apart.
LABEL_OFFSETS = {"chewinggum": (4, -12), "pipe_fryum": (6, 8), "cashew": (6, -10)}


def collect(results: Path, seed: str) -> list[dict]:
    rows = []
    for mpath in sorted((results / "E0_baseline" / seed).glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        d = np.load(mpath.with_name("maps.npz"))
        val, test = d["split"] == "val", d["split"] == "test"
        youden = evaluate(d["label"][val], d["score"][val], d["label"][test], d["score"][test], method="youden")
        e = json.loads((results / "E1a_posthoc_mask" / seed / m["category"] / "metrics.json").read_text())
        rows.append({
            "category": m["category"],
            "image_auroc": m["image_auroc_official_test"],
            "pixel_auroc": m["pixel_auroc_official_test"],
            "f1_maxf1": m["test"]["f1"],
            "recall_maxf1": m["test"]["anomaly_recall"],
            "specificity_maxf1": m["test"]["normal_specificity"],
            "f1_youden": youden["test"]["f1"],
            "recall_youden": youden["test"]["anomaly_recall"],
            "specificity_youden": youden["test"]["normal_specificity"],
            "auroc_mapmax_unmasked": e["unmasked_map_max"]["image_auroc_official_test"],
            "auroc_mapmax_sam": e["masked_map_max"]["image_auroc_official_test"],
            "defect_coverage": e["defect_pixel_coverage"],
            "n_train": m["n_train"],
            "fit_seconds": m["fit_seconds"],
            "gpu_ms": m["device_ms_per_image"],
            "cpu_ms": m.get("cpu_ms_median"),
        })
    return rows


def tradeoff_figure(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for r in rows:
        delta = r["auroc_mapmax_sam"] - r["auroc_mapmax_unmasked"]
        ax.scatter(r["defect_coverage"], delta, s=36, color=UP if delta >= 0 else DOWN, zorder=3)
        dx, dy = LABEL_OFFSETS.get(r["category"], (5, 3))
        ax.annotate(r["category"], (r["defect_coverage"], delta), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=INK, ha="right" if dx < 0 else "left")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("defect pixels kept inside the SAM foreground")
    ax.set_ylabel("image AUROC change with SAM masking")
    ax.set_title("SAM foreground masking: gain vs. defect coverage (VisA, seed 0)", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--examples", nargs="+", default=["macaroni2", "candle", "capsules"])
    args = parser.parse_args()

    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    rows = collect(args.results, args.seed)
    with open(args.out / "results" / f"e0_e1a_{args.seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tradeoff_figure(rows, args.out / "figures" / "sam_tradeoff.png")

    for c in args.examples:
        src = args.results / "E0_baseline" / args.seed / c / "figures" / "examples.png"
        img = Image.open(src).convert("RGB")
        img.thumbnail((900, 4000))
        img.save(args.out / "figures" / f"examples_{c}.jpg", quality=85)
    shutil.copy(args.results / "E1a_posthoc_mask" / "sam_qa.csv", args.out / "results" / "sam_mask_qa.csv")

    keys = [k for k in rows[0] if k != "category"]
    mean = {k: np.mean([r[k] for r in rows if r[k] is not None]) for k in keys}
    print(json.dumps({k: round(float(v), 3) for k, v in mean.items()}, indent=1))


if __name__ == "__main__":
    main()
