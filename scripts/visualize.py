"""Figures from a saved run (maps.npz + metrics.json).

    .venv/bin/python scripts/visualize.py results/reduced_max200/candle

Writes into <run_dir>/figures/:
- examples.png: original | ground truth | anomaly map | overlay, for correct
  detections, misses, and false alarms on the test split
- scores.png:   per-image scores on val and test with the val-selected threshold,
  plus the test ROC curve
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

NORMAL = "#3B7DD8"
ANOMALY = "#E8710A"
INK = "#222222"
MUTED = "#8A8A8A"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(run_dir: Path):
    d = np.load(run_dir / "maps.npz")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return d, metrics


def pick_examples(d, per_kind: int) -> list[tuple[str, int]]:
    """Indices of test images: correct detections, misses, false alarms, correct normals."""
    test = np.where(d["split"] == "test")[0]
    thr = float(d["threshold"])
    score, label = d["score"], d["label"]
    tp = [i for i in test if label[i] == 1 and score[i] >= thr]
    fn = [i for i in test if label[i] == 1 and score[i] < thr]
    fp = [i for i in test if label[i] == 0 and score[i] >= thr]
    tn = [i for i in test if label[i] == 0 and score[i] < thr]
    picks = []
    picks += [("Detected anomaly", i) for i in sorted(tp, key=lambda i: -score[i])[:per_kind]]
    picks += [("Missed anomaly", i) for i in sorted(fn, key=lambda i: score[i])[:per_kind]]
    picks += [("False alarm", i) for i in sorted(fp, key=lambda i: -score[i])[:per_kind]]
    picks += [("Correct normal", i) for i in sorted(tn, key=lambda i: score[i])[:1]]
    return picks


def examples_figure(d, metrics, root: Path, out: Path, per_kind: int) -> None:
    picks = pick_examples(d, per_kind)
    size = int(d["image_size"])
    test = d["split"] == "test"
    vmin, vmax = np.percentile(d["map"][test].astype(np.float32), [1, 99.5])
    thr = float(d["threshold"])

    fig, axes = plt.subplots(len(picks), 4, figsize=(8.4, 2.15 * len(picks)), dpi=150)
    axes = np.atleast_2d(axes)
    for c, title in enumerate(["Original", "Ground truth", "Anomaly map", "Overlay"]):
        axes[0, c].set_title(title, color=INK, fontsize=10, pad=8)

    for r, (kind, i) in enumerate(picks):
        img = np.asarray(Image.open(root / str(d["image"][i])).convert("RGB").resize((size, size)))
        amap = d["map"][i].astype(np.float32)
        mask = d["mask"][i]
        color = ANOMALY if d["label"][i] == 1 else NORMAL

        axes[r, 0].imshow(img)
        axes[r, 1].imshow(img)
        if mask.any():
            axes[r, 1].contour(mask, levels=[0.5], colors=["#FF2D55"], linewidths=1.2)
        axes[r, 2].imshow(amap, cmap="jet", vmin=vmin, vmax=vmax)
        axes[r, 3].imshow(img)
        axes[r, 3].imshow(amap, cmap="jet", vmin=vmin, vmax=vmax, alpha=0.45)
        if mask.any():
            axes[r, 3].contour(mask, levels=[0.5], colors=["white"], linewidths=1.0)

        relation = "≥" if d["score"][i] >= thr else "<"
        axes[r, 0].set_ylabel(
            f"{kind}\nscore {d['score'][i]:.1f} {relation} {thr:.1f}", color=color, fontsize=9, labelpad=8
        )
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    t = metrics["test"]
    fig.suptitle(
        f"PatchCore on VisA {metrics['category']} — test F1 {t['f1']:.2f} "
        f"(recall {t['anomaly_recall']:.2f}, specificity {t['normal_specificity']:.2f}), "
        f"threshold chosen on val only",
        fontsize=10,
        color=INK,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out / "examples.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scores_figure(d, metrics, out: Path) -> None:
    thr = float(d["threshold"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150, gridspec_kw={"width_ratios": [1, 1, 0.8]})

    for ax, split, name in [(axes[0], "val", "Validation (threshold selected here)"), (axes[1], "test", "Test (threshold applied once)")]:
        idx = np.where(d["split"] == split)[0]
        idx = idx[np.argsort(d["label"][idx], kind="stable")]
        x = np.arange(len(idx))
        for y, color, label in [(0, NORMAL, "normal"), (1, ANOMALY, "anomaly")]:
            m = d["label"][idx] == y
            ax.scatter(x[m], d["score"][idx][m], s=12, color=color, label=label, alpha=0.85, linewidths=0)
        ax.axhline(thr, color=INK, linestyle="--", linewidth=1)
        ax.text(len(idx) - 1, thr, f" threshold {thr:.1f}", va="bottom", ha="right", color=INK, fontsize=8)
        f1 = metrics[split]["f1"]
        ax.set_title(f"{name}\nF1 {f1:.2f}", color=INK, fontsize=9)
        ax.set_xlabel("image (grouped by label)")
        ax.set_xticks([])
    axes[0].set_ylabel("image anomaly score")
    axes[0].legend(frameon=False, loc="upper left")

    test = d["split"] == "test"
    fpr, tpr, _ = roc_curve(d["label"][test], d["score"][test])
    axes[2].plot(fpr, tpr, color=ANOMALY, linewidth=1.6)
    axes[2].plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    axes[2].set_title(f"Test ROC — AUROC {metrics['image_auroc_test']:.3f}", color=INK, fontsize=9)
    axes[2].set_xlabel("false positive rate")
    axes[2].set_ylabel("true positive rate")
    axes[2].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out / "scores.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/VisA"))
    parser.add_argument("--per-kind", type=int, default=2)
    args = parser.parse_args()

    d, metrics = load(args.run_dir)
    out = args.run_dir / "figures"
    out.mkdir(exist_ok=True)
    examples_figure(d, metrics, args.root, out, args.per_kind)
    scores_figure(d, metrics, out)
    print(f"wrote {out / 'examples.png'} and {out / 'scores.png'}")


if __name__ == "__main__":
    main()
