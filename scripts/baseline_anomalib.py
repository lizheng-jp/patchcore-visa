"""anomalib PatchCore on our leakage-free VisA splits.

Uses anomalib's PatchcoreModel directly (no Lightning Engine), so the data split,
threshold selection, and metrics all come from this repository.

    .venv/bin/python scripts/baseline_anomalib.py --category candle
    .venv/bin/python scripts/baseline_anomalib.py --category all --seed 1 --out results/E0_baseline/seed1

Options used by the experiment plan (experiments/run_all.sh):
    --fg-masks DIR     mask the background of every image (train and eval) with
                       cached foreground masks before feature extraction
    --cpu-latency N    after evaluation, time N test images one by one on CPU
    --save-memory-bank keep the coreset memory bank (large) for reuse
"""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anomalib  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402

from iad.metrics import evaluate, pixel_auroc  # noqa: E402
from iad.splits import Sample, read_splits  # noqa: E402


def fg_mask_path(fg_dir: Path, image: str) -> Path:
    return fg_dir / Path(image).with_suffix(".png")


class ImageSet(Dataset):
    def __init__(self, root: Path, samples: list[Sample], transform, size: int, fg_dir: Path | None = None):
        self.root, self.samples, self.transform, self.size, self.fg_dir = root, samples, transform, size, fg_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        image = to_dtype(pil_to_tensor(Image.open(self.root / s.image).convert("RGB")), torch.float32, scale=True)
        if self.fg_dir is not None:
            fg = pil_to_tensor(Image.open(fg_mask_path(self.fg_dir, s.image)).convert("L")) > 0
            image = image * fg
        if s.mask:
            mask = pil_to_tensor(Image.open(self.root / s.mask).convert("L"))[None].float()
            mask = (F.interpolate(mask, size=(self.size, self.size), mode="nearest")[0, 0] > 0).to(torch.uint8)
        else:
            mask = torch.zeros(self.size, self.size, dtype=torch.uint8)
        return self.transform(image), mask


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def cpu_latency_ms(model, loader: DataLoader, n: int, threads: int) -> dict:
    """Median single-image latency on CPU (model forward only, data loading excluded)."""
    torch.set_num_threads(threads)
    model = model.to("cpu").eval()
    times = []
    for images, _ in loader:
        for img in images:
            t = time.perf_counter()
            model(img[None])
            times.append(1000 * (time.perf_counter() - t))
            if len(times) >= n + 2:
                break
        if len(times) >= n + 2:
            break
    times = times[2:]  # warm-up
    return {"cpu_threads": threads, "cpu_ms_median": round(float(np.median(times)), 1), "cpu_n": len(times)}


@torch.no_grad()
def run_category(category: str, samples: list[Sample], args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)  # KCenterGreedy picks its first center with torch.randint
    np.random.seed(args.seed)

    lightning = Patchcore(
        backbone=args.backbone, layers=args.layers, coreset_sampling_ratio=args.coreset, num_neighbors=args.neighbors
    )
    transform = Patchcore.configure_pre_processor(image_size=(args.image_size, args.image_size)).transform
    model = lightning.model.to(device)

    def make_loader(subset: list[Sample]) -> DataLoader:
        return DataLoader(
            ImageSet(args.root, subset, transform, args.image_size, args.fg_masks),
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

    def split_samples(split: str) -> list[Sample]:
        return [s for s in samples if s.category == category and s.split == split]

    train = split_samples("train")
    if args.max_train and len(train) > args.max_train:
        train = sorted(random.Random(args.seed).sample(train, args.max_train), key=lambda s: s.image)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train()
    for images, _ in make_loader(train):
        model(images.to(device))
    model.subsample_embedding(args.coreset)
    sync(device)
    fit_seconds = time.perf_counter() - t0

    model.eval()
    outputs = {}
    for split in ("val", "test"):
        subset = split_samples(split)
        scores, maps, masks, model_seconds = [], [], [], 0.0
        for images, mask in make_loader(subset):
            images = images.to(device)
            sync(device)
            t1 = time.perf_counter()
            pred = model(images)
            sync(device)
            model_seconds += time.perf_counter() - t1
            scores.append(pred.pred_score.flatten().cpu())
            maps.append(pred.anomaly_map.squeeze(1).cpu())
            masks.append(mask)
        outputs[split] = {
            "samples": subset,
            "labels": np.array([s.label for s in subset]),
            "scores": torch.cat(scores).numpy(),
            "maps": torch.cat(maps).numpy(),
            "masks": torch.cat(masks).numpy(),
            "ms_per_image": 1000 * model_seconds / len(subset),
        }

    v, t = outputs["val"], outputs["test"]
    result = evaluate(v["labels"], v["scores"], t["labels"], t["scores"], method=args.threshold_method)
    result.update(
        {
            "category": category,
            "seed": args.seed,
            "pixel_auroc_official_test": pixel_auroc(
                np.concatenate([v["masks"], t["masks"]]), np.concatenate([v["maps"], t["maps"]])
            ),
            "n_train": len(train),
            "memory_bank_size": int(model.memory_bank.shape[0]),
            "memory_bank_mb": round(model.memory_bank.numel() * model.memory_bank.element_size() / 2**20, 1),
            "fit_seconds": round(fit_seconds, 1),
            "device": str(device),
            "device_ms_per_image": round((v["ms_per_image"] + t["ms_per_image"]) / 2, 2),
        }
    )
    if device.type == "cuda":
        result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)

    out_dir = args.out / category
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.csv", "w") as f:
        f.write("split,label,score,image\n")
        for split in ("val", "test"):
            o = outputs[split]
            for s, y, score in zip(o["samples"], o["labels"], o["scores"]):
                f.write(f"{split},{y},{score:.6f},{s.image}\n")
    # Saved for visualization (scripts/visualize.py) and post-hoc analysis without refitting.
    np.savez_compressed(
        out_dir / "maps.npz",
        split=np.array([sp for sp in ("val", "test") for _ in outputs[sp]["samples"]]),
        image=np.array([s.image for sp in ("val", "test") for s in outputs[sp]["samples"]]),
        label=np.concatenate([v["labels"], t["labels"]]),
        score=np.concatenate([v["scores"], t["scores"]]),
        map=np.concatenate([v["maps"], t["maps"]]).astype(np.float16),
        mask=np.concatenate([v["masks"], t["masks"]]),
        threshold=result["test"]["threshold"],
        image_size=args.image_size,
    )
    if args.save_memory_bank:
        torch.save(model.memory_bank.cpu(), out_dir / "memory_bank.pt")
    if args.cpu_latency:
        result.update(cpu_latency_ms(model, make_loader(split_samples("test")), args.cpu_latency, args.cpu_threads))

    # Written last: its presence marks the category as complete for --skip-existing.
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="all", help="'all' or comma-separated categories")
    parser.add_argument("--root", type=Path, default=Path("data/VisA"))
    parser.add_argument("--splits", type=Path, default=Path("splits/visa_seed0.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/anomalib_patchcore"))
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--neighbors", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold-method", default="f1", choices=["f1", "youden"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int, default=0, help="reduced setting: random subset of train images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fg-masks", type=Path, default=None, help="foreground mask dir from scripts/sam_masks.py")
    parser.add_argument("--cpu-latency", type=int, default=0, help="time N test images on CPU after evaluation")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--save-memory-bank", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    samples = read_splits(args.splits)
    all_categories = sorted({s.category for s in samples})
    categories = all_categories if args.category == "all" else args.category.split(",")

    run_info = {
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "machine": platform.machine(),
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps(run_info, indent=2))

    for category in categories:
        if args.skip_existing and (args.out / category / "metrics.json").exists():
            print(f"{category:12s} skipped (exists)", flush=True)
            continue
        r = run_category(category, samples, args, device)
        print(
            f"{category:12s} img AUROC {r['image_auroc_official_test']:.3f} | pixel AUROC "
            f"{r['pixel_auroc_official_test']:.3f} | test F1 {r['test']['f1']:.3f} "
            f"(recall {r['test']['anomaly_recall']:.3f}, spec {r['test']['normal_specificity']:.3f}) | "
            f"fit {r['fit_seconds']}s, {r['device_ms_per_image']} ms/img",
            flush=True,
        )


if __name__ == "__main__":
    main()
