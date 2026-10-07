"""Leakage-free train / val / test splits for VisA.

VisA's official 1-class split (split_csv/1cls.csv) has a normal-only train set
and a mixed test set. Thresholds must not be tuned on the data used to report
results, so the official test set is divided, per category and per label, into:

- val:  used only to select the decision threshold
- test: used once to report thresholded metrics

AUROC is threshold-free, so it is also reported on the full official test set
for comparison with published results.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

NORMAL = "normal"
ANOMALY = "anomaly"


@dataclass(frozen=True)
class Sample:
    category: str
    split: str  # train | val | test
    label: int  # 0 = normal, 1 = anomaly
    image: str  # path relative to the dataset root
    mask: str  # "" for normal samples


def read_official_split(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {"object", "split", "label", "image", "mask"}
    missing = expected - set(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    return rows


def make_splits(rows: list[dict[str, str]], val_fraction: float = 0.5, seed: int = 0) -> list[Sample]:
    """Divide the official test set into val/test, stratified by category and label."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    samples: list[Sample] = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["label"] not in (NORMAL, ANOMALY):
            raise ValueError(f"unexpected label {row['label']!r} for {row['image']}")
        if row["split"] == "train":
            if row["label"] != NORMAL:
                raise ValueError(f"anomalous image in the train split: {row['image']}")
            samples.append(Sample(row["object"], "train", 0, row["image"], ""))
        elif row["split"] == "test":
            groups.setdefault((row["object"], row["label"]), []).append(row)
        else:
            raise ValueError(f"unexpected split {row['split']!r} for {row['image']}")

    rng = random.Random(seed)
    for (category, label), group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r["image"])
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        y = 1 if label == ANOMALY else 0
        for i, row in enumerate(group):
            split = "val" if i < n_val else "test"
            samples.append(Sample(category, split, y, row["image"], row.get("mask", "") if y else ""))
    return samples


def write_splits(samples: list[Sample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "split", "label", "image", "mask"])
        for s in samples:
            writer.writerow([s.category, s.split, s.label, s.image, s.mask])


def read_splits(path: Path) -> list[Sample]:
    with open(path, newline="") as f:
        return [
            Sample(r["category"], r["split"], int(r["label"]), r["image"], r["mask"])
            for r in csv.DictReader(f)
        ]
