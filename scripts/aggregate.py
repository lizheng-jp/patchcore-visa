"""Collect metrics.json files into one table.

    .venv/bin/python scripts/aggregate.py results

Writes results/summary.csv (one row per experiment, seed, category) and
results/summary.md (per experiment: mean ± std over seeds, per category and the
category average).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS = [
    ("image_auroc", lambda m: m["image_auroc_official_test"]),
    ("pixel_auroc", lambda m: m["pixel_auroc_official_test"]),
    ("test_f1", lambda m: m["test"]["f1"]),
    ("test_recall", lambda m: m["test"]["anomaly_recall"]),
    ("test_specificity", lambda m: m["test"]["normal_specificity"]),
    ("val_f1", lambda m: m["val"]["f1"]),
]
COST = ["fit_seconds", "device_ms_per_image", "cpu_ms_median", "memory_bank_mb", "peak_gpu_mb"]


def rows(results: Path):
    for path in sorted(results.rglob("metrics.json")):
        rel = path.relative_to(results).parts  # <experiment>/<seed or variant>/<category>/metrics.json
        if len(rel) < 3:
            continue
        experiment, variant, category = rel[0], "/".join(rel[1:-2]), rel[-2]
        m = json.loads(path.read_text())
        if "masked_map_max" in m:  # post-hoc masking: two comparable entries per file
            for name in ("unmasked_map_max", "masked_map_max"):
                yield {"experiment": f"{experiment}:{name}", "variant": variant, "category": category,
                       **{k: f(m[name]) for k, f in FIELDS}, "defect_pixel_coverage": m["defect_pixel_coverage"]}
        else:
            yield {"experiment": experiment, "variant": variant, "category": category,
                   **{k: f(m) for k, f in FIELDS}, **{k: m.get(k) for k in COST}}


def fmt(values: list[float]) -> str:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return "–"
    return f"{v.mean():.3f}" if v.size == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    all_rows = list(rows(args.results))
    if not all_rows:
        raise SystemExit("no metrics.json found")
    keys = sorted({k for r in all_rows for k in r}, key=lambda k: (k not in ("experiment", "variant", "category"), k))
    with open(args.results / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    lines = ["# Results summary", "", "Mean ± std over seeds where several seeds exist. "
             "Thresholds are selected on val only; test metrics use that fixed threshold.", ""]
    by_exp = defaultdict(list)
    for r in all_rows:
        by_exp[r["experiment"]].append(r)
    metric_names = [k for k, _ in FIELDS]
    for exp, rs in sorted(by_exp.items()):
        lines += [f"## {exp}", "", "| category | " + " | ".join(metric_names) + " |",
                  "|---" * (len(metric_names) + 1) + "|"]
        cats = sorted({r["category"] for r in rs})
        for c in cats:
            cr = [r for r in rs if r["category"] == c]
            lines.append(f"| {c} | " + " | ".join(fmt([r[k] for r in cr]) for k in metric_names) + " |")
        # category average per variant (seed), then mean ± std across variants
        variants = sorted({r["variant"] for r in rs})
        avg = {k: [np.mean([r[k] for r in rs if r["variant"] == v]) for v in variants] for k in metric_names}
        lines.append("| **mean** | " + " | ".join(f"**{fmt(avg[k])}**" for k in metric_names) + " |")
        lines.append("")
    (args.results / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
