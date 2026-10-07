"""Create splits/visa_seed{seed}.csv from VisA's official 1-class split."""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iad.splits import make_splits, read_official_split, write_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/VisA"))
    parser.add_argument("--val-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_official_split(args.root / "split_csv" / "1cls.csv")
    samples = make_splits(rows, args.val_fraction, args.seed)
    out = Path("splits") / f"visa_seed{args.seed}.csv"
    write_splits(samples, out)

    counts = Counter((s.category, s.split, s.label) for s in samples)
    for category in sorted({s.category for s in samples}):
        parts = [
            f"{split}: {counts[(category, split, 0)]} normal / {counts[(category, split, 1)]} anomaly"
            for split in ("train", "val", "test")
        ]
        print(f"{category:12s} " + " | ".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
