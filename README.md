# patchcore-visa

SAM-masked PatchCore for industrial visual anomaly detection, evaluated on the public [VisA](https://github.com/amazon-science/spot-diff) dataset with a protocol designed to avoid optimistic results.

## Why this project

I previously built a segmentation + PatchCore inspection pipeline on proprietary manufacturing data. Re-examining that work showed two common evaluation mistakes: random image-level splits that leak lots and devices across train and test, and decision thresholds tuned on the test set. This repository reimplements the approach on open data, from scratch, with those mistakes designed out.

## Evaluation protocol

- **Train** on the official normal-only train split.
- **Split the official test set** per category and per label into `val` and `test` (seeded, see `src/iad/splits.py`).
- **Select the threshold on `val` only** (max F1 or Youden J), then apply it once to `test`.
- **Report** image AUROC on the full official test set (threshold-free, comparable with papers), plus thresholded recall, precision, specificity, and F1 on `test`.
- Label `1` always means anomalous. Metrics name the class explicitly (`anomaly_recall`, `normal_specificity`).

## Setup

```sh
python -m venv .venv && .venv/bin/pip install -e .
scripts/download_visa.sh            # ~1.8 GB, CC BY 4.0
.venv/bin/python scripts/make_splits.py --seed 0
.venv/bin/python -m unittest discover -s tests
```

## Roadmap

- [x] Leakage-free split and evaluation protocol with tests
- [x] anomalib PatchCore baseline script on the same splits; reduced-setting run on candle verified locally
- [x] Experiment plan E0–E4 scripted (`experiments/`); full runs pending on GPU
- [ ] PatchCore reimplementation: Wide ResNet-50 features, greedy coreset, FAISS kNN, heatmaps
- [x] SAM foreground mask generation with defect-coverage QA (`scripts/sam_masks.py`)
- [ ] Ablations: masking on/off; coreset ratio vs. CPU latency
- [ ] Pixel-level metrics: pixel AUROC done; PRO pending
- [ ] Demo: upload an image, get a score and heatmap
- [ ] Optional: vision-language model explanation of detected defects, evaluated against ground-truth masks

## Results

Not yet available. See `experiments/README.md` for the plan and what each run saves. Every number added here will state the split, seed, and threshold method used.

## Data and license

VisA is released by Amazon under CC BY 4.0. The dataset is not redistributed in this repository. No proprietary data, code, or weights are used.
