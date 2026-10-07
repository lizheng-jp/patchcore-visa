# Experiment plan

All experiments use `splits/visa_seed0.csv`: official normal-only train set; the official test set divided per category and label into `val` (threshold selection only) and `test` (reported once). Image AUROC is also reported on the full official test set for comparison with published numbers.

Run everything with `bash experiments/run_all.sh` (resumable), or selected phases, e.g. `bash experiments/run_all.sh E0 E2`.

| ID | Question | Setting | Scope |
|---|---|---|---|
| E0 | Baseline: how good is standard PatchCore under a leakage-free protocol? | Wide ResNet-50, layer2+3, 256 px, coreset 0.1 | 12 categories × seeds 0, 1, 2 |
| E1a | Does SAM foreground masking reduce background false alarms? | E0 anomaly maps, scored only inside the SAM foreground (no refit) | 12 categories × 3 seeds |
| E1b | Does removing the background *before* feature extraction help? | E0 setting, background blacked out in train and eval images | candle, pcb1, macaroni1 × 3 seeds |
| E2 | Accuracy vs CPU latency of a lighter backbone | ResNet-18 vs Wide ResNet-50 (E0 seed 0) | 12 categories, seed 0 |
| E3 | Memory bank size vs accuracy and latency | coreset 0.01 and 0.25 vs 0.1 (E0 seed 0) | 12 categories, seed 0 |
| E4 | Does higher resolution recover tiny missed defects? | 256 vs 512 px, both with 200 train images | candle, capsules, pcb1, pcb4, seed 0 |

## Saved per run (`results/<experiment>/<variant>/<category>/`)

| File | Contents |
|---|---|
| `metrics.json` | image/pixel AUROC, val and test thresholded metrics (threshold, recall, precision, specificity, F1, confusion matrix), train size, memory bank size/MB, fit time, GPU ms/image, CPU median ms/image (where measured), peak GPU MB |
| `scores.csv` | per-image split, label, score, image path |
| `maps.npz` | per-image anomaly maps (float16), ground-truth masks, labels, scores, threshold; input to `visualize.py` and `posthoc_mask_eval.py` |
| `figures/` | `examples.png` (detections, misses, false alarms) and `scores.png` (val/test score distribution with threshold, ROC); generated for E0 seed 0 and E1b |

Also: `results/<experiment>/<variant>/run.json` (library versions, device, GPU, all arguments), `data/fg_masks/sam-vit-base/qa.csv` (per-image foreground fraction and defect coverage of SAM masks) with `preview/` overlays, `results/summary.csv` and `results/summary.md` (all runs; mean ± std across seeds), `results/logs/` (console logs, commands, `pip freeze`).

Memory banks are not saved by default (about 0.5 GB per category at full size); add `--save-memory-bank` if needed.
