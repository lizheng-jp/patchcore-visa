#!/usr/bin/env bash
# Full experiment plan. Resumable: finished categories are skipped (--skip-existing),
# so the script can be rerun after an interruption.
#
#   bash experiments/run_all.sh            # everything
#   bash experiments/run_all.sh E0 E1      # selected phases
#
# Expects: .venv with `pip install -e . anomalib transformers`, data/VisA, splits/visa_seed0.csv
set -euo pipefail
cd "$(dirname "$0")/.."
export TQDM_DISABLE=1
PY=.venv/bin/python
R=results
FG=data/fg_masks/sam-vit-base
SEEDS="0 1 2"
SAM_CATS="candle,pcb1,macaroni1"        # input-masking subset (train masks are the expensive part)
SMALL_DEFECT_CATS="candle,capsules,pcb1,pcb4"
PHASES="${*:-E0 E1 E2 E3 E4 SUMMARY}"
mkdir -p $R/logs

run() { echo ">>> $*" | tee -a $R/logs/commands.log; "$@" 2>&1 | grep --line-buffered -v -i warn | tee -a "$R/logs/$LOG.log"; }

for phase in $PHASES; do case $phase in

E0)  # Baseline: anomalib PatchCore defaults, all categories, 3 seeds
  LOG=E0
  for s in $SEEDS; do
    extra=""; [ "$s" = 0 ] && extra="--cpu-latency 20"
    run $PY scripts/baseline_anomalib.py --category all --seed $s --out $R/E0_baseline/seed$s --skip-existing $extra
  done ;;

E1)  # SAM foreground masking
  LOG=E1
  run $PY scripts/sam_masks.py --category all --splits val,test --out $FG --skip-existing
  run $PY scripts/sam_masks.py --category $SAM_CATS --splits train --out $FG --skip-existing
  for s in $SEEDS; do  # E1a: masks applied to saved E0 anomaly maps (no refit)
    run $PY scripts/posthoc_mask_eval.py $R/E0_baseline/seed$s --fg-masks $FG --out $R/E1a_posthoc_mask/seed$s
  done
  for s in $SEEDS; do  # E1b: background removed before feature extraction (train and eval)
    run $PY scripts/baseline_anomalib.py --category $SAM_CATS --seed $s --fg-masks $FG \
      --out $R/E1b_input_mask/seed$s --skip-existing
  done ;;

E2)  # Backbone: lighter ResNet-18 vs Wide ResNet-50 (E0 seed0), accuracy vs CPU latency
  LOG=E2
  run $PY scripts/baseline_anomalib.py --category all --seed 0 --backbone resnet18 \
    --cpu-latency 20 --out $R/E2_backbone/resnet18 --skip-existing ;;

E3)  # Coreset ratio: memory bank size and latency vs accuracy (0.1 is E0 seed0)
  LOG=E3
  for c in 0.01 0.25; do
    run $PY scripts/baseline_anomalib.py --category all --seed 0 --coreset $c \
      --cpu-latency 20 --out $R/E3_coreset/c$c --skip-existing
  done ;;

E4)  # Input resolution for small defects (200 train images each, so 256 vs 512 is a fair pair)
  LOG=E4
  for size in 256 512; do
    run $PY scripts/baseline_anomalib.py --category $SMALL_DEFECT_CATS --seed 0 --max-train 200 \
      --image-size $size --batch-size 8 --out $R/E4_resolution/s$size --skip-existing
  done ;;

SUMMARY)
  LOG=summary
  run $PY scripts/aggregate.py $R
  for d in $R/E0_baseline/seed0/*/; do run $PY scripts/visualize.py "$d"; done
  for d in $R/E1b_input_mask/seed0/*/; do run $PY scripts/visualize.py "$d"; done
  $PY -m pip freeze > $R/logs/pip-freeze.txt ;;

*) echo "unknown phase $phase"; exit 1 ;;
esac; done
