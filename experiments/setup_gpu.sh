#!/usr/bin/env bash
# One-time setup on a fresh GPU machine (e.g. a Runpod pod), from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e . anomalib transformers
[ -f data/VisA/split_csv/1cls.csv ] || bash scripts/download_visa.sh
cp splits/visa_seed0.csv /tmp/expected_splits.csv
.venv/bin/python scripts/make_splits.py --seed 0
cmp splits/visa_seed0.csv /tmp/expected_splits.csv && echo "splits identical to the committed file"
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
