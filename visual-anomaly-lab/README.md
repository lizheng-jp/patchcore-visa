# Visual Anomaly Lab

Live PatchCore inference on three VisA categories (candle, pcb4, macaroni2), a held-out score chart for each model, and an EfficientSAM foreground-masking study. Live at [vision.lizheng.cc](https://vision.lizheng.cc/). Styled to match [Signals](https://signals.lizheng.cc); the two demos link to each other.

## Run locally

Double-click `start.command`, or:

```sh
../.venv/bin/python server.py            # http://127.0.0.1:8766/?lang=en  (or ?lang=ja, &model=macaroni2)
```

## What the page shows

- **Live inference.** Each model was built from 200 normal training images (`models/reduced_max200/<category>`). Bundled samples are sent by id and scored from the original file, so their scores match `scores.csv` exactly. Heatmaps use a fixed colour range per category (`showcase.json`), so colours can be compared across images.
- **Discrimination chart.** Every held-out test image of the selected model, with the threshold chosen on val.
- **SAM study (macaroni2).** `tools/sam_clicks.py` prompts EfficientSAM with one point per pasta piece, the way a visitor clicks, on all 200 val/test images. `tools/sam_study.py` selects both thresholds on val and writes `sam-study.json` and `cases/*.jpg`.

| Macaroni2, held-out test | Full image | SAM foreground |
|---|---|---|
| Image AUROC (val + test) | 0.634 | **0.797** |
| False alarms (of 50 normal) | 38 | **23** |
| Missed defects (of 50 anomalous) | **6** | 9 |

Defect pixels inside the mask average 99.9%. The extra misses are images that were flagged only because of the background before masking (see the counterexample card). One run, one seed.

Samples were chosen as typical correctly classified test images (20th, 50th and 80th percentile of each class), not the easiest ones. For macaroni2 they were chosen to demonstrate the SAM effect. The chart shows every test image, including the errors.

## Deploy next to Signals (Lightsail)

The image holds code only. Copy `models/`, `samples/` and `runtime/weights/` (about 700 MB) to the host and mount them:

```sh
docker build -t visual-anomaly-lab .
docker run -d --name lab -p 127.0.0.1:8766:8766 \
  -v "$PWD/models:/app/models:ro" -v "$PWD/samples:/app/samples:ro" \
  -v "$PWD/runtime/weights:/app/runtime/weights:ro" visual-anomaly-lab
```

The production target is `https://vision.lizheng.cc`. Use the separate Compose project in `compose.prod.yml`, attached to the existing `dev-productivity-signals_default` network:

```sh
docker buildx build --platform linux/amd64 -t visual-anomaly-lab:<tag in compose.prod.yml> --load .
docker compose -f compose.prod.yml up -d --no-build
```

Copy `deploy/Caddyfile.vision` to `deploy/sites/vision.caddy` in the Signals deployment directory on the server. Signals' Caddyfile imports `deploy/sites/*.caddy`, and its CI deploy never overwrites that directory; do not append the block to the Signals Caddyfile, which every Signals deploy replaces. Keep this app in its own Compose project so a Signals deployment using `--remove-orphans` does not remove it. Port 8766 is bound only to localhost; visitors use HTTPS through Caddy. Model weights and samples are mounted read-only. The container has a 1.5 GiB memory limit, a 2 GiB RAM-plus-swap limit, and a 1.5 CPU limit.

Host-specific operations notes are kept outside the repository (`deploy/DEPLOYMENT.md` is git-ignored). User-uploaded inference images are processed in memory and are not saved by this application.

Measured with `docker stats` (image 2.4 GB): about 670 MB idle and 820 MB peak during inference with EfficientSAM. The shared host has 4 GB RAM plus 2 GB swap. First inference takes about 5 s (weight loading), then 0.3 to 2 s on 2 CPU threads. Inference serialises behind one lock: concurrent visitors queue.

Sample source: [Amazon VisA](https://github.com/amazon-science/spot-diff), CC BY 4.0. No proprietary manufacturing images or weights are included.
