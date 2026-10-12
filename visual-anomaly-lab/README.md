# Visual Anomaly Lab

Live demo for reproducing, on public data, a finding from manufacturing inspection: where false alarms come from the background, SAM background removal improves anomaly detection. PatchCore inference on three VisA categories (candle, pcb4, macaroni2), a held-out score chart for each model, and an EfficientSAM masking study on Macaroni2. Live at [vision.lizheng.cc](https://vision.lizheng.cc/).

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

## Deploy (CI/CD)

`.github/workflows/vision.yml` runs on every pull request and push: unit tests for the evaluation protocol, syntax checks for the server, tools and page scripts, and a Docker build. On `main`, when `visual-anomaly-lab/` or the workflow changed, it also:

1. pushes `ghcr.io/lizheng-jp/patchcore-visa-vision:<commit>` (and `:latest`) built on GitHub's amd64 runners;
2. copies `compose.prod.yml` to the server and runs `docker compose pull && up -d` with that tag (the GHCR login uses the job token over stdin and is logged out afterwards), then waits for `/api/health`;
3. syncs `deploy/Caddyfile.vision` to Signals' `deploy/sites/vision.caddy`, validates the Caddy config, and reloads Caddy only if the site changed (the previous site is restored if validation fails);
4. checks `https://vision.lizheng.cc/api/health`.

Secrets: `DEPLOY_SSH_KEY`, `DEPLOY_KNOWN_HOSTS`, `DEPLOY_HOST`, `DEPLOY_USER`, `VISION_PATH`, `SIGNALS_PATH`. The image holds code only. `models/`, `samples/` and `runtime/weights/` (about 700 MB) live on the server and are mounted read-only; copy them by hand when a model changes.

Run locally with Docker:

```sh
docker build -t visual-anomaly-lab .
docker run -d --name lab -p 127.0.0.1:8766:8766 \
  -v "$PWD/models:/app/models:ro" -v "$PWD/samples:/app/samples:ro" \
  -v "$PWD/runtime/weights:/app/runtime/weights:ro" visual-anomaly-lab
```

Vision runs as its own Compose project (`compose.prod.yml`) on the existing `dev-productivity-signals_default` network, so a Signals deploy using `--remove-orphans` does not remove it. Port 8766 is bound only to localhost; visitors use HTTPS through Caddy. The container has a 1.5 GiB memory limit, a 2 GiB RAM-plus-swap limit, and a 1.5 CPU limit.

Host-specific operations notes are kept outside the repository (`deploy/DEPLOYMENT.md` is git-ignored). User-uploaded inference images are processed in memory and are not saved by this application.

Measured with `docker stats` (image 2.4 GB): about 670 MB idle and 820 MB peak during inference with EfficientSAM. The shared host has 4 GB RAM plus 2 GB swap. First inference takes about 5 s (weight loading), then 0.3 to 2 s on 2 CPU threads. Inference serialises behind one lock: concurrent visitors queue.

Sample source: [Amazon VisA](https://github.com/amazon-science/spot-diff), CC BY 4.0. No proprietary manufacturing images or weights are included.
