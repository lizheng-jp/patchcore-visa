"""Visual Anomaly Lab: live PatchCore inference on three VisA categories, with optional EfficientSAM masking."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime-cache/mpl'))
import sys, io, json, base64, time, threading, csv
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, parse_qs
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype
from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

WEIGHT = Path(os.environ.get('PATCHCORE_BACKBONE', str(ROOT / 'models/backbone.safetensors')))
MODEL_DIRS = {c: ROOT / 'models/reduced_max200' / c for c in ['candle', 'pcb4', 'macaroni2']}
# Curated samples and the fixed heatmap colour range per category (see showcase.json).
SHOWCASE = json.loads((ROOT / 'showcase.json').read_text())
STATIC = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
    '/i18n.js': ('i18n.js', 'text/javascript; charset=utf-8'),
    '/style.css': ('style.css', 'text/css; charset=utf-8'),
    '/favicon.svg': ('favicon.svg', 'image/svg+xml'),
    '/recorded-experiments.json': ('recorded-experiments.json', 'application/json; charset=utf-8'),
    '/sam-study.json': ('sam-study.json', 'application/json; charset=utf-8'),
}
CASE_DIR = ROOT / 'cases'


def model_info(category):
    if category not in MODEL_DIRS:
        raise ValueError('bad_category')
    directory = MODEL_DIRS[category]
    if not (directory / 'memory_bank.pt').exists():
        raise ValueError('model_unavailable')
    return json.loads((directory / 'metrics.json').read_text())


def category_samples(category):
    model_info(category)
    return [{'id': i, 'category': category, **s} for i, s in enumerate(SHOWCASE[category]['samples'])]


def distribution(category):
    """Held-out test scores of the loaded model, for the normal-vs-anomalous chart."""
    info = model_info(category)
    with (MODEL_DIRS[category] / 'scores.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
    return {'category': category, 'threshold': info['test']['threshold'],
            'image_auroc': info.get('image_auroc_official_test'),
            'points': [{'score': float(r['score']), 'label': int(r['label']), 'image': r['image'].rsplit('/', 1)[-1]} for r in rows]}


lock = threading.Lock(); model = None; sam = None; active_category = None


def png(a):
    b = io.BytesIO(); Image.fromarray(a.astype(np.uint8)).save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def get_model(category='candle'):
    global model, active_category
    if model is None:
        torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '2')))
        m = PatchcoreModel(layers=['layer2', 'layer3'], backbone='wide_resnet50_2', pre_trained=False, num_neighbors=9)
        from safetensors.torch import load_file
        state = load_file(str(WEIGHT), device='cpu')
        extractor = m.feature_extractor.feature_extractor
        extractor.load_state_dict({k: v for k, v in state.items() if k in extractor.state_dict()}, strict=True)
        model = m.eval()
    if active_category != category:
        model_info(category)
        model.memory_bank = torch.load(MODEL_DIRS[category] / 'memory_bank.pt', map_location='cpu', weights_only=True)
        active_category = category
    return model


@torch.inference_mode()
def segment(rgb, points, scope):
    global sam
    if not points:
        raise ValueError('need_points')
    if sam is None:
        sys.path.insert(0, str(ROOT / 'runtime'))
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
        sam = build_efficient_sam_vitt().eval()
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
    p = torch.tensor(points, dtype=torch.float32)[None, :, None, :]
    logits, iou = sam(x, p, torch.ones((1, len(points), 1), dtype=torch.int64))
    candidates = logits[0] >= 0
    if scope == 'object':
        area = candidates.float().mean((-2, -1))
        acceptable = (area > 0) & (area <= 0.4)
        best = area.masked_fill(~acceptable, -1).argmax(-1)
        fallback = ~acceptable.any(-1)
        best[fallback] = iou[0].argmax(-1)[fallback]
    else:
        best = iou[0].argmax(-1)
    masks = candidates[torch.arange(len(points)), best]
    return masks.any(0).numpy()


def heatmap(amap, lo, hi):
    """Colour with a fixed per-category range so normal and anomalous images are comparable."""
    norm = np.clip((amap - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]


@torch.inference_mode()
def predict(im, points, scope='object', category='candle'):
    info = model_info(category); threshold = info['test']['threshold']
    lo, hi = SHOWCASE[category]['display_range']
    start = time.perf_counter(); rgb = np.asarray(im.resize((256, 256))).copy()
    transform = Patchcore.configure_pre_processor(image_size=(256, 256)).transform
    pred = get_model(category)(transform(to_dtype(pil_to_tensor(im), torch.float32, scale=True))[None])
    amap = pred.anomaly_map[0, 0].numpy(); score = float(pred.pred_score[0])
    if not np.isfinite(amap).all() or not np.isfinite(score):
        raise RuntimeError('nonfinite')
    heat = heatmap(amap, lo, hi)
    out = {'score': score, 'category': category, 'threshold': threshold,
           'decision': 'anomalous' if score >= threshold else 'normal',
           'map_max': float(amap.max()), 'map_min': float(amap.min()), 'display_range': [lo, hi],
           'map': amap.round(5).tolist(), 'original': png(rgb), 'heat': png(heat),
           'overlay': png(rgb * .58 + heat * .42), 'elapsed': 0, 'sam': None}
    if points:
        fg = segment(rgb, points, scope)
        masked = np.where(fg, amap, 0)
        out['sam'] = {'score': float(masked.max()), 'baseline_score': float(amap.max()),
                      'foreground_fraction': float(fg.mean()), 'mask': png(fg * 255),
                      'overlay': png(np.where(fg[..., None], rgb * .58 + heat * .42, rgb * .15)),
                      'map': masked.round(5).tolist()}
        # Thresholds selected on val for both map-maximum scores (sam-study.json); categories without them stay uncalibrated.
        thresholds = SHOWCASE[category].get('sam_thresholds')
        if thresholds:
            out['sam']['thresholds'] = thresholds
            out['sam']['decision_full'] = 'anomalous' if out['sam']['baseline_score'] >= thresholds['full'] else 'normal'
            out['sam']['decision'] = 'anomalous' if out['sam']['score'] >= thresholds['sam'] else 'normal'
    out['elapsed'] = round(time.perf_counter() - start, 2)
    return out


ERROR_CODES = ['not_found', 'bad_size', 'bad_points', 'bad_scope', 'need_points', 'bad_category', 'model_unavailable']


class Handler(BaseHTTPRequestHandler):
    def send(self, data, status=200, ctype='application/json', cache='no-store'):
        body = json.dumps(data, ensure_ascii=False).encode() if ctype == 'application/json' else data
        self.send_response(status); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', cache)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path); category = parse_qs(url.query).get('category', ['candle'])[0]
        try:
            if url.path == '/api/models':
                return self.send([{'category': c, 'metrics': model_info(c)} for c in MODEL_DIRS if (MODEL_DIRS[c] / 'metrics.json').exists()])
            if url.path == '/api/samples':
                return self.send(category_samples(category))
            if url.path == '/api/distribution':
                return self.send(distribution(category))
            if url.path.startswith('/api/sample/'):
                index = int(url.path.rsplit('/', 1)[1]); items = category_samples(category)
                if index < 0 or index >= len(items):
                    raise ValueError('not_found')
                sample = items[index]; im = Image.open(ROOT / 'samples' / sample['image']).convert('RGB')
                im.thumbnail((512, 512))
                return self.send({**sample, 'image': png(np.asarray(im))})
        except (ValueError, IndexError, OSError):
            return self.send({'error_code': 'model_unavailable'}, 400)
        if url.path == '/api/health':
            return self.send({'status': 'ready', 'model_loaded': model is not None, 'active_category': active_category, 'categories': list(MODEL_DIRS)})
        if url.path in STATIC:
            p, t = STATIC[url.path]
            return self.send((ROOT / p).read_bytes(), ctype=t, cache='no-cache')
        if url.path.startswith('/cases/'):
            f = (CASE_DIR / url.path.removeprefix('/cases/')).resolve()
            if f.parent == CASE_DIR.resolve() and f.suffix == '.jpg' and f.exists():
                return self.send(f.read_bytes(), ctype='image/jpeg', cache='public, max-age=86400')
        return self.send({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send({'error': 'Not found'}, 404)
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if n <= 0 or n > 16 * 1024 * 1024:
                raise ValueError('bad_size')
            data = json.loads(self.rfile.read(n))
            category = data.get('category', 'candle'); model_info(category)
            if isinstance(data.get('sample'), int):
                # Bundled samples are scored from the original file, so results match scores.csv exactly.
                items = category_samples(category)
                if not 0 <= data['sample'] < len(items):
                    raise ValueError('not_found')
                im = Image.open(ROOT / 'samples' / items[data['sample']]['image']).convert('RGB')
            else:
                raw = base64.b64decode(data['image'].split(',', 1)[1], validate=True)
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))); im.load(); im = im.convert('RGB')
            points = data.get('points', [])
            if not isinstance(points, list) or len(points) > 16 or any(
                    not isinstance(p, list) or len(p) != 2 or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 or v > 255 for v in p)
                    for p in points):
                raise ValueError('bad_points')
            scope = data.get('sam_scope', 'object')
            if scope not in ['object', 'detail']:
                raise ValueError('bad_scope')
            with lock:
                out = predict(im, points, scope, category)
            return self.send(out)
        except (ValueError, KeyError, IndexError, Image.DecompressionBombError, OSError) as e:
            return self.send({'error': 'Invalid image request.', 'error_code': str(e) if str(e) in ERROR_CODES else 'unreadable'}, 400)
        except Exception:
            import traceback; traceback.print_exc()
            return self.send({'error': 'Inference failed.', 'error_code': 'inference_error'}, 500)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, default=8766); a = p.parse_args()
    print(f'Visual Anomaly Lab: http://127.0.0.1:{a.port}', flush=True)
    ThreadingHTTPServer.request_queue_size = 64  # the page fetches six samples at once; the default of 5 resets some
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer((os.environ.get('BIND_HOST', '127.0.0.1'), a.port), Handler).serve_forever()
