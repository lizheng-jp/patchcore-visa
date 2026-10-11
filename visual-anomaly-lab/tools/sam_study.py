"""Build sam-study.json and cases/*.jpg for the SAM section from tools/sam_clicks.py output.

    ../.venv/bin/python tools/sam_study.py .runtime-cache/macaroni2_sam.json
"""
import sys, json, io, base64
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw
LAB = Path(__file__).resolve().parents[1]; VISA = LAB.parent / 'data/VisA'
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / 'src'))
import server
from iad.metrics import evaluate
from sklearn.metrics import roc_auc_score
sim = json.load(open(sys.argv[1]))
L = np.array([r['label'] for r in sim]); sp = np.array([r['split'] for r in sim])
full = np.array([r['map_max'] for r in sim]); sam = np.array([r['masked'] for r in sim])
v, t = sp == 'val', sp == 'test'
res = {k: evaluate(L[v], s[v], L[t], s[t], method='f1') for k, s in [('full', full), ('sam', sam)]}
def dec(b64): return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64.split(',')[1]))).convert('RGB'))
cases = [('false-alarm-removed', 'macaroni2/Data/Images/Normal/0844.JPG'),
         ('defect-kept', 'macaroni2/Data/Images/Anomaly/038.JPG'),
         ('counterexample', 'macaroni2/Data/Images/Anomaly/059.JPG')]
by = {r['image']: r for r in sim}
(LAB / 'cases').mkdir(exist_ok=True)
out_cases = []
for key, img in cases:
    r = by[img]; im = Image.open(VISA / img).convert('RGB')
    res_i = server.predict(im, r['points'], 'object', 'macaroni2')
    amap = np.array(res_i['map']); fg = dec(res_i['sam']['mask'])[..., 0] > 0
    y, x = np.unravel_index(amap.argmax(), amap.shape)
    base = dec(res_i['original']).copy()
    if r['label'] == 1:
        gt = np.asarray(Image.open(VISA / (img.replace('Images/Anomaly', 'Masks/Anomaly').rsplit('.', 1)[0] + '.png')).convert('L').resize((256, 256), Image.NEAREST)) > 0
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, cs, -1, (240, 71, 107), 2)
    orig = Image.fromarray(base); d = ImageDraw.Draw(orig)
    for px, py in r['points']:
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill='white', outline='#3b63f0', width=2)
    a = Image.fromarray(dec(res_i['overlay'])); b = Image.fromarray(dec(res_i['sam']['overlay']))
    for panel, m in [(a, amap), (b, np.array(res_i['sam']['map']))]:
        yy, xx = np.unravel_index(m.argmax(), m.shape)
        ImageDraw.Draw(panel).ellipse([xx - 11, yy - 11, xx + 11, yy + 11], outline='white', width=3)
    canvas = Image.new('RGB', (256 * 3 + 16, 256), 'white')
    for i, p in enumerate([orig, a, b]): canvas.paste(p, (i * 264, 0))
    canvas.save(LAB / 'cases' / f'macaroni2-{key}.jpg', quality=88)
    out_cases.append({'key': key, 'image': img.rsplit('/', 1)[-1], 'label': r['label'], 'full': r['map_max'], 'sam': r['masked'],
                      'full_max_in_foreground': bool(fg[y, x]), 'defect_coverage': r.get('coverage')})
    print(key, 'max in fg:', fg[y, x], round(r['map_max'], 2), '->', round(r['masked'], 2))
def summary(k):
    s = full if k == 'full' else sam; te = res[k]['test']
    return {'threshold': te['threshold'], 'image_auroc_val_test': roc_auc_score(L, s), 'accuracy': te['accuracy'],
            'anomaly_recall': te['anomaly_recall'], 'normal_specificity': te['normal_specificity'], 'confusion': te['confusion']}
study = {'category': 'macaroni2', 'model': 'reduced_max200 live model (200 normal training images)',
         'protocol': 'One EfficientSAM prompt point per pasta piece (placed automatically, as a visitor would click), Whole object scope. Score = maximum of the same anomaly map over the full image or inside the foreground. Threshold = max F1 on val, applied once to test.',
         'n_val': int(v.sum()), 'n_test': int(t.sum()),
         'mean_defect_coverage': float(np.mean([r['coverage'] for r in sim if r.get('coverage') is not None])),
         'full': summary('full'), 'sam': summary('sam'), 'cases': out_cases,
         'test_points': [{'image': r['image'].rsplit('/', 1)[-1], 'label': r['label'], 'full': round(r['map_max'], 3), 'sam': round(r['masked'], 3)} for r in sim if r['split'] == 'test']}
json.dump(study, open(LAB / 'sam-study.json', 'w'), indent=1)
print(json.dumps({k: study[k] for k in ['full', 'sam', 'mean_defect_coverage']}, indent=1))
