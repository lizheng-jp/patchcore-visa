"""Score every val/test image with and without EfficientSAM, prompting one point per object
the way a visitor clicks in the page (largest connected components of a colour mask).

    ../.venv/bin/python tools/sam_clicks.py macaroni2 .runtime-cache/macaroni2_sam.json
"""
import sys, csv, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
LAB = Path(__file__).resolve().parents[1]
VISA = LAB.parent / 'data/VisA'
sys.path.insert(0, str(LAB))
import server

def prompts(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    fg = ((r - g) > -10) & (r > 120)          # yellow pasta vs green mat
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:4]
    pts = []
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < 60: continue
        dt = cv2.distanceTransform((lab == i).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(dt.argmax(), dt.shape)
        pts.append([float(x), float(y)])
    return pts

cat = sys.argv[1]
rows = list(csv.DictReader(open(LAB / f'models/reduced_max200/{cat}/scores.csv')))
out = []
for k, row in enumerate(rows):
    im = Image.open(VISA / row['image']).convert('RGB')
    rgb = np.asarray(im.resize((256, 256)))
    pts = prompts(rgb)
    with server.lock:
        res = server.predict(im, pts, 'object', cat)
    rec = dict(image=row['image'], split=row['split'], label=int(row['label']), score=res['score'],
               map_max=res['map_max'], masked=res['sam']['score'] if res['sam'] else None,
               fg=res['sam']['foreground_fraction'] if res['sam'] else None, n_points=len(pts), points=pts)
    if rec['label'] == 1:
        mp = VISA / row['image'].replace('Images/Anomaly', 'Masks/Anomaly').rsplit('.', 1)[0]
        mp = Path(str(mp) + '.png')
        if mp.exists():
            gt = np.asarray(Image.open(mp).convert('L').resize((256, 256), Image.NEAREST)) > 0
            fgm = np.asarray(Image.open(__import__('io').BytesIO(__import__('base64').b64decode(res['sam']['mask'].split(',')[1])))) > 0
            rec['coverage'] = float((gt & fgm).sum() / max(gt.sum(), 1))
    out.append(rec)
    if k % 20 == 0: print(k, rec['image'], round(rec['score'], 2), rec['masked'] and round(rec['masked'], 2), flush=True)
json.dump(out, open(sys.argv[2], 'w'))
