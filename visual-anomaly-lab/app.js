const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';

const params = new URLSearchParams(location.search);
let lang = params.get('lang') || ((navigator.language || '').startsWith('ja') ? 'ja' : 'en');
if (!translations[lang]) lang = 'en';

let category = 'candle', modelMetadata = {}, distributions = {}, study = null, archive = null;
let input = null, sampleIndex = null, sampleLabel = null, points = [], baseImage = null, result = null;
let busy = false, requestId = 0, statusKey = 'initial';

function t(key, args = {}) {
  let text = translations[lang][key] ?? translations.en[key] ?? translations[lang].error;
  for (const [k, v] of Object.entries(args)) text = text.replaceAll('{' + k + '}', v);
  return text;
}
const pct = x => (100 * x).toFixed(0) + '%';
const fix = (x, n = 2) => Number(x).toFixed(n);

function setStatus(key) { statusKey = key; $('status').textContent = t(key); }

/* ---------- language ---------- */
function applyLanguage() {
  document.documentElement.lang = lang;
  $('language').value = lang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    if (el.id === 'filename' && input) return;
    el.textContent = t(el.dataset.i18n);
  });
  $('pointcount').textContent = t('points_count', { n: points.length });
  document.querySelectorAll('.sample').forEach(b => {
    const label = t(b.dataset.label);
    b.title = t('sample_title', { n: +b.dataset.id + 1, label });
    b.querySelector('img').alt = b.title;
    b.querySelector('small').textContent = label;
  });
  setStatus(statusKey);
  renderModelText();
  renderResult();
  renderDistribution();
  renderStudy();
  renderArchive();
}
$('language').onchange = () => {
  lang = $('language').value;
  const url = new URL(location.href); url.searchParams.set('lang', lang);
  history.replaceState(null, '', url);
  applyLanguage();
};

/* ---------- model summary ---------- */
function renderModelText() {
  const m = modelMetadata[category];
  $('model-hint').textContent = t('hint_' + category);
  if (!m) return;
  const c = m.test.confusion;
  $('stat-auroc').textContent = fix(m.image_auroc_official_test, 3);
  $('stats-label').textContent = t('stats_label', { model: $('model').selectedOptions[0].textContent });
  $('stat-fa').textContent = `${c.fp}/${c.fp + c.tn}`;
  $('stat-miss').textContent = `${c.fn}/${c.fn + c.tp}`;
  $('model-evidence').textContent = t('model_evidence', { category, train: m.n_train, bank: m.memory_bank_size.toLocaleString() });
  $('threshold-value').textContent = fix(m.test.threshold);
}

/* ---------- canvases ---------- */
function image(src) {
  return new Promise((resolve, reject) => { const im = new Image(); im.onload = () => resolve(im); im.onerror = reject; im.src = src; });
}
async function draw(id, src) {
  const im = await image(src), ctx = $(id).getContext('2d');
  ctx.clearRect(0, 0, 256, 256); ctx.drawImage(im, 0, 0, 256, 256);
}
function clearCanvas(id) { $(id).getContext('2d').clearRect(0, 0, 256, 256); }

function drawPoints() {
  if (!baseImage) return;
  const ctx = $('original').getContext('2d');
  ctx.drawImage(baseImage, 0, 0, 256, 256);
  for (const [x, y] of points) {
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#3b63f0'; ctx.lineWidth = 2; ctx.stroke();
  }
  $('pointcount').textContent = t('points_count', { n: points.length });
}

function resetResult() {
  result = null;
  for (const id of ['overlay', 'region', 'masked']) clearCanvas(id);
  for (const id of ['score', 'decision', 'elapsed']) $(id).textContent = '—';
  $('truth').textContent = '';
  $('comparison').hidden = true;
  $('threshold').disabled = true;
  $('samlabel').textContent = t($('sam').checked ? 'waiting' : 'disabled');
  renderDistribution();
}

async function select(src, name, index = null, label = null) {
  if (busy) return;
  requestId++;
  input = src; sampleIndex = index; sampleLabel = label;
  baseImage = await image(src); points = [];
  drawPoints();
  $('placeholder').hidden = true;
  $('filename').textContent = name;
  resetResult();
  setStatus('ready');
}

async function upload(file) {
  if (!file || busy) return;
  if (file.size > 10 * 1024 * 1024) { setStatus('too_large'); return; }
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      await select(reader.result, file.name);
      document.querySelectorAll('.sample').forEach(b => b.classList.remove('selected'));
    } catch { setStatus('unreadable'); }
  };
  reader.readAsDataURL(file);
}
$('file').onchange = e => upload(e.target.files[0]);
const drop = $('drop');
drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } };
drop.ondragover = e => { e.preventDefault(); drop.classList.add('drag'); };
drop.ondragleave = () => drop.classList.remove('drag');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('drag'); upload(e.dataTransfer.files[0]); };

$('original').onclick = e => {
  if (!input || busy || !$('sam').checked || points.length >= 16) return;
  const r = e.target.getBoundingClientRect();
  points.push([Math.min(255, (e.clientX - r.left) / r.width * 256), Math.min(255, (e.clientY - r.top) / r.height * 256)]);
  drawPoints(); setStatus('points_updated');
};
$('clear').onclick = () => { if (busy) return; points = []; drawPoints(); };
$('sam').onchange = () => {
  $('original').classList.toggle('clickable', $('sam').checked);
  $('samlabel').textContent = t($('sam').checked ? 'waiting' : 'disabled');
  setStatus($('sam').checked ? 'prompt' : 'full_image');
};

function regions() {
  if (!result) return;
  const threshold = +$('threshold').value;
  $('pixelvalue').textContent = threshold.toFixed(1);
  const ctx = $('region').getContext('2d');
  ctx.drawImage(baseImage, 0, 0, 256, 256);
  const frame = ctx.getImageData(0, 0, 256, 256), d = frame.data;
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
    const k = (y * 256 + x) * 4;
    if (result.map[y][x] >= threshold) { d[k] = 240; d[k + 1] *= .3; d[k + 2] = d[k + 2] * .3 + 60; }
    else { d[k] *= .45; d[k + 1] *= .45; d[k + 2] *= .5; }
  }
  ctx.putImageData(frame, 0, 0);
}
$('threshold').oninput = regions;

function decisionPill(decision) {
  return `<span class="pill ${decision}">${t(decision)}</span>`;
}

function renderResult() {
  if (!result) return;
  $('score').textContent = fix(result.score);
  $('decision').innerHTML = decisionPill(result.decision);
  $('truth').textContent = sampleLabel ? t('truth', { label: t(sampleLabel) }) : '';
  const s = result.sam;
  if (!s) { $('comparison').hidden = true; $('samlabel').textContent = t('disabled'); return; }
  $('samlabel').textContent = t('foreground', { n: (100 * s.foreground_fraction).toFixed(1) });
  $('comparison').hidden = false;
  if (s.thresholds) {
    $('comparison').innerHTML = t('comparison_calibrated', {
      full: fix(s.baseline_score), masked: fix(s.score), tfull: fix(s.thresholds.full), tsam: fix(s.thresholds.sam),
      dfull: decisionPill(s.decision_full), dsam: decisionPill(s.decision),
    });
  } else {
    $('comparison').textContent = t('comparison', { full: fix(s.baseline_score), masked: fix(s.score) }) + t('comparison_uncalibrated');
  }
}

$('run').onclick = async () => {
  if (!input) { setStatus('need_image'); return; }
  if ($('sam').checked && !points.length) { setStatus('need_points'); return; }
  busy = true;
  for (const id of ['model', 'run', 'file', 'sam']) $(id).disabled = true;
  const id = ++requestId;
  setStatus('running');
  try {
    const body = { category, points: $('sam').checked ? points : [], sam_scope: $('sam_scope').value };
    if (sampleIndex !== null) body.sample = sampleIndex; else body.image = input;
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const out = await response.json();
    if (!response.ok) throw Object.assign(new Error(out.error), { code: out.error_code });
    if (id !== requestId) return;
    result = out;
    $('elapsed').textContent = out.elapsed + ' s';
    $('range-lo').textContent = fix(out.display_range[0], 0);
    $('range-hi').textContent = fix(out.display_range[1], 0);
    await draw('overlay', out.overlay);
    const slider = $('threshold');
    slider.min = Math.floor(out.display_range[0]); slider.max = Math.ceil(Math.max(out.map_max, out.display_range[1]));
    slider.value = Math.min(out.threshold, out.map_max - .1).toFixed(1);
    slider.disabled = false;
    regions();
    if (out.sam) await draw('masked', out.sam.overlay); else clearCanvas('masked');
    renderResult();
    renderDistribution();
    setStatus('done');
  } catch (e) {
    setStatus(e.code && translations[lang][e.code] ? e.code : 'error');
  } finally {
    busy = false;
    for (const id of ['model', 'run', 'file', 'sam']) $(id).disabled = false;
  }
};

/* ---------- strip charts ---------- */
function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
}

/* rows: [{label, points: [{score, label}], threshold}] drawn on one shared x axis. */
function stripChart(svg, rows, marker) {
  svg.replaceChildren();
  const width = Math.max(320, svg.clientWidth || 800);
  const left = rows.length > 1 ? 112 : 16, right = 16, rowH = 92, top = 10;
  const height = top + rows.length * rowH + 28;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const all = rows.flatMap(r => r.points.map(p => p.score)).concat(rows.map(r => r.threshold));
  if (marker != null) all.push(marker);
  const lo = Math.floor(Math.min(...all) - .5), hi = Math.ceil(Math.max(...all) + .5);
  const x = v => left + (v - lo) / (hi - lo) * (width - left - right);

  rows.forEach((row, r) => {
    const y0 = top + r * rowH, mid = y0 + rowH / 2;
    el('rect', { x: x(row.threshold), y: y0 + 4, width: x(hi) - x(row.threshold), height: rowH - 8, class: 'zone' }, svg);
    if (rows.length > 1) el('text', { x: 0, y: mid + 4, class: 'row-label' }, svg).textContent = row.label;
    // Normal dots above the midline, anomalous below; deterministic jitter keeps the layout stable.
    row.points.forEach((p, i) => {
      const jitter = ((i * 37) % 17) / 17 * 24;
      const cy = p.label ? mid + 6 + jitter : mid - 6 - jitter;
      el('circle', { cx: x(p.score), cy, r: 4.2, fill: p.label ? 'var(--anomaly)' : 'var(--normal)', 'fill-opacity': .72, stroke: '#fff', 'stroke-width': .8 }, svg)
        .append(Object.assign(document.createElementNS(SVG, 'title'), { textContent: `${p.image ?? ''} ${fix(p.score)}` }));
    });
    el('line', { x1: x(row.threshold), x2: x(row.threshold), y1: y0 + 2, y2: y0 + rowH - 2, class: 'thr' }, svg);
    el('text', { x: x(row.threshold) + 5, y: y0 + 14 }, svg).textContent = `${t('threshold_label')} ${fix(row.threshold)}`;
  });

  const axisY = top + rows.length * rowH + 4;
  el('line', { x1: left, x2: width - right, y1: axisY, y2: axisY, class: 'axis' }, svg);
  const step = (hi - lo) > 24 ? 5 : 2;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    el('line', { x1: x(v), x2: x(v), y1: axisY, y2: axisY + 4, class: 'axis' }, svg);
    el('text', { x: x(v), y: axisY + 17, 'text-anchor': 'middle' }, svg).textContent = v;
  }
  if (marker != null) {
    el('line', { x1: x(marker), x2: x(marker), y1: top, y2: axisY, class: 'you' }, svg);
    el('text', { x: Math.min(x(marker) + 5, width - 90), y: axisY - 6, class: 'you-label' }, svg).textContent = `${t('you')} ${fix(marker)}`;
  }
}

function renderDistribution() {
  const d = distributions[category], m = modelMetadata[category];
  if (!d || !m) return;
  const c = m.test.confusion;
  $('dist-summary').textContent = t('dist_summary', {
    auroc: fix(d.image_auroc, 3), acc: pct(m.test.accuracy), fp: c.fp, nn: c.fp + c.tn, fn: c.fn, na: c.fn + c.tp,
  });
  $('dist').setAttribute('aria-label', $('dist-heading').textContent);
  stripChart($('dist'), [{ label: '', points: d.points, threshold: d.threshold }], result && result.category === category ? result.score : null);
}

/* ---------- SAM study ---------- */
function renderStudy() {
  if (!study) return;
  const toPoints = key => study.test_points.map(p => ({ score: p[key], label: p.label, image: p.image }));
  stripChart($('sam-dist'), [
    { label: t('row_full'), points: toPoints('full'), threshold: study.full.threshold },
    { label: t('row_sam'), points: toPoints('sam'), threshold: study.sam.threshold },
  ]);
  const f = study.full, s = study.sam;
  const rows = [
    ['m_auroc', fix(f.image_auroc_val_test, 3), fix(s.image_auroc_val_test, 3), s.image_auroc_val_test > f.image_auroc_val_test],
    ['m_acc', pct(f.accuracy), pct(s.accuracy), s.accuracy > f.accuracy],
    ['m_fp', `${f.confusion.fp} / 50`, `${s.confusion.fp} / 50`, s.confusion.fp < f.confusion.fp],
    ['m_fn', `${f.confusion.fn} / 50`, `${s.confusion.fn} / 50`, s.confusion.fn < f.confusion.fn],
  ];
  $('sam-table').replaceChildren(...rows.map(([key, a, b, better]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row">${t(key)}</th><td>${a}</td><td class="${better ? 'better' : 'worse'}">${b}</td>`;
    return tr;
  }));
  $('sam-protocol').textContent = t('sam_protocol', { coverage: (100 * study.mean_defect_coverage).toFixed(1) });

  const tri = t('tri').split('|').map(s => `<span>${s}</span>`).join('');
  const meta = { 'false-alarm-removed': ['c1', 'ok'], 'defect-kept': ['c2', 'ok'], 'counterexample': ['c3', 'bad'] };
  $('cases').replaceChildren(...study.cases.map(c => {
    const [k, tone] = meta[c.key];
    const thrF = study.full.threshold, thrS = study.sam.threshold;
    const d1 = c.full >= thrF ? 'anomalous' : 'normal', d2 = c.sam >= thrS ? 'anomalous' : 'normal';
    const card = document.createElement('article');
    card.className = 'card case';
    card.innerHTML = `<img src="/cases/macaroni2-${c.key}.jpg" alt="${t(k + '_title')}" loading="lazy" width="784" height="256">
      <div class="tri">${tri}</div>
      <span class="tag ${tone}">${t(k + '_tag')} · ${c.image}</span>
      <h3>${t(k + '_title')}</h3>
      <p>${t(k + '_text')}</p>
      <div class="scores">${fix(c.full)} ${decisionPill(d1)} <span class="arrow">→</span> ${fix(c.sam)} ${decisionPill(d2)}</div>`;
    return card;
  }));
}

function renderArchive() {
  if (!archive) return;
  const cards = archive.categories.map(c => {
    const a = c.unmasked_map_max, b = c.masked_map_max;
    const rows = [
      ['m_auroc', fix(a.image_auroc_official_test, 3), fix(b.image_auroc_official_test, 3)],
      ['m_acc', pct(a.test.accuracy), pct(b.test.accuracy)],
      ['m_fp', `${a.test.confusion.fp} / ${a.test.confusion.fp + a.test.confusion.tn}`, `${b.test.confusion.fp} / ${b.test.confusion.fp + b.test.confusion.tn}`],
      ['m_fn', `${a.test.confusion.fn} / ${a.test.confusion.fn + a.test.confusion.tp}`, `${b.test.confusion.fn} / ${b.test.confusion.fn + b.test.confusion.tp}`],
    ];
    const card = document.createElement('div');
    card.className = 'card study';
    card.style.display = 'block';
    card.innerHTML = `<h3 style="margin:0 0 12px;font-size:15px">${t(c.category === 'pcb4' ? 'archive_pcb' : 'archive_pasta')}</h3>
      <table><thead><tr><th></th><th>${t('case_before')}</th><th>${t('case_after')}</th></tr></thead>
      <tbody>${rows.map(([k, x, y]) => `<tr><th scope="row">${t(k)}</th><td>${x}</td><td>${y}</td></tr>`).join('')}</tbody></table>`;
    return card;
  });
  $('archive').replaceChildren(...cards);
}

/* ---------- loading ---------- */
async function loadSamples() {
  busy = true;
  $('run').disabled = true; $('model').disabled = true;
  input = null; sampleIndex = null; sampleLabel = null; points = []; baseImage = null;
  $('samples').replaceChildren();
  clearCanvas('original');
  resetResult();
  $('filename').textContent = t('no_image');
  $('placeholder').hidden = false;
  renderModelText();
  try {
    if (!distributions[category]) distributions[category] = await (await fetch('/api/distribution?category=' + category)).json();
    renderDistribution();
    const samples = await (await fetch('/api/samples?category=' + category)).json();
    const loaded = await Promise.all(samples.map(s => fetch(`/api/sample/${s.id}?category=${category}`).then(r => r.json())));
    for (const [i, sample] of samples.entries()) {
      const d = loaded[i];
      const b = document.createElement('button');
      b.className = 'sample'; b.disabled = true; b.dataset.id = sample.id; b.dataset.label = sample.label;
      b.title = t('sample_title', { n: sample.id + 1, label: t(sample.label) });
      const im = document.createElement('img'); im.src = d.image; im.alt = b.title;
      const caption = document.createElement('small'); caption.textContent = t(sample.label);
      b.append(im, caption);
      b.onclick = async () => {
        if (busy) return;
        await select(d.image, sample.image.split('/').pop(), sample.id, sample.label);
        document.querySelectorAll('.sample').forEach(x => x.classList.remove('selected'));
        b.classList.add('selected');
      };
      $('samples').append(b);
    }
    setStatus('initial');
  } catch {
    setStatus('sample_error');
  } finally {
    busy = false;
    $('run').disabled = false; $('model').disabled = false;
    document.querySelectorAll('.sample').forEach(b => b.disabled = false);
  }
}
$('model').onchange = async () => {
  category = $('model').value;
  const url = new URL(location.href); url.searchParams.set('model', category); history.replaceState(null, '', url);
  await loadSamples();
};

let resizeTimer;
addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderDistribution(); renderStudy(); }, 150); });

(async () => {
  applyLanguage();
  try {
    const [models, studyData, archiveData] = await Promise.all([
      fetch('/api/models').then(r => r.json()),
      fetch('/sam-study.json').then(r => r.json()).catch(() => null),
      fetch('/recorded-experiments.json').then(r => r.json()).catch(() => null),
    ]);
    modelMetadata = Object.fromEntries(models.map(m => [m.category, m.metrics]));
    study = studyData; archive = archiveData;
    for (const option of $('model').options) option.disabled = !modelMetadata[option.value];
    const requested = params.get('model');
    if (requested && modelMetadata[requested]) { category = requested; $('model').value = requested; }
    renderStudy(); renderArchive();
    await loadSamples();
  } catch {
    setStatus('model_unavailable');
  }
})();
