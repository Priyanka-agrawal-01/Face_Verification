import {
  loadModels, analyzeFaces, distance, meanDescriptor, loadImage, loadImageFromFile,
  DEFAULT_THRESHOLD, THRESHOLDS, parseLfwPairs, generatePairs,
  scoreAt, bestThreshold, equalErrorRate, rocCurve, crossValidate,
} from './core.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const fmt = (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : '—');
const pct = (n, d = 1) => (Number.isFinite(n) ? `${(n * 100).toFixed(d)}%` : '—');
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

// Animated number counter for smooth value transitions.
function animateNumber(el, target, duration = 500) {
  const start = parseFloat(el.textContent) || 0;
  if (!Number.isFinite(target)) { el.textContent = '—'; return; }
  const startTime = performance.now();
  const step = (now) => {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - (1 - progress) ** 3;
    el.textContent = fmt(start + (target - start) * eased);
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  el.classList.add('counting');
  setTimeout(() => el.classList.remove('counting'), duration);
}

const state = {
  threshold: DEFAULT_THRESHOLD,
  verify: { a: null, b: null, shown: null },
  people: [],
  query: null,
  sampleSet: null,
  eval: { sources: [], source: null, cache: new Map(), running: false, stop: false, last: null },
};

const GAUGE_MAX = 1.2;
const datasetUrl = (name, rel) => `/data/${encodeURIComponent(name)}/` + rel.split('/').map(encodeURIComponent).join('/');
const personName = (rel) => rel.split('/')[0].replace(/_/g, ' ');

// ------------------------------------------------------------------ shell

// Tabs follow the WAI-ARIA pattern: arrow keys move between them, and the
// URL hash (#verify, #gallery…) makes each one linkable and reload-safe.
const tabs = $$('nav.tabs [role="tab"]');
function selectTab(name, { focus = false, remember = false } = {}) {
  const tab = tabs.find((t) => t.dataset.tab === name) ?? tabs[0];
  for (const t of tabs) {
    const selected = t === tab;
    t.setAttribute('aria-selected', String(selected));
    t.tabIndex = selected ? 0 : -1;
    $(`#panel-${t.dataset.tab}`).hidden = !selected;
  }
  if (focus) tab.focus();
  if (remember) history.replaceState(null, '', `#${tab.dataset.tab}`);
}
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => selectTab(tab.dataset.tab, { remember: true }));
  tab.addEventListener('keydown', (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    const jump = { Home: 0, End: tabs.length - 1 }[e.key];
    if (step === undefined && jump === undefined) return;
    e.preventDefault();
    selectTab(tabs[jump ?? (i + step + tabs.length) % tabs.length].dataset.tab, { focus: true, remember: true });
  });
});
window.addEventListener('hashchange', () => {
  selectTab(location.hash.slice(1));
  window.scrollTo({ top: 0 });
});
selectTab(location.hash.slice(1));

// Light/dark toggle. Until the user picks one, the OS preference applies.
const themeToggle = $('#theme-toggle');
const prefersDark = matchMedia('(prefers-color-scheme: dark)');
const isDark = () => (document.documentElement.dataset.theme ?? (prefersDark.matches ? 'dark' : 'light')) === 'dark';
const syncThemeToggle = () => themeToggle.setAttribute('aria-pressed', String(isDark()));
themeToggle.addEventListener('click', () => {
  const theme = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('theme', theme); } catch {}
  syncThemeToggle();
});
prefersDark.addEventListener('change', syncThemeToggle);
syncThemeToggle();

// About 12 MB of weights on the first visit, so loading gets a visible state
// and a failure gets a retry instead of a silently broken page.
const pill = $('#model-status');
const BACKENDS = { webgl: 'GPU', wasm: 'WebAssembly', cpu: 'CPU, slower' };
function startModels() {
  pill.dataset.state = 'loading';
  pill.lastElementChild.textContent = 'Loading models…';
  $('#model-alert').hidden = true;
  loadModels()
    .then((backend) => {
      pill.dataset.state = 'ready';
      pill.lastElementChild.textContent = `Ready · ${BACKENDS[backend] ?? backend}`;
    })
    .catch((err) => {
      console.error(err);
      pill.dataset.state = 'error';
      pill.lastElementChild.textContent = 'Models failed to load';
      $('#model-alert').hidden = false;
    });
}
$('#model-retry').addEventListener('click', startModels);
startModels();

function setThreshold(t) {
  state.threshold = t;
  $$('[data-threshold]').forEach((input) => (input.value = t));
  $$('[data-threshold-label]').forEach((label) => (label.textContent = t.toFixed(2)));
  renderVerify();
  renderIdentify();
}
$$('[data-threshold]').forEach((input) =>
  input.addEventListener('input', () => setThreshold(Number(input.value))),
);

// Click, keyboard, drag-and-drop and paste file picking for a drop zone.
const dropHandlers = new Map();
function bindDrop(zone, onFiles, multiple = false) {
  dropHandlers.set(zone, onFiles);
  const input = Object.assign(document.createElement('input'), {
    type: 'file', accept: 'image/jpeg,image/png,image/webp', multiple, hidden: true,
  });
  zone.after(input);
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => {
    if (input.files.length) onFiles([...input.files]);
    input.value = '';
  });
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragging');
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'));
    if (files.length) onFiles(files);
  });
}

// A pasted image goes to the focused or hovered drop zone, else the first
// empty one on the current tab.
document.addEventListener('paste', (e) => {
  const files = [...e.clipboardData.files].filter((f) => f.type.startsWith('image/'));
  const zones = [...dropHandlers.keys()].filter((z) => z.offsetParent);
  const zone = zones.find((z) => z.matches(':focus, :hover'))
    ?? zones.find((z) => !z.classList.contains('has-result')) ?? zones[0];
  if (!files.length || !zone) return;
  e.preventDefault();
  dropHandlers.get(zone)(files);
});

// Big phone photos are scaled down before detection: faster, same result.
function toWorkingCanvas(img, max = 1024) {
  const scale = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas;
}

// Preview with every detected face boxed; the chosen face is solid with landmarks.
function drawFaces(zone, source, faces, primary) {
  const scale = Math.min(1, 520 / source.width, 380 / source.height);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(source.width * scale);
  canvas.height = Math.round(source.height * scale);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  const accent = css('--accent');
  for (const face of faces) {
    const b = face.detection.box;
    ctx.lineWidth = 2;
    ctx.setLineDash(face === primary ? [] : [6, 4]);
    ctx.strokeStyle = face === primary ? accent : 'rgba(255,255,255,0.85)';
    ctx.strokeRect(b.x * scale, b.y * scale, b.width * scale, b.height * scale);
  }
  if (primary) {
    ctx.fillStyle = accent;
    for (const p of primary.landmarks.positions) {
      ctx.beginPath();
      ctx.arc(p.x * scale, p.y * scale, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  const hint = Object.assign(document.createElement('span'), {
    className: 'replace-hint', textContent: 'Click, drop or paste to replace',
  });
  zone.replaceChildren(canvas, hint);
}

function describeDetection({ faces, primary, padded }) {
  if (!faces.length) return 'No face found. Try a clearer, front-facing photo.';
  const note = padded ? ' Found after padding the image edges.' : '';
  if (faces.length === 1) return `1 face found (detector confidence ${pct(primary.detection.score, 0)}).${note}`;
  return `${faces.length} faces found. Using the largest one (solid box).${note}`;
}

function analyzeFile(file) {
  return loadImageFromFile(file).then((img) => analyzeFaces(toWorkingCanvas(img)));
}

// Run the pipeline on one image and show the boxes in its drop zone.
// Resolves to the chosen face's descriptor, or null if there is none.
async function analyzeInto(zone, meta, loadImg, label = '') {
  meta.textContent = 'Finding faces…';
  zone.classList.add('scanning');
  zone.classList.remove('has-result');
  try {
    const result = await analyzeFaces(toWorkingCanvas(await loadImg()));
    zone.classList.add('has-result');
    drawFaces(zone, result.input, result.faces, result.primary);
    meta.textContent = (label ? `${label}. ` : '') + describeDetection(result);
    return result.primary?.descriptor ?? null;
  } catch (err) {
    meta.textContent = err.message;
    return null;
  } finally {
    zone.classList.remove('scanning');
  }
}

// ------------------------------------------------------------------ verify

const verifySlots = {};
for (const slot of ['a', 'b']) {
  const zone = $(`.drop[data-slot="${slot}"]`);
  const meta = $(`[data-meta="${slot}"]`);
  verifySlots[slot] = async (loadImg, label) => {
    state.verify[slot] = null;
    renderVerify();
    state.verify[slot] = await analyzeInto(zone, meta, loadImg, label);
    renderVerify();
  };
  bindDrop(zone, ([file]) => verifySlots[slot](() => loadImageFromFile(file)));
}

// Random pairs from the dataset's pairs.txt, so the tab can be tried without photos.
let samplePairs = null;
async function loadSample(same) {
  const set = state.sampleSet;
  samplePairs ??= fetch(datasetUrl(set, 'pairs.txt'))
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`pairs.txt: HTTP ${r.status}`))))
    .then((text) => parseLfwPairs(text).pairs)
    .catch((err) => { samplePairs = null; throw err; });
  const pool = (await samplePairs).filter((p) => p.same === same);
  const pair = pool[Math.floor(Math.random() * pool.length)];
  await Promise.all([
    verifySlots.a(() => loadImage(datasetUrl(set, pair.a)), personName(pair.a)),
    verifySlots.b(() => loadImage(datasetUrl(set, pair.b)), personName(pair.b)),
  ]);
}
$$('[data-sample]').forEach((btn) => btn.addEventListener('click', async () => {
  const buttons = $$('[data-sample]');
  buttons.forEach((b) => (b.disabled = true));
  try {
    await loadSample(btn.dataset.sample === 'same');
  } catch (err) {
    console.error(err);
    $('[data-meta="a"]').textContent = `Could not load a sample pair: ${err.message}`;
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}));

function renderVerify() {
  const { a, b } = state.verify;
  const card = $('#verdict-card');
  const badge = $('#verify-badge');
  const margin = $('#verify-margin');
  const gauge = $('#verify-gauge');
  const at = (d) => `${Math.min(d / GAUGE_MAX, 1) * 100}%`;
  gauge.style.setProperty('--threshold', at(state.threshold));
  $('.mark span', gauge).textContent = `threshold ${state.threshold.toFixed(2)}`;
  if (!a || !b) {
    state.verify.shown = null;
    delete card.dataset.verdict;
    $('#verify-distance').textContent = '—';
    badge.className = 'badge';
    badge.textContent = 'Waiting for two faces';
    margin.textContent = 'Add a photo on each side to compare them.';
    return;
  }
  const d = distance(a, b);
  const same = d <= state.threshold;
  // Only count up when the pair changes, not on every threshold slider tick.
  if (d !== state.verify.shown) animateNumber($('#verify-distance'), d);
  state.verify.shown = d;
  gauge.style.setProperty('--value', at(d));
  card.dataset.verdict = same ? 'same' : 'different';
  badge.className = `badge ${same ? 'good' : 'bad'}`;
  badge.textContent = same ? '✓ Same person' : '✕ Different people';
  const gap = Math.abs(d - state.threshold);
  margin.textContent = `${fmt(gap)} ${same ? 'below' : 'above'} the threshold` +
    (gap < 0.05 ? '. A close call: small changes in lighting or pose could flip it.' : '.');
}

// ------------------------------------------------------------------ gallery

function faceThumb(canvas, face, size = 80) {
  const b = face.detection.box;
  const pad = Math.max(b.width, b.height) * 0.25;
  const side = Math.max(b.width, b.height) + pad * 2;
  const thumb = document.createElement('canvas');
  thumb.width = thumb.height = size;
  thumb.getContext('2d').drawImage(
    canvas, b.x + b.width / 2 - side / 2, b.y + b.height / 2 - side / 2, side, side, 0, 0, size, size,
  );
  return thumb.toDataURL('image/jpeg', 0.85);
}

$('#enrol-pick').addEventListener('click', () => {
  if (!$('#enrol-name').value.trim()) {
    $('#enrol-status').textContent = 'Enter a name first.';
    $('#enrol-name').focus();
    return;
  }
  $('#enrol-files').click();
});
$('#enrol-name').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); $('#enrol-pick').click(); }
});

$('#enrol-files').addEventListener('change', async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  const status = $('#enrol-status');
  const name = $('#enrol-name').value.trim();
  if (!name) { status.textContent = 'Enter a name first.'; return; }
  let person = state.people.find((p) => p.name.toLowerCase() === name.toLowerCase());
  const added = [];
  let skipped = 0;
  for (const [i, file] of files.entries()) {
    status.textContent = `Embedding photo ${i + 1} of ${files.length}…`;
    try {
      const { input, primary } = await analyzeFile(file);
      if (!primary) { skipped++; continue; }
      added.push({ descriptor: primary.descriptor, thumb: faceThumb(input, primary) });
    } catch {
      skipped++;
    }
  }
  if (added.length) {
    if (!person) state.people.push((person = { name, entries: [] }));
    person.entries.push(...added);
    person.template = meanDescriptor(person.entries.map((x) => x.descriptor));
  }
  status.textContent = `Added ${added.length} photo${added.length === 1 ? '' : 's'} for ${name}` +
    (skipped ? `. Skipped ${skipped} with no detectable face.` : '.');
  renderPeople();
  renderIdentify(true);
});

function renderPeople() {
  const list = $('#people');
  list.replaceChildren(...state.people.map((person, i) => {
    const row = document.createElement('div');
    row.className = 'person';
    row.style.animationDelay = `${i * 0.08}s`;
    row.innerHTML = `<img alt="" src="${person.entries[0].thumb}"><div class="who"><strong>${escapeHtml(person.name)}</strong><span class="muted small">${person.entries.length} photo${person.entries.length === 1 ? '' : 's'}</span></div>`;
    const remove = Object.assign(document.createElement('button'), {
      className: 'icon-btn', textContent: '✕', title: `Remove ${person.name}`,
    });
    remove.setAttribute('aria-label', `Remove ${person.name}`);
    remove.addEventListener('click', () => { state.people.splice(i, 1); renderPeople(); renderIdentify(true); });
    row.append(remove);
    return row;
  }));
}

{
  const zone = $('.drop[data-slot="query"]');
  const meta = $('[data-meta="query"]');
  bindDrop(zone, async ([file]) => {
    state.query = await analyzeInto(zone, meta, () => loadImageFromFile(file));
    renderIdentify(true);
  });
}

// `animate` replays the row entrance; threshold changes re-render without it.
function renderIdentify(animate = false) {
  $('#identify-table').classList.toggle('animate-rows', animate);
  const body = $('#identify-table tbody');
  if (!state.query || !state.people.length) {
    body.innerHTML = `<tr><td colspan="5" class="muted">${state.people.length ? 'Choose a query photo.' : 'Enrol people and choose a query photo.'}</td></tr>`;
    return;
  }
  const ranked = state.people
    .map((p) => ({
      name: p.name,
      closest: Math.min(...p.entries.map((x) => distance(state.query, x.descriptor))),
      template: distance(state.query, p.template),
    }))
    .sort((x, y) => x.closest - y.closest);
  body.innerHTML = ranked.map((r, i) => {
    const match = r.closest <= state.threshold;
    const result = match
      ? `<span class="badge ${i === 0 ? 'good' : ''}">${i === 0 ? '✓ Best match' : 'Also within threshold'}</span>`
      : `<span class="muted">${i === 0 ? 'Unknown: nobody close enough' : '—'}</span>`;
    return `<tr style="animation-delay:${i * 0.06}s"${i === 0 ? ' class="highlight"' : ''}><td>${i + 1}</td><td>${escapeHtml(r.name)}</td><td class="num">${fmt(r.closest)}</td><td class="num">${fmt(r.template)}</td><td>${result}</td></tr>`;
  }).join('');
}

// ------------------------------------------------------------------ evaluate

const sourceSelect = $('#eval-source');
const protocolSelect = $('#eval-protocol');

async function loadSources() {
  const datasets = await fetch('/api/datasets').then((r) => r.json()).catch(() => []);
  state.eval.sources = datasets.map((d) => ({
    id: `server:${d.name}`,
    label: `${d.name} (${d.people.toLocaleString()} people, ${d.images.toLocaleString()} photos)`,
    kind: 'server', name: d.name, hasPairs: d.hasPairs,
  }));
  state.sampleSet = datasets.find((d) => d.hasPairs)?.name ?? null;
  $('#samples').hidden = !state.sampleSet;
  renderSourceOptions();
}

function renderSourceOptions() {
  const options = state.eval.sources.map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.label)}</option>`);
  // Without a placeholder, a server with no datasets would leave "Choose a folder…"
  // as the only option, and picking it would never fire a change event.
  if (!options.length) options.push('<option value="" disabled>No datasets on this server</option>');
  options.push('<option value="pick">Choose a folder from your computer…</option>');
  sourceSelect.innerHTML = options.join('');
  sourceSelect.value = state.eval.source?.id ?? state.eval.sources[0]?.id ?? '';
  applySource();
}

function onSourceChange() {
  if (sourceSelect.value === 'pick') { $('#eval-folder').click(); return; }
  applySource();
}

function applySource() {
  state.eval.source = state.eval.sources.find((s) => s.id === sourceSelect.value) ?? null;
  const official = protocolSelect.querySelector('option[value="official"]');
  official.disabled = !state.eval.source?.hasPairs;
  if (official.disabled) protocolSelect.value = 'generated';
  onProtocolChange();
}

function onProtocolChange() {
  const official = protocolSelect.value === 'official';
  $('#eval-folds-field').hidden = !official;
  $('#eval-max-field').hidden = official;
}

sourceSelect.addEventListener('change', onSourceChange);
protocolSelect.addEventListener('change', onProtocolChange);
$('#eval-folder').addEventListener('cancel', renderSourceOptions);

// A local folder: <root>/<person>/<photo>, with an optional <root>/pairs.txt.
$('#eval-folder').addEventListener('change', (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  if (!files.length) { renderSourceOptions(); return; }
  const people = {};
  const fileMap = new Map();
  let pairsFile = null;
  let root = '';
  for (const file of files) {
    const parts = file.webkitRelativePath.split('/');
    root = parts[0];
    if (parts.length === 2 && parts[1].toLowerCase() === 'pairs.txt') pairsFile = file;
    if (parts.length === 3 && /\.(jpe?g|png|webp)$/i.test(parts[2])) {
      (people[parts[1]] ??= []).push(parts[2]);
      fileMap.set(`${parts[1]}/${parts[2]}`, file);
    }
  }
  Object.values(people).forEach((list) => list.sort());
  const id = `local:${root}:${Date.now()}`;
  const source = {
    id, kind: 'local', name: root, people, fileMap, pairsFile, hasPairs: !!pairsFile,
    label: `Folder: ${root} (${Object.keys(people).length} people, ${fileMap.size} photos)`,
  };
  state.eval.sources = [...state.eval.sources.filter((s) => s.kind !== 'local'), source];
  state.eval.source = source;
  renderSourceOptions();
});

async function resolveSource(source) {
  if (source.kind === 'local') {
    return {
      people: source.people,
      pairsText: source.pairsFile ? await source.pairsFile.text() : null,
      image: (rel) => loadImageFromFile(source.fileMap.get(rel)),
      thumbUrl: (rel) => URL.createObjectURL(source.fileMap.get(rel)),
    };
  }
  const url = (rel) => datasetUrl(source.name, rel);
  const dataset = await fetch(`/api/dataset?name=${encodeURIComponent(source.name)}`).then((r) => r.json());
  return {
    people: dataset.files,
    pairsText: source.hasPairs ? await fetch(url('pairs.txt')).then((r) => r.text()) : null,
    image: (rel) => loadImage(url(rel)),
    thumbUrl: url,
  };
}

$('#eval-stop').addEventListener('click', () => { state.eval.stop = true; });
$('#eval-run').addEventListener('click', () => runEvaluation().catch((err) => {
  console.error(err);
  $('#eval-status').textContent = `Evaluation failed: ${err.message}`;
  setRunning(false);
}));

const formatDuration = (s) => (s < 60 ? `${Math.max(1, Math.round(s))} s` : `${Math.round(s / 60)} min`);

function setRunning(running) {
  state.eval.running = running;
  $('#eval-run').disabled = running;
  $('#eval-stop').disabled = !running;
  sourceSelect.disabled = protocolSelect.disabled = running;
}

async function runEvaluation() {
  const source = state.eval.source;
  const status = $('#eval-status');
  const bar = $('#eval-bar');
  if (!source) { status.textContent = 'Pick a dataset first.'; return; }
  setRunning(true);
  state.eval.stop = false;
  status.textContent = 'Preparing pairs…';
  await loadModels();
  const data = await resolveSource(source);

  // 1. Build the list of pairs to score.
  const available = new Set(Object.entries(data.people).flatMap(([name, files]) => files.map((f) => `${name}/${f}`)));
  let protocol;
  let folds = 1;
  if (protocolSelect.value === 'official' && data.pairsText) {
    const parsed = parseLfwPairs(data.pairsText);
    folds = Math.max(1, Math.min(parsed.folds, Number($('#eval-folds').value) || 1));
    protocol = { name: `official pairs.txt, ${folds} fold${folds > 1 ? 's' : ''}`, pairs: parsed.pairs.filter((p) => p.fold < folds) };
  } else {
    const generated = generatePairs(data.people, Number($('#eval-max').value) || 500);
    protocol = { name: 'generated pairs', pairs: generated.pairs };
  }
  const pairs = protocol.pairs.filter((p) => available.has(p.a) && available.has(p.b));
  const missing = protocol.pairs.length - pairs.length;
  if (!pairs.length) throw new Error('No usable pairs. A dataset needs at least one person with two photos.');

  // 2. Embed every image once (cached across runs).
  const images = [...new Set(pairs.flatMap((p) => [p.a, p.b]))];
  const key = (rel) => `${source.id}|${rel}`;
  const started = performance.now();
  let done = 0;
  for (const rel of images) {
    if (state.eval.stop) break;
    if (!state.eval.cache.has(key(rel))) {
      try {
        const img = await data.image(rel);
        const { primary } = await analyzeFaces(toWorkingCanvas(img));
        state.eval.cache.set(key(rel), primary?.descriptor ?? null);
      } catch {
        state.eval.cache.set(key(rel), null);
      }
    }
    done++;
    if (done % 5 === 0 || done === images.length) {
      const rate = done / ((performance.now() - started) / 1000);
      const left = (images.length - done) / rate;
      const percent = (done / images.length) * 100;
      bar.style.width = `${percent}%`;
      $('#eval-progress').setAttribute('aria-valuenow', percent.toFixed(0));
      status.textContent = `Embedding faces: ${done.toLocaleString()} / ${images.length.toLocaleString()} photos (${rate.toFixed(1)} per second` +
        (done < images.length ? `, about ${formatDuration(left)} left)` : ')');
    }
  }
  const stopped = state.eval.stop;

  // 3. Score pairs where both photos produced an embedding.
  const scored = [];
  const noFace = new Set();
  for (const p of pairs) {
    const da = state.eval.cache.get(key(p.a));
    const db = state.eval.cache.get(key(p.b));
    if (da === null) noFace.add(p.a);
    if (db === null) noFace.add(p.b);
    if (da && db) scored.push({ ...p, distance: distance(da, db) });
  }
  setRunning(false);
  if (!scored.some((p) => p.same) || !scored.some((p) => !p.same)) {
    status.textContent = 'Not enough scored pairs of both kinds to evaluate. Try more photos or folds.';
    return;
  }

  const result = {
    source: source.label, protocol: protocol.name, folds, scored, data,
    skipped: pairs.length - scored.length, missing, noFace: [...noFace],
    atDefault: scoreAt(scored, DEFAULT_THRESHOLD),
    best: bestThreshold(scored),
    eer: equalErrorRate(scored),
    roc: rocCurve(scored),
    cv: folds > 1 ? crossValidate(scored, folds) : null,
  };
  state.eval.last = result;
  const secs = ((performance.now() - started) / 1000).toFixed(0);
  status.textContent = `${stopped ? 'Stopped early. ' : ''}Scored ${scored.length.toLocaleString()} pairs from ${source.label} using ${protocol.name} in ${secs}s.`;
  renderResults(result);
}

function renderResults(r) {
  $('#eval-results').hidden = false;
  $('#eval-empty').hidden = true;
  const genuine = r.scored.filter((p) => p.same).length;
  const tiles = [
    ['Pairs scored', r.scored.length.toLocaleString(), `${genuine} same · ${r.scored.length - genuine} different · ${r.skipped} skipped (no face)`],
    [`Accuracy at ${DEFAULT_THRESHOLD.toFixed(2)}`, pct(r.atDefault.accuracy), `false accepts ${pct(r.atDefault.far)} · false rejects ${pct(r.atDefault.frr)}`],
    ['Best threshold', r.best.threshold.toFixed(2), `${pct(r.best.accuracy)} on these same pairs (optimistic)`],
    r.cv
      ? ['Cross-validated accuracy', pct(r.cv.mean, 2), `± ${pct(r.cv.std, 2)} over ${r.cv.results.length} folds, threshold chosen on the other folds`]
      : ['Cross-validated accuracy', '—', 'Use 2+ official folds to test on held-out data'],
    ['Equal error rate', pct(r.eer.eer, 2), `where false accepts ≈ false rejects (threshold ${r.eer.threshold.toFixed(2)})`],
    ['ROC AUC', fmt(r.roc.auc, 4), '1.0 is perfect, 0.5 is guessing'],
  ];
  $('#eval-tiles').innerHTML = tiles
    .map(([label, value, sub], i) => `<div class="tile tile-visible" style="animation-delay:${i * 0.08}s"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`)
    .join('');

  renderHistogram($('#hist-chart'), r.scored, r.best.threshold);
  renderRoc($('#roc-chart'), r.roc);
  renderThresholdTable(r);
  renderMistakes(r);
}

// ------------------------------------------------------------------ charts

const SVG = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent?.appendChild(node);
  return node;
}
const tooltip = $('#tooltip');
function showTip(e, html) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const x = Math.min(e.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
  tooltip.style.left = `${x + window.scrollX}px`;
  tooltip.style.top = `${e.clientY + window.scrollY + 14}px`;
}
const hideTip = () => (tooltip.hidden = true);

function niceMax(v) {
  const step = v <= 0.1 ? 0.02 : v <= 0.25 ? 0.05 : 0.1;
  return Math.ceil(v / step) * step;
}

// Bar with rounded top corners only, anchored to the baseline.
function topRoundedBar(x, y, w, h, r = 2) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function renderHistogram(host, scored, threshold) {
  const W = 560, H = 290, m = { l: 46, r: 12, t: 16, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const binW = 0.04, maxD = 1.2, nBins = Math.round(maxD / binW);
  const counts = { true: new Array(nBins).fill(0), false: new Array(nBins).fill(0) };
  const totals = { true: 0, false: 0 };
  for (const p of scored) {
    const bin = Math.min(nBins - 1, Math.floor(p.distance / binW));
    counts[p.same][bin]++;
    totals[p.same]++;
  }
  const share = (kind, i) => (totals[kind] ? counts[kind][i] / totals[kind] : 0);
  const yMax = niceMax(Math.max(...counts.true.map((_, i) => Math.max(share(true, i), share(false, i)))));
  const x = (d) => m.l + (d / maxD) * iw;
  const y = (s) => m.t + ih - (s / yMax) * ih;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Histogram of embedding distances for same-person and different-person pairs' });
  const axis = svgEl('g', { class: 'axis' }, svg);
  for (let i = 0; i <= 4; i++) {
    const v = (yMax / 4) * i;
    svgEl('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, axis);
    svgEl('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, axis).textContent = `${Math.round(v * 100)}%`;
  }
  for (let d = 0; d <= maxD + 1e-9; d += 0.2) {
    svgEl('text', { x: x(d), y: H - m.b + 16, 'text-anchor': 'middle' }, axis).textContent = d.toFixed(1);
  }
  svgEl('text', { x: m.l + iw / 2, y: H - 4, 'text-anchor': 'middle' }, axis).textContent = 'Embedding distance';

  const gap = 2;
  const slot = (iw / nBins - gap) / 2;
  for (let i = 0; i < nBins; i++) {
    [[true, '--series-1'], [false, '--series-2']].forEach(([kind, color], k) => {
      const s = share(kind, i);
      if (!s) return;
      const bx = x(i * binW) + gap / 2 + k * slot;
      const bar = svgEl('path', { d: topRoundedBar(bx, y(s), Math.max(1, slot - 0.5), y(0) - y(s)), fill: `var(${color})`, class: 'bar-animated' }, svg);
      bar.style.animationDelay = `${i * 0.025}s`;
    });
    const hit = svgEl('rect', { x: x(i * binW), y: m.t, width: iw / nBins, height: ih, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', (e) => showTip(e,
      `<strong>${(i * binW).toFixed(2)}–${((i + 1) * binW).toFixed(2)}</strong><br>` +
      `Same person: ${pct(share(true, i))} (${counts.true[i]})<br>Different people: ${pct(share(false, i))} (${counts.false[i]})`));
    hit.addEventListener('mouseleave', hideTip);
  }

  // Direct labels at each distribution's peak.
  for (const [kind, text] of [[true, 'Same person'], [false, 'Different people']]) {
    const peak = counts[kind].reduce((best, _, i) => (share(kind, i) > share(kind, best) ? i : best), 0);
    svgEl('text', { class: 'label-ink', x: x((peak + 0.5) * binW), y: y(share(kind, peak)) - 6, 'text-anchor': 'middle' }, svg).textContent = text;
  }
  svgEl('line', { class: 'threshold-line', x1: x(threshold), x2: x(threshold), y1: m.t, y2: y(0) }, svg);
  svgEl('text', { class: 'label-ink', x: x(threshold) + 4, y: m.t + 10 }, svg).textContent = `threshold ${threshold.toFixed(2)}`;
  host.replaceChildren(svg);
}

function renderRoc(host, roc) {
  const W = 560, H = 290, m = { l: 46, r: 16, t: 16, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (v) => m.l + v * iw;
  const y = (v) => m.t + ih - v * ih;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `ROC curve, area under curve ${fmt(roc.auc, 4)}` });
  const axis = svgEl('g', { class: 'axis' }, svg);
  for (let i = 0; i <= 4; i++) {
    const v = i / 4;
    svgEl('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, axis);
    svgEl('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, axis).textContent = `${v * 100}%`;
    svgEl('text', { x: x(v), y: H - m.b + 16, 'text-anchor': 'middle' }, axis).textContent = `${v * 100}%`;
  }
  svgEl('text', { x: m.l + iw / 2, y: H - 4, 'text-anchor': 'middle' }, axis).textContent = 'False accept rate';
  svgEl('text', { x: 12, y: m.t + ih / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${m.t + ih / 2})` }, axis).textContent = 'True accept rate';

  svgEl('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, svg);
  const d = roc.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.fpr).toFixed(1)},${y(p.tpr).toFixed(1)}`).join('');
  const rocPath = svgEl('path', { d, fill: 'none', stroke: 'var(--series-1)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  requestAnimationFrame(() => {
    const len = rocPath.getTotalLength();
    rocPath.style.setProperty('--line-length', len);
    rocPath.classList.add('roc-line-animated');
  });
  svgEl('text', { class: 'label-ink', x: x(0.55), y: y(0.12) }, svg).textContent = `AUC ${fmt(roc.auc, 4)}`;

  // Crosshair + nearest point tooltip.
  const cross = svgEl('line', { y1: m.t, y2: y(0), stroke: 'var(--muted)', 'stroke-width': 1, visibility: 'hidden' }, svg);
  const dot = svgEl('circle', { r: 4.5, fill: 'var(--series-1)', stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' }, svg);
  const hit = svgEl('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }, svg);
  hit.addEventListener('mousemove', (e) => {
    const box = svg.getBoundingClientRect();
    const fpr = Math.max(0, Math.min(1, ((e.clientX - box.left) * (W / box.width) - m.l) / iw));
    let near = roc.points[0];
    for (const p of roc.points) if (Math.abs(p.fpr - fpr) < Math.abs(near.fpr - fpr) || (p.fpr === near.fpr && p.tpr > near.tpr)) near = p;
    for (const [n, k, v] of [[cross, 'x1', x(near.fpr)], [cross, 'x2', x(near.fpr)], [dot, 'cx', x(near.fpr)], [dot, 'cy', y(near.tpr)]]) n.setAttribute(k, v);
    cross.setAttribute('visibility', 'visible');
    dot.setAttribute('visibility', 'visible');
    showTip(e, `Threshold ${fmt(near.threshold, 3)}<br>True accept ${pct(near.tpr)} · False accept ${pct(near.fpr)}`);
  });
  hit.addEventListener('mouseleave', () => {
    hideTip();
    cross.setAttribute('visibility', 'hidden');
    dot.setAttribute('visibility', 'hidden');
  });
  host.replaceChildren(svg);
}

function renderThresholdTable(r) {
  const rows = new Set(THRESHOLDS.filter((t) => t >= 0.3 && t <= 0.9 && Math.round(t * 100) % 5 === 0));
  rows.add(r.best.threshold);
  const html = [...rows].sort((a, b) => a - b).map((t) => {
    const s = scoreAt(r.scored, t);
    const tag = t === r.best.threshold ? ' (best)' : t === DEFAULT_THRESHOLD ? ' (default)' : '';
    return `<tr${t === r.best.threshold ? ' class="highlight"' : ''}><td class="num">${t.toFixed(2)}${tag}</td><td class="num">${pct(s.accuracy, 2)}</td><td class="num">${pct(s.far, 2)}</td><td class="num">${pct(s.frr, 2)}</td><td class="num">${s.tp}</td><td class="num">${s.fp}</td><td class="num">${s.tn}</td><td class="num">${s.fn}</td></tr>`;
  });
  $('#threshold-table tbody').innerHTML = html.join('');
}

function renderMistakes(r) {
  const t = r.best.threshold;
  const rejects = r.scored.filter((p) => p.same && p.distance > t).sort((a, b) => b.distance - a.distance).slice(0, 6);
  const accepts = r.scored.filter((p) => !p.same && p.distance <= t).sort((a, b) => a.distance - b.distance).slice(0, 6);
  const label = (rel) => escapeHtml(personName(rel));
  const card = (p, kind) => `<div class="mistake"><div class="imgs"><img alt="" loading="lazy" src="${r.data.thumbUrl(p.a)}"><img alt="" loading="lazy" src="${r.data.thumbUrl(p.b)}"></div><strong>${kind}</strong> · distance ${fmt(p.distance)}<br>${label(p.a)}${p.same ? '' : ` vs ${label(p.b)}`}</div>`;
  const cards = [...rejects.map((p) => card(p, 'False reject')), ...accepts.map((p) => card(p, 'False accept'))];
  $('#mistakes').innerHTML = cards.length ? cards.join('') : '<p class="muted">No mistakes at this threshold.</p>';
  $$('#mistakes .mistake').forEach((m, i) => {
    m.style.animationDelay = `${i * 0.06}s`;
    m.classList.add('mistake-visible');
  });
  $('#eval-noface').innerHTML = r.noFace.length
    ? `<details><summary>${r.noFace.length} photo${r.noFace.length === 1 ? '' : 's'} had no detectable face, so ${r.skipped} pair${r.skipped === 1 ? ' was' : 's were'} skipped</summary><p class="small">${r.noFace.map(escapeHtml).join(' · ')}</p></details>`
    : 'Every photo produced a face embedding. No pairs were skipped.';
}

$('#eval-export').addEventListener('click', () => {
  const r = state.eval.last;
  if (!r) return;
  const lines = ['image_a,image_b,same_person,fold,distance', ...r.scored.map((p) => `${p.a},${p.b},${p.same ? 1 : 0},${p.fold},${p.distance.toFixed(6)}`)];
  const link = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' })),
    download: 'pair-distances.csv',
  });
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

renderVerify();
loadSources();

// ------------------------------------------------------------------ animations

// Button ripple effect
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.btn');
  if (!btn || btn.disabled) return;
  const rect = btn.getBoundingClientRect();
  const ripple = document.createElement('span');
  ripple.className = 'ripple-effect';
  ripple.style.left = `${e.clientX - rect.left - 10}px`;
  ripple.style.top = `${e.clientY - rect.top - 10}px`;
  btn.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
});
