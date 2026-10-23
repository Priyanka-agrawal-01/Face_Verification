// Face verification core: detection -> landmarks/alignment -> 128-d embedding -> distance.
// Everything runs in the browser; images never leave this machine.
import * as faceapi from '/vendor/face-api/face-api.esm.js';

export { faceapi };

// face-api's recognition net is trained so that the same person usually lands
// below 0.6 Euclidean distance. The Evaluate tab measures this on real data.
export const DEFAULT_THRESHOLD = 0.6;

let modelsReady = null;

export function loadModels() {
  modelsReady ??= (async () => {
    await faceapi.tf.ready();
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    ]);
    return faceapi.tf.getBackend();
  })().catch((err) => {
    modelsReady = null;
    throw err;
  });
  return modelsReady;
}

// Detect every face, then pick the largest as the subject of the photo.
// A lower-confidence retry rescues small or partly turned faces, and a padded
// retry rescues faces cropped tight to the image edge. `input` is the image the
// returned coordinates refer to.
export async function analyzeFaces(image) {
  await loadModels();
  let input = image;
  let faces = await detect(input, 0.5);
  if (!faces.length) faces = await detect(input, 0.2);
  if (!faces.length) {
    input = padImage(image, 0.25);
    faces = await detect(input, 0.2);
  }
  faces.sort((a, b) => area(b.detection.box) - area(a.detection.box));
  return { faces, primary: faces[0] ?? null, input, padded: input !== image };
}

function padImage(image, fraction) {
  const w = image.naturalWidth ?? image.width;
  const h = image.naturalHeight ?? image.height;
  const pad = Math.round(Math.max(w, h) * fraction);
  const canvas = document.createElement('canvas');
  canvas.width = w + pad * 2;
  canvas.height = h + pad * 2;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, pad, pad, w, h);
  return canvas;
}

function detect(input, minConfidence) {
  return faceapi
    .detectAllFaces(input, new faceapi.SsdMobilenetv1Options({ minConfidence }))
    .withFaceLandmarks()
    .withFaceDescriptors();
}

const area = (box) => box.width * box.height;

export function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

// Mean of several embeddings of one person: a more stable "template".
export function meanDescriptor(descriptors) {
  const mean = new Float32Array(descriptors[0].length);
  for (const d of descriptors) for (let i = 0; i < d.length; i++) mean[i] += d[i] / descriptors.length;
  return mean;
}

export function loadImageFromFile(file) {
  return loadImage(URL.createObjectURL(file), true);
}

export function loadImage(src, revoke = false) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      if (revoke) URL.revokeObjectURL(src);
      resolve(img);
    };
    img.onerror = () => {
      if (revoke) URL.revokeObjectURL(src);
      reject(new Error('Could not open this image.'));
    };
    img.src = src;
  });
}

// ---------------------------------------------------------------- evaluation

// Small seeded PRNG so generated impostor pairs are reproducible.
export function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// LFW pairs.txt: header "folds<TAB>n", then per fold n matched lines
// "name i j" followed by n mismatched lines "name1 i name2 j".
export function parseLfwPairs(text) {
  const lines = text.trim().split(/\r?\n/);
  const [folds, perSide] = lines[0].trim().split(/\s+/).map(Number);
  const lfwFile = (name, n) => `${name}/${name}_${String(n).padStart(4, '0')}.jpg`;
  const pairs = [];
  let row = 1;
  for (let fold = 0; fold < folds; fold++) {
    for (let k = 0; k < perSide; k++, row++) {
      const [name, i, j] = lines[row].trim().split(/\s+/);
      pairs.push({ a: lfwFile(name, +i), b: lfwFile(name, +j), same: true, fold });
    }
    for (let k = 0; k < perSide; k++, row++) {
      const [n1, i, n2, j] = lines[row].trim().split(/\s+/);
      pairs.push({ a: lfwFile(n1, +i), b: lfwFile(n2, +j), same: false, fold });
    }
  }
  return { folds, pairs };
}

// For datasets without pairs.txt (e.g. your own photos): every same-person
// pair up to a cap, plus an equal number of random different-person pairs.
export function generatePairs(people, maxGenuine = 1500, seed = 42) {
  const rand = mulberry32(seed);
  const names = Object.keys(people).filter((n) => people[n].length > 0);
  const genuine = [];
  for (const name of names) {
    const files = people[name];
    for (let i = 0; i < files.length; i++)
      for (let j = i + 1; j < files.length; j++)
        genuine.push({ a: `${name}/${files[i]}`, b: `${name}/${files[j]}`, same: true, fold: 0 });
  }
  shuffle(genuine, rand);
  genuine.length = Math.min(genuine.length, maxGenuine);

  const impostor = [];
  const seen = new Set();
  const wanted = names.length > 1 ? genuine.length : 0;
  for (let tries = 0; impostor.length < wanted && tries < wanted * 20; tries++) {
    const n1 = names[Math.floor(rand() * names.length)];
    const n2 = names[Math.floor(rand() * names.length)];
    if (n1 === n2) continue;
    const a = `${n1}/${pick(people[n1], rand)}`;
    const b = `${n2}/${pick(people[n2], rand)}`;
    const key = a < b ? a + '|' + b : b + '|' + a;
    if (seen.has(key)) continue;
    seen.add(key);
    impostor.push({ a, b, same: false, fold: 0 });
  }
  return { folds: 1, pairs: [...genuine, ...impostor] };
}

const pick = (arr, rand) => arr[Math.floor(rand() * arr.length)];
function shuffle(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Confusion counts at one threshold: "same person" means distance <= threshold.
export function scoreAt(scored, threshold) {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (const p of scored) {
    const accept = p.distance <= threshold;
    if (p.same) accept ? tp++ : fn++;
    else accept ? fp++ : tn++;
  }
  const total = tp + fn + fp + tn;
  return {
    threshold,
    tp, fn, fp, tn,
    accuracy: total ? (tp + tn) / total : 0,
    far: fp + tn ? fp / (fp + tn) : 0, // false accept rate: strangers let through
    frr: tp + fn ? fn / (tp + fn) : 0, // false reject rate: same person turned away
  };
}

export const THRESHOLDS = Array.from({ length: 121 }, (_, i) => Math.round(i) / 100);

export function bestThreshold(scored) {
  let best = null;
  for (const t of THRESHOLDS) {
    const s = scoreAt(scored, t);
    if (!best || s.accuracy > best.accuracy) best = s;
  }
  return best;
}

// Equal error rate: where false accepts and false rejects balance.
export function equalErrorRate(scored) {
  let best = null;
  for (const t of THRESHOLDS) {
    const s = scoreAt(scored, t);
    const gap = Math.abs(s.far - s.frr);
    if (!best || gap < best.gap) best = { ...s, gap, eer: (s.far + s.frr) / 2 };
  }
  return best;
}

// ROC points sweep every distinct distance, so the curve is exact.
export function rocCurve(scored) {
  const sorted = [...scored].sort((x, y) => x.distance - y.distance);
  const pos = sorted.filter((p) => p.same).length;
  const neg = sorted.length - pos;
  const points = [{ fpr: 0, tpr: 0, threshold: 0 }];
  let tp = 0, fp = 0;
  for (let i = 0; i < sorted.length; i++) {
    sorted[i].same ? tp++ : fp++;
    if (i === sorted.length - 1 || sorted[i + 1].distance !== sorted[i].distance)
      points.push({ fpr: neg ? fp / neg : 0, tpr: pos ? tp / pos : 0, threshold: sorted[i].distance });
  }
  let auc = 0;
  for (let i = 1; i < points.length; i++)
    auc += (points[i].fpr - points[i - 1].fpr) * (points[i].tpr + points[i - 1].tpr) / 2;
  return { points, auc };
}

// Standard LFW protocol: for each fold, choose the threshold on the other
// folds and measure accuracy on the held-out fold.
export function crossValidate(scored, folds) {
  const results = [];
  for (let f = 0; f < folds; f++) {
    const train = scored.filter((p) => p.fold !== f);
    const test = scored.filter((p) => p.fold === f);
    if (!train.length || !test.length) continue;
    const t = bestThreshold(train).threshold;
    results.push({ fold: f, threshold: t, accuracy: scoreAt(test, t).accuracy });
  }
  if (!results.length) return null;
  const mean = results.reduce((s, r) => s + r.accuracy, 0) / results.length;
  const std = Math.sqrt(results.reduce((s, r) => s + (r.accuracy - mean) ** 2, 0) / results.length);
  return { results, mean, std };
}
