// Minimal static server for the face verification demo.
// Serves the UI, the face-api library and model weights from node_modules,
// and lists datasets placed under ./data (or DATA_DIR). Binds to localhost
// unless HOST is set (e.g. HOST=0.0.0.0 on Render or another Node host).
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, readdir, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { gzip } from 'node:zlib';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const PORT = Number(process.env.PORT) || 5180;
const HOST = process.env.HOST || '127.0.0.1';
const DATA_DIR = resolve(process.env.DATA_DIR || join(ROOT, 'data'));

const MOUNTS = {
  '/vendor/face-api/': join(ROOT, 'node_modules/@vladmandic/face-api/dist'),
  '/models/': join(ROOT, 'node_modules/@vladmandic/face-api/model'),
  '/data/': DATA_DIR,
  '/': join(ROOT, 'public'),
};

// The library and weights are pinned by package-lock.json, so browsers may keep
// them for a day. The app's own files are revalidated on every load (a cheap 304
// thanks to ETags), so a new deploy shows up immediately.
const CACHE = {
  '/vendor/face-api/': 'public, max-age=86400',
  '/models/': 'public, max-age=86400',
  '/data/': 'public, max-age=3600',
  '/': 'no-cache',
};

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.map': 'application/json; charset=utf-8',
};

// Text compresses well (face-api.esm.js: 1.3 MB → about 0.3 MB). Images and the
// model weights barely shrink, so they are sent as they are.
const COMPRESSIBLE = new Set(['.html', '.js', '.mjs', '.css', '.json', '.txt', '.svg', '.map']);

const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp']);

// Everything runs in the page and only talks to this server. Inline styles are
// allowed because the markup and charts use style attributes; scripts are not.
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join('; ');

// Sent with every response. The page needs no camera, mic or location.
const SECURITY_HEADERS = {
  'Content-Security-Policy': CSP,
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'X-Frame-Options': 'DENY',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
};

const gzipAsync = promisify(gzip);
const gzipCache = new Map(); // file path -> { mtimeMs, body }

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// Resolve a URL path inside a mount, refusing anything that escapes it.
function resolveInside(base, rel) {
  const target = resolve(base, normalize(decodeURIComponent(rel)).replace(/^([/\\])+/, ''));
  return target === base || target.startsWith(base + sep) ? target : null;
}

// A dataset is a folder of person sub-folders, each holding face images.
// An optional pairs.txt in LFW format sits at the dataset root.
async function listDatasets() {
  const out = [];
  for (const entry of await readdir(DATA_DIR, { withFileTypes: true }).catch(() => [])) {
    if (!entry.isDirectory() || entry.name === 'download') continue;
    const summary = await describeDataset(entry.name, false);
    if (summary.people > 0) out.push(summary);
  }
  return out;
}

async function describeDataset(name, withFiles) {
  const dir = resolveInside(DATA_DIR, name);
  if (!dir || dir === DATA_DIR) throw new HttpError(400, 'Invalid dataset name');
  const people = {};
  let images = 0;
  for (const person of await readdir(dir, { withFileTypes: true })) {
    if (!person.isDirectory()) continue;
    const files = (await readdir(join(dir, person.name)))
      .filter((f) => IMAGE_EXT.has(extname(f).toLowerCase()))
      .sort();
    if (!files.length) continue;
    people[person.name] = files;
    images += files.length;
  }
  const hasPairs = await stat(join(dir, 'pairs.txt')).then((s) => s.isFile(), () => false);
  const summary = { name, people: Object.keys(people).length, images, hasPairs };
  return withFiles ? { ...summary, files: people } : summary;
}

const acceptsGzip = (req) => /\bgzip\b/.test(req.headers['accept-encoding'] ?? '');

function securityHeaders(req) {
  // Render and most hosts terminate TLS in front of the app and say so in this header.
  return req.headers['x-forwarded-proto'] === 'https'
    ? { ...SECURITY_HEADERS, 'Strict-Transport-Security': 'max-age=31536000' }
    : SECURITY_HEADERS;
}

async function send(req, res, status, body, { type = 'application/json; charset=utf-8', headers = {} } = {}) {
  let buf = Buffer.from(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
  const out = { ...securityHeaders(req), ...headers, 'Content-Type': type, 'Cache-Control': 'no-store', Vary: 'Accept-Encoding' };
  if (buf.length > 1024 && acceptsGzip(req)) {
    buf = await gzipAsync(buf);
    out['Content-Encoding'] = 'gzip';
  }
  out['Content-Length'] = buf.length;
  res.writeHead(status, out);
  res.end(req.method === 'HEAD' ? undefined : buf);
}

// Browsers asking for a page get a page; scripts and fetches get JSON.
function notFound(req, res) {
  if (!req.headers.accept?.includes('text/html')) return send(req, res, 404, { error: 'Not found' });
  return send(req, res, 404, `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Page not found · Face Verification Lab</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;font:16px/1.6 system-ui,sans-serif;background:#f6f6f4;color:#0b0b0b;text-align:center}
@media (prefers-color-scheme:dark){body{background:#0e0e0d;color:#f0efec}}a{color:#2a78d6}h1{font-size:1.6rem;margin:0 0 6px}</style>
<main><h1>Page not found</h1><p>This address doesn't exist. <a href="/">Back to Face Verification Lab</a></p></main>`, { type: 'text/html; charset=utf-8' });
}

async function gzipped(file, info) {
  const hit = gzipCache.get(file);
  if (hit?.mtimeMs === info.mtimeMs) return hit.body;
  const body = await gzipAsync(await readFile(file));
  gzipCache.set(file, { mtimeMs: info.mtimeMs, body });
  return body;
}

async function serveFile(req, res, file, info, cacheControl) {
  const ext = extname(file).toLowerCase();
  const etag = `W/"${info.size.toString(16)}-${Math.trunc(info.mtimeMs).toString(16)}"`;
  const headers = {
    ...securityHeaders(req),
    'Content-Type': TYPES[ext] ?? 'application/octet-stream',
    'Cache-Control': cacheControl,
    ETag: etag,
    'Last-Modified': info.mtime.toUTCString(),
  };
  const compressible = COMPRESSIBLE.has(ext);
  if (compressible) headers.Vary = 'Accept-Encoding';

  const known = (req.headers['if-none-match'] ?? '').split(',').map((s) => s.trim());
  if (known.includes(etag)) {
    res.writeHead(304, headers);
    return res.end();
  }
  if (compressible && acceptsGzip(req)) {
    const body = await gzipped(file, info);
    res.writeHead(200, { ...headers, 'Content-Encoding': 'gzip', 'Content-Length': body.length });
    return res.end(req.method === 'HEAD' ? undefined : body);
  }
  res.writeHead(200, { ...headers, 'Content-Length': info.size });
  if (req.method === 'HEAD') return res.end();
  // Stream large files (the weights are several MB) instead of buffering them.
  createReadStream(file).on('error', () => res.destroy()).pipe(res);
}

const server = createServer(async (req, res) => {
  try {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return await send(req, res, 405, { error: 'Method not allowed' }, { headers: { Allow: 'GET, HEAD' } });
    }
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/healthz') return await send(req, res, 200, 'ok', { type: 'text/plain; charset=utf-8' });
    if (url.pathname === '/api/datasets') return await send(req, res, 200, await listDatasets());
    if (url.pathname === '/api/dataset') {
      return await send(req, res, 200, await describeDataset(url.searchParams.get('name') ?? '', true));
    }

    const mount = Object.keys(MOUNTS).find((m) => url.pathname.startsWith(m));
    let file = resolveInside(MOUNTS[mount], url.pathname.slice(mount.length));
    if (!file) return await send(req, res, 403, { error: 'Forbidden' });
    if (url.pathname === '/') file = join(MOUNTS['/'], 'index.html');

    const info = await stat(file).catch(() => null);
    if (!info?.isFile()) return await notFound(req, res);
    await serveFile(req, res, file, info, CACHE[mount]);
  } catch (err) {
    if (res.headersSent) return res.destroy();
    // Never echo internal error text (it can contain file paths) to the client.
    if (err instanceof URIError) return send(req, res, 400, { error: 'Bad request' });
    if (err instanceof HttpError) return send(req, res, err.status, { error: err.message });
    if (err.code === 'ENOENT' || err.code === 'ENOTDIR') return notFound(req, res);
    console.error(err);
    send(req, res, 500, { error: 'Internal server error' });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Face verification demo running at http://${HOST}:${PORT}`);
});

// Hosts send SIGTERM before replacing an instance: finish open requests, then exit.
for (const signal of ['SIGTERM', 'SIGINT']) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 5000).unref();
  });
}
