// Apply a saved light/dark choice before the first paint. It lives in its own
// file so the Content-Security-Policy can forbid inline scripts.
try { const t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; } catch {}
