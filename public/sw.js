// EchoSight V8.8 Offline-First Service Worker
// Caches the App Shell (same-origin JS/CSS/HTML), TensorFlow.js COCO-SSD weights,
// EchoDepthNet ONNX weights, and ONNX Runtime Web WASM binaries so blind users can
// continue running local vision & 3D audio in subways, elevators, or airplane mode.

const SHELL_CACHE = 'echosight-v8-shell';
const MODEL_CACHE = 'echosight-v8-models';
const PRECACHE_URLS = ['./', './index.html', './manifest.json', './icon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== SHELL_CACHE && k !== MODEL_CACHE)
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // 1. Cache-First for TensorFlow.js COCO-SSD weights, EchoDepthNet ONNX weights, and ORT WASM runtime
  const isModelAsset =
    (url.hostname === 'storage.googleapis.com' && url.pathname.includes('/tfjs-models/')) ||
    (url.origin === self.location.origin &&
      (url.pathname.endsWith('.onnx') || url.pathname.endsWith('.wasm')));
  if (isModelAsset) {
    event.respondWith(
      caches.open(MODEL_CACHE).then(async (cache) => {
        const cached = await cache.match(req);
        if (cached) return cached;
        const fresh = await fetch(req);
        if (fresh && fresh.ok) {
          cache.put(req, fresh.clone()).catch(() => {});
        }
        return fresh;
      })
    );
    return;
  }

  // Only intercept same-origin app assets
  if (url.origin !== self.location.origin) return;

  // 2. Network-First for HTML navigation (fresh deployments when online, cached shell when offline)
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(SHELL_CACHE).then((c) => c.put(req, copy).catch(() => {}));
          }
          return res;
        })
        .catch(async () => {
          const cached = (await caches.match(req)) || (await caches.match('./index.html'));
          return cached || Response.error();
        })
    );
    return;
  }

  // 3. Stale-While-Revalidate for same-origin JS/CSS/icons
  event.respondWith(
    caches.open(SHELL_CACHE).then(async (cache) => {
      const cached = await cache.match(req);
      const networkPromise = fetch(req)
        .then((res) => {
          if (res && res.ok) {
            cache.put(req, res.clone()).catch(() => {});
          }
          return res;
        })
        .catch(() => cached);
      return cached || networkPromise;
    })
  );
});
