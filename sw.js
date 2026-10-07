// Cache-first service worker. Bump CACHE_NAME on every release.
const CACHE_NAME = 'yoga-flow-cache-v3';
const ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './icon.svg',
  './images/kalyanasundaram.jpg',
  './images/sattianadan.jpg',
  './images/vidyasagar.jpg',
  './images/saravanan.jpg'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request).then(fetchRes => {
        if (fetchRes && fetchRes.status === 200) {
          const copy = fetchRes.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        }
        return fetchRes;
      });
    }).catch(() => caches.match('./index.html'))
  );
});
