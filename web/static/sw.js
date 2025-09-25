// Enhanced VANI Service Worker with caching
const CACHE_NAME = 'vani-v1.0.1';
const STATIC_CACHE_NAME = 'vani-static-v1.0.1';

// Assets to cache immediately
const STATIC_ASSETS = [
    '/',
    '/static/css/style.css',
    '/static/favicon.svg'
];

// API endpoints to cache (GET-only, with short TTL). Do NOT cache /query (POST).
const API_CACHE_PATTERNS = [
    /\/health/
];

self.addEventListener('install', (event) => {
    console.log('VANI SW: Installing...');
    event.waitUntil(
        caches.open(STATIC_CACHE_NAME)
            .then(async (cache) => {
                console.log('VANI SW: Caching static assets');
                // Cache assets individually; skip any that fail to avoid aborting install
                await Promise.all(
                    STATIC_ASSETS.map(async (url) => {
                        try {
                            const req = new Request(url, { cache: 'reload' });
                            const resp = await fetch(req);
                            if (resp && resp.ok) {
                                await cache.put(req, resp);
                            } else {
                                console.warn('VANI SW: Skipping asset (not ok):', url);
                            }
                        } catch (e) {
                            console.warn('VANI SW: Skipping missing asset:', url, e);
                        }
                    })
                );
            })
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    console.log('VANI SW: Activating...');
    event.waitUntil(
        caches.keys().then(async cacheNames => {
            await Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== STATIC_CACHE_NAME && cacheName !== CACHE_NAME) {
                        console.log('VANI SW: Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
            // Additionally purge any stale cached /query entries from existing caches
            await Promise.all(
                cacheNames.map(async cacheName => {
                    try {
                        const cache = await caches.open(cacheName);
                        const requests = await cache.keys();
                        await Promise.all(
                            requests.map(req => {
                                try {
                                    const u = new URL(req.url);
                                    if (u.pathname === '/query') {
                                        return cache.delete(req);
                                    }
                                } catch (e) {}
                                return Promise.resolve(false);
                            })
                        );
                    } catch (e) {}
                })
            );
        }).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Handle static assets - cache-first strategy
    if (STATIC_ASSETS.includes(event.request.url.replace(self.location.origin, ''))) {
        event.respondWith(
            caches.match(event.request)
                .then(response => {
                    if (response) {
                        return response;
                    }
                    return fetch(event.request).then(response => {
                        if (response.ok) {
                            const responseClone = response.clone();
                            caches.open(STATIC_CACHE_NAME).then(cache => {
                                cache.put(event.request, responseClone);
                            });
                        }
                        return response;
                    });
                })
                .catch(() => {
                    // Fallback for offline
                    if (event.request.destination === 'document') {
                        return caches.match('/');
                    }
                })
        );
        return;
    }

    // Handle API requests (GET-only) - network-first with fallback
    if (event.request.method === 'GET' && API_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname))) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    if (response.ok) {
                        // Cache successful responses for short term
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then(cache => {
                            // Add timestamp for cache invalidation
                            const responseWithTimestamp = new Response(responseClone.body, {
                                status: responseClone.status,
                                statusText: responseClone.statusText,
                                headers: {
                                    ...Object.fromEntries(responseClone.headers),
                                    'sw-cache-time': Date.now()
                                }
                            });
                            cache.put(event.request, responseWithTimestamp);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Try cache if network fails
                    return caches.match(event.request).then(response => {
                        if (response) {
                            // Check if cache is still fresh (5 minutes)
                            const cacheTime = response.headers.get('sw-cache-time');
                            if (cacheTime && (Date.now() - parseInt(cacheTime)) < 5 * 60 * 1000) {
                                return response;
                            }
                        }
                        // Return offline message for API failures
                        return new Response(
                            JSON.stringify({
                                success: false,
                                error: 'You are offline. Please check your connection and try again.',
                                offline: true
                            }),
                            {
                                headers: { 'Content-Type': 'application/json' },
                                status: 503
                            }
                        );
                    });
                })
        );
        return;
    }

    // Default: network-first for everything else
    event.respondWith(
        fetch(event.request)
            .catch(() => {
                // Try cache for HTML pages
                if (event.request.destination === 'document') {
                    return caches.match('/').then(response => {
                        if (response) {
                            return response;
                        }
                        return new Response('Offline - Please check your connection', {
                            status: 503,
                            headers: { 'Content-Type': 'text/plain' }
                        });
                    });
                }
                // Return offline message for other failures
                return new Response('Offline', { status: 503 });
            })
    );
});

// Clean up old caches periodically
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'CLEAN_CACHE') {
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== STATIC_CACHE_NAME && cacheName !== CACHE_NAME) {
                        console.log('VANI SW: Cleaning up cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        });
    }
});
