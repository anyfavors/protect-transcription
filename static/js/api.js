// Thin fetch wrappers. Errors carry the API's `detail` message so toasts can
// say what went wrong.

async function request(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    }
    let res;
    try {
        res = await fetch(url, opts);
    } catch {
        throw new Error('Can’t reach the server. Check your connection and try again.');
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : `HTTP ${res.status}`;
        throw new Error(detail);
    }
    return data;
}

export const api = {
    get: (url) => request('GET', url),
    post: (url, body) => request('POST', url, body),
    put: (url, body) => request('PUT', url, body),
    patch: (url, body) => request('PATCH', url, body),
    del: (url) => request('DELETE', url),
};

export function qs(params) {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null && v !== '' && v !== false) p.set(k, v);
    }
    return p.toString();
}

// Poll a background job until it finishes. onTick gets the job record each time.
export async function waitForJob(job, onTick = () => {}) {
    let current = job;
    while (current.status === 'running') {
        onTick(current);
        await new Promise(r => setTimeout(r, 2000));
        current = await api.get(`/api/jobs/${current.id}`);
    }
    onTick(current);
    if (current.status === 'error') throw new Error(current.error || 'The job failed');
    return current.result;
}
