// Global UI state: routing, live status, toasts, confirm dialog, theme.
// Pages are separate Alpine components that read and write $store.ui.

import { api } from './api.js';

// Hash routes: #/day/2026-11-14?clip=12 · #/summaries/daily/2026-11-14 ·
// #/search?q=hund · #/insights · #/system/settings
function parseHash() {
    const raw = location.hash.replace(/^#\/?/, '');
    const [path, query = ''] = raw.split('?');
    const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
    return { page: parts[0] || 'day', parts: parts.slice(1), query: Object.fromEntries(new URLSearchParams(query)) };
}

export function createStore() {
    return {
        route: parseHash(),
        stats: { queued: 0, processing: 0, completed: 0, errors: 0, total: 0, rate_per_hour: 0, eta_seconds: null },
        health: null,
        me: null,
        cameras: [],
        toasts: [],
        confirmBox: null,
        theme: localStorage.getItem('theme') || 'system',
        wsOnline: false,
        _toastId: 0,
        _statsTimer: null,

        init() {
            window.addEventListener('hashchange', () => { this.route = parseHash(); });
            this.applyTheme();
            matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.applyTheme());
            this.refreshStats();
            this.refreshHealth();
            setInterval(() => this.refreshHealth(), 60000);
            setInterval(() => { if (!this.wsOnline) this.refreshStats(); }, 15000);
            api.get('/api/me').then(me => { this.me = me; }).catch(() => {});
            api.get('/api/cameras').then(d => { this.cameras = d.cameras; }).catch(() => {});
            this.connect();
        },

        // ── Navigation ───────────────────────────────────────────
        go(page, parts = [], query = {}) {
            const q = new URLSearchParams(Object.entries(query).filter(([, v]) => v !== '' && v != null && v !== false));
            const path = [page, ...parts].map(encodeURIComponent).join('/');
            location.hash = `#/${path}${q.toString() ? '?' + q : ''}`;
        },
        setQuery(changes, { replace = false } = {}) {
            const query = { ...this.route.query, ...changes };
            const q = new URLSearchParams(Object.entries(query).filter(([, v]) => v !== '' && v != null && v !== false));
            const url = `#/${[this.route.page, ...this.route.parts].map(encodeURIComponent).join('/')}${q.toString() ? '?' + q : ''}`;
            if (replace) { history.replaceState(null, '', url); this.route = parseHash(); }
            else location.hash = url;
        },
        openClip(id) { this.setQuery({ clip: id }); },
        closeClip() { this.setQuery({ clip: '' }); },
        get clipId() { return this.route.query.clip ? Number(this.route.query.clip) : null; },

        // ── Live updates ─────────────────────────────────────────
        connect() {
            const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
            ws.onopen = () => { this.wsOnline = true; this._wsDelay = 1000; };
            ws.onclose = () => {
                this.wsOnline = false;
                setTimeout(() => this.connect(), this._wsDelay || 1000);
                this._wsDelay = Math.min((this._wsDelay || 1000) * 2, 30000);
            };
            ws.onmessage = (e) => {
                let msg;
                try { msg = JSON.parse(e.data); } catch { return; }
                if (msg.type !== 'transcription_update') return;
                window.dispatchEvent(new CustomEvent('clip-changed', { detail: msg }));
                clearTimeout(this._statsTimer);
                this._statsTimer = setTimeout(() => this.refreshStats(), 2000);
            };
        },
        async refreshStats() {
            try { this.stats = await api.get('/api/stats'); } catch { /* strip shows stale numbers */ }
        },
        async refreshHealth() {
            try { this.health = await api.get('/api/system/health'); } catch { this.health = null; }
        },
        get queueProgress() {
            const s = this.stats;
            const left = (s.queued || 0) + (s.processing || 0);
            return s.total ? Math.round(((s.total - left) / s.total) * 100) : 100;
        },

        // ── Feedback ─────────────────────────────────────────────
        toast(message, kind = 'info') {
            const id = ++this._toastId;
            this.toasts.push({ id, message, kind });
            setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, kind === 'error' ? 7000 : 3500);
        },
        // Resolves true/false; `typed` requires the user to type that word first.
        confirm({ title, message, action = 'Confirm', danger = false, typed = '' }) {
            return new Promise(resolve => { this.confirmBox = { title, message, action, danger, typed, input: '', resolve }; });
        },
        closeConfirm(ok) {
            const box = this.confirmBox;
            this.confirmBox = null;
            box?.resolve(ok);
        },

        // ── Theme: system → light → dark ─────────────────────────
        cycleTheme() {
            this.theme = { system: 'light', light: 'dark', dark: 'system' }[this.theme];
            localStorage.setItem('theme', this.theme);
            this.applyTheme();
        },
        applyTheme() {
            const dark = this.theme === 'dark' || (this.theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
            document.documentElement.classList.toggle('dark', dark);
            window.dispatchEvent(new Event('theme-changed'));
        },
    };
}
