import { api, qs } from '../api.js';
import { addDays, clock, dayTitle, duration, longDate, minuteOfDay, speechTotal, today } from '../format.js';

const HIDDEN = new Set(['error', 'filtered', 'archived']);

export default function dayPage() {
    return {
        clips: [],
        loading: false,
        showHidden: false,
        location: '',
        _loadedFor: null,
        _reloadTimer: null,

        init() {
            this.$watch('$store.ui.route', () => this.sync());
            this.sync();
            window.addEventListener('clip-changed', (e) => {
                if (String(e.detail.timestamp || '').slice(0, 10) !== this.date) return;
                clearTimeout(this._reloadTimer);
                this._reloadTimer = setTimeout(() => this.load({ quiet: true }), 1500);
            });
        },

        get active() { return this.$store.ui.route.page === 'day'; },
        get date() {
            const d = this.$store.ui.route.parts[0];
            return d && /^\d{4}-\d{2}-\d{2}$/.test(d) ? d : today();
        },

        sync() {
            if (!this.active) return;
            if (this._loadedFor !== this.date) this.load();
        },

        async load({ quiet = false } = {}) {
            const date = this.date;
            if (!quiet) { this.loading = true; this.clips = []; }
            try {
                const data = await api.get(`/api/transcriptions?${qs({ date, per_page: 500 })}`);
                if (date !== this.date) return;
                this.clips = data.transcriptions.slice().reverse(); // oldest first, like a day reads
                this._loadedFor = date;
                const at = this.$store.ui.route.query.at;
                if (at && !this.$store.ui.clipId) this.openNearest(at);
            } catch (e) {
                this.$store.ui.toast(`Couldn’t load ${longDate(date)}: ${e.message}`, 'error');
            } finally {
                this.loading = false;
            }
        },

        // ── Derived data ─────────────────────────────────────────
        get visible() {
            return this.clips.filter(c => (this.showHidden || !HIDDEN.has(c.status)) && (!this.location || c.camera_name === this.location));
        },
        get hiddenCount() { return this.clips.filter(c => HIDDEN.has(c.status)).length; },
        get locations() {
            const seen = new Set(this.clips.map(c => c.camera_name));
            const all = this.$store.ui.cameras.length ? this.$store.ui.cameras : [...seen];
            // Locations with activity today first, then the rest (dimmed).
            return [...all].sort((a, b) => (seen.has(b) - seen.has(a)) || a.localeCompare(b));
        },
        lane(name) {
            return this.clips.filter(c => c.camera_name === name && (this.showHidden || !HIDDEN.has(c.status)));
        },
        get speechSeconds() {
            return this.clips.filter(c => c.status === 'completed').reduce((s, c) => s + (c.duration_seconds || 0), 0);
        },
        blockStyle(c) {
            const left = (minuteOfDay(c.timestamp) / 1440) * 100;
            const width = Math.max(((c.duration_seconds || 30) / 86400) * 100, 0.35);
            return `left:${left.toFixed(3)}%;width:${width.toFixed(3)}%`;
        },
        preview(c) {
            if (c.status === 'pending') return c.transcription && c.edited_at ? c.transcription : 'Waiting to be transcribed…';
            if (c.status === 'processing') return 'Transcribing now…';
            return c.transcription || '';
        },

        // ── Navigation ───────────────────────────────────────────
        shiftDay(n) { this.$store.ui.go('day', [addDays(this.date, n)]); },
        goToday() { this.$store.ui.go('day', [today()]); },
        pickDate(value) { if (value) this.$store.ui.go('day', [value]); },
        toggleLocation(name) { this.location = this.location === name ? '' : name; },
        openNearest(hhmm) {
            const [h, m] = hhmm.split(':').map(Number);
            const target = h * 60 + m;
            let best = null;
            for (const c of this.visible) {
                if (!best || Math.abs(minuteOfDay(c.timestamp) - target) < Math.abs(minuteOfDay(best.timestamp) - target)) best = c;
            }
            if (best) this.$store.ui.setQuery({ at: '', clip: best.id }, { replace: true });
        },
        step(n) {
            const list = this.visible;
            if (!list.length) return;
            const i = list.findIndex(c => c.id === this.$store.ui.clipId);
            const next = list[Math.min(Math.max(i + n, 0), list.length - 1)] || list[0];
            this.$store.ui.openClip(next.id);
            this.$nextTick(() => document.getElementById(`row-${next.id}`)?.scrollIntoView({ block: 'nearest' }));
        },

        // Expose formatters to the template.
        clock, duration, dayTitle, longDate, speechTotal, today,
    };
}
