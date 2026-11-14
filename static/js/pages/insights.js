import { api, qs } from '../api.js';
import { number, parseDate, speechTotal } from '../format.js';

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export default function insightsPage() {
    return {
        WEEKDAYS,
        days: 90,
        location: '',
        cells: [],
        daily: [],
        loading: false,
        _loadedFor: null,

        init() {
            this.$watch('$store.ui.route', () => this.sync());
            this.sync();
        },
        get active() { return this.$store.ui.route.page === 'insights'; },
        sync() { if (this.active && this._loadedFor !== this.days) this.load(); },

        async load() {
            this.loading = true;
            try {
                const [heat, daily] = await Promise.all([
                    api.get(`/api/analytics/heatmap?${qs({ days: this.days })}`),
                    api.get(`/api/analytics/daily?${qs({ days: Math.min(this.days, 365) })}`),
                ]);
                this.cells = heat.cells;
                this.daily = daily.days;
                this._loadedFor = this.days;
            } catch (e) {
                this.$store.ui.toast(`Couldn’t load insights: ${e.message}`, 'error');
            } finally {
                this.loading = false;
            }
        },
        setDays(d) { this.days = d; this.load(); },

        // ── Weekday × hour heatmap (minutes of speech) ───────────
        get locations() { return [...new Set(this.cells.map(c => c.camera))].sort(); },
        get grid() {
            const g = Array.from({ length: 7 }, () => Array(24).fill(0));
            for (const c of this.cells) {
                if (!this.location || c.camera === this.location) g[c.weekday][c.hour] += c.seconds;
            }
            return g;
        },
        get gridMax() { return Math.max(1, ...this.grid.flat()); },
        cellStyle(seconds) {
            if (!seconds) return '';
            const a = 0.12 + 0.88 * Math.sqrt(seconds / this.gridMax);
            return `background-color: rgb(var(--accent-rgb) / ${a.toFixed(2)})`;
        },
        cellTitle(wd, h, seconds) {
            return `${WEEKDAYS[wd]} ${String(h).padStart(2, '0')}:00–${String(h + 1).padStart(2, '0')}:00 · ${speechTotal(seconds)} of speech`;
        },

        // ── Per location totals ──────────────────────────────────
        get perLocation() {
            const totals = {};
            for (const c of this.cells) {
                totals[c.camera] ??= { camera: c.camera, seconds: 0, count: 0 };
                totals[c.camera].seconds += c.seconds;
                totals[c.camera].count += c.count;
            }
            return Object.values(totals).sort((a, b) => b.seconds - a.seconds);
        },
        get locationMax() { return Math.max(1, ...this.perLocation.map(l => l.seconds)); },

        // ── Speech per day (SVG bars) ────────────────────────────
        get dayBars() {
            const max = Math.max(1, ...this.daily.map(d => d.seconds || 0));
            const n = this.daily.length || 1;
            const w = 100 / n;
            return this.daily.map((d, i) => ({
                x: i * w,
                w: Math.max(w - 0.15, 0.1),
                h: ((d.seconds || 0) / max) * 100,
                title: `${parseDate(d.day).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}: ${speechTotal(d.seconds)}, ${d.completed} clips`,
                day: d.day,
            }));
        },
        // Alpine can't x-for inside <svg>, so the bars are rendered as markup (escaped values only).
        get barsSvg() {
            const esc = (t) => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
            const base = '<line x1="0" y1="40" x2="100" y2="40" class="stroke-slate-200 dark:stroke-slate-700" stroke-width="1" vector-effect="non-scaling-stroke"/>';
            return base + this.dayBars.map(b => `<a href="#/day/${esc(b.day)}"><rect x="${b.x}" width="${b.w}" y="${40 - b.h * 0.4}" height="${Math.max(b.h * 0.4, 0)}" class="fill-teal-500 hover:fill-orange-500"><title>${esc(b.title)}</title></rect></a>`).join('');
        },
        get dailyMaxLabel() { return speechTotal(Math.max(0, ...this.daily.map(d => d.seconds || 0))); },
        get totalSpeech() { return this.daily.reduce((s, d) => s + (d.seconds || 0), 0); },
        get totalClips() { return this.daily.reduce((s, d) => s + (d.completed || 0), 0); },

        number, speechTotal,
    };
}
