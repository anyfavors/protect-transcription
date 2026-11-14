import { api, qs, waitForJob } from '../api.js';
import { addDays, isoDate, markdown, number, parseDate, periodLabel, relative, today, weekKey } from '../format.js';

const PERIODS = [
    { id: 'daily', label: 'Days' },
    { id: 'weekly', label: 'Weeks' },
    { id: 'monthly', label: 'Months' },
    { id: 'yearly', label: 'Years' },
];

export default function summariesPage() {
    return {
        PERIODS,
        items: [],
        loading: false,
        month: today().slice(0, 7),   // calendar month shown on the Days tab
        _loadedFor: null,

        init() {
            this.$watch('$store.ui.route', () => this.sync());
            this.sync();
        },
        get active() { return this.$store.ui.route.page === 'summaries'; },
        get period() {
            const p = this.$store.ui.route.parts[0];
            return PERIODS.some(x => x.id === p) ? p : 'daily';
        },
        get selectedKey() { return this.$store.ui.route.parts[1] || this.items[0]?.period_key || null; },
        get selected() { return this.items.find(i => i.period_key === this.selectedKey) || null; },

        sync() {
            if (!this.active) return;
            if (this._loadedFor !== this.period) this.load();
            const key = this.$store.ui.route.parts[1];
            if (this.period === 'daily' && key) this.month = key.slice(0, 7);
        },
        async load() {
            const period = this.period;
            this.loading = true;
            try {
                const data = await api.get(`/api/summaries?${qs({ period, limit: 2000 })}`);
                if (period !== this.period) return;
                this.items = data.items.map(i => ({ ...i, job: null }));
                this._loadedFor = period;
            } catch (e) {
                this.$store.ui.toast(`Couldn’t load summaries: ${e.message}`, 'error');
            } finally {
                this.loading = false;
            }
        },
        select(key) { this.$store.ui.go('summaries', [this.period, key]); },
        setPeriod(p) { this.$store.ui.go('summaries', [p]); },

        state(item) {
            if (!item) return '';
            if (item.generating || item.job) return 'building';
            if (!item.summary) return 'missing';
            return item.stale ? 'stale' : 'fresh';
        },
        stateLabel(item) {
            return { building: 'Building…', missing: 'Not built yet', stale: 'Out of date · rebuilds automatically', fresh: '' }[this.state(item)];
        },

        async rebuild(item) {
            if (!item || item.job) return;
            item.job = true;
            try {
                const job = await api.post('/api/summaries/generate', { period_type: this.period, period_key: item.period_key });
                const result = await waitForJob(job);
                Object.assign(item, { summary: result.summary, stale: false, generated_at: new Date().toISOString() });
                this.$store.ui.toast(`${periodLabel(this.period, item.period_key)} rebuilt`, 'success');
            } catch (e) {
                this.$store.ui.toast(`Rebuild failed: ${e.message}`, 'error');
            } finally {
                item.job = null;
                item.generating = false;
            }
        },

        // Daily summaries mention times ("At 07:54 …"): link them to that moment on the day page.
        summaryHtml(item) {
            if (!item?.summary) return '';
            if (this.period !== 'daily') return markdown(item.summary);
            const date = item.period_key;
            return markdown(item.summary, line => line.replace(/\b([01]\d|2[0-3])[:.]([0-5]\d)\b/g,
                (m, h, mm) => `<a class="timeref" href="#/day/${date}?at=${h}:${mm}">${m}</a>`));
        },

        // ── Calendar for the Days tab ────────────────────────────
        get byKey() { return Object.fromEntries(this.items.map(i => [i.period_key, i])); },
        get calendar() {
            const first = parseDate(`${this.month}-01`);
            const lead = (first.getDay() + 6) % 7;
            const cells = [];
            for (let i = 0; i < lead; i++) cells.push(null);
            for (let d = 1; ; d++) {
                const date = new Date(first.getFullYear(), first.getMonth(), d);
                if (date.getMonth() !== first.getMonth()) break;
                const key = isoDate(date);
                cells.push({ key, day: d, item: this.byKey[key] || null, week: weekKey(key) });
            }
            return cells;
        },
        get maxCount() { return Math.max(1, ...this.items.filter(i => i.period_key.startsWith(this.month)).map(i => i.count)); },
        heat(item) {
            if (!item) return 0;
            const r = item.count / this.maxCount;
            return r > 0.66 ? 3 : r > 0.33 ? 2 : 1;
        },
        shiftMonth(n) {
            const d = parseDate(`${this.month}-01`);
            this.month = isoDate(new Date(d.getFullYear(), d.getMonth() + n, 1)).slice(0, 7);
        },
        get monthLabel() { return periodLabel('monthly', this.month); },

        periodLabel, relative, number, addDays, today,
    };
}
