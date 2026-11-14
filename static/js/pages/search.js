import { api, qs, waitForJob } from '../api.js';
import { clock, duration, escapeHtml, longDate, snippetHtml } from '../format.js';

const ASK_HISTORY_KEY = 'askHistory';

function loadHistory() {
    try { return JSON.parse(localStorage.getItem(ASK_HISTORY_KEY) || '[]'); } catch { return []; }
}

export default function searchPage() {
    return {
        // Search
        q: '',
        location: '',
        from: '',
        to: '',
        saved: false,
        results: [],
        total: 0,
        page: 1,
        loading: false,
        _timer: null,
        _key: '',
        // Ask
        question: '',
        asking: false,
        askStatus: '',
        answer: null,
        history: loadHistory(),

        init() {
            this.$watch('$store.ui.route', () => this.sync());
            this.sync();
        },
        get active() { return this.$store.ui.route.page === 'search'; },
        get mode() { return this.$store.ui.route.query.mode === 'ask' ? 'ask' : 'search'; },

        sync() {
            if (!this.active || this.mode !== 'search') return;
            const r = this.$store.ui.route.query;
            this.q = r.q || '';
            this.location = r.loc || '';
            this.from = r.from || '';
            this.to = r.to || '';
            this.saved = r.saved === '1';
            const key = JSON.stringify([this.q, this.location, this.from, this.to, this.saved]);
            if (key !== this._key) { this._key = key; this.run(); }
        },

        // Typing updates the URL (debounced); the route watcher runs the search.
        changed() {
            clearTimeout(this._timer);
            this._timer = setTimeout(() => {
                this.$store.ui.setQuery({ q: this.q.trim(), loc: this.location, from: this.from, to: this.to, saved: this.saved ? '1' : '', clip: '' }, { replace: true });
                this.sync();
            }, 300);
        },
        get hasCriteria() { return this.q.trim() || this.saved || this.location || this.from || this.to; },

        async run(append = false) {
            if (!this.hasCriteria) { this.results = []; this.total = 0; return; }
            this.loading = true;
            this.page = append ? this.page + 1 : 1;
            try {
                const data = await api.get(`/api/transcriptions?${qs({
                    search: this.q.trim(), camera: this.location, date_from: this.from, date_to: this.to,
                    starred: this.saved, visible: true, page: this.page, per_page: 30,
                })}`);
                this.results = append ? [...this.results, ...data.transcriptions] : data.transcriptions;
                this.total = data.total;
            } catch (e) {
                this.$store.ui.toast(`Search failed: ${e.message}`, 'error');
            } finally {
                this.loading = false;
            }
        },
        resultHtml(r) {
            if (r.snippet) return snippetHtml(r.snippet);
            const text = r.transcription || '';
            return escapeHtml(text.length > 220 ? text.slice(0, 220) + '…' : text);
        },
        clearFilters() {
            this.$store.ui.setQuery({ q: '', loc: '', from: '', to: '', saved: '' });
        },

        // ── Ask the archive ──────────────────────────────────────
        setMode(mode) { this.$store.ui.setQuery({ mode: mode === 'ask' ? 'ask' : '', clip: '' }); },
        async ask(text) {
            const question = (text ?? this.question).trim();
            if (!question || this.asking) return;
            this.question = question;
            this.asking = true;
            this.answer = null;
            this.askStatus = 'Finding search words…';
            try {
                const job = await api.post('/api/ask', { question });
                const started = Date.now();
                const result = await waitForJob(job, () => {
                    const s = Math.round((Date.now() - started) / 1000);
                    this.askStatus = s < 20 ? 'Searching the transcripts…' : `Reading the matching clips… (${s} s)`;
                });
                this.answer = { question, ...result };
                this.history = [question, ...this.history.filter(h => h !== question)].slice(0, 8);
                try { localStorage.setItem(ASK_HISTORY_KEY, JSON.stringify(this.history)); } catch { /* private mode */ }
            } catch (e) {
                this.$store.ui.toast(`Couldn’t answer: ${e.message}`, 'error');
            } finally {
                this.asking = false;
                this.askStatus = '';
            }
        },
        // Answer text with [#123] citations turned into buttons that open the clip.
        answerHtml() {
            if (!this.answer?.answer) return '';
            const ids = new Set(this.answer.sources.map(s => String(s.id)));
            return escapeHtml(this.answer.answer)
                .replace(/\[#?(\d+)\]/g, (m, id) => ids.has(id)
                    ? `<button type="button" class="cite" data-clip="${id}">#${id}</button>`
                    : m)
                .replace(/\n+/g, '<br>');
        },
        citeClick(e) {
            const id = e.target.closest('[data-clip]')?.dataset.clip;
            if (id) this.$store.ui.openClip(Number(id));
        },

        clock, duration, longDate,
    };
}
