import { api, qs } from '../api.js';
import { bytes, eta, number, relative } from '../format.js';

const TABS = [
    { id: 'status', label: 'Status' },
    { id: 'recording', label: 'Recording' },
    { id: 'transcription', label: 'Transcription' },
    { id: 'summaries', label: 'Summaries' },
    { id: 'alerts', label: 'Alerts' },
    { id: 'maintenance', label: 'Maintenance' },
];

// Which setting keys each tab saves.
const TAB_KEYS = {
    recording: ['protect_host', 'protect_verify_ssl', 'buffer_before', 'buffer_after', 'min_audio_energy', 'audio_compression_days', 'auto_sync_interval_minutes', 'auto_sync_hours', 'processing_timeout_minutes'],
    transcription: ['whisper_model', 'language', 'vad_filter'],
    summaries: ['ollama_url', 'ollama_model', 'summary_language', 'auto_summary_time'],
    alerts: ['alert_keywords', 'alert_url'],
};

export default function systemPage() {
    return {
        TABS,
        settings: {},
        saved: {},
        languages: [],
        models: [],
        installedIds: [],
        modelsError: '',
        storage: null,
        audit: [],
        syncHours: 24,
        busy: '',
        _loaded: false,

        init() {
            this.$watch('$store.ui.route', () => this.sync());
            this.sync();
        },
        get active() { return this.$store.ui.route.page === 'system'; },
        get tab() {
            const t = this.$store.ui.route.parts[0];
            return TABS.some(x => x.id === t) ? t : 'status';
        },
        sync() {
            if (!this.active) return;
            if (!this._loaded) { this._loaded = true; this.loadSettings(); }
            if (this.tab === 'status') { this.$store.ui.refreshHealth(); this.loadStorage(); }
            if (this.tab === 'transcription' && !this.models.length) this.loadModels();
            if (this.tab === 'maintenance' || this.tab === 'alerts') this.loadAudit();
        },
        setTab(t) { this.$store.ui.go('system', [t]); },

        async loadSettings() {
            try {
                const data = await api.get('/api/settings');
                this.settings = { ...data.settings };
                this.saved = { ...data.settings };
                this.languages = data.available_languages;
            } catch (e) { this.$store.ui.toast(`Couldn’t load settings: ${e.message}`, 'error'); }
        },
        dirty(tab) { return (TAB_KEYS[tab] || []).some(k => String(this.settings[k] ?? '') !== String(this.saved[k] ?? '')); },
        async save(tab) {
            const body = Object.fromEntries((TAB_KEYS[tab] || []).map(k => [k, this.settings[k] ?? '']));
            this.busy = 'save';
            try {
                const data = await api.put('/api/settings', body);
                this.settings = { ...data.settings };
                this.saved = { ...data.settings };
                this.$store.ui.toast('Settings saved. They apply to the next clip.', 'success');
                this.$store.ui.refreshHealth();
            } catch (e) { this.$store.ui.toast(`Not saved: ${e.message}`, 'error'); }
            finally { this.busy = ''; }
        },
        reset(tab) { for (const k of TAB_KEYS[tab] || []) this.settings[k] = this.saved[k]; },
        toggle(key) { this.settings[key] = this.settings[key] === 'true' ? 'false' : 'true'; },

        // ── Status ───────────────────────────────────────────────
        async loadStorage() { try { this.storage = await api.get('/api/storage'); } catch { this.storage = null; } },
        get services() {
            const h = this.$store.ui.health || {};
            return [
                { id: 'protect', name: 'UniFi Protect NVR', role: 'Delivers speech events and recordings', check: h.protect },
                { id: 'whisper', name: 'Speech recognition', role: 'speaches · turns audio into text', check: h.whisper },
                { id: 'ollama', name: 'Language model', role: 'Ollama · summaries and answers', check: h.ollama },
            ];
        },

        // ── Recording explanations ───────────────────────────────
        get clipExample() {
            const b = Number(this.settings.buffer_before || 0);
            const a = Number(this.settings.buffer_after || 0);
            return `A conversation from 16:03:00 to 16:04:00 is saved as a clip from ${this.addSec('16:03:00', -b)} to ${this.addSec('16:04:00', a)} (${60 + a + b} s).`;
        },
        addSec(hms, delta) {
            const [h, m, s] = hms.split(':').map(Number);
            const t = h * 3600 + m * 60 + s + delta;
            return [Math.floor(t / 3600), Math.floor((t % 3600) / 60), t % 60].map(n => String(n).padStart(2, '0')).join(':');
        },

        // ── Transcription models ─────────────────────────────────
        async loadModels() {
            this.modelsError = '';
            try {
                const data = await api.get('/api/settings/speaches-models');
                this.models = data.models;
                this.installedIds = data.installed_ids || [];
            } catch (e) { this.modelsError = `Couldn’t reach the speech server: ${e.message}`; }
        },
        get installedModels() {
            const ids = [...this.installedIds];
            if (this.settings.whisper_model && !ids.includes(this.settings.whisper_model)) ids.unshift(this.settings.whisper_model);
            return ids;
        },
        async testWhisper() {
            this.busy = 'whisper';
            try {
                const r = await api.post('/api/settings/test-whisper');
                this.$store.ui.toast(r.status === 'connected' ? 'Speech server answered' : r.message, r.status === 'connected' ? 'success' : 'error');
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
            finally { this.busy = ''; }
        },
        async testProtect() {
            this.busy = 'protect';
            try {
                if (this.dirty('recording')) await this.save('recording');
                const r = await api.post('/api/settings/test-protect');
                if (r.status === 'connected') this.$store.ui.toast(`Connected to ${r.nvr_name} (${r.camera_count} cameras)`, 'success');
                else this.$store.ui.toast(r.message, 'error');
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
            finally { this.busy = ''; }
        },

        // ── Alerts ───────────────────────────────────────────────
        async testAlert() {
            this.busy = 'alert';
            try {
                if (this.dirty('alerts')) await this.save('alerts');
                await api.post('/api/alerts/test');
                this.$store.ui.toast('Test alert sent', 'success');
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
            finally { this.busy = ''; }
        },
        get alertLog() { return this.audit.filter(a => a.action === 'keyword_alert'); },
        alertDetails(entry) {
            try {
                const d = JSON.parse(entry.details);
                return { id: d.id, text: `${d.keywords.join(', ')} · ${d.camera} · HTTP ${d.http_status}` };
            } catch { return { id: null, text: entry.details }; }
        },

        // ── Maintenance ──────────────────────────────────────────
        async loadAudit() { try { this.audit = (await api.get('/api/audit-log?limit=200')).entries; } catch { this.audit = []; } },
        async run(label, fn) {
            this.busy = label;
            try { await fn(); this.$store.ui.refreshStats(); this.loadAudit(); }
            catch (e) { this.$store.ui.toast(e.message, 'error'); }
            finally { this.busy = ''; }
        },
        syncNow() {
            return this.run('sync', async () => {
                const r = await api.post(`/api/sync?${qs({ hours: this.syncHours })}`);
                this.$store.ui.toast(`Checked ${r.speech_events_found} speech events, queued ${r.events_queued} new`, 'success');
            });
        },
        retryErrors() {
            return this.run('retry', async () => {
                const r = await api.post('/api/transcriptions/retry-errors');
                this.$store.ui.toast(r.message, 'success');
            });
        },
        archiveErrors() {
            return this.run('archive', async () => {
                const r = await api.post('/api/transcriptions/archive-errors');
                this.$store.ui.toast(r.message, 'success');
            });
        },
        async retranscribeAll() {
            const ok = await this.$store.ui.confirm({
                title: 'Re-transcribe everything?',
                message: `All ${number(this.$store.ui.stats.total)} clips go back into the queue with the current model. New recordings are always handled first. Text you corrected by hand is kept.`,
                action: 'Re-transcribe all',
            });
            if (ok) return this.run('retranscribe', async () => {
                const r = await api.post('/api/transcriptions/retranscribe-all', { include_errors: true, refetch: true });
                this.$store.ui.toast(r.message, 'success');
            });
        },
        async resetDatabase() {
            const ok = await this.$store.ui.confirm({
                title: 'Delete all transcripts and audio?',
                message: 'Every clip, transcript, summary and audio file is deleted. This can’t be undone.',
                action: 'Delete everything',
                danger: true,
                typed: 'RESET',
            });
            if (ok) return this.run('reset', async () => {
                await api.post('/api/database/reset');
                this.$store.ui.toast('Database reset', 'success');
            });
        },
        exportUrl(kind) { return `/api/export/${kind}`; },

        bytes, eta, number, relative,
    };
}
