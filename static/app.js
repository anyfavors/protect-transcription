function transcribeApp() {
    return {
        loading: false,
        transcriptions: [],
        cameras: [],
        dates: [],
        stats: { total: 0, completed: 0, processing: 0, queued: 0, errors: 0, filtered: 0, today: 0, rate_per_hour: 0, eta_seconds: null },
        filters: { search: '', camera: '', date: '', status: '' },
        pagination: { page: 1, per_page: 20, total: 0, pages: 0 },
        confirmModal: { show: false, action: '', title: '', message: '', item: null, loading: false },
        // Settings
        showSettings: false,
        settingsTab: 'transcription',
        settingsLoading: false,
        settingsSaved: false,
        whisperStatus: null,
        protectStatus: null,
        // Speaches model management
        speachesModels: [],
        speachesModelsLoading: false,
        speachesModelsError: null,
        downloadConfirm: { show: false, model: null },
        // Sync
        syncLoading: false,
        syncHours: '24',
        syncResult: null,
        // Reset modal
        showResetConfirm: false,
        resetConfirmText: '',
        resetLoading: false,
        resetResult: null,
        retryingAllErrors: false,
        retranscribing: false,
        retranscribeResult: null,
        // Summaries modal
        showSummaries: false,
        summaryPeriod: 'daily',
        summaryItems: [],
        summariesLoading: false,
        // Analytics modal
        showAnalytics: false,
        analyticsTab: 'hourly',
        analyticsLoading: false,
        analyticsData: null,
        analyticsDays: 30,
        // Storage stats
        storageStats: null,
        // Dark mode
        darkMode: document.documentElement.classList.contains('dark'),
        // Toast stack
        toasts: [],
        _toastId: 0,
        // Audio playback
        activeAudio: { id: null, time: 0 },
        // Expanded segment panels
        expandedSegments: {},
        // Active long operations
        activeOps: [],
        _opsTimerId: null,
        // Onboarding status
        onboarding: { checked: false, nvrOk: null, whisperOk: null, loading: false },
        // Bulk selection
        selectedIds: [],
        // WebSocket
        _ws: null,
        _wsRetryDelay: 1000,
        settings: {
            whisper_model: 'Systran/faster-whisper-large-v3',
            language: 'da',
            buffer_before: '5',
            buffer_after: '10',
            vad_filter: 'true',
            protect_host: '',
            ollama_url: '',
            ollama_model: '',
            summary_language: 'English',
            min_audio_energy: '0.005',
            audio_compression_days: '7',
        },
        availableLanguages: [],

        _pollTimer: null,

        async init() {
            await this.loadAll();
            await this.loadSettings();
            this.checkOnboarding();
            this.schedulePoll();
            this._connectWebSocket();
        },

        // ── WebSocket ──────────────────────────────────────────────
        _connectWebSocket() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${proto}//${location.host}/ws`;
            try {
                this._ws = new WebSocket(url);
                this._ws.onopen = () => { this._wsRetryDelay = 1000; };
                this._ws.onmessage = (e) => {
                    try {
                        const msg = JSON.parse(e.data);
                        if (msg.type === 'transcription_update') this._queueQuietRefresh();
                    } catch { /* ignore */ }
                };
                this._ws.onclose = () => {
                    setTimeout(() => this._connectWebSocket(), this._wsRetryDelay);
                    this._wsRetryDelay = Math.min(this._wsRetryDelay * 2, 30000);
                };
                this._ws.onerror = () => { this._ws?.close(); };
            } catch { /* WebSocket not available, rely on polling */ }
        },

        // Background refresh: at most one every 3 s, no spinner, selection kept.
        _queueQuietRefresh() {
            if (this._quietTimer) return;
            this._quietTimer = setTimeout(async () => {
                this._quietTimer = null;
                await this.loadAll({ quiet: true });
            }, 3000);
        },

        schedulePoll() {
            if (this._pollTimer) clearTimeout(this._pollTimer);
            // The WebSocket pushes changes; polling is only a fallback.
            const wsOpen = this._ws && this._ws.readyState === WebSocket.OPEN;
            const busy = this.stats.processing > 0 || this.stats.queued > 0;
            const interval = !wsOpen && busy ? 10000 : 60000;
            this._pollTimer = setTimeout(async () => {
                await this.loadAll({ quiet: true });
                this.schedulePoll();
            }, interval);
        },

        async loadAll({ quiet = false } = {}) {
            await Promise.all([
                this.loadStats(),
                this.loadCameras(),
                this.loadDates(),
                this.loadTranscriptions({ quiet })
            ]);
        },

        async refresh() { await this.loadAll(); this.schedulePoll(); },

        async loadStats() {
            try { this.stats = await (await fetch('/api/stats')).json(); }
            catch (e) { console.error(e); }
        },

        async loadCameras() {
            try { this.cameras = (await (await fetch('/api/cameras')).json()).cameras; }
            catch (e) { console.error(e); }
        },

        async loadDates() {
            try { this.dates = (await (await fetch('/api/dates')).json()).dates; }
            catch (e) { console.error(e); }
        },

        async loadTranscriptions({ quiet = false } = {}) {
            if (!quiet) { this.loading = true; this.selectedIds = []; }
            try {
                const p = new URLSearchParams({ page: this.pagination.page, per_page: this.pagination.per_page });
                if (this.filters.search) p.append('search', this.filters.search);
                if (this.filters.camera) p.append('camera', this.filters.camera);
                if (this.filters.date)   p.append('date', this.filters.date);
                if (this.filters.status) p.append('status', this.filters.status);
                const data = await (await fetch(`/api/transcriptions?${p}`)).json();
                this.transcriptions = data.transcriptions;
                if (quiet) {
                    const ids = new Set(data.transcriptions.map(t => t.id));
                    this.selectedIds = this.selectedIds.filter(id => ids.has(id));
                }
                this.pagination.total = data.total;
                this.pagination.pages = data.pages;
            } catch (e) { console.error(e); }
            finally { this.loading = false; }
        },

        get groupedTranscriptions() {
            const groups = {};
            for (const item of this.transcriptions) {
                const date = item.timestamp ? item.timestamp.split('T')[0] : 'unknown';
                if (!groups[date]) groups[date] = [];
                groups[date].push(item);
            }
            return groups;
        },

        formatDate(dateStr) {
            if (!dateStr || dateStr === 'unknown') return 'Unknown date';
            const date = new Date(dateStr);
            const today = new Date();
            const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1);
            if (date.toDateString() === today.toDateString()) return 'Today';
            if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
            return date.toLocaleDateString('en-GB', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
        },

        formatTime(timestamp) {
            if (!timestamp) return '';
            return new Date(timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
        },

        formatRelativeTime(timestamp) {
            if (!timestamp) return '';
            const diffSec = Math.floor((new Date() - new Date(timestamp)) / 1000);
            if (diffSec < 60) return 'just now';
            if (diffSec < 3600) return `${Math.floor(diffSec/60)}m ago`;
            if (diffSec < 86400) return `${Math.floor(diffSec/3600)}h ago`;
            const days = Math.floor(diffSec/86400);
            if (days < 7) return `${days}d ago`;
            return this.formatTime(timestamp);
        },

        formatFullTimestamp(timestamp) {
            if (!timestamp) return '';
            return new Date(timestamp).toLocaleString('en-GB', {
                weekday: 'short', year: 'numeric', month: 'short',
                day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
        },

        formatDuration(seconds) {
            if (!seconds) return '';
            const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
            return m > 0 ? `${m}:${s.toString().padStart(2,'0')}` : `${s}s`;
        },

        formatSegmentTime(seconds) {
            if (seconds === undefined || seconds === null) return '0:00';
            const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
            return `${m}:${s.toString().padStart(2,'0')}`;
        },

        seekAudio(itemId, seconds) {
            const audio = document.getElementById('audio-' + itemId);
            if (audio) { audio.currentTime = seconds; audio.play(); this.activeAudio = { id: itemId, time: seconds }; }
        },

        updateActiveSegment(itemId, currentTime) {
            this.activeAudio = { id: itemId, time: currentTime };
            const item = this.transcriptions.find(t => t.id === itemId);
            if (item && item.segments) {
                for (let i = 0; i < item.segments.length; i++) {
                    const seg = item.segments[i];
                    if (currentTime >= seg.start && currentTime < (seg.end || seg.start + 5)) {
                        document.getElementById(`seg-${itemId}-${i}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                        break;
                    }
                }
            }
        },

        isSegmentActive(itemId, start, end) {
            if (this.activeAudio.id !== itemId) return false;
            return this.activeAudio.time >= start && this.activeAudio.time < (end || start + 5);
        },

        async copyToClipboard(text) {
            try { await navigator.clipboard.writeText(text); this.showToast('Copied!', 'success'); }
            catch { this.showToast('Copy failed', 'error'); }
        },

        showToast(message, type = 'info') {
            const id = ++this._toastId;
            this.toasts.push({ id, message, type, show: true });
            if (this.toasts.length > 5) this.toasts.splice(0, this.toasts.length - 5);
            setTimeout(() => { this.dismissToast(id); }, 3000);
        },

        dismissToast(id) {
            const t = this.toasts.find(t => t.id === id);
            if (t) t.show = false;
            setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 200);
        },

        startOp(id, label) {
            this.activeOps = this.activeOps.filter(o => o.id !== id);
            this.activeOps.push({ id, label, startedAt: Date.now() });
            if (!this._opsTimerId) {
                this._opsTimerId = setInterval(() => { this.activeOps = [...this.activeOps]; }, 1000);
            }
        },

        endOp(id) {
            this.activeOps = this.activeOps.filter(o => o.id !== id);
            if (this.activeOps.length === 0 && this._opsTimerId) {
                clearInterval(this._opsTimerId);
                this._opsTimerId = null;
            }
        },

        formatElapsed(startedAt) {
            const s = Math.floor((Date.now() - startedAt) / 1000);
            if (s < 60) return s + 's';
            return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
        },

        escapeHtml(text) {
            return String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
        },

        // Result is used with x-html: escape first, then add <mark>.
        highlightSearch(text) {
            const safe = this.escapeHtml(text);
            if (!safe || !this.filters.search) return safe;
            const term = this.escapeHtml(this.filters.search).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            return safe.replace(new RegExp(`(${term})`, 'gi'), '<mark>$1</mark>');
        },

        // Minimal Markdown for LLM summaries: headings, bold, bullet lists. Escapes first.
        renderMarkdown(text) {
            const out = [];
            let inList = false;
            for (const raw of this.escapeHtml(text).split('\n')) {
                const line = raw.trim().replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                const bullet = line.match(/^[-*•]\s+(.*)/);
                if (bullet) {
                    if (!inList) { out.push('<ul class="list-disc pl-5 space-y-1">'); inList = true; }
                    out.push(`<li>${bullet[1]}</li>`);
                    continue;
                }
                if (inList) { out.push('</ul>'); inList = false; }
                const heading = line.match(/^#{1,6}\s+(.*)/);
                if (heading) out.push(`<p class="font-semibold mt-3">${heading[1]}</p>`);
                else if (line) out.push(`<p class="mt-2">${line}</p>`);
            }
            if (inList) out.push('</ul>');
            return out.join('');
        },

        localToday() {
            const d = new Date();
            return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
        },

        formatEta(seconds) {
            if (!seconds) return '';
            if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min`;
            const h = Math.round(seconds / 3600);
            return h < 48 ? `${h} h` : `${Math.floor(h / 24)} d ${h % 24} h`;
        },

        setQuickFilter(filter) {
            this.filters.date = '';
            this.filters.status = '';
            if (filter === 'today') this.filters.date = this.localToday();
            else if (filter === 'errors') this.filters.status = 'error';
            else if (filter === 'processing') this.filters.status = 'processing';
            else if (filter === 'queued') this.filters.status = 'pending';
            this.pagination.page = 1;
            this.loadTranscriptions();
        },

        prevPage() { if (this.pagination.page > 1) { this.pagination.page--; this.loadTranscriptions(); } },
        nextPage() { if (this.pagination.page < this.pagination.pages) { this.pagination.page++; this.loadTranscriptions(); } },

        toggleDarkMode() {
            this.darkMode = !this.darkMode;
            document.documentElement.classList.toggle('dark', this.darkMode);
            localStorage.setItem('theme', this.darkMode ? 'dark' : 'light');
        },

        toggleSelect(id) {
            const idx = this.selectedIds.indexOf(id);
            if (idx === -1) this.selectedIds.push(id);
            else this.selectedIds.splice(idx, 1);
        },

        toggleSelectAll() {
            if (this.selectedIds.length === this.transcriptions.length) {
                this.selectedIds = [];
            } else {
                this.selectedIds = this.transcriptions.map(t => t.id);
            }
        },

        clearSelection() { this.selectedIds = []; },

        async bulkDelete() {
            if (!this.selectedIds.length) return;
            this.startOp('bulk-delete', `Deleting ${this.selectedIds.length} transcriptions`);
            try {
                const r = await fetch('/api/transcriptions/bulk-delete', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids: [...this.selectedIds] })
                });
                if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `HTTP ${r.status}`);
                const data = await r.json();
                this.showToast(data.message, 'success');
                this.selectedIds = [];
                await this.loadAll();
            } catch (e) { this.showToast('Bulk delete failed: ' + e.message, 'error'); }
            finally { this.endOp('bulk-delete'); }
        },

        async bulkRetry() {
            if (!this.selectedIds.length) return;
            this.startOp('bulk-retry', `Retrying ${this.selectedIds.length} transcriptions`);
            try {
                const r = await fetch('/api/transcriptions/bulk-retry', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids: [...this.selectedIds] })
                });
                if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `HTTP ${r.status}`);
                const data = await r.json();
                this.showToast(data.message, 'success');
                this.selectedIds = [];
                await this.loadAll();
            } catch (e) { this.showToast('Bulk retry failed: ' + e.message, 'error'); }
            finally { this.endOp('bulk-retry'); }
        },

        confirmRetry(item) {
            this.confirmModal = { show: true, action: 'retry', title: 'Retry Transcription?',
                message: 'This will re-fetch the audio and create a new transcription.', item, loading: false };
        },

        confirmDelete(item) {
            this.confirmModal = { show: true, action: 'delete', title: 'Delete Transcription?',
                message: 'This will permanently delete the transcription and audio file.', item, loading: false };
        },

        async executeAction() {
            if (!this.confirmModal.item) return;
            this.confirmModal.loading = true;
            const { item, action } = this.confirmModal;
            try {
                if (action === 'delete') {
                    const r = await fetch(`/api/transcriptions/${item.id}`, { method: 'DELETE' });
                    if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `Delete failed (${r.status})`);
                    this.transcriptions = this.transcriptions.filter(t => t.id !== item.id);
                    this.showToast('Deleted', 'success');
                } else if (action === 'retry') {
                    const r = await fetch(`/api/transcriptions/${item.id}/retry`, { method: 'POST' });
                    if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `Retry failed (${r.status})`);
                    this.showToast('Retry queued', 'success');
                }
                this.confirmModal.show = false;
                await this.loadAll();
            } catch (e) { this.showToast(e.message, 'error'); }
            finally { this.confirmModal.loading = false; }
        },

        async retryAllErrors() {
            this.retryingAllErrors = true;
            try {
                const r = await fetch('/api/transcriptions/retry-errors', { method: 'POST' });
                if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `Failed (${r.status})`);
                this.showToast((await r.json()).message, 'success');
                await this.loadAll();
            } catch (e) { this.showToast(e.message, 'error'); }
            finally { this.retryingAllErrors = false; }
        },

        async retranscribeAll(includeErrors) {
            this.retranscribing = true; this.retranscribeResult = null;
            this.startOp('retranscribe', 'Re-transcribing all');
            try {
                const r = await fetch('/api/transcriptions/retranscribe-all', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ include_errors: includeErrors })
                });
                if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `HTTP ${r.status}`);
                const data = await r.json();
                this.retranscribeResult = data.message;
                this.showToast(data.message, 'success');
                await this.loadAll();
            } catch (e) { this.showToast('Failed: ' + e.message, 'error'); }
            finally { this.retranscribing = false; this.endOp('retranscribe'); }
        },

        openSummaries() { this.showSummaries = true; this.loadSummaries(); },

        async loadSummaries() {
            this.summariesLoading = true; this.summaryItems = [];
            try {
                const data = await (await fetch(`/api/summaries?period=${this.summaryPeriod}`)).json();
                this.summaryItems = data.items.map(i => ({ ...i, generating: false }));
            } catch (e) { this.showToast('Failed to load summaries: ' + e.message, 'error'); }
            finally { this.summariesLoading = false; }
        },

        async generateSummary(item) {
            item.generating = true;
            try {
                const r = await fetch('/api/summaries/generate', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ period_type: this.summaryPeriod, period_key: item.period_key })
                });
                if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || `HTTP ${r.status}`);
                const data = await r.json();
                item.summary = data.summary;
                item.generated_at = new Date().toISOString().replace('T',' ').slice(0,19);
                item.stale = false;
                this.showToast('Summary generated', 'success');
            } catch (e) { this.showToast('Failed: ' + e.message, 'error'); }
            finally { item.generating = false; }
        },

        async loadSettings() {
            try {
                const data = await (await fetch('/api/settings')).json();
                this.settings = data.settings;
                this.availableLanguages = data.available_languages;
            } catch (e) { console.error(e); }
        },

        async checkOnboarding() {
            if (this.stats.total > 0 || this.onboarding.checked) return;
            this.onboarding.loading = true;
            try {
                const [nvrRes, whisperRes] = await Promise.allSettled([
                    fetch('/api/settings/test-protect', { method: 'POST' }).then(r => r.json()),
                    fetch('/api/settings/test-whisper', { method: 'POST' }).then(r => r.json()),
                ]);
                this.onboarding.nvrOk = nvrRes.status === 'fulfilled' && nvrRes.value.status === 'connected';
                this.onboarding.whisperOk = whisperRes.status === 'fulfilled' && whisperRes.value.status === 'connected';
            } catch { /* ignore */ }
            finally { this.onboarding.loading = false; this.onboarding.checked = true; }
        },

        openSettings() {
            this.showSettings = true;
            if (this.speachesModels.length === 0) this.loadSpeachesModels();
        },

        async loadSpeachesModels() {
            this.speachesModelsLoading = true;
            this.speachesModelsError = null;
            try {
                const data = await (await fetch('/api/settings/speaches-models')).json();
                this.speachesModels = data.models.map(m => ({ ...m, downloading: false }));
            } catch (e) {
                this.speachesModelsError = 'Could not reach speaches server';
            } finally {
                this.speachesModelsLoading = false;
            }
        },

        confirmDownloadModel(model) {
            this.downloadConfirm = { show: true, model };
        },

        async downloadModel() {
            const model = this.downloadConfirm.model;
            if (!model) return;
            this.downloadConfirm = { show: false, model: null };

            const m = this.speachesModels.find(x => x.id === model.id);
            if (m) m.downloading = true;
            this.startOp('model-dl', `Downloading model ${model.id}`);
            try {
                const r = await fetch(`/api/settings/speaches-models/${model.id}`, { method: 'POST' });
                if (r.ok) {
                    if (m) { m.installed = true; m.downloading = false; }
                    this.settings.whisper_model = model.id;
                    this.showToast(`Model downloaded: ${model.id}`, 'success');
                } else {
                    const err = await r.json();
                    this.showToast(`Download failed: ${err.detail || r.status}`, 'error');
                    if (m) m.downloading = false;
                }
            } catch (e) {
                this.showToast(`Download failed: ${e.message}`, 'error');
                if (m) m.downloading = false;
            } finally {
                this.endOp('model-dl');
            }
        },

        async saveSettings() {
            this.settingsLoading = true; this.settingsSaved = false;
            try {
                const r = await fetch('/api/settings', {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.settings)
                });
                if (!r.ok) throw new Error((await r.json()).detail || 'Failed to save');
                this.settings = (await r.json()).settings;
                this.settingsSaved = true;
                this.showToast('Settings saved', 'success');
                setTimeout(() => { this.settingsSaved = false; }, 3000);
            } catch (e) { this.showToast('Save failed: ' + e.message, 'error'); }
            finally { this.settingsLoading = false; }
        },

        async testWhisper() {
            this.whisperStatus = { loading: true };
            try { this.whisperStatus = await (await fetch('/api/settings/test-whisper', { method: 'POST' })).json(); }
            catch (e) { this.whisperStatus = { status: 'error', message: e.message }; }
        },

        async testProtect() {
            this.protectStatus = { loading: true };
            try {
                await this.saveSettings();
                this.protectStatus = await (await fetch('/api/settings/test-protect', { method: 'POST' })).json();
            } catch (e) { this.protectStatus = { status: 'error', message: e.message }; }
        },

        async runSync() {
            this.syncLoading = true; this.syncResult = null;
            this.startOp('sync', 'Syncing from NVR');
            try {
                const r = await fetch(`/api/sync?hours=${this.syncHours}`, { method: 'POST' });
                const data = await r.json();
                this.syncResult = r.ok ? data : { status: 'error', message: data.detail || 'Sync failed' };
                if (r.ok && data.events_queued > 0) {
                    this.showToast(`Queued ${data.events_queued} events`, 'success');
                    setTimeout(() => this.loadAll(), 2000);
                }
            } catch (e) { this.syncResult = { status: 'error', message: e.message }; }
            finally { this.syncLoading = false; this.endOp('sync'); }
        },

        // ── Analytics ────────────────────────────────────────────
        openAnalytics() {
            this.showAnalytics = true;
            this.loadAnalytics();
        },

        async loadAnalytics() {
            this.analyticsLoading = true;
            this.analyticsData = null;
            try {
                const tab = this.analyticsTab;
                let url;
                if (tab === 'hourly') url = `/api/analytics/hourly?days=${this.analyticsDays}`;
                else if (tab === 'daily') url = `/api/analytics/daily?days=${this.analyticsDays}`;
                else if (tab === 'cameras') url = '/api/analytics/cameras';
                else if (tab === 'languages') url = '/api/analytics/languages';
                this.analyticsData = await (await fetch(url)).json();
            } catch (e) { this.showToast('Failed to load analytics: ' + e.message, 'error'); }
            finally { this.analyticsLoading = false; }
        },

        availableModelsForLang() {
            const lang = this.settings.language;
            return this.speachesModels.filter(m =>
                !m.installed && (
                    !m.language || m.language.length === 0 || m.language.includes(lang)
                )
            );
        },

        maxOf(arr, key) {
            return Math.max(1, ...arr.map(i => i[key] || 0));
        },

        barPercent(val, max) {
            return max > 0 ? Math.round((val / max) * 100) : 0;
        },

        // ── Storage stats ────────────────────────────────────────
        async loadStorageStats() {
            try {
                this.storageStats = await (await fetch('/api/storage')).json();
            } catch { /* ignore */ }
        },

        formatBytes(bytes) {
            if (!bytes || bytes === 0) return '0 B';
            const units = ['B', 'KB', 'MB', 'GB'];
            let i = 0;
            let b = bytes;
            while (b >= 1024 && i < units.length - 1) { b /= 1024; i++; }
            return b.toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
        },

        // ── Export ───────────────────────────────────────────────
        exportCsv() {
            const p = this._exportParams();
            window.open(`/api/export/csv?${p}`, '_blank');
        },

        exportJson() {
            const p = this._exportParams();
            window.open(`/api/export/json?${p}`, '_blank');
        },

        exportSrtZip() {
            const p = this._exportParams();
            window.open(`/api/export/srt?${p}`, '_blank');
        },

        _exportParams() {
            const p = new URLSearchParams();
            if (this.filters.camera) p.append('camera', this.filters.camera);
            if (this.filters.date) { p.append('from', this.filters.date); p.append('to', this.filters.date); }
            if (this.filters.status) p.append('status', this.filters.status);
            if (this.filters.search) p.append('search', this.filters.search);
            return p.toString();
        },

        async performReset() {
            if (this.resetConfirmText !== 'RESET') return;
            this.resetLoading = true; this.resetResult = null;
            try {
                const r = await fetch('/api/database/reset', { method: 'POST' });
                const data = await r.json();
                this.resetResult = data;
                if (r.ok) {
                    this.showToast('Database reset', 'success');
                    setTimeout(() => {
                        this.showResetConfirm = false; this.showSettings = false;
                        this.resetConfirmText = ''; this.resetResult = null;
                        this.loadAll();
                    }, 2000);
                }
            } catch (e) { this.resetResult = { status: 'error', message: e.message }; }
            finally { this.resetLoading = false; }
        }
    }
}
