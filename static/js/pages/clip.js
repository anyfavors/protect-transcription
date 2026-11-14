import { api } from '../api.js';
import { clock, duration, escapeHtml, highlight, longDate, relative, segTime } from '../format.js';

const SPEEDS = [1, 1.5, 2];

export default function clipPanel() {
    return {
        clip: null,
        loading: false,
        editing: false,
        draft: '',
        note: '',
        showOriginal: false,
        playing: false,
        time: 0,
        speed: 1,
        ws: null,
        waveError: '',
        _id: null,

        init() {
            this.$watch('$store.ui.clipId', () => this.sync());
            this.sync();
            window.addEventListener('clip-changed', (e) => {
                if (e.detail.id === this._id && !this.editing) this.load(this._id, { keepPlayer: true });
            });
            window.addEventListener('theme-changed', () => this.ws && this.ws.setOptions(this.waveColors()));
        },

        get open() { return this.$store.ui.clipId !== null; },

        sync() {
            const id = this.$store.ui.clipId;
            if (id === this._id) return;
            this._id = id;
            this.destroyPlayer();
            this.clip = null;
            this.editing = false;
            this.showOriginal = false;
            if (id) this.load(id);
        },

        async load(id, { keepPlayer = false } = {}) {
            this.loading = !keepPlayer;
            try {
                const clip = await api.get(`/api/transcriptions/${id}`);
                if (id !== this._id) return;
                const audioChanged = !this.clip || this.clip.audio_file !== clip.audio_file;
                this.clip = clip;
                this.note = clip.note || '';
                if (!keepPlayer || audioChanged) this.$nextTick(() => this.createPlayer());
            } catch (e) {
                this.$store.ui.toast(`Couldn’t open clip: ${e.message}`, 'error');
                this.$store.ui.closeClip();
            } finally {
                this.loading = false;
            }
        },

        // ── Player ───────────────────────────────────────────────
        waveColors() {
            const css = getComputedStyle(document.documentElement);
            return {
                waveColor: css.getPropertyValue('--wave').trim() || '#94a3b8',
                progressColor: css.getPropertyValue('--wave-progress').trim() || '#0d9488',
                cursorColor: css.getPropertyValue('--wave-cursor').trim() || '#ea580c',
            };
        },
        createPlayer() {
            this.destroyPlayer();
            if (!this.clip?.audio_file || !window.WaveSurfer || !this.$refs.wave) return;
            this.waveError = '';
            this.ws = window.WaveSurfer.create({
                container: this.$refs.wave,
                url: `/audio/${encodeURIComponent(this.clip.audio_file)}`,
                height: 64,
                barWidth: 2,
                barGap: 1,
                barRadius: 2,
                normalize: true,
                ...this.waveColors(),
            });
            this.ws.on('timeupdate', (t) => { this.time = t; this.followSegment(); });
            this.ws.on('play', () => { this.playing = true; });
            this.ws.on('pause', () => { this.playing = false; });
            this.ws.on('finish', () => { this.playing = false; });
            this.ws.on('ready', () => this.ws.setPlaybackRate(this.speed));
            this.ws.on('error', () => { this.waveError = 'The audio file couldn’t be loaded.'; });
        },
        destroyPlayer() {
            if (this.ws) { this.ws.destroy(); this.ws = null; }
            this.playing = false;
            this.time = 0;
        },
        toggle() { this.ws?.playPause(); },
        seek(seconds) { if (this.ws) { this.ws.setTime(seconds); this.ws.play(); } },
        cycleSpeed() {
            this.speed = SPEEDS[(SPEEDS.indexOf(this.speed) + 1) % SPEEDS.length];
            this.ws?.setPlaybackRate(this.speed);
        },

        // ── Transcript ───────────────────────────────────────────
        get segments() {
            if (!this.clip || this.clip.edited_at || this.showOriginal) return [];
            return this.clip.segments || [];
        },
        get text() {
            if (!this.clip) return '';
            return this.showOriginal ? this.clip.original_transcription : this.clip.transcription;
        },
        isActive(seg) { return this.playing || this.time ? this.time >= seg.start && this.time < (seg.end || seg.start + 5) : false; },
        followSegment() {
            const i = this.segments.findIndex(s => this.isActive(s));
            if (i >= 0) document.getElementById(`seg-${i}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        },
        textHtml(t) { return highlight(t, this.$store.ui.route.page === 'search' ? this.$store.ui.route.query.q : ''); },

        startEdit() { this.draft = this.clip.transcription || ''; this.editing = true; this.$nextTick(() => this.$refs.editor?.focus()); },
        async saveEdit() {
            try {
                this.clip = await api.patch(`/api/transcriptions/${this.clip.id}`, { transcription: this.draft });
                this.editing = false;
                this.$store.ui.toast('Text saved. Summaries for this day will update.', 'success');
                window.dispatchEvent(new CustomEvent('clip-changed', { detail: { id: this.clip.id, timestamp: this.clip.timestamp, local: true } }));
            } catch (e) { this.$store.ui.toast(`Couldn’t save: ${e.message}`, 'error'); }
        },
        async revert() {
            try {
                this.clip = await api.patch(`/api/transcriptions/${this.clip.id}`, { revert: true });
                this.showOriginal = false;
                this.$store.ui.toast('Restored the machine transcript', 'success');
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
        },
        async toggleStar() {
            try {
                this.clip = await api.patch(`/api/transcriptions/${this.clip.id}`, { starred: !this.clip.starred });
                window.dispatchEvent(new CustomEvent('clip-changed', { detail: { id: this.clip.id, timestamp: this.clip.timestamp, local: true } }));
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
        },
        async saveNote() {
            if ((this.clip.note || '') === this.note.trim()) return;
            try {
                this.clip = await api.patch(`/api/transcriptions/${this.clip.id}`, { note: this.note });
                this.$store.ui.toast('Note saved', 'success');
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
        },
        async copy() {
            try { await navigator.clipboard.writeText(this.text || ''); this.$store.ui.toast('Copied', 'success'); }
            catch { this.$store.ui.toast('Your browser blocked copying', 'error'); }
        },
        async retranscribe() {
            try {
                await api.post(`/api/transcriptions/${this.clip.id}/retry`);
                this.$store.ui.toast('Queued for re-transcription. It goes ahead of the backlog.', 'success');
                this.load(this.clip.id, { keepPlayer: true });
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
        },
        async remove() {
            const ok = await this.$store.ui.confirm({
                title: 'Delete this clip?',
                message: 'The transcript and the audio file are deleted for good.',
                action: 'Delete clip',
                danger: true,
            });
            if (!ok) return;
            try {
                const { id, timestamp } = this.clip;
                await api.del(`/api/transcriptions/${id}`);
                this.$store.ui.closeClip();
                this.$store.ui.toast('Clip deleted', 'success');
                window.dispatchEvent(new CustomEvent('clip-changed', { detail: { id, timestamp, local: true } }));
            } catch (e) { this.$store.ui.toast(e.message, 'error'); }
        },
        statusNote() {
            const c = this.clip;
            if (!c) return '';
            if (c.status === 'pending') return 'Queued for transcription. The current text (if any) stays until it is replaced.';
            if (c.status === 'processing') return 'Being transcribed right now.';
            if (c.status === 'filtered') return c.transcription || 'No speech found in this clip.';
            if (c.status === 'error' || c.status === 'archived') return c.transcription || 'Transcription failed.';
            return '';
        },

        clock, duration, escapeHtml, longDate, relative, segTime,
    };
}
