// Formatting helpers shared by all pages. Everything that ends up in x-html
// goes through escapeHtml first.

export function escapeHtml(text) {
    return String(text ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

// FTS snippet from the API marks matches with \x02 … \x03.
export function snippetHtml(snippet) {
    return escapeHtml(snippet).replace(/\x02/g, '<mark>').replace(/\x03/g, '</mark>');
}

export function highlight(text, query) {
    const safe = escapeHtml(text);
    const words = String(query || '').toLowerCase().match(/[\p{L}\p{N}]+/gu);
    if (!words) return safe;
    const re = new RegExp(`(?<![\\p{L}\\p{N}])(${words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'giu');
    return safe.replace(re, '<mark>$1</mark>');
}

// Minimal Markdown for LLM output: headings, bold, bullets. Optional linkify(line) runs
// on already-escaped text.
export function markdown(text, linkify = s => s) {
    const out = [];
    let inList = false;
    for (const raw of escapeHtml(text).split('\n')) {
        const line = linkify(raw.trim().replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>'));
        const bullet = line.match(/^[-*•]\s+(.*)/) || line.match(/^\d+\.\s+(.*)/);
        if (bullet) {
            if (!inList) { out.push('<ul class="list-disc pl-5 space-y-1 my-2">'); inList = true; }
            out.push(`<li>${bullet[1]}</li>`);
            continue;
        }
        if (inList) { out.push('</ul>'); inList = false; }
        const heading = line.match(/^#{1,6}\s+(.*)/);
        if (heading) out.push(`<h4 class="font-semibold mt-4 mb-1">${heading[1]}</h4>`);
        else if (line) out.push(`<p class="my-2">${line}</p>`);
    }
    if (inList) out.push('</ul>');
    return out.join('');
}

const pad = n => String(n).padStart(2, '0');

export function isoDate(d) {
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function today() { return isoDate(new Date()); }

export function addDays(dateStr, days) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return isoDate(new Date(y, m - 1, d + days));
}

export function parseDate(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return new Date(y, m - 1, d);
}

export function dayTitle(dateStr) {
    if (dateStr === today()) return 'Today';
    if (dateStr === addDays(today(), -1)) return 'Yesterday';
    return parseDate(dateStr).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' });
}

export function longDate(dateStr) {
    return parseDate(dateStr).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
}

// Stored timestamps are local ISO strings ("2026-11-14T16:03:08.000+01:00"):
// read the wall-clock parts directly so the display matches the house's time.
export function clock(ts, seconds = false) {
    if (!ts) return '';
    return String(ts).slice(11, seconds ? 19 : 16);
}

export function minuteOfDay(ts) {
    const s = String(ts);
    return Number(s.slice(11, 13)) * 60 + Number(s.slice(14, 16)) + Number(s.slice(17, 19)) / 60;
}

export function duration(seconds) {
    if (!seconds && seconds !== 0) return '';
    const s = Math.round(seconds);
    if (s < 60) return `${s} s`;
    return `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

export function speechTotal(seconds) {
    const m = Math.round((seconds || 0) / 60);
    if (m < 60) return `${m} min`;
    return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function segTime(seconds) {
    const s = Math.floor(seconds || 0);
    return `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

export function eta(seconds) {
    if (!seconds) return '';
    if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min`;
    const h = Math.round(seconds / 3600);
    return h < 48 ? `${h} h` : `${Math.floor(h / 24)} d ${h % 24} h`;
}

export function relative(utcLike) {
    if (!utcLike) return '';
    // SQLite CURRENT_TIMESTAMP is UTC without a zone marker.
    const s = String(utcLike);
    const d = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(s) ? s : s.replace(' ', 'T') + 'Z');
    const sec = Math.round((Date.now() - d) / 1000);
    if (sec < 60) return 'just now';
    if (sec < 3600) return `${Math.floor(sec / 60)} min ago`;
    if (sec < 86400) return `${Math.floor(sec / 3600)} h ago`;
    const days = Math.floor(sec / 86400);
    return days === 1 ? 'yesterday' : `${days} days ago`;
}

export function bytes(n) {
    if (!n) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function number(n) { return Number(n || 0).toLocaleString('en-GB'); }

// Period keys as the API uses them: 2026-11-14, 2026-W45, 2026-11, 2026.
export function periodLabel(type, key) {
    if (type === 'daily') return longDate(key);
    if (type === 'weekly') {
        const [y, w] = key.split('-W');
        return `Week ${Number(w)}, ${y}`;
    }
    if (type === 'monthly') {
        const [y, m] = key.split('-').map(Number);
        return new Date(y, m - 1, 1).toLocaleDateString('en-GB', { month: 'long', year: 'numeric' });
    }
    return key;
}

// SQLite strftime('%W'): week of year, weeks start Monday, days before the first Monday are week 00.
export function weekKey(dateStr) {
    const d = parseDate(dateStr);
    const jan1 = new Date(d.getFullYear(), 0, 1);
    const yday = Math.round((d - jan1) / 86400000);
    const wdayMon = (d.getDay() + 6) % 7;
    const week = Math.floor((yday + 7 - wdayMon) / 7);
    return `${d.getFullYear()}-W${pad(week)}`;
}
