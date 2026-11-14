// Entry point: registers the store, page components and helpers before Alpine starts.
// Loaded as a module ahead of the deferred Alpine script, so 'alpine:init' fires after this runs.

import { icon } from './icons.js';
import { createStore } from './store.js';
import clipPanel from './pages/clip.js';
import dayPage from './pages/day.js';
import insightsPage from './pages/insights.js';
import searchPage from './pages/search.js';
import summariesPage from './pages/summaries.js';
import systemPage from './pages/system.js';

document.addEventListener('alpine:init', () => {
    const Alpine = window.Alpine;
    Alpine.store('ui', createStore());
    Alpine.magic('icon', () => icon);
    Alpine.data('dayPage', dayPage);
    Alpine.data('clipPanel', clipPanel);
    Alpine.data('searchPage', searchPage);
    Alpine.data('summariesPage', summariesPage);
    Alpine.data('insightsPage', insightsPage);
    Alpine.data('systemPage', systemPage);
});

// Keyboard shortcuts. Ignored while typing in a field.
document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const el = e.target instanceof Element ? e.target : document.body;
    if (el.closest('input, textarea, select, [contenteditable="true"]')) {
        if (e.key === 'Escape') el.blur();
        return;
    }
    const ui = window.Alpine?.store('ui');
    if (!ui) return;
    const send = (name) => window.dispatchEvent(new CustomEvent(`shortcut:${name}`));
    switch (e.key) {
        case '/': e.preventDefault(); ui.go('search'); setTimeout(() => document.getElementById('search-input')?.focus(), 50); break;
        case 'j': send('next'); break;
        case 'k': send('prev'); break;
        case 'ArrowRight': if (ui.route.page === 'day') send('next-day'); break;
        case 'ArrowLeft': if (ui.route.page === 'day') send('prev-day'); break;
        case 't': ui.go('day'); break;
        case ' ': if (ui.clipId) { e.preventDefault(); send('play'); } break;
        case 's': if (ui.clipId) send('star'); break;
        case 'e': if (ui.clipId) { e.preventDefault(); send('edit'); } break;
        case 'Escape': if (ui.confirmBox) ui.closeConfirm(false); else if (ui.clipId) ui.closeClip(); break;
        case '?': send('help'); break;
        default: return;
    }
});
