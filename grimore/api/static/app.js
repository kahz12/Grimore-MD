// =================================================================
// Grimore — Codex front-end controller.
// Vanilla JS. Streams answers via SSE, parses [[citation]] markers
// into clickable glyphs, opens notes in a side drawer. No build step.
// =================================================================

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
    health: null,
    categories: [],
    busy: false,
    lastSearchHits: [],
};

// ── Sanitisation helpers ─────────────────────────────────────────

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

// Convert raw streamed text into HTML with [[citations]] turned into
// clickable spans. Always called on full collected text so the parser
// sees complete citation tokens.
const CITE_RE = /\[\[([^\[\]]+)\]\]/g;

function renderCitedHtml(text) {
    const escaped = escapeHtml(text);
    return escaped.replace(CITE_RE, (_, label) =>
        `<span class="cite" data-label="${escapeHtml(label)}">${escapeHtml(label)}</span>`
    );
}

function renderProse(raw) {
    const paragraphs = raw.split(/\n{2,}/).filter(p => p.trim());
    if (paragraphs.length === 0) return '';
    return paragraphs.map(p =>
        `<p>${renderCitedHtml(p).replace(/\n/g, '<br>')}</p>`
    ).join('');
}

// Lower-roman for marginalia indices: i, ii, iii…
function romanize(n) {
    const map = ['', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'];
    return map[n] || String(n);
}

// ── Health (the candle) ─────────────────────────────────────────

async function refreshHealth() {
    const el = $('#health');
    try {
        const r = await fetch('/api/health', { cache: 'no-store' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        state.health = data;
        if (data.ok) {
            el.classList.add('is-ok');
            el.classList.remove('is-err');
            $('.health__label', el).textContent = `v${data.version} · awake`;
        } else {
            throw new Error('unhealthy');
        }
    } catch (e) {
        el.classList.add('is-err');
        el.classList.remove('is-ok');
        $('.health__label', el).textContent = 'unreachable';
    }
}

// ── Categories (the verso catalogue) ────────────────────────────

async function loadCategories() {
    try {
        const r = await fetch('/api/categories');
        const data = await r.json();
        state.categories = data.categories || [];
        renderCategories();
    } catch (e) {
        // Non-fatal — the rest of the UI keeps working.
        const ul = $('#categories');
        ul.innerHTML = '<li class="categories__loading">— catalogue unavailable —</li>';
    }
}

function renderCategories() {
    const ul = $('#categories');
    ul.innerHTML = '';
    if (state.categories.length === 0) {
        const li = document.createElement('li');
        li.className = 'categories__loading';
        li.textContent = '— no categories indexed —';
        ul.appendChild(li);
        return;
    }
    for (const row of state.categories) {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.textContent = row.category || '— uncategorised —';
        const count = document.createElement('span');
        count.className = 'count';
        count.textContent = String(row.count);
        li.append(name, count);
        li.addEventListener('click', () => {
            const q = $('#question');
            q.value = `What does the codex contain on ${row.category}?`;
            q.focus();
            q.setSelectionRange(q.value.length, q.value.length);
        });
        ul.appendChild(li);
    }
}

// ── Marginalia (right column glosses) ───────────────────────────

function renderMarginalia(hits) {
    const wrap = $('#marginalia');
    wrap.innerHTML = '';
    if (!hits || hits.length === 0) {
        wrap.innerHTML = `
            <div class="marginalia__placeholder">
                <span class="marginalia__symbol" aria-hidden="true">⚜</span>
                <p>Marginal annotations appear here when you invoke a question.</p>
            </div>`;
        return;
    }
    const head = document.createElement('h3');
    head.className = 'rubric rubric--sm';
    head.textContent = 'Marginal Gloss';
    head.style.marginBottom = '24px';
    wrap.appendChild(head);

    hits.slice(0, 4).forEach((h, i) => {
        const div = document.createElement('div');
        div.className = 'marg-entry';
        div.innerHTML = `
            <div class="marg-entry__index">${escapeHtml(romanize(i + 1))} · note ${escapeHtml(String(h.note_id))}</div>
            <div class="marg-entry__title">${escapeHtml(h.title || `#${h.note_id}`)}</div>
            <div class="marg-entry__snippet">${escapeHtml(h.snippet || '')}</div>
            <span class="marg-entry__score">score · ${Number(h.score || 0).toFixed(3)}</span>
        `;
        div.addEventListener('click', () => openNote(h.note_id));
        wrap.appendChild(div);
    });
}

// ── Hits (centre column, full list) ─────────────────────────────

function renderHits(hits) {
    const wrap = $('#results');
    wrap.innerHTML = '';
    if (!hits || hits.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'hits__empty';
        empty.textContent = '— no passages summoned for this question —';
        wrap.appendChild(empty);
        return;
    }
    for (const h of hits) {
        const li = document.createElement('li');
        li.className = 'hit';
        li.innerHTML = `
            <div class="hit__head">
                <div class="hit__title">${escapeHtml(h.title || `#${h.note_id}`)}</div>
                <div class="hit__score">score · ${Number(h.score || 0).toFixed(3)}</div>
            </div>
            <div class="hit__snippet">${escapeHtml(h.snippet || '')}</div>
        `;
        li.addEventListener('click', () => openNote(h.note_id));
        wrap.appendChild(li);
    }
}

// ── Note drawer ─────────────────────────────────────────────────

function closeDrawer() {
    const drawer = $('#drawer');
    const backdrop = $('#drawer-backdrop');
    if (drawer) drawer.remove();
    if (backdrop) backdrop.remove();
    document.removeEventListener('keydown', escClosesDrawer);
}

function escClosesDrawer(e) {
    if (e.key === 'Escape') closeDrawer();
}

async function openNote(noteId) {
    closeDrawer();

    const backdrop = document.createElement('div');
    backdrop.id = 'drawer-backdrop';
    backdrop.className = 'drawer-backdrop';
    backdrop.addEventListener('click', closeDrawer);
    document.body.appendChild(backdrop);

    const drawer = document.createElement('aside');
    drawer.id = 'drawer';
    drawer.className = 'drawer';
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-label', 'Note');
    drawer.innerHTML = `
        <button class="drawer__close" aria-label="Close">close ✕</button>
        <div class="drawer__rubric">— Folio —</div>
        <h2 class="drawer__title">Summoning…</h2>
        <div class="drawer__path"></div>
        <div class="drawer__body drawer__body--loading">— consulting the codex —</div>
    `;
    document.body.appendChild(drawer);
    $('.drawer__close', drawer).addEventListener('click', closeDrawer);
    document.addEventListener('keydown', escClosesDrawer);

    try {
        const r = await fetch(`/api/notes/${encodeURIComponent(noteId)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        $('.drawer__title', drawer).textContent = data.title || `Note #${noteId}`;
        $('.drawer__path', drawer).textContent = data.path || '';
        const body = $('.drawer__body', drawer);
        body.classList.remove('drawer__body--loading');
        body.textContent = data.body || '— (this folio is empty) —';
    } catch (e) {
        $('.drawer__title', drawer).textContent = 'The folio resists.';
        const body = $('.drawer__body', drawer);
        body.classList.remove('drawer__body--loading');
        body.classList.add('drawer__body--error');
        body.textContent = `Could not retrieve note ${noteId}: ${e.message}`;
    }
}

// Resolve a citation label ([[Title]] or [[Title#anchor]]) by searching
// for the bare title and opening the best match.
async function openNoteByLabel(label) {
    const title = label.split('#')[0].trim();
    // Prefer the cached search hits when one of them matches — avoids a
    // round trip when the citation came from this turn's retrieval.
    const cached = state.lastSearchHits.find(h => h.title === title);
    if (cached) {
        openNote(cached.note_id);
        return;
    }
    try {
        const r = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: title, top_k: 5 }),
        });
        const data = await r.json();
        const exact = (data.hits || []).find(h => h.title === title);
        const hit = exact || (data.hits || [])[0];
        if (hit) openNote(hit.note_id);
    } catch (e) {
        // Silently fail — the user can still click the related hit list.
    }
}

// Re-wire .cite spans after streaming/render completes.
function wireCitations() {
    $$('.cite').forEach(el => {
        el.addEventListener('click', () => openNoteByLabel(el.dataset.label));
    });
}

// ── Ask flow ────────────────────────────────────────────────────

async function handleAsk(event) {
    event.preventDefault();
    if (state.busy) return;

    const q = $('#question').value.trim();
    if (!q) return;
    const topK = parseInt($('#top-k').value, 10) || 5;

    state.busy = true;
    const btn = $('#ask-btn');
    btn.disabled = true;
    const btnText = $('.invocation__btn-text', btn);
    const originalLabel = btnText.textContent;
    btnText.textContent = 'Inscribing…';

    // Reveal panels.
    const panel  = $('#answer');
    const body   = $('#answer-body');
    const sources = $('#sources');
    const sourcesList = $('#sources-list');
    const dropped = $('#dropped');

    panel.hidden = false;
    sources.hidden = true;
    sourcesList.innerHTML = '';
    dropped.hidden = true;
    dropped.textContent = '';

    // Initial: just a scribe cursor pulsing.
    const textNode = document.createElement('span');
    const cursorNode = document.createElement('span');
    cursorNode.className = 'scribe-cursor';
    body.innerHTML = '';
    body.append(textNode, cursorNode);

    // Fire the search in parallel — the marginalia + hits don't depend
    // on the answer.
    const searchPromise = fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 10 }),
    }).then(r => r.json()).catch(() => ({ hits: [] }));

    let collected = '';
    let streamedOk = false;

    try {
        const r = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, top_k: topK, stream: true }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        if (!r.body) throw new Error('no stream body');

        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // SSE frames are separated by blank lines.
            let frameEnd;
            while ((frameEnd = buffer.indexOf('\n\n')) !== -1) {
                const frame = buffer.slice(0, frameEnd);
                buffer = buffer.slice(frameEnd + 2);

                for (const line of frame.split('\n')) {
                    if (!line.startsWith('data: ')) continue;
                    const payload = line.slice(6);
                    let ev;
                    try { ev = JSON.parse(payload); } catch { continue; }

                    if (ev.type === 'token' && ev.text) {
                        collected += ev.text;
                        textNode.textContent = collected;
                    } else if (ev.type === 'done') {
                        renderSources(ev.sources || []);
                        const drop = ev.dropped_citations || 0;
                        if (drop > 0) {
                            dropped.hidden = false;
                            dropped.textContent =
                                `${drop} citation${drop === 1 ? '' : 's'} the model produced ` +
                                `were not in the retrieved context and have been stripped.`;
                        }
                    }
                }
            }
            streamedOk = true;
        }

        // Final render: paragraph + citation styling, drop the cursor.
        if (collected.trim()) {
            body.innerHTML = renderProse(collected);
            wireCitations();
        } else if (streamedOk) {
            body.innerHTML = '<p><em>— the codex returned no tokens. Is the model awake? —</em></p>';
        }
    } catch (e) {
        body.innerHTML = '<p><em>— the codex falls silent. The backend may be unreachable. —</em></p>';
        // Surface the error in the console for debugging, but keep the UI calm.
        console.error('ask failed:', e);
    }

    // Wait for the parallel search to complete and render its results.
    const search = await searchPromise;
    state.lastSearchHits = search.hits || [];
    renderHits(state.lastSearchHits);
    renderMarginalia(state.lastSearchHits);

    state.busy = false;
    btn.disabled = false;
    btnText.textContent = originalLabel;
}

function renderSources(sourceLabels) {
    const wrap = $('#sources');
    const list = $('#sources-list');
    list.innerHTML = '';
    if (!sourceLabels || sourceLabels.length === 0) {
        wrap.hidden = true;
        return;
    }
    wrap.hidden = false;
    for (const s of sourceLabels) {
        const li = document.createElement('li');
        li.textContent = `[[${s}]]`;
        li.dataset.label = s;
        li.addEventListener('click', () => openNoteByLabel(s));
        list.appendChild(li);
    }
}

// ── Boot ────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    refreshHealth();
    loadCategories();

    const form = $('#ask-form');
    form.addEventListener('submit', handleAsk);

    // Ctrl/Cmd + Enter from the textarea submits.
    $('#question').addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            form.requestSubmit();
        }
    });

    // Refresh the candle every 30 s so the user sees if the backend dies.
    setInterval(refreshHealth, 30_000);
});
