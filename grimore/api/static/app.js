// Grimore web UI — vanilla JS, no build step.
// Talks to /api/* with fetch. Streaming endpoint not wired in yet —
// the non-streaming /api/ask round-trip is faster to render and easier
// to read for the typical ask-a-thing flow this UI targets.

const $ = (sel) => document.querySelector(sel);

async function refreshHealth() {
    const el = $("#health");
    try {
        const r = await fetch("/api/health");
        const data = await r.json();
        if (data.ok) {
            el.textContent = `${data.version} · ${data.vault}`;
            el.classList.add("ok");
        } else {
            el.textContent = "unhealthy";
            el.classList.add("err");
        }
    } catch (e) {
        el.textContent = "offline";
        el.classList.add("err");
    }
}

async function loadCategories() {
    try {
        const r = await fetch("/api/categories");
        const data = await r.json();
        const ul = $("#categories");
        ul.innerHTML = "";
        for (const row of data.categories || []) {
            const li = document.createElement("li");
            const name = document.createElement("span");
            name.textContent = row.category || "(uncategorised)";
            const count = document.createElement("span");
            count.className = "count";
            count.textContent = row.count;
            li.appendChild(name);
            li.appendChild(count);
            ul.appendChild(li);
        }
    } catch (e) {
        // Categories are nice-to-have; failure shouldn't break the rest of the UI.
    }
}

function renderHits(hits) {
    const wrap = $("#results");
    wrap.innerHTML = "";
    for (const h of hits || []) {
        const div = document.createElement("div");
        div.className = "hit";
        const title = document.createElement("span");
        title.className = "title";
        title.textContent = h.title || `#${h.note_id}`;
        const score = document.createElement("span");
        score.className = "score";
        score.textContent = h.score.toFixed(3);
        const head = document.createElement("div");
        head.appendChild(title);
        head.appendChild(score);
        const snippet = document.createElement("div");
        snippet.className = "snippet";
        snippet.textContent = h.snippet || "";
        div.appendChild(head);
        div.appendChild(snippet);
        wrap.appendChild(div);
    }
}

function renderAnswer(data) {
    const panel = $("#answer");
    panel.classList.remove("hidden");
    $("#answer-body").textContent = data.answer || "(no answer)";

    const sources = $("#sources");
    sources.innerHTML = "";
    for (const s of data.sources || []) {
        const chip = document.createElement("span");
        chip.className = "source-chip";
        chip.textContent = `[[${s}]]`;
        sources.appendChild(chip);
    }

    const dropped = $("#dropped");
    if (data.dropped_citations) {
        dropped.classList.remove("hidden");
        dropped.textContent =
            `${data.dropped_citations} citation(s) in the answer weren't ` +
            "among the retrieved sources and were stripped.";
    } else {
        dropped.classList.add("hidden");
    }
}

async function handleAsk(event) {
    event.preventDefault();
    const question = $("#question").value.trim();
    if (!question) return;
    const topK = parseInt($("#top-k").value, 10) || 5;
    const btn = $("#ask-btn");
    btn.disabled = true;
    btn.textContent = "…";

    try {
        const [askResp, searchResp] = await Promise.all([
            fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, top_k: topK }),
            }),
            fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: question, top_k: 10 }),
            }),
        ]);
        if (askResp.ok) {
            renderAnswer(await askResp.json());
        }
        if (searchResp.ok) {
            const data = await searchResp.json();
            renderHits(data.hits);
        }
    } finally {
        btn.disabled = false;
        btn.textContent = "Ask";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    refreshHealth();
    loadCategories();
    $("#ask-form").addEventListener("submit", handleAsk);
});
