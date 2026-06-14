/* Spaced-repetition flashcards (Anki-style SM-2) for the docs.
 *
 * Loads docs/flashcards/deck.json (a shipped static asset — studying needs no server),
 * schedules with the SM-2 algorithm, persists per-card state in localStorage, and shows
 * cards you don't know far more often: a low grade (0–2) reschedules the card within the
 * same session; a high grade pushes it days/weeks out. Mounts into <div id="fc-app">. */

(function () {
  "use strict";
  const PROGRESS_KEY = "fc-progress-v1";
  const CFG_KEY = "fc-config-v1";
  const DAY = 86400000;
  const SOON = 60000; // 1 min — "show again this session"
  const API =
    localStorage.getItem("studyApiBase") ||
    (location.port === "8002" ? "" : "http://localhost:8002");

  const GRADES = [
    { q: 0, cls: "fc-g0", label: "Blackout", sub: "no idea" },
    { q: 1, cls: "fc-g1", label: "Wrong", sub: "guessed" },
    { q: 2, cls: "fc-g2", label: "Hard", sub: "barely" },
    { q: 3, cls: "fc-g3", label: "OK", sub: "effort" },
    { q: 4, cls: "fc-g4", label: "Good", sub: "recalled" },
    { q: 5, cls: "fc-g5", label: "Easy", sub: "instant" },
  ];

  const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) || d; } catch (e) { return d; } };
  const save = (k, v) => localStorage.setItem(k, JSON.stringify(v));

  let deck = [];
  let progress = load(PROGRESS_KEY, {});
  const cfg = Object.assign({ newPerSession: 20 }, load(CFG_KEY, {}));

  // ── SM-2 ────────────────────────────────────────────────────────────────
  function schedule(state, q) {
    const s = state || { ease: 2.5, interval: 0, reps: 0, lapses: 0 };
    if (q < 3) {
      s.reps = 0;
      s.interval = 0;
      s.lapses = (s.lapses || 0) + 1;
      s.ease = Math.max(1.3, s.ease - 0.2);
      s.due = Date.now() + SOON; // resurfaces within the session
    } else {
      s.reps = (s.reps || 0) + 1;
      s.interval = s.reps === 1 ? 1 : s.reps === 2 ? 6 : Math.round(s.interval * s.ease);
      s.ease = Math.min(2.7, Math.max(1.3, s.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))));
      s.due = Date.now() + s.interval * DAY;
    }
    s.last = Date.now();
    return s;
  }

  const isNew = (c) => !progress[c.id];
  const isDue = (c) => progress[c.id] && progress[c.id].due <= Date.now();
  const isLearned = (c) => progress[c.id] && progress[c.id].reps >= 2 && progress[c.id].due - Date.now() > DAY;

  function buildSession() {
    const due = deck.filter(isDue).sort((a, b) => progress[a.id].due - progress[b.id].due);
    const fresh = deck.filter(isNew).slice(0, cfg.newPerSession);
    // interleave new cards in, but lead with what's already due
    const ids = new Set([...due, ...fresh].map((c) => c.id));
    return deck.filter((c) => ids.has(c.id));
  }

  // ── rendering ─────────────────────────────────────────────────────────────
  let root, session, revealed;

  function nextCard() {
    const now = Date.now();
    const ready = session
      .filter((c) => isNew(c) || (progress[c.id] && progress[c.id].due <= now))
      .sort((a, b) => (progress[a.id]?.due || 0) - (progress[b.id]?.due || 0));
    return ready[0] || null;
  }

  function render() {
    const total = deck.length;
    const dueN = deck.filter(isDue).length;
    const newN = deck.filter(isNew).length;
    const learnedN = deck.filter(isLearned).length;

    const card = nextCard();
    root.innerHTML = "";
    root.appendChild(stats(dueN, newN, learnedN, total));

    const toolbar = node("div", "fc-toolbar");
    toolbar.appendChild(btn("study-btn study-btn--ghost", "Reset progress", resetProgress));
    root.appendChild(toolbar);

    if (!card) {
      root.appendChild(
        node("div", "fc-done",
          `<h2>✅ All caught up!</h2><p>No cards are due right now. ` +
          `${newN ? newN + " new cards remain — " : ""}come back later for review, or ` +
          `<a href="#" id="fc-more">study ${Math.min(newN, cfg.newPerSession) || 0} more now</a>.</p>`)
      );
      const more = document.getElementById("fc-more");
      if (more) more.addEventListener("click", (e) => { e.preventDefault(); cfg.newPerSession += 20; save(CFG_KEY, cfg); session = buildSession(); render(); });
      return;
    }

    const c = node("div", "fc-card");
    c.appendChild(node("div", "fc-front", c2(card.front)));
    if (revealed) {
      c.appendChild(node("div", "fc-back", c2(card.back)));
      c.appendChild(node("div", "fc-meta",
        `${(card.tags || []).map((t) => "#" + t).join(" ")} · ${card.page_title || card.page}`));
    }
    root.appendChild(c);

    const controls = node("div", "fc-controls");
    if (!revealed) {
      const show = btn("study-btn", "Show answer  (Space)", () => { revealed = true; render(); });
      show.style.flex = "1 1 100%";
      controls.appendChild(show);
    } else {
      GRADES.forEach((g) => {
        const b = node("button", "fc-grade " + g.cls, `${g.q} · ${g.label}<small>${g.sub}</small>`);
        b.addEventListener("click", () => grade(card, g.q));
        controls.appendChild(b);
      });
    }
    root.appendChild(controls);
  }

  function grade(card, q) {
    const st = schedule(progress[card.id], q);
    progress[card.id] = st;
    save(PROGRESS_KEY, progress);
    // Log the review for the Progress dashboard (mastered-over-time + activity).
    const hist = load("fc-history-v1", []);
    hist.push({ t: Date.now(), id: card.id, q, reps: st.reps, interval: st.interval, page: card.page });
    if (hist.length > 8000) hist.splice(0, hist.length - 8000);
    save("fc-history-v1", hist);
    revealed = false;
    render();
  }

  function resetProgress() {
    if (!confirm("Reset all flashcard progress? This cannot be undone.")) return;
    progress = {};
    save(PROGRESS_KEY, progress);
    session = buildSession();
    revealed = false;
    render();
  }

  // keyboard: Space reveals / grades vary
  function onKey(e) {
    const card = nextCard();
    if (!card) return;
    if (!revealed && (e.code === "Space" || e.code === "Enter")) { e.preventDefault(); revealed = true; render(); }
    else if (revealed && e.key >= "0" && e.key <= "5") { e.preventDefault(); grade(card, parseInt(e.key, 10)); }
  }

  // ── helpers ────────────────────────────────────────────────────────────────
  function node(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function btn(cls, label, fn) { const b = node("button", cls, label); b.addEventListener("click", fn); return b; }
  function stats(d, n, l, t) {
    const s = node("div", "fc-stats");
    s.innerHTML =
      `<div class="fc-stat"><b>${d}</b> due</div>` +
      `<div class="fc-stat"><b>${n}</b> new</div>` +
      `<div class="fc-stat"><b>${l}</b> learned</div>` +
      `<div class="fc-stat"><b>${t}</b> total</div>`;
    return s;
  }
  // minimal, safe text->html (escape, keep code ticks readable)
  function c2(s) {
    const esc = (s || "").replace(/[&<>]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));
    return esc.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>");
  }

  async function loadDeck() {
    let base = null;
    try {
      const r = await fetch("deck.json", { cache: "no-cache" });
      if (r.ok) base = (await r.json()).cards || [];
    } catch (e) {}
    if (!base) {
      try {
        const r = await fetch(API + "/api/flashcards");
        if (r.ok) base = (await r.json()).cards || [];
      } catch (e) {}
    }
    // Merge user-added cards (created from weak exam answers). These work even with no
    // generated deck, so a learner can build a deck purely from their exam mistakes.
    const user = load("fc-user-cards-v1", []);
    if (base) return base.concat(user);
    return user.length ? user : null;
  }

  async function init() {
    root = document.getElementById("fc-app");
    if (!root) return;
    root.innerHTML = `<p><span class="study-spinner"></span>Loading deck…</p>`;
    const cards = await loadDeck();
    if (!cards) {
      root.innerHTML =
        `<div class="study-error"><b>No flashcard deck found.</b><br>` +
        `Generate it with <code>make flashcards</code> (uses the local <code>claude -p</code> ` +
        `to build cards from every docs page), then reload this page.</div>`;
      return;
    }
    deck = cards;
    session = buildSession();
    revealed = false;
    document.addEventListener("keydown", onKey);
    render();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
