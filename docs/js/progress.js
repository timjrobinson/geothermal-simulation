/* Progress dashboard — visualizes your study data from localStorage:
 *   fc-progress-v1   SM-2 state per flashcard (current mastery)
 *   fc-history-v1    every flashcard review {t,id,q,reps,interval} (mastered-over-time, activity)
 *   exam-history-v1  every graded exam {t,page,score_pct,n} (per-topic exam scores)
 *   fc-user-cards-v1 cards added from weak exam answers (counted in totals)
 * Plus the shipped deck (deck.json) for totals + per-topic grouping. Hand-rolled SVG
 * charts, no dependencies. Mounts into <div id="progress-app">. */

(function () {
  "use strict";
  const DAY = 86400000;
  const MASTER_DAYS = 21; // interval >= 3 weeks counts as "mastered"
  const API = localStorage.getItem("studyApiBase") || (location.port === "8002" ? "" : "http://localhost:8002");

  const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) || d; } catch (e) { return d; } };
  const dayKey = (t) => new Date(t).toISOString().slice(0, 10);
  function node(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  async function loadDeck() {
    let base = [];
    try { const r = await fetch("deck.json", { cache: "no-cache" }); if (r.ok) base = (await r.json()).cards || []; } catch (e) {}
    if (!base.length) { try { const r = await fetch(API + "/api/flashcards"); if (r.ok) base = (await r.json()).cards || []; } catch (e) {} }
    return base.concat(load("fc-user-cards-v1", []));
  }

  // ── SVG chart helpers ───────────────────────────────────────────────────────
  const SVG = "http://www.w3.org/2000/svg";
  function svg(w, h) { const s = document.createElementNS(SVG, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("width", "100%"); s.classList.add("pg-chart"); return s; }
  function add(parent, tag, attrs, text) { const e = document.createElementNS(SVG, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; parent.appendChild(e); return e; }
  const FG = "var(--md-default-fg-color--light)", GRID = "var(--md-default-fg-color--lightest)", ACC = "var(--md-primary-fg-color)";

  function lineChart(points, { color = ACC, fmtY = (v) => v, yMax = null } = {}) {
    const W = 640, H = 220, P = 36;
    const s = svg(W, H);
    if (!points.length) { add(s, "text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: FG, "font-size": 13 }, "no data yet"); return s; }
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs) || x0 + 1;
    const ymax = yMax != null ? yMax : Math.max(1, ...ys);
    const sx = (x) => P + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * P);
    const sy = (y) => H - P - (y / ymax) * (H - 2 * P);
    [0, 0.5, 1].forEach((f) => { const y = H - P - f * (H - 2 * P); add(s, "line", { x1: P, y1: y, x2: W - P, y2: y, stroke: GRID, "stroke-width": 1 }); add(s, "text", { x: P - 6, y: y + 4, "text-anchor": "end", fill: FG, "font-size": 10 }, fmtY(Math.round(f * ymax))); });
    const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    add(s, "path", { d, fill: "none", stroke: color, "stroke-width": 2.5 });
    points.forEach((p) => add(s, "circle", { cx: sx(p.x), cy: sy(p.y), r: 3, fill: color }));
    add(s, "text", { x: P, y: H - 8, fill: FG, "font-size": 10 }, dayKey(x0));
    add(s, "text", { x: W - P, y: H - 8, "text-anchor": "end", fill: FG, "font-size": 10 }, dayKey(x1));
    return s;
  }

  function barChart(bars, { color = ACC, fmtY = (v) => v } = {}) {
    const W = 640, H = 200, P = 32;
    const s = svg(W, H);
    if (!bars.length) { add(s, "text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: FG, "font-size": 13 }, "no data yet"); return s; }
    const ymax = Math.max(1, ...bars.map((b) => b.value));
    const bw = (W - 2 * P) / bars.length;
    bars.forEach((b, i) => {
      const h = (b.value / ymax) * (H - 2 * P);
      add(s, "rect", { x: P + i * bw + 1, y: H - P - h, width: Math.max(1, bw - 2), height: h, fill: color, rx: 1 });
    });
    add(s, "text", { x: P, y: H - 8, fill: FG, "font-size": 10 }, bars[0].label);
    add(s, "text", { x: W - P, y: H - 8, "text-anchor": "end", fill: FG, "font-size": 10 }, bars[bars.length - 1].label);
    add(s, "text", { x: P - 6, y: P, "text-anchor": "end", fill: FG, "font-size": 10 }, fmtY(ymax));
    return s;
  }

  function section(root, title, sub) { root.appendChild(node("h2", null, title)); if (sub) root.appendChild(node("p", "pg-sub", sub)); }

  // ── build dashboard ─────────────────────────────────────────────────────────
  function render(root, deck) {
    const progress = load("fc-progress-v1", {});
    const history = load("fc-history-v1", []);
    const exams = load("exam-history-v1", []);

    const total = deck.length;
    const stateOf = (id) => progress[id];
    const mastered = deck.filter((c) => stateOf(c.id) && stateOf(c.id).interval >= MASTER_DAYS).length;
    const learning = deck.filter((c) => stateOf(c.id) && stateOf(c.id).interval < MASTER_DAYS).length;
    const fresh = total - mastered - learning;

    root.innerHTML = "";

    // Summary stats
    const stats = node("div", "fc-stats");
    [["Total cards", total], ["Mastered", mastered], ["Learning", learning], ["New", fresh],
     ["Reviews", history.length], ["Exams taken", exams.length]].forEach(([l, v]) =>
      stats.appendChild(node("div", "fc-stat", `<b>${v}</b> ${l}`)));
    root.appendChild(stats);

    // Mastery breakdown bar
    section(root, "Flashcard mastery");
    const pct = (n) => total ? Math.round((n / total) * 100) : 0;
    root.appendChild(node("div", "pg-mastery",
      `<div class="pg-bar"><span class="pg-seg pg-mastered" style="width:${pct(mastered)}%" title="mastered"></span>` +
      `<span class="pg-seg pg-learning" style="width:${pct(learning)}%" title="learning"></span>` +
      `<span class="pg-seg pg-new" style="width:${pct(fresh)}%" title="new"></span></div>` +
      `<div class="pg-legend"><span><i class="pg-mastered"></i> mastered ${pct(mastered)}%</span>` +
      `<span><i class="pg-learning"></i> learning ${pct(learning)}%</span>` +
      `<span><i class="pg-new"></i> new ${pct(fresh)}%</span></div>`));

    // Cards mastered over time (first day each card's interval reached MASTER_DAYS)
    section(root, "Cards mastered over time", "When each card first reached a ≥3-week interval.");
    const masterDay = {};
    history.forEach((h) => { if (h.interval >= MASTER_DAYS && !(h.id in masterDay)) masterDay[h.id] = dayKey(h.t); });
    const byDay = {};
    Object.values(masterDay).forEach((d) => (byDay[d] = (byDay[d] || 0) + 1));
    const days = Object.keys(byDay).sort();
    let cum = 0;
    const masteredPts = days.map((d) => ({ x: new Date(d).getTime(), y: (cum += byDay[d]) }));
    root.appendChild(lineChart(masteredPts, { fmtY: (v) => v }));

    // Review activity last 30 days
    section(root, "Review activity (last 30 days)");
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const start = today.getTime() - 29 * DAY;
    const counts = {};
    history.forEach((h) => { if (h.t >= start) counts[dayKey(h.t)] = (counts[dayKey(h.t)] || 0) + 1; });
    const actBars = [];
    for (let t = start; t <= today.getTime(); t += DAY) { const k = dayKey(t); actBars.push({ label: k.slice(5), value: counts[k] || 0 }); }
    root.appendChild(barChart(actBars));

    // Exam scores over time
    section(root, "Exam scores over time");
    const examPts = exams.slice().sort((a, b) => a.t - b.t).map((e) => ({ x: e.t, y: e.score_pct }));
    root.appendChild(lineChart(examPts, { color: "#43a047", yMax: 100, fmtY: (v) => v + "%" }));

    // Per-topic table
    section(root, "By topic", "Flashcard mastery and best/latest exam score for each page.");
    const byPage = {};
    deck.forEach((c) => {
      const key = c.page || "(misc)";
      const b = (byPage[key] = byPage[key] || { title: c.page_title || key, total: 0, mastered: 0, exams: [] });
      b.total++;
      if (stateOf(c.id) && stateOf(c.id).interval >= MASTER_DAYS) b.mastered++;
    });
    exams.forEach((e) => { const b = byPage[e.page]; if (b) b.exams.push(e); });
    const rows = Object.values(byPage).sort((a, b) => a.title.localeCompare(b.title));
    const table = node("table", "pg-table");
    table.innerHTML = "<thead><tr><th>Topic</th><th>Cards mastered</th><th>Latest exam</th><th>Best exam</th></tr></thead>";
    const tb = node("tbody");
    rows.forEach((r) => {
      const latest = r.exams.length ? r.exams.slice().sort((a, b) => b.t - a.t)[0].score_pct + "%" : "—";
      const best = r.exams.length ? Math.max(...r.exams.map((e) => e.score_pct)) + "%" : "—";
      const cp = r.total ? Math.round((r.mastered / r.total) * 100) : 0;
      tb.appendChild(node("tr", null,
        `<td>${r.title}</td><td>${r.mastered}/${r.total} <span class="pg-mini">(${cp}%)</span></td><td>${latest}</td><td>${best}</td>`));
    });
    table.appendChild(tb);
    root.appendChild(table);

    if (!history.length && !exams.length) {
      root.appendChild(node("div", "study-error",
        "No study activity yet. Take an exam (the 📝 button on any page) or review some " +
        "<a href='../flashcards/'>flashcards</a>, then come back — your progress will appear here."));
    }
  }

  async function init() {
    const root = document.getElementById("progress-app");
    if (!root) return;
    root.innerHTML = `<p><span class="study-spinner"></span>Loading your progress…</p>`;
    render(root, await loadDeck());
  }
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
