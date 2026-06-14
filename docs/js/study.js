/* "Generate Exam" widget injected at the bottom of every docs page.
 *
 * Calls the local study server (study/server.py, default http://localhost:8002), which
 * uses `claude -p` to generate an exam from THIS page and to grade your free-text answers.
 * Persists each graded exam to localStorage (exam-history-v1) for the Progress dashboard,
 * and lets you turn weak (partial/incorrect) answers into flashcards (fc-user-cards-v1).
 * If the server isn't running, the button explains how to start it (`make study`). */

(function () {
  "use strict";
  const API =
    localStorage.getItem("studyApiBase") ||
    (location.port === "8002" ? "" : "http://localhost:8002");
  const EXAM_HISTORY = "exam-history-v1";
  const USER_CARDS = "fc-user-cards-v1";

  const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) || d; } catch (e) { return d; } };
  const save = (k, v) => localStorage.setItem(k, JSON.stringify(v));
  function hash(s) { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0; return (h >>> 0).toString(36); }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function isFlashcardsPage() {
    return /\/(flashcards|progress)\/?$/.test(location.pathname) || document.getElementById("fc-app") || document.getElementById("progress-app");
  }

  async function postJSON(path, body) {
    const r = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) {}
      throw new Error(detail);
    }
    return r.json();
  }

  function serverDownMsg(err) {
    return (
      `<div class="study-error"><b>Couldn't reach the study server.</b><br>` +
      `Start it in a terminal with <code>make study</code> (serves docs + exams) or ` +
      `<code>make study-api</code> (API only), then retry.<br>` +
      `<small>${(err && err.message) || err}</small></div>`
    );
  }

  function mount(article) {
    const box = el("div", "study-exam");
    const intro = el(
      "div",
      null,
      `<button class="study-btn study-exam__cta">📝 Generate Exam</button>` +
        `<div class="study-hint">An AI examiner (local <code>claude -p</code>) writes ` +
        `questions from this page, you answer in your own words, and it grades you.</div>`
    );
    const panel = el("div");
    box.appendChild(intro);
    box.appendChild(panel);
    article.appendChild(box);

    let questions = [];
    let examPage = location.pathname;

    intro.querySelector("button").addEventListener("click", async () => {
      panel.innerHTML = `<p><span class="study-spinner"></span>Writing your exam…</p>`;
      try {
        const data = await postJSON("/api/exam/generate", { path: location.pathname, n_questions: 6 });
        questions = data.questions || [];
        examPage = data.page || location.pathname;
        renderQuestions(panel, questions, examPage);
      } catch (err) {
        panel.innerHTML = serverDownMsg(err);
      }
    });
  }

  function renderQuestions(panel, questions, examPage) {
    panel.innerHTML = "";
    questions.forEach((q, i) => {
      const wrap = el("div", "study-q");
      const tag = q.type ? `<span class="study-tag">${q.type}</span>` : "";
      wrap.appendChild(el("div", "study-q__prompt", `<span class="study-q__num">${i + 1}.</span>${q.prompt}${tag}`));
      const ta = el("textarea");
      ta.dataset.qid = q.id;
      ta.placeholder = "Your answer…";
      wrap.appendChild(ta);
      panel.appendChild(wrap);
    });
    const submit = el("button", "study-btn", "Submit answers for grading");
    panel.appendChild(submit);
    submit.addEventListener("click", async () => {
      const answers = {};
      panel.querySelectorAll("textarea").forEach((t) => (answers[t.dataset.qid] = t.value));
      submit.disabled = true;
      submit.innerHTML = `<span class="study-spinner"></span>Grading…`;
      try {
        const res = await postJSON("/api/exam/evaluate", {
          path: location.pathname,
          questions: questions.map((q) => ({ id: q.id, prompt: q.prompt, type: q.type })),
          answers,
        });
        renderResults(panel, questions, res, examPage);
      } catch (err) {
        submit.disabled = false;
        submit.textContent = "Submit answers for grading";
        panel.appendChild(el("div", null, serverDownMsg(err)));
      }
    });
  }

  function renderResults(panel, questions, res, examPage) {
    panel.innerHTML = "";
    const o = res.overall || {};
    const byId = {};
    (res.results || []).forEach((r) => (byId[r.id] = r));

    // Persist for the Progress dashboard.
    if (o.score_pct != null) {
      const hist = load(EXAM_HISTORY, []);
      hist.push({ t: Date.now(), page: examPage, score_pct: o.score_pct, n: questions.length });
      if (hist.length > 2000) hist.splice(0, hist.length - 2000);
      save(EXAM_HISTORY, hist);
    }

    panel.appendChild(
      el("div", "study-overall",
        `Score: ${o.score_pct != null ? o.score_pct + "%" : "—"} — ${o.summary || ""}` +
        (o.study_tips ? `<div class="study-hint">Study tips: ${o.study_tips}</div>` : ""))
    );

    questions.forEach((q, i) => {
      const r = byId[q.id] || { verdict: "incorrect", feedback: "No grade returned." };
      const card = el("div", `study-result study-result--${r.verdict}`);
      card.innerHTML =
        `<div><span class="study-verdict">${r.verdict}</span>` +
        (r.score != null ? ` · ${Math.round(r.score * 100)}%` : "") + `</div>` +
        `<div><b>${i + 1}. ${q.prompt}</b></div><div>${r.feedback || ""}</div>` +
        (r.ideal_answer ? `<div class="study-ideal"><b>Ideal answer:</b> ${r.ideal_answer}</div>` : "");
      panel.appendChild(card);
    });

    // Weak answers -> flashcards.
    const weak = questions
      .map((q) => ({ q, r: byId[q.id] }))
      .filter((x) => x.r && x.r.verdict !== "correct" && (x.r.ideal_answer || x.r.feedback));
    const actions = el("div", null, "");
    actions.style.marginTop = "1rem";
    if (weak.length) {
      const addBtn = el("button", "study-btn", `➕ Add ${weak.length} weak answer${weak.length > 1 ? "s" : ""} to flashcards`);
      addBtn.addEventListener("click", () => {
        const cards = load(USER_CARDS, []);
        const have = new Set(cards.map((c) => c.id));
        let added = 0;
        weak.forEach(({ q, r }) => {
          const front = q.prompt;
          const id = "exam-" + hash(examPage + "|" + front);
          if (have.has(id)) return;
          cards.push({
            id, front,
            back: r.ideal_answer || r.feedback,
            tags: ["exam", "weak"],
            page: examPage,
            page_title: examPage.replace(/\.md$/, ""),
            source: "exam",
          });
          have.add(id);
          added++;
        });
        save(USER_CARDS, cards);
        addBtn.outerHTML = `<span class="study-tag" style="font-size:.78rem">✓ Added ${added} card${added !== 1 ? "s" : ""} — review them on the <a href="${flashHref()}">Flashcards</a> page</span>`;
      });
      actions.appendChild(addBtn);
    }
    const again = el("button", "study-btn study-btn--ghost", "↻ New exam");
    again.style.marginLeft = weak.length ? "0.5rem" : "0";
    again.addEventListener("click", () => panel.closest(".study-exam").querySelector(".study-exam__cta").click());
    actions.appendChild(again);
    panel.appendChild(actions);
  }

  // Best-effort relative link to the flashcards page from any docs page.
  function flashHref() {
    const m = location.pathname.match(/\//g);
    const depth = location.pathname.endsWith("/") ? (m ? m.length - 1 : 0) : (m ? m.length : 0);
    return "../".repeat(Math.max(depth - 0, 0)) + "flashcards/";
  }

  function init() {
    if (isFlashcardsPage()) return;
    const article = document.querySelector(".md-content article") || document.querySelector("article");
    if (article && !article.querySelector(".study-exam")) mount(article);
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
