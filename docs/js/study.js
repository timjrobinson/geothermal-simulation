/* "Generate Exam" widget injected at the bottom of every docs page.
 *
 * Calls the local study server (study/server.py, default http://localhost:8002), which
 * uses `claude -p` to generate an exam from THIS page and to grade your free-text answers.
 * If the server isn't running, the button explains how to start it (`make study`). */

(function () {
  "use strict";
  const API =
    localStorage.getItem("studyApiBase") ||
    (location.port === "8002" ? "" : "http://localhost:8002");

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function isFlashcardsPage() {
    return /\/flashcards\/?$/.test(location.pathname) || document.getElementById("fc-app");
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

    intro.querySelector("button").addEventListener("click", async () => {
      panel.innerHTML = `<p><span class="study-spinner"></span>Writing your exam…</p>`;
      try {
        const data = await postJSON("/api/exam/generate", {
          path: location.pathname,
          n_questions: 6,
        });
        questions = data.questions || [];
        renderQuestions(panel, questions);
      } catch (err) {
        panel.innerHTML = serverDownMsg(err);
      }
    });
  }

  function renderQuestions(panel, questions) {
    panel.innerHTML = "";
    questions.forEach((q, i) => {
      const wrap = el("div", "study-q");
      const tag = q.type ? `<span class="study-tag">${q.type}</span>` : "";
      wrap.appendChild(
        el(
          "div",
          "study-q__prompt",
          `<span class="study-q__num">${i + 1}.</span>${q.prompt}${tag}`
        )
      );
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
        renderResults(panel, questions, res);
      } catch (err) {
        submit.disabled = false;
        submit.textContent = "Submit answers for grading";
        panel.appendChild(el("div", null, serverDownMsg(err)));
      }
    });
  }

  function renderResults(panel, questions, res) {
    panel.innerHTML = "";
    const o = res.overall || {};
    panel.appendChild(
      el(
        "div",
        "study-overall",
        `Score: ${o.score_pct != null ? o.score_pct + "%" : "—"} — ${o.summary || ""}` +
          (o.study_tips ? `<div class="study-hint">Study tips: ${o.study_tips}</div>` : "")
      )
    );
    const byId = {};
    (res.results || []).forEach((r) => (byId[r.id] = r));
    questions.forEach((q, i) => {
      const r = byId[q.id] || { verdict: "incorrect", feedback: "No grade returned." };
      const card = el("div", `study-result study-result--${r.verdict}`);
      card.innerHTML =
        `<div><span class="study-verdict">${r.verdict}</span>` +
        (r.score != null ? ` · ${Math.round(r.score * 100)}%` : "") +
        `</div><div><b>${i + 1}. ${q.prompt}</b></div>` +
        `<div>${r.feedback || ""}</div>` +
        (r.ideal_answer
          ? `<div class="study-ideal"><b>Ideal answer:</b> ${r.ideal_answer}</div>`
          : "");
      panel.appendChild(card);
    });
    const again = el("button", "study-btn study-btn--ghost", "↻ New exam");
    again.style.marginTop = "1rem";
    again.addEventListener("click", () => panel.closest(".study-exam").querySelector(".study-exam__cta").click());
    panel.appendChild(again);
  }

  function init() {
    if (isFlashcardsPage()) return;
    const article = document.querySelector(".md-content article") || document.querySelector("article");
    if (article && !article.querySelector(".study-exam")) mount(article);
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
