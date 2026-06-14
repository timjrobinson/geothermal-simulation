"""Study server — AI exams + flashcard deck for the docs site.

A tiny FastAPI service the documentation site calls to:
  * POST /api/exam/generate  — make an exam from a docs page (via `claude -p`)
  * POST /api/exam/evaluate  — grade the learner's free-text answers (via `claude -p`)
  * GET  /api/flashcards     — serve the generated flashcard deck (study/decks/deck.json)
  * GET  /api/health         — liveness + whether the `claude` CLI is available

It uses the **local Claude Code model** via `claude -p ... --output-format json` (no API
key needed — it reuses your Claude Code auth). The subprocess runs in a neutral temp cwd
so it does not load the project's CLAUDE.md context on every call.

Run it with `make study` (serves the built docs site too) or `make study-api` (API only).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
# The deck is a SHIPPED docs asset (docs/flashcards/deck.json) so the flashcards page
# works purely client-side off the static site; the API just re-serves it as a convenience.
DECK = REPO / "docs" / "flashcards" / "deck.json"
SITE = REPO / "site"  # built docs (mkdocs build); served at / when present

CLAUDE_TIMEOUT = 240
MAX_PAGE_CHARS = 24_000  # keep prompts bounded


# ─────────────────────────────── claude -p plumbing ───────────────────────────────
def claude(prompt: str, *, timeout: int = CLAUDE_TIMEOUT) -> str:
    """Run a one-shot headless Claude Code call and return the model's text result."""
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=tempfile.gettempdir(),  # neutral cwd → no project CLAUDE.md context
        )
    except FileNotFoundError as e:
        raise HTTPException(503, "the `claude` CLI is not installed / not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise HTTPException(504, f"claude timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise HTTPException(502, f"claude failed: {proc.stderr.strip()[:500]}")
    try:
        outer = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise HTTPException(502, "could not parse claude output envelope") from e
    if outer.get("is_error"):
        raise HTTPException(502, f"claude error: {str(outer.get('result'))[:500]}")
    return str(outer.get("result", ""))


def extract_json(text: str):
    """Pull a JSON object out of a model reply (tolerating ```json fences / prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"claude did not return valid JSON: {e}")


def resolve_page(url_path: str) -> tuple[str, str]:
    """Map a site URL path (e.g. '/survey-methods/electrical/') to its docs markdown
    file, returning (relative_md_path, markdown_text). Path-traversal safe."""
    rel = url_path.strip("/")
    rel = re.sub(r"\.html?$", "", rel)
    candidates = []
    if not rel:
        candidates = ["index.md"]
    else:
        candidates = [f"{rel}.md", f"{rel}/index.md"]
    for cand in candidates:
        target = (DOCS / cand).resolve()
        if DOCS in target.parents and target.is_file():
            return cand, target.read_text(encoding="utf-8")
    raise HTTPException(404, f"no docs page found for path {url_path!r}")


# ─────────────────────────────── request models ───────────────────────────────
class ExamGenReq(BaseModel):
    path: str  # site URL path of the page, e.g. "/survey-methods/electrical/"
    n_questions: int = 6


class ExamQuestion(BaseModel):
    id: str
    prompt: str
    type: str = "short"


class ExamEvalReq(BaseModel):
    path: str
    questions: list[ExamQuestion]
    answers: dict[str, str]


# ─────────────────────────────── prompts ───────────────────────────────
_GEN_PROMPT = """You are an examiner for a geothermal-engineering course. Using ONLY the
documentation page below, write {n} exam questions that test genuine understanding (a mix
of recall and "explain why / how" conceptual questions) for a learner who knows programming
but is new to geoscience. Vary difficulty. Do NOT include the answers.

Return ONLY a JSON object of this exact shape (no prose, no code fence):
{{"questions": [{{"id": "q1", "type": "short"|"concept", "prompt": "...", "points": 1}}]}}

=== DOCUMENTATION PAGE: {title} ===
{page}
"""

_EVAL_PROMPT = """You are grading a learner's free-text answers for a geothermal-engineering
course, using ONLY the reference documentation page below as ground truth. Be fair but
rigorous; award partial credit; explain mistakes clearly and kindly; give the ideal answer.

Return ONLY a JSON object of this exact shape (no prose, no code fence):
{{"results": [{{"id": "q1", "verdict": "correct"|"partial"|"incorrect",
  "score": 0.0-1.0, "feedback": "...", "ideal_answer": "..."}}],
  "overall": {{"score_pct": 0-100, "summary": "...", "study_tips": "..."}}}}

=== REFERENCE PAGE: {title} ===
{page}

=== QUESTIONS AND THE LEARNER'S ANSWERS ===
{qa}
"""


# ─────────────────────────────── app ───────────────────────────────
app = FastAPI(title="GeoSim Study Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local single-user dev tool
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    have_claude = subprocess.run(
        ["which", "claude"], capture_output=True, text=True
    ).returncode == 0
    return {"ok": True, "claude_available": have_claude, "deck_exists": DECK.is_file()}


@app.post("/api/exam/generate")
def exam_generate(req: ExamGenReq) -> dict:
    rel, page = resolve_page(req.path)
    n = max(3, min(req.n_questions, 12))
    prompt = _GEN_PROMPT.format(n=n, title=rel, page=page[:MAX_PAGE_CHARS])
    data = extract_json(claude(prompt))
    questions = data.get("questions", []) if isinstance(data, dict) else []
    # normalize ids
    for i, q in enumerate(questions, 1):
        q.setdefault("id", f"q{i}")
        q.setdefault("type", "short")
    return {"page": rel, "questions": questions}


@app.post("/api/exam/evaluate")
def exam_evaluate(req: ExamEvalReq) -> dict:
    rel, page = resolve_page(req.path)
    qa_lines = []
    for q in req.questions:
        ans = req.answers.get(q.id, "").strip() or "(no answer given)"
        qa_lines.append(f"[{q.id}] Q: {q.prompt}\n     A: {ans}")
    prompt = _EVAL_PROMPT.format(
        title=rel, page=page[:MAX_PAGE_CHARS], qa="\n\n".join(qa_lines)
    )
    return extract_json(claude(prompt))


@app.get("/api/flashcards")
def flashcards() -> JSONResponse:
    if not DECK.is_file():
        raise HTTPException(404, "no flashcard deck yet — run `make flashcards` to build it")
    return JSONResponse(json.loads(DECK.read_text(encoding="utf-8")))


# Serve the built docs site at / when it exists (so `make study` is one origin/port).
if SITE.is_dir():
    app.mount("/", StaticFiles(directory=str(SITE), html=True), name="site")
