"""Generate the flashcard deck from the docs, using the local Claude Code model.

For every documentation page under ``docs/`` this runs ``claude -p`` to extract the most
important facts a student must memorize as concise front/back flashcards, then writes a
single deck to ``docs/flashcards/deck.json`` (a shipped static asset the flashcards study
page loads directly — no server required to *study*; the server is only for AI exams).

Usage:
    python study/generate_flashcards.py [--per-page N] [--jobs K] [--limit N] [--pages glob]

The number of cards scales with how much there is to know (``--per-page`` is a target the
model may adjust per page), typically landing in the 400–900 range across the ~24 pages.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT = DOCS / "flashcards" / "deck.json"
MAX_PAGE_CHARS = 24_000

_PROMPT = """You are building spaced-repetition flashcards for a geothermal-engineering
course, from the documentation page below, for a learner who knows programming but is new
to geoscience. Extract the MOST IMPORTANT facts, definitions, relationships, formulas, and
"why" insights a student must know cold. Each card: a focused question/cue on the FRONT and
a concise, complete answer on the BACK (1–4 sentences; include the key formula/units where
relevant). Prefer atomic cards (one idea each). Aim for about {n} cards for THIS page, but
use fewer if the page is short or more if it is dense — quality over quota.

Return ONLY a JSON object (no prose, no code fence):
{{"cards": [{{"front": "...", "back": "...", "tags": ["topic", ...]}}]}}

=== DOCUMENTATION PAGE: {title} ===
{page}
"""


def claude(prompt: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL, cwd=tempfile.gettempdir(),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:500] or "claude failed")
    outer = json.loads(proc.stdout)
    if outer.get("is_error"):
        raise RuntimeError(str(outer.get("result"))[:500])
    return str(outer.get("result", ""))


def extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL) or re.search(
        r"(\{.*\}|\[.*\])", text, re.DOTALL
    )
    if m:
        text = m.group(1)
    return json.loads(text)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def page_title(md: str, rel: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    return m.group(1).strip() if m else rel


def cards_for_page(path: Path, per_page: int) -> list[dict]:
    rel = str(path.relative_to(DOCS))
    md = path.read_text(encoding="utf-8")
    title = page_title(md, rel)
    try:
        data = extract_json(claude(_PROMPT.format(n=per_page, title=rel, page=md[:MAX_PAGE_CHARS])))
    except Exception as e:  # noqa: BLE001 — keep going on a single bad page
        print(f"  ! {rel}: {e}")
        return []
    raw = data.get("cards", []) if isinstance(data, dict) else []
    out = []
    base = slug(rel.replace(".md", ""))
    for i, c in enumerate(raw, 1):
        front, back = (c.get("front") or "").strip(), (c.get("back") or "").strip()
        if not front or not back:
            continue
        out.append({
            "id": f"{base}-{i:03d}",
            "front": front,
            "back": back,
            "tags": c.get("tags") or [],
            "page": rel,
            "page_title": title,
        })
    print(f"  ✓ {rel}: {len(out)} cards")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-page", type=int, default=30, help="target cards per page")
    ap.add_argument("--jobs", type=int, default=4, help="parallel claude calls")
    ap.add_argument("--limit", type=int, default=0, help="limit number of pages (0 = all)")
    ap.add_argument("--pages", default="**/*.md", help="glob under docs/ to include")
    args = ap.parse_args()

    pages = sorted(
        p for p in DOCS.glob(args.pages)
        if p.is_file() and p.name not in {"flashcards.md"} and "flashcards" not in p.parts
    )
    if args.limit:
        pages = pages[: args.limit]
    print(f"Generating flashcards from {len(pages)} pages (≈{args.per_page}/page, {args.jobs} parallel)…")

    deck: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for cards in ex.map(lambda p: cards_for_page(p, args.per_page), pages):
            deck.extend(cards)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"version": 1, "cards": deck}, indent=2), encoding="utf-8")
    print(f"\nWrote {len(deck)} cards to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
