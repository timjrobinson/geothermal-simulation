# Flashcards

A spaced-repetition deck covering the essential facts from across these docs — definitions,
formulas, units, the physics of each survey method, and the "why" behind the integration
pipeline. It works like Anki: you grade how well you knew each card from **0 (blackout)** to
**5 (instant recall)**, and the [SM-2 algorithm](inversion.md) shows the cards you find hard
far more often, so everything sticks over time.

!!! tip "How it works"
    - Read the prompt, recall the answer, then **Show answer** (or press <kbd>Space</kbd>).
    - Grade yourself **0–5** (keys <kbd>0</kbd>–<kbd>5</kbd>). Low grades (0–2) bring the card
      back within this session; high grades push it out days, then weeks.
    - Progress is saved in your browser (`localStorage`) — no account, no server needed to study.
    - The deck is generated from the docs by the local Claude model. If you see "no deck found",
      run `make flashcards` once to build it.

<div id="fc-app" class="fc-app"></div>
