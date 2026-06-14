# Progress

Your personal study dashboard — flashcard **mastery**, **cards mastered over time**,
**review activity**, and **exam scores by topic**. Everything is computed in your browser
from the reviews and exams you've done (`localStorage`); nothing is uploaded.

!!! tip "How to fill it up"
    - Take exams with the **📝 Generate Exam** button at the bottom of any page (needs the
      study server: `make study`). Each graded exam is recorded per topic.
    - Review the [Flashcards](flashcards.md) deck — every self-grade feeds the mastery and
      activity charts. A card counts as **mastered** once its review interval reaches ≥ 3 weeks.

<div id="progress-app" class="fc-app"></div>
