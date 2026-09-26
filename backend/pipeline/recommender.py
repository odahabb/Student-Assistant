"""
backend/pipeline/recommender.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Extension of the pipeline: ADAPTIVE DIFFICULTY AND REVISION RECOMMENDATIONS
Tracks a student's mastery per topic from their quiz answers, picks the
difficulty of the next question, and recommends which topics to revise.

Model: a one-parameter item response theory (Rasch) model, updated online in
the style of an Elo rating. Each topic t has a student ability theta_t
(starting at PRIOR_ABILITY) and each quiz level a fixed difficulty b
(LEVEL_DIFFICULTY, keyed by quiz.LEVELS):

    P(correct) = 1 / (1 + exp(-(theta_t - b)))

After an answer, theta_t moves towards the evidence:

    theta_t += k_n * (correct - P(correct)),   k_n = BASE_K / sqrt(1 + n)

where n is the number of earlier answers on that topic, so the first answers
on a topic move the estimate furthest and later ones refine it.

  mastery(t)     = P(correct) on a level-2 question, reported as a fraction
  next_level(t)  = the level whose predicted success is closest to
                   TARGET_SUCCESS. A wrong answer lowers theta, so the next
                   question is easier; a right answer raises it.
  recommend()    = topics ordered by mastery, weakest first, with unattempted
                   topics at the prior (50%) and ties broken by attempts.

Progress is a plain JSON file per subject; nothing here loads a model.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LEVEL_DIFFICULTY = {1: -1.0, 2: 0.0, 3: 1.0}
REFERENCE_LEVEL = 2          # the level mastery is reported at
TARGET_SUCCESS = 0.7         # predicted success next_level aims for
BASE_K = 1.5                 # step size on a topic's first answer
PRIOR_ABILITY = 0.0          # ability assumed before any answer


def success_probability(ability: float, level: int) -> float:
    """The Rasch probability of answering a question of `level` correctly."""
    return 1.0 / (1.0 + math.exp(-(ability - LEVEL_DIFFICULTY[level])))


@dataclass
class Recommendation:
    """One topic to revise, with the figures the interface shows beside it."""

    topic_id: str
    mastery: float
    attempts: int
    reason: str


class Progress:
    """One student's quiz history and ability estimates for one subject."""

    def __init__(self, attempts: Optional[List[dict]] = None):
        self.attempts: List[dict] = []
        self.ability: Dict[str, float] = {}
        self.count: Dict[str, int] = {}
        for attempt in attempts or []:
            self._apply(attempt["topic_id"], attempt["level"], attempt["correct"])
            self.attempts.append(attempt)

    # persistence

    @classmethod
    def load(cls, path) -> "Progress":
        """Replay the attempts saved at `path`; an empty Progress if none."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("attempts", []))

    def save(self, path) -> None:
        """Write the attempts and the current estimates to `path` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"attempts": self.attempts,
                   "ability": self.ability, "count": self.count}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # updates

    def _apply(self, topic_id: str, level: int, correct: bool) -> float:
        """Move a topic's ability by one answer and return the new value."""
        ability = self.ability.get(topic_id, PRIOR_ABILITY)
        n = self.count.get(topic_id, 0)
        k = BASE_K / math.sqrt(1 + n)
        ability += k * (float(correct) - success_probability(ability, level))
        self.ability[topic_id] = ability
        self.count[topic_id] = n + 1
        return ability

    def record(self, topic_id: str, level: int, correct: bool, **details) -> float:
        """Log an answer and update the topic's ability. Returns the new ability."""
        if level not in LEVEL_DIFFICULTY:
            raise ValueError(f"Unknown level {level}")
        ability = self._apply(topic_id, level, bool(correct))
        self.attempts.append({"topic_id": topic_id, "level": level,
                              "correct": bool(correct), "time": time.time(),
                              **details})
        return ability

    # queries

    def attempts_on(self, topic_id: str) -> int:
        """How many questions on this topic have been answered."""
        return self.count.get(topic_id, 0)

    def mastery(self, topic_id: str) -> float:
        """Predicted chance of a correct answer at REFERENCE_LEVEL, 0..1."""
        return success_probability(self.ability.get(topic_id, PRIOR_ABILITY),
                                   REFERENCE_LEVEL)

    def accuracy(self, topic_id: str) -> Optional[float]:
        """Plain share of answers on this topic that were correct, or None."""
        results = [a["correct"] for a in self.attempts if a["topic_id"] == topic_id]
        return sum(results) / len(results) if results else None

    def next_level(self, topic_id: str) -> int:
        """The level whose predicted success is nearest TARGET_SUCCESS."""
        ability = self.ability.get(topic_id, PRIOR_ABILITY)
        return min(LEVEL_DIFFICULTY, key=lambda level: (
            abs(success_probability(ability, level) - TARGET_SUCCESS), level))

    def recommend(self, topic_ids: Iterable[str], n: int = 3) -> List[Recommendation]:
        """The n weakest topics, weakest first, each with a short reason."""
        ranked = sorted(topic_ids, key=lambda t: (self.mastery(t), self.attempts_on(t)))
        out = []
        for t in ranked[:n]:
            attempts = self.attempts_on(t)
            mastery = self.mastery(t)
            if attempts == 0:
                reason = "Not practised yet"
            else:
                correct = sum(a["correct"] for a in self.attempts if a["topic_id"] == t)
                reason = (f"{correct}/{attempts} correct — estimated mastery "
                          f"{mastery:.0%}")
            out.append(Recommendation(t, mastery, attempts, reason))
        return out
