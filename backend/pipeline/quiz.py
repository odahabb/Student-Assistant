"""
backend/pipeline/quiz.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Extension of the pipeline: QUIZ GENERATION AND GRADING
Turns a subject's indexed chunks into short quiz questions, grouped by topic
(document section), and grades a student's answers. Adaptive difficulty and
revision recommendations live in recommender.py.

Question generation reuses the pipeline's own models, so the quiz adds no new
model to the system:

  1. a chunk is picked from a topic;
  2. flan-t5-large writes a question about a fact in that chunk;
  3. generator.generate() answers the question from that chunk alone — this is
     the reference answer;
  4. round-trip check: the question is answered again through normal
     retrieval over the whole subject, and the item is kept only if the two
     answers agree (the "roundtrip consistency" filter of Alberti et al.,
     2019). This drops questions that are ambiguous, or whose answer only
     makes sense with the passage in view.

Difficulty comes from the answer format rather than the question wording,
because flan-t5-large does not reliably follow instructions to write harder
("why"/"how") questions:

  level 1 — multiple choice (recognition)
  level 2 — short answer, with the source section shown as a hint
  level 3 — short answer, no hint (recall)

Grading compares the student's answer with the reference answer: containment
in either direction counts as correct outright; an answer missing a number
the reference states is wrong; otherwise the score is the higher of token F1
and MiniLM cosine similarity, and GRADE_THRESHOLD decides correctness
(calibrated in backend/scripts/eval_grader.py).
"""

import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

LEVELS = {
    1: "Multiple choice",
    2: "Short answer (with hint)",
    3: "Short answer",
}

WHOLE_DOCUMENT = "Whole document"
SKIPPED_SECTIONS = re.compile(
    r"^(references|bibliography|works cited|acknowledge?ments?)$", re.IGNORECASE)

MIN_CHUNK_WORDS = 40
MAX_QUESTION_WORDS = 30
MAX_ANSWER_WORDS = 12
QUESTION_PROMPT = ("Write a question about a specific fact stated in the passage "
                   "below.\n\nPassage: {passage}\n\nQuestion:")
GRADE_THRESHOLD = 0.7
MC_OPTIONS = 4

# Bibliography text makes poor quiz material ("Who wrote ... (2021)?"). Chunk
# text is tokenizer-decoded, so punctuation is space-separated: "( 2024 )".
_REFERENCE_PATTERNS = [
    r"\(\s*(?:19|20)\d\d\s*[a-z]?\s*\)",
    r"\barxiv\b",
    r"\bdoi\b",
    r"\bhttps?\b",
    r"\bet al\b",
    r"\bproceedings\b",
    r"\bin advances in\b",
    r"\bpreprint\b",
    r"\bconference\b",
    r"\bjournal\b",
]
REFERENCE_SIGNAL_THRESHOLD = 4
_UNUSABLE_ANSWERS = {"", "unanswerable", "none", "unknown", "no answer", "n/a"}


def reference_signals(text: str) -> int:
    """Count of bibliography markers (citation years, arXiv, DOI, URLs, et al.)."""
    t = " ".join(str(text).lower().split())
    return sum(len(re.findall(p, t)) for p in _REFERENCE_PATTERNS)


def looks_like_reference_list(text: str) -> bool:
    return reference_signals(text) >= REFERENCE_SIGNAL_THRESHOLD


def normalize(text: str) -> str:
    text = str(text).lower()
    # keep "0.83" as one token so it cannot match a bare "0"
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", "_", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Topics

@dataclass
class Topic:
    id: str
    source_file: Optional[str]
    section: str
    chunk_indices: List[int] = field(default_factory=list)


def topic_id(source_file: Optional[str], section: Optional[str]) -> str:
    return f"{source_file or 'Untitled'} › {section or WHOLE_DOCUMENT}"


def usable_for_quiz(chunk) -> bool:
    section = getattr(chunk, "section", None) or ""
    return (not SKIPPED_SECTIONS.match(section.strip())
            and len(str(chunk).split()) >= MIN_CHUNK_WORDS
            and not looks_like_reference_list(chunk))


def build_topics(chunks: Sequence) -> List[Topic]:
    """
    Group a subject's chunks into topics — one per (document, section) — in
    the order they first appear. Reference lists and very short chunks are
    left out; a topic with no usable chunk is dropped.
    """
    topics: Dict[str, Topic] = {}
    for i, chunk in enumerate(chunks):
        if not usable_for_quiz(chunk):
            continue
        source_file = getattr(chunk, "source_file", None)
        section = getattr(chunk, "section", None) or WHOLE_DOCUMENT
        tid = topic_id(source_file, section)
        if tid not in topics:
            topics[tid] = Topic(tid, source_file, section)
        topics[tid].chunk_indices.append(i)
    return list(topics.values())


# Grading

@dataclass
class Grade:
    score: float
    correct: bool
    method: str


def token_f1(answer: str, reference: str) -> float:
    a, r = normalize(answer).split(), normalize(reference).split()
    if not a or not r:
        return 0.0
    same = sum((Counter(a) & Counter(r)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(a), same / len(r)
    return 2 * precision * recall / (precision + recall)


def _numbers(text: str) -> List[str]:
    """
    Numbers in a string, with thousands separators and decoding spaces removed
    ("680, 000" -> "680000", "0. 83" -> "0.83").
    """
    text = re.sub(r"(?<=\d),?\s?(?=\d{3}\b)", "", str(text))
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def _cosine(a: str, b: str) -> float:
    from backend.pipeline.embedder import embed
    # Always MiniLM: GRADE_THRESHOLD was calibrated on its similarities.
    vectors = embed([a, b], model="minilm")
    return float(vectors[0] @ vectors[1])   # vectors are unit-norm


def grade(answer: str, reference: str, threshold: float = GRADE_THRESHOLD,
          similarity: Optional[Callable[[str, str], float]] = None) -> Grade:
    """
    Score a free-text answer against the reference answer.

    similarity defaults to MiniLM cosine similarity; tests pass a stub so
    grading can be checked without loading a model.
    """
    a, r = normalize(answer), normalize(reference)
    if not a or not r:
        return Grade(0.0, False, "empty")

    # Embeddings rate "861 hours" close to "680,000 hours", so when the
    # reference states numbers, the answer must state the same ones.
    missing = set(_numbers(reference)) - set(_numbers(answer))
    if missing:
        return Grade(0.0, False, "number_mismatch")

    if f" {r} " in f" {a} " or (len(a.split()) >= max(1, len(r.split()) // 2)
                                and f" {a} " in f" {r} "):
        return Grade(1.0, True, "containment")

    f1 = token_f1(a, r)
    cosine = (similarity or _cosine)(answer, reference)
    score = max(f1, cosine)
    return Grade(round(score, 4), score >= threshold,
                 "token_f1" if f1 >= cosine else "cosine")


# Question generation

@dataclass
class QuizItem:
    topic_id: str
    question: str
    answer: str
    source_file: Optional[str]
    page: Optional[int]
    section: Optional[str]
    passage: str
    roundtrip_answer: Optional[str] = None
    roundtrip_score: Optional[float] = None

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "QuizItem":
        return cls(**record)


def well_formed(question: str, answer: str) -> Optional[str]:
    """None if the pair is usable, otherwise the reason it is rejected."""
    q, a = question.strip(), answer.strip()
    if not q.endswith("?"):
        return "question does not end with '?'"
    if not 4 <= len(q.split()) <= MAX_QUESTION_WORDS:
        return "question length out of range"
    if normalize(a) in _UNUSABLE_ANSWERS:
        return "no answer"
    if len(a.split()) > MAX_ANSWER_WORDS:
        return "answer too long"
    if f" {normalize(a)} " in f" {normalize(q)} ":
        return "answer given away by the question"
    return None


def generate_item(chunk, answer_fn: Optional[Callable] = None,
                  retrieve_fn: Optional[Callable] = None,
                  question_fn: Optional[Callable[[str], str]] = None,
                  similarity: Optional[Callable[[str, str], float]] = None):
    """
    Try to make one quiz item from a chunk.

    answer_fn(question, chunks) defaults to generator.generate and
    question_fn(prompt) to generator.complete. retrieve_fn(question) should
    return the chunks normal retrieval would give for the question; when it
    is None the round-trip check is skipped.

    Returns (item, None) on success or (None, reason) on rejection.
    """
    if answer_fn is None or question_fn is None:
        from backend.pipeline.generator import complete, generate
        answer_fn = answer_fn or generate
        question_fn = question_fn or (lambda prompt: complete(prompt, max_new_tokens=48))

    question = question_fn(QUESTION_PROMPT.format(passage=str(chunk))).strip()
    answer = answer_fn(question, [chunk]).strip()
    problem = well_formed(question, answer)
    if problem:
        return None, problem

    item = QuizItem(
        topic_id=topic_id(getattr(chunk, "source_file", None),
                          getattr(chunk, "section", None)),
        question=question,
        answer=answer,
        source_file=getattr(chunk, "source_file", None),
        page=getattr(chunk, "page", None),
        section=getattr(chunk, "section", None),
        passage=str(chunk),
    )

    if retrieve_fn is not None and not roundtrip_check(item, answer_fn, retrieve_fn,
                                                       similarity):
        return None, "failed round-trip check"

    return item, None


def roundtrip_check(item: QuizItem, answer_fn: Callable, retrieve_fn: Callable,
                    similarity: Optional[Callable[[str, str], float]] = None) -> bool:
    """
    Answer the item's question through normal retrieval and compare with its
    reference answer. Records the outcome on the item; True when they agree.
    """
    item.roundtrip_answer = answer_fn(item.question, retrieve_fn(item.question)).strip()
    check = grade(item.roundtrip_answer, item.answer, similarity=similarity)
    item.roundtrip_score = check.score
    return check.correct


def build_pool(chunks: Sequence, topics: Sequence[Topic], per_topic: int = 2,
               max_attempts_per_topic: int = 6, seed: int = 0,
               progress: Optional[Callable[[int, int], None]] = None,
               **generate_kwargs) -> List[QuizItem]:
    """
    Generate up to per_topic items for every topic.

    Chunks are tried in a seeded random order so the pool covers a topic
    rather than always starting from its first chunk. Duplicate questions are
    skipped. progress(done, total), if given, is called after each topic.
    """
    rng = random.Random(seed)
    pool: List[QuizItem] = []
    seen = set()
    for done, topic in enumerate(topics, start=1):
        order = list(topic.chunk_indices)
        rng.shuffle(order)
        made = 0
        for i in order[:max_attempts_per_topic]:
            if made >= per_topic:
                break
            item, _ = generate_item(chunks[i], **generate_kwargs)
            if item is None or normalize(item.question) in seen:
                continue
            item.topic_id = topic.id
            seen.add(normalize(item.question))
            pool.append(item)
            made += 1
        if progress:
            progress(done, len(topics))
    return pool


def multiple_choice(item: QuizItem, pool: Sequence[QuizItem],
                    rng: Optional[random.Random] = None,
                    n_options: int = MC_OPTIONS) -> List[str]:
    """
    Options for a level-1 question: the reference answer plus distractors
    taken from other items' answers, preferring ones from the same topic.
    May return fewer than n_options when the pool is small.
    """
    rng = rng or random.Random()
    taken = {normalize(item.answer)}
    same_topic = [o.answer for o in pool if o.topic_id == item.topic_id]
    other = [o.answer for o in pool if o.topic_id != item.topic_id]
    rng.shuffle(same_topic)
    rng.shuffle(other)

    options = [item.answer]
    for candidate in same_topic + other:
        key = normalize(candidate)
        if key in taken or not key:
            continue
        taken.add(key)
        options.append(candidate)
        if len(options) == n_options:
            break
    rng.shuffle(options)
    return options
