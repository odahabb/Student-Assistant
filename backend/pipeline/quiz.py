"""
backend/pipeline/quiz.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Extension of the pipeline: QUIZ GENERATION AND GRADING
Turns a subject's indexed chunks into short quiz questions, grouped by topic
(document section), and grades a student's answers. Adaptive difficulty and
revision recommendations live in recommender.py.

Question generation reuses the pipeline's own models, so the quiz loads no
model of its own:

  1. a chunk is picked from a topic;
  2. Qwen2.5-1.5B-Instruct writes a question about a fact in that chunk;
  3. generator.answer_short() answers it from that chunk alone, which becomes
     the reference answer;
  4. round-trip check: the question is answered again through normal
     retrieval over the whole subject, and the item is kept only if the two
     answers agree (roundtrip consistency, Alberti et al., 2019). This drops
     questions that are ambiguous, or whose answer only makes sense with the
     passage in view.

Difficulty comes from the answer format rather than the question's wording:

  level 1 — multiple choice, with distractors from other items' answers
  level 2 — short answer, with the source section shown as a hint
  level 3 — short answer, no hint

Grading compares the student's answer with the reference answer. Containment
in either direction is correct outright; an answer missing a number the
reference states is wrong; otherwise the score is the higher of token F1 and
MiniLM cosine similarity, and GRADE_THRESHOLD decides correctness.

recommender.py turns the graded answers into the mastery estimate that picks
the level and the next topic.
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

# The section name given to chunks that have none.
WHOLE_DOCUMENT = "Whole document"
# The section name of a topic covering a whole file rather than one section.
WHOLE_FILE = "Everything in this file"
SKIPPED_SECTIONS = re.compile(
    r"^(references|bibliography|works cited|acknowledge?ments?)$", re.IGNORECASE)

MIN_CHUNK_WORDS = 40       # chunks shorter than this are not quizzed on
MAX_QUESTION_WORDS = 30    # bounds well_formed() applies to a written
MAX_ANSWER_WORDS = 12      # ... question and its reference answer
QUESTION_PROMPT = ("Write one question about a specific fact stated in the "
                   "passage below. It must be answerable from the passage "
                   "alone, in a few words. Reply with the question only."
                   "\n\nPassage: {passage}\n\nQuestion:")
# The "Question:" label an instruction-tuned model puts on what it writes.
_QUESTION_PREFIX = re.compile(r"^\s*(?:question\s*\d*\s*[:.\-]|q\s*[:.\-])\s*",
                              re.IGNORECASE)
GRADE_THRESHOLD = 0.7      # score at or above which an answer is correct
MC_OPTIONS = 4             # options on a level-1 question, including the answer

# Markers of bibliography text, which is skipped rather than quizzed on. Chunk
# text is tokenizer-decoded, so its punctuation is space-separated: "( 2024 )".
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
    """How many bibliography markers a passage holds: citation years, arXiv,
    DOI, URLs, "et al." and the names of venues."""
    t = " ".join(str(text).lower().split())
    return sum(len(re.findall(p, t)) for p in _REFERENCE_PATTERNS)


def looks_like_reference_list(text: str) -> bool:
    """True once a passage carries REFERENCE_SIGNAL_THRESHOLD markers."""
    return reference_signals(text) >= REFERENCE_SIGNAL_THRESHOLD


def normalize(text: str) -> str:
    """Lower-case, punctuation-free text for comparing two answers."""
    text = str(text).lower()
    # "0.83" is kept as one token, so it cannot match a bare "0"
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", "_", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Topics

@dataclass
class Topic:
    """One thing to be quizzed on, and the chunks its questions come from."""

    id: str
    source_file: Optional[str]
    section: str
    chunk_indices: List[int] = field(default_factory=list)
    # "section" — one section of one document; "document" — the whole file.
    scope: str = "section"


def topic_id(source_file: Optional[str], section: Optional[str]) -> str:
    """A topic's id: its file and section, joined by a separator."""
    return f"{source_file or 'Untitled'} › {section or WHOLE_DOCUMENT}"


def usable_for_quiz(chunk) -> bool:
    """Whether a chunk is long enough, and ordinary enough, to quiz on."""
    section = getattr(chunk, "section", None) or ""
    return (not SKIPPED_SECTIONS.match(section.strip())
            and len(str(chunk).split()) >= MIN_CHUNK_WORDS
            and not looks_like_reference_list(chunk))


def build_topics(chunks: Sequence, whole_documents: bool = False) -> List[Topic]:
    """
    Group a subject's chunks into topics, one per (document, section), in
    the order they first appear. Reference lists and very short chunks are
    left out, and a topic with no usable chunk is dropped.

    With whole_documents, each file of more than one section also gets a
    topic covering all of its chunks, listed before its sections.
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
    if not whole_documents:
        return list(topics.values())
    return with_document_topics(topics.values())


def with_document_topics(section_topics: Sequence[Topic]) -> List[Topic]:
    """
    Put a whole-file topic in front of each document's section topics. A
    file of one section already is that topic, so it is left alone.
    """
    by_document: Dict[Optional[str], List[Topic]] = {}
    for topic in section_topics:
        by_document.setdefault(topic.source_file, []).append(topic)

    taken = {t.id for t in section_topics}
    out: List[Topic] = []
    for source_file, sections in by_document.items():
        tid = topic_id(source_file, WHOLE_FILE)
        if len(sections) > 1 and tid not in taken:
            out.append(Topic(tid, source_file, WHOLE_FILE,
                             sorted(i for t in sections for i in t.chunk_indices),
                             scope="document"))
        out.extend(sections)
    return out


# Grading

@dataclass
class Grade:
    """The outcome of grading one answer, and which rule decided it."""

    score: float
    correct: bool
    method: str


def token_f1(answer: str, reference: str) -> float:
    """F1 over the normalised tokens the two strings share."""
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
    """Cosine similarity of the two strings' MiniLM embeddings."""
    from backend.pipeline.embedder import embed
    # Always MiniLM, whatever SA_EMBEDDER selects for retrieval, since
    # GRADE_THRESHOLD is a threshold on its similarities.
    vectors = embed([a, b], model="minilm")
    return float(vectors[0] @ vectors[1])   # vectors are unit-norm


def grade(answer: str, reference: str, threshold: float = GRADE_THRESHOLD,
          similarity: Optional[Callable[[str, str], float]] = None) -> Grade:
    """
    Score a free-text answer against the reference answer.

    Containment in either direction scores 1.0; an answer missing a number
    the reference states scores 0.0; otherwise the score is the higher of
    token F1 and cosine similarity, and `threshold` decides correctness.

    similarity defaults to MiniLM cosine similarity; tests pass a stub so
    grading can be checked without loading a model.
    """
    a, r = normalize(answer), normalize(reference)
    if not a or not r:
        return Grade(0.0, False, "empty")

    # Embeddings rate "861 hours" close to "680,000 hours", so a reference
    # that states numbers requires the answer to state the same ones.
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
    """One generated question with its reference answer and its source."""

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


def clean_question(text: str) -> str:
    """
    The question alone: the first line of what the model wrote, without a
    "Question:" label or surrounding quotes. well_formed() decides whether
    what is left is usable.
    """
    line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    return _QUESTION_PREFIX.sub("", line).strip().strip('"“”').strip()


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

    answer_fn(question, chunks) defaults to generator.answer_short and
    question_fn(prompt) to generator.complete. retrieve_fn(question) should
    return the chunks normal retrieval would give for the question; with it
    None, the round-trip check is skipped.

    Returns (item, None) on success or (None, reason) on rejection.
    """
    if answer_fn is None or question_fn is None:
        # answer_short rather than generate: a reference answer has to be
        # short enough to compare with what the student types, whatever style
        # the chat view is answering in.
        from backend.pipeline.generator import answer_short, complete
        answer_fn = answer_fn or answer_short
        question_fn = question_fn or (lambda prompt: complete(prompt, max_new_tokens=48))

    question = clean_question(question_fn(QUESTION_PROMPT.format(passage=str(chunk))))
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
    Answer the item's question through normal retrieval and compare that
    with its reference answer. The answer and score are recorded on the item;
    returns True when the two agree.
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
    Generate up to per_topic items for every topic, at most
    max_attempts_per_topic tries each.

    Chunks are tried in an order seeded by `seed`, so the pool covers a topic
    rather than always starting at its first chunk. Questions duplicating one
    already in the pool are skipped. progress(done, total), if given, is
    called after each topic.
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
    Options for a level-1 question, shuffled: the reference answer plus
    distractors taken from other items' answers, those from the same topic
    first. Fewer than n_options come back when the pool is small.
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
