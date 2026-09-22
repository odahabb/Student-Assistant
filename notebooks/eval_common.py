"""
notebooks/eval_common.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Helpers shared by the evaluation notebooks.

  repo_root()  the repository root, whether a notebook runs from notebooks/
               or from the root, so every notebook reads and writes the same
               data/ paths
  judge()      the answer-grading rule every end-to-end number in this project
               uses (notebooks 01, 03 and 11), with its normalisation helpers.
               It lived in backend/scripts/generation_analysis.py and moved
               here unchanged when the evaluations became notebooks.
"""

import re
from pathlib import Path


def repo_root() -> Path:
    """The repository root: the parent of notebooks/, found from the cwd."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "backend" / "pipeline").is_dir():
            return candidate
    raise RuntimeError("Run the notebooks from inside the repository.")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, and unify number formatting (3,197 -> 3197)."""
    text = str(text).lower()
    text = re.sub(r'(?<=\d),(?=\d)', '', text)          # thousands separators
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)        # "1 550" -> "1550"
    text = re.sub(r'[^a-z0-9.]+', ' ', text)
    text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)     # keep decimal points only
    return re.sub(r'\s+', ' ', text).strip()


def numbers_in(text: str):
    return set(re.findall(r'\d+(?:\.\d+)?', normalise(text)))


# Stop words carry no claim, so they say nothing about whether an answer
# stuck to its passages.
_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "that", "this", "these", "those", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "as", "by", "from", "at", "which", "into",
    "can", "may", "might", "will", "would", "should", "such", "than", "then",
    "they", "their", "them", "there", "here", "when", "while", "how", "what",
    "each", "other", "more", "most", "some", "any", "all", "also", "based",
    "using", "used", "use", "have", "has", "had", "not", "no", "so", "we",
}


def grounded_share(answer: str, context: str) -> float:
    """
    Share of the answer's content words that appear in the retrieved passages.

    A blunt instrument: it cannot tell a paraphrase from an invention, and a
    wrong claim made in the passages' own vocabulary scores well. It is
    reported as a symptom — an answer far below the rest is worth reading —
    not as a measure of faithfulness.
    """
    words = [w for w in normalise(answer).split()
             if len(w) > 3 and w not in _FUNCTION_WORDS]
    if not words:
        return 1.0
    haystack = set(normalise(context).split())
    return round(sum(1 for w in words if w in haystack) / len(words), 3)


def token_f1(a: str, b: str) -> float:
    ta, tb = normalise(a).split(), normalise(b).split()
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for token in ta:
        if token in pool:
            pool.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(ta), common / len(tb)
    return 2 * precision * recall / (precision + recall)


def judge(expected: str, produced: str) -> dict:
    """
    Decide whether a generated answer carries the ground-truth answer.

    flan-t5 answers tersely and extractively ("3197" for "3,197 sentence
    pairs"), so exact string equality is far too strict. The rule is:
    containment either way, or a high token overlap, or - when the expected
    answer is numeric - every one of its numbers appearing in the output.
    Every signal is stored so a borderline call can be re-judged by hand.
    """
    exp, got = normalise(expected), normalise(produced)
    expected_numbers = numbers_in(expected)

    signals = {
        "exact_match": exp == got,
        "expected_in_answer": bool(exp) and exp in got,
        "answer_in_expected": bool(got) and len(got) >= 2 and got in exp,
        "token_f1": round(token_f1(expected, produced), 3),
        # Every content word of the expected answer appears somewhere in the
        # output. A paragraph that explains the right fact passes this where
        # strict containment fails on wording ("CoT" vs "chain-of-thought").
        "expected_words_present": all(
            w in set(normalise(produced).split())
            for w in normalise(expected).split() if len(w) > 2) if expected else False,
        "expected_numbers": sorted(expected_numbers),
        "all_expected_numbers_present": bool(expected_numbers)
        and expected_numbers <= numbers_in(produced),
    }

    correct = (
        signals["exact_match"]
        or signals["expected_in_answer"]
        or signals["answer_in_expected"]
        or signals["token_f1"] >= 0.6
        or signals["all_expected_numbers_present"]
    )
    # Flag the grey zone rather than pretending the rule is unambiguous.
    signals["needs_human_review"] = bool(
        not correct and 0.3 <= signals["token_f1"] < 0.6)
    return correct, signals


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())
