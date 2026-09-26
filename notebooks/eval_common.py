"""
notebooks/eval_common.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Helpers shared by the evaluation notebooks.

  repo_root()       the repository root, whether a notebook runs from
                    notebooks/ or from the root, so every notebook reads and
                    writes the same data/ paths
  judge()           the answer-grading rule the end-to-end numbers use
                    (notebooks 01, 03 and 11)
  grounded_share()  how much of an answer's vocabulary comes from its passages
  normalise(), numbers_in(), token_f1(), squash()
                    the string handling judge() is built from
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
    """Every number in a string, after normalisation."""
    return set(re.findall(r'\d+(?:\.\d+)?', normalise(text)))


# Words grounded_share() ignores, since they carry no claim of their own.
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
    Share of the answer's content words that appear in the retrieved
    passages, counting words over three letters that are not in
    _FUNCTION_WORDS. An answer with no content words scores 1.0.

    This is a vocabulary overlap, not a measure of faithfulness: it cannot
    tell a paraphrase from an invention, and a wrong claim made in the
    passages' own words scores well.
    """
    words = [w for w in normalise(answer).split()
             if len(w) > 3 and w not in _FUNCTION_WORDS]
    if not words:
        return 1.0
    haystack = set(normalise(context).split())
    return round(sum(1 for w in words if w in haystack) / len(words), 3)


def token_f1(a: str, b: str) -> float:
    """F1 over the normalised tokens two strings share, counting repeats."""
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

    Returns (correct, signals). An answer counts as correct on any of:
    exact match, containment in either direction, token F1 of 0.6 or more,
    or — for a numeric expected answer — every one of its numbers appearing
    in the output. Every signal is returned as well, so a borderline call can
    be re-judged by hand.
    """
    exp, got = normalise(expected), normalise(produced)
    expected_numbers = numbers_in(expected)

    signals = {
        "exact_match": exp == got,
        "expected_in_answer": bool(exp) and exp in got,
        "answer_in_expected": bool(got) and len(got) >= 2 and got in exp,
        "token_f1": round(token_f1(expected, produced), 3),
        # Every content word of the expected answer appears somewhere in the
        # output, in any order — recorded, but not one of the rules below.
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
    # Marks the band where the token-F1 rule is closest to its threshold.
    signals["needs_human_review"] = bool(
        not correct and 0.3 <= signals["token_f1"] < 0.6)
    return correct, signals


def squash(text: str) -> str:
    """A string reduced to its lower-case letters and digits, for comparing
    two labels that may be punctuated differently."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())
