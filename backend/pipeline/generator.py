"""
backend/pipeline/generator.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 6 of pipeline: GENERATION
Generates an answer from retrieved context using google/flan-t5-large.
Runs on CPU by default; supports optional Intel Arc GPU / NPU acceleration
via the SA_DEVICE env var (see backend/pipeline/device.py).
"""

import logging
import re
from typing import List

from transformers import AutoTokenizer

from backend.pipeline.device import get_torch_device, should_use_npu

log = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-large"

_tokenizer = None
_model = None
_model_is_ov = False


def _get_model():
    global _tokenizer, _model, _model_is_ov
    if _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if should_use_npu():
        try:
            from optimum.intel.openvino import OVModelForSeq2SeqLM
            _model = OVModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True, device="NPU")
            _model_is_ov = True
            return _tokenizer, _model
        except Exception as e:
            log.warning(f"NPU generator load failed ({e}), falling back to torch CPU/GPU")

    from transformers import AutoModelForSeq2SeqLM
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _model.to(get_torch_device())
    return _tokenizer, _model


MAX_INPUT_TOKENS = 1024
_CONTEXT_SAFETY_MARGIN = 10


_RANK_DECAY = 0.85


def _allocate_budget(lengths: List[int], budget: int) -> List[int]:
    """
    Split `budget` tokens across chunks whose lengths are given in retrieval
    order (most similar first).

    Each chunk's share is weighted by _RANK_DECAY ** rank, and any surplus from
    a chunk shorter than its share is redistributed to the rest (water-filling).
    Two things matter here:

      - every chunk gets a share, so none is dropped outright for being last;
      - the weighting keeps that from being paid for entirely by the top-ranked
        chunk, which retrieval says is the most likely to hold the answer.

    An even split does the first but not the second: with 8 chunks over budget
    it cut the top-ranked chunk to roughly half, which can remove the answer
    span from the very chunk retrieval ranked first.
    """
    n = len(lengths)
    allocation = [0] * n
    weights = [_RANK_DECAY ** i for i in range(n)]
    remaining = budget
    active = [i for i, length in enumerate(lengths) if length > 0]

    while active and remaining > 0:
        weight_sum = sum(weights[i] for i in active)
        pool = remaining
        progressed = False
        for i in list(active):
            share = int(pool * weights[i] / weight_sum)
            take = min(share, lengths[i] - allocation[i], remaining)
            if take > 0:
                allocation[i] += take
                remaining -= take
                progressed = True
            if allocation[i] >= lengths[i]:
                active.remove(i)
        if not progressed:
            # Shares have rounded down to zero — hand what's left to the
            # highest-ranked chunks still short of their full length.
            for i in active[:remaining]:
                allocation[i] += 1
            break

    return allocation


def _budget_context(tokenizer, query: str, context_chunks: List[str]) -> str:
    """
    Fit the retrieved chunks into the token budget left over after the prompt
    template, and return them as one context string.

    When everything fits, the chunks are joined unchanged. When it doesn't, the
    budget is shared out across chunks (see _allocate_budget) so every retrieved
    chunk is still represented. The earlier version concatenated the chunks
    first and then cut the tail off the combined string, which silently deleted
    whole low-ranked chunks — the chunk holding the answer could disappear from
    the prompt entirely while the model still produced a confident-looking
    answer from the chunks that survived.
    """
    if isinstance(context_chunks, str):
        context_chunks = [context_chunks]

    # Tokens consumed by the fixed parts of the template (question + answer tag)
    shell = f"Question: {query}\nContext: \nAnswer:"
    shell_tokens = len(tokenizer.encode(shell, add_special_tokens=True))
    budget = MAX_INPUT_TOKENS - shell_tokens - _CONTEXT_SAFETY_MARGIN
    if budget <= 0:
        return ""

    chunk_ids = [tokenizer.encode(c, add_special_tokens=False) for c in context_chunks]
    lengths = [len(ids) for ids in chunk_ids]
    # One token reserved per " " joining two chunks together.
    separator_cost = max(0, len(chunk_ids) - 1)

    if sum(lengths) + separator_cost <= budget:
        # Nothing to trim. The encode/decode round trip is redundant here, but
        # it is what the previous implementation did to every context, and it
        # normalises some OCR artefacts — keeping it means this function is a
        # byte-for-byte no-op versus the old behaviour whenever the context
        # fits, so previously recorded eval results remain comparable.
        joined = " ".join(context_chunks)
        return tokenizer.decode(
            tokenizer.encode(joined, add_special_tokens=False),
            skip_special_tokens=True,
        )

    allocation = _allocate_budget(lengths, budget - separator_cost)
    dropped = sum(1 for n in allocation if n == 0)
    log.info(
        f"Context over budget ({sum(lengths)} > {budget} tokens) — trimming "
        f"{len(chunk_ids)} chunks to {allocation} tokens each"
        + (f" ({dropped} chunk(s) too small a share to keep)" if dropped else "")
    )

    context = " ".join(
        tokenizer.decode(ids[:n], skip_special_tokens=True)
        for ids, n in zip(chunk_ids, allocation) if n > 0
    )

    # Sentencepiece can merge tokens across the join boundaries, so the
    # reassembled string may re-tokenize a few tokens longer than the sum of
    # its parts. Verify against the budget and hard-trim as a last resort.
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(context_ids) > budget:
        context = tokenizer.decode(context_ids[:budget], skip_special_tokens=True)

    return context


def _fix_number_spacing(text: str) -> str:
    """
    flan-t5's tokenizer splits digits into subword pieces, and decoding those
    back can leave stray spaces around punctuation inside numbers and times
    (e.g. "0. 28", "11 : 39 a. m.", "$ 975. 00"). Collapse spacing immediately
    around '.', ',', and ':' when both sides are digits.
    """
    text = re.sub(r'(\d)\s*([.,:])\s*(\d)', r'\1\2\3', text)
    text = re.sub(r'([$#])\s+(\d)', r'\1\2', text)
    return text


def generate(query: str, context_chunks: List[str]) -> str:
    """
    Build a prompt from the query and context chunks, run inference on CPU
    using flan-t5-large, and return the generated answer string.
    """
    tokenizer, model = _get_model()

    context = _budget_context(tokenizer, query, context_chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    device = "cpu" if _model_is_ov else get_torch_device()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, max_length=MAX_INPUT_TOKENS).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _fix_number_spacing(answer)
