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
import os
import re
import threading
from typing import Iterator, List, Optional

from transformers import AutoTokenizer

from backend.pipeline.device import get_torch_device, should_use_npu

log = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-large"

# Two answering styles, because the two places an answer is used want
# opposite things.
#
#   "short"   — FLAN-T5-Large, extractive: a span, a number, a few words.
#               This is what the quiz compares a student's answer against and
#               what every Chapter 5 measurement was taken on.
#   "explain" — a small instruction-tuned model writing a short paragraph, for
#               the chat view, where a student asking "what is a fitness
#               function?" wants the idea explained rather than a phrase
#               lifted off a slide. FLAN-T5 cannot do this: asked for three to
#               four sentences it returns one of ten words, and sampling does
#               not change that, because its distribution is too peaked.
#
# SA_ANSWER_STYLE selects it. The library default stays "short" so the
# evaluation scripts keep measuring the configuration they were written for;
# backend/service.py turns on "explain" for the application.
ANSWER_STYLE = os.environ.get("SA_ANSWER_STYLE", "short").lower()
CHAT_MODEL_NAME = os.environ.get("SA_CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
# Enough freedom to phrase an explanation, not enough to wander off the
# passages: every claim is still supposed to come from the context.
CHAT_TEMPERATURE = 0.6
CHAT_TOP_P = 0.9
CHAT_MAX_TOKENS = 260

CHAT_SYSTEM = (
    "You are helping a student revise from their own course material. Answer "
    "the question using only the passages provided, which come from documents "
    "the student uploaded. Write three to five sentences of plain prose that "
    "explain the idea, rather than copying the wording of the passages. Do not "
    "add facts that are not in the passages. If the passages do not answer the "
    "question, say so in one sentence."
)

_tokenizer = None
_model = None
_model_is_ov = False
_chat_tokenizer = None
_chat_model = None


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


def _get_chat_model():
    """
    Lazy-load the instruction-tuned model used for explanations. Loaded only
    when SA_ANSWER_STYLE=explain, so the quiz and the evaluation scripts never
    pay for it.
    """
    global _chat_tokenizer, _chat_model
    if _chat_model is not None:
        return _chat_tokenizer, _chat_model

    import torch
    from transformers import AutoModelForCausalLM

    log.info(f"Loading {CHAT_MODEL_NAME} (first use)...")
    device = get_torch_device()
    _chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
    _chat_model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL_NAME,
        torch_dtype=torch.float16 if device == "xpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    try:
        _chat_model.to(device)
    except Exception as e:      # an unusable GPU must not cost the answer
        log.warning(f"Could not place the chat model on {device} ({e}) — using cpu")
        _chat_model.to("cpu")
    return _chat_tokenizer, _chat_model


def _chat_inputs(query: str, context_chunks: List[str]):
    """The question and its passages, in the model's chat format."""
    tokenizer, model = _get_chat_model()
    if isinstance(context_chunks, str):
        context_chunks = [context_chunks]
    passages = "\n\n".join(f"[{i + 1}] {chunk}"
                           for i, chunk in enumerate(context_chunks))
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {query}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt").to(model.device)


def _chat_settings() -> dict:
    return {"max_new_tokens": CHAT_MAX_TOKENS, "do_sample": True,
            "temperature": CHAT_TEMPERATURE, "top_p": CHAT_TOP_P}


def explain(query: str, context_chunks: List[str]) -> str:
    """A short paragraph answering the question from the retrieved passages."""
    tokenizer, model = _get_chat_model()
    inputs = _chat_inputs(query, context_chunks)
    output = model.generate(**inputs, **_chat_settings())
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
    return _fix_number_spacing(answer).strip()


def explain_stream(query: str, context_chunks: List[str]) -> Iterator[str]:
    """explain(), piece by piece, for the web interface."""
    from transformers import TextIteratorStreamer

    tokenizer, model = _get_chat_model()
    inputs = _chat_inputs(query, context_chunks)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    failure = []

    def run():
        try:
            model.generate(**inputs, **_chat_settings(), streamer=streamer)
        except Exception as exc:
            failure.append(exc)
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield from streamer
    finally:
        worker.join()
    if failure:
        raise failure[0]


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


def complete(prompt: str, max_new_tokens: int = 128) -> str:
    """
    Run flan-t5-large on an arbitrary prompt (greedy decoding) and return the
    decoded output. Shared by generate() and the quiz layer, which prompts the
    same model to write questions. Prompts longer than the model's input limit
    are truncated from the end, so callers should keep their passage short.
    """
    tokenizer, model = _get_model()

    device = "cpu" if _model_is_ov else get_torch_device()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def stream(query: str, context_chunks: List[str]) -> Iterator[str]:
    """
    Same prompt and decoding as generate(), but yields the answer piece by
    piece as the model produces it, for the web interface. Joining the pieces
    and passing them through _fix_number_spacing gives generate()'s output.
    """
    if ANSWER_STYLE == "explain":
        yield from explain_stream(query, context_chunks)
        return

    from transformers import TextIteratorStreamer

    tokenizer, model = _get_model()
    context = _budget_context(tokenizer, query, context_chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    device = "cpu" if _model_is_ov else get_torch_device()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS).to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    failure = []

    def run():
        try:
            model.generate(**inputs, max_new_tokens=128, streamer=streamer)
        except Exception as exc:
            failure.append(exc)
            streamer.end()   # otherwise the loop below waits forever

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield from streamer
    finally:
        # Also reached when the caller stops early (e.g. the browser tab was
        # closed): wait for the model to finish before anyone else uses it.
        worker.join()
    if failure:
        raise failure[0]


def generate(query: str, context_chunks: List[str]) -> str:
    """
    Answer the query from the retrieved chunks, in whichever style
    SA_ANSWER_STYLE selects (see ANSWER_STYLE above).
    """
    if ANSWER_STYLE == "explain":
        return explain(query, context_chunks)
    return answer_short(query, context_chunks)


def answer_short(query: str, context_chunks: List[str]) -> str:
    """
    The extractive answer: flan-t5-large, greedy, a span or a few words.
    Used by the quiz layer, which compares a reference answer with a
    student's, and by the evaluation scripts, whose numbers describe it.
    """
    tokenizer, _ = _get_model()

    context = _budget_context(tokenizer, query, context_chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    return _fix_number_spacing(complete(prompt, max_new_tokens=128))
