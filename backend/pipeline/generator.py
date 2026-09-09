"""
backend/pipeline/generator.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 6 of pipeline: GENERATION
Generates an answer from retrieved context using Qwen2.5-1.5B-Instruct — the
same instruction-tuned model that writes the quiz questions, so the system
loads one language model rather than two.
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

MODEL_NAME = os.environ.get("SA_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Where the model runs. "transformers" loads the weights into this process,
# which is what the application does and what every recorded result
# describes. "ollama" sends the same messages to a model already served on
# this machine by Ollama, which is how a model too large to hold in process —
# qwen3:14b, say — can be put through the identical pipeline for comparison.
# Nothing leaves the machine either way: Ollama listens on 127.0.0.1.
BACKEND = os.environ.get("SA_BACKEND", "transformers").lower()
OLLAMA_URL = os.environ.get("SA_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SA_OLLAMA_MODEL", "qwen3:14b")
OLLAMA_TIMEOUT = 900
if BACKEND == "ollama":
    MODEL_NAME = OLLAMA_MODEL
# Context is budgeted with the same tokenizer whichever backend answers, so
# the passages a model is given are identical and only the model differs.
# Loading a tokenizer does not load any weights.
BUDGET_TOKENIZER = os.environ.get("SA_BUDGET_TOKENIZER",
                                  "Qwen/Qwen2.5-1.5B-Instruct")

# Two answering styles, because the two places an answer is used want
# opposite things. Both are now the same model under different instructions
# and decoding settings, so only one set of weights is ever in memory.
#
#   "short"   — extractive: a span, a number, a few words, decoded greedily.
#               This is what the quiz compares a student's answer against.
#   "explain" — a short paragraph, sampled, for the chat view, where a student
#               asking "what is a fitness function?" wants the idea explained
#               rather than a phrase lifted off a slide.
#
# SA_ANSWER_STYLE selects it. The library default stays "short" so the
# evaluation scripts keep measuring the configuration they were written for;
# backend/service.py turns on "explain" for the application.
#
# Until 2026-09-21 the short style and the quiz ran on google/flan-t5-large.
# Chapter 5's recorded numbers describe that model; SA_MODEL=google/flan-t5-large
# reproduces them (the seq2seq class is still selected automatically).
ANSWER_STYLE = os.environ.get("SA_ANSWER_STYLE", "short").lower()
CHAT_MODEL_NAME = os.environ.get("SA_CHAT_MODEL", MODEL_NAME)

# The shape of a short answer follows the shape of the question. Until
# 2026-09-22 this asked for a span and nothing else, so a question a reader
# would answer "yes" got a phrase lifted off the page instead: on QASPER,
# boolean questions scored 0.0000 and questions wanting a sentence 0.04, while
# span questions scored 0.22. The model knew the answers and returned them in
# the wrong form. The two rules below are what a person would do, not a fit to
# that benchmark — a student asking "does BERT use absolute position
# embeddings?" wants yes or no, not a clause from the middle of a paragraph.
SHORT_SYSTEM = (
    "You answer comprehension questions about passages from a student's own "
    "course material. Reply with the words from the passage that answer the "
    "question — a name, a number, a date or a short phrase — and nothing "
    "else: no sentence, no explanation, no label. Give every part the question "
    "asks for, and write numbers and units exactly as the passage does. Answer "
    "unanswerable only when the passages genuinely do not contain the answer."
)
# A short answer has to take the shape of its question, and asking one prompt
# to choose between three shapes does not work at this model size: given the
# three rules as bullets, Qwen2.5-1.5B answered "No" to "how many TPUs were
# used?". The question is therefore classified here, in code, and each shape
# gets a prompt that asks for one thing.
#
# This is not a fit to QASPER, where the defect showed up as boolean questions
# scoring 0.0000 and how/why questions 0.04 against 0.22 for spans. A student
# asking "does BERT use absolute position embeddings?" wants yes or no, and
# one asking "why does it use them?" wants a sentence.
# Spelling out when to decline, and warning off outside knowledge, was tried
# and cost boolean answers a tenth (0.50 to 0.40 on QASPER). The short form
# is what ships.
BOOLEAN_SYSTEM = (
    "You answer a yes or no question from the passages given. Reply with "
    "exactly one word: Yes, or No, or unanswerable if the passages do not "
    "settle it. Nothing else."
)
SENTENCE_SYSTEM = (
    "You answer a question about passages from a student's own course "
    "material, in one short sentence of your own words drawn only from the "
    "passages. No preamble, no list, no second sentence. Answer unanswerable "
    "only when the passages genuinely do not contain the answer."
)
SENTENCE_MAX_TOKENS = 60

_BOOLEAN_OPENERS = re.compile(
    r"^\s*(do|does|did|is|are|was|were|can|could|will|would|has|have|had|"
    r"should|shall|must|may|might|am)\b", re.IGNORECASE)
# "How many" and its relatives want a number, not a sentence.
_COUNTING = re.compile(r"^\s*how\s+(many|much|long|often|far|big|large|old)\b",
                       re.IGNORECASE)
_EXPLAINING = re.compile(r"^\s*(how|why|in what way|for what reason)\b",
                         re.IGNORECASE)


def question_shape(query: str) -> str:
    """"boolean", "sentence" or "span" — what form the answer should take."""
    q = str(query).strip()
    if _BOOLEAN_OPENERS.match(q):
        return "boolean"
    if _COUNTING.match(q):
        return "span"
    if _EXPLAINING.match(q):
        return "sentence"
    return "span"


SHORT_MAX_TOKENS = 48

# Saying "I don't know" is a skill the model has to be asked for separately.
# Given three passages it will answer almost anything, which is wrong twice
# over: it invents facts for a student, and it forfeits every question whose
# answer is not in the documents at all.
#
# SA_ABSTAIN selects how that is handled, because the choice is a real
# trade-off rather than a bug with one fix, and both directions are measured:
#
#   "off"   — the wording above and nothing more. What every result recorded
#             before 2026-09-21 describes.
#   "firm"  — the same, plus an explicit instruction not to guess.
#   "check" — a separate yes/no question first: do these passages contain the
#             answer? Only then is the answer asked for. Two generations
#             instead of one, and the judgement is made without the pressure
#             of having to produce an answer in the same breath.
#
# Abstention cannot help on a question set where everything is answerable; it
# can only lose answers the model would have got right. The gain is on
# question sets that contain unanswerable questions, and on a student's real
# material, where a confident wrong answer is worse than none.
ABSTAIN = os.environ.get("SA_ABSTAIN", "check").lower()
ABSTAIN_ANSWER = "unanswerable"
FIRM_CLAUSE = (
    " Do not guess and do not answer from your own knowledge: if the answer is "
    "not stated in the passages, the only correct reply is unanswerable."
)
# "state the answer" was too literal a test. A yes/no question's answer is
# never written down anywhere — a passage says what a model does, not "yes" —
# so the gate declined boolean questions almost always, and on a probe where
# the model answers 4 of 4 correctly with the gate off it answered none with
# it on. It now asks whether the passages carry the information the question
# is about, which is the thing the gate was always meant to test.
SUPPORT_SYSTEM = (
    "You decide whether a question can be answered from the passages given, "
    "and nothing else. Reply with one word, yes or no. Reply yes if the "
    "passages contain the information the question asks about, even when the "
    "answer has to be read off rather than copied out. Reply no if answering "
    "would need information the passages do not contain, or your own "
    "knowledge."
)
SUPPORT_MAX_TOKENS = 4

INSTRUCTION_SYSTEM = (
    "Follow the instruction exactly and reply with the requested text only, "
    "with no preamble, label or explanation."
)
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

# Loaded models, by name. Short answers, quiz questions and explanations all
# use MODEL_NAME, so they share one set of weights; SA_CHAT_MODEL can still
# point the chat view at a different one.
_loaded = {}
_load_lock = threading.Lock()
_model_is_ov = False


def _is_seq2seq(name: str) -> bool:
    """
    Encoder-decoder families, which load through a different class and have no
    chat template. Only flan-t5 is expected here, as the reproduction path for
    Chapter 5's measurements.
    """
    return "t5" in name.lower()


def _load(name: str):
    """Tokenizer and model for `name`, loaded once and kept."""
    global _model_is_ov
    with _load_lock:
        if name in _loaded:
            return _loaded[name]

        import torch

        log.info(f"Loading {name} (first use)...")
        device = get_torch_device()
        tokenizer = AutoTokenizer.from_pretrained(name)
        seq2seq = _is_seq2seq(name)

        if should_use_npu():
            try:
                if seq2seq:
                    from optimum.intel.openvino import OVModelForSeq2SeqLM as OVModel
                else:
                    from optimum.intel.openvino import OVModelForCausalLM as OVModel
                model = OVModel.from_pretrained(name, export=True, device="NPU")
                _model_is_ov = True
                _loaded[name] = (tokenizer, model)
                return _loaded[name]
            except Exception as e:
                log.warning(f"NPU generator load failed ({e}), falling back to torch CPU/GPU")

        if seq2seq:
            from transformers import AutoModelForSeq2SeqLM as AutoModel
        else:
            from transformers import AutoModelForCausalLM as AutoModel
        model = AutoModel.from_pretrained(
            name,
            torch_dtype=torch.float16 if device == "xpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        try:
            model.to(device)
        except Exception as e:      # an unusable GPU must not cost the answer
            log.warning(f"Could not place {name} on {device} ({e}) — using cpu")
            model.to("cpu")
        _loaded[name] = (tokenizer, model)
        return _loaded[name]


def _budget_tokenizer():
    """
    The tokenizer used only to measure context against MAX_INPUT_TOKENS. Under
    the transformers backend it is the answering model's own; under Ollama
    there is no local tokenizer, so the default one is loaded and the passages
    come out identical to a transformers run.
    """
    if BACKEND == "ollama":
        global _budget_tok
        if _budget_tok is None:
            _budget_tok = AutoTokenizer.from_pretrained(BUDGET_TOKENIZER)
        return _budget_tok
    return _load(MODEL_NAME)[0]


_budget_tok = None


def _ollama_reply(messages: List[dict], max_new_tokens: int,
                  sampling: Optional[dict] = None) -> str:
    """
    One reply from the model Ollama is serving, given the same messages the
    transformers path would build.

    think=False suppresses the reasoning block Qwen3 emits by default, which
    would otherwise arrive wrapped around every short answer.
    """
    import json as _json
    import urllib.request

    options = {"num_predict": max_new_tokens}
    if sampling:
        options.update({"temperature": sampling.get("temperature", 0.0),
                        "top_p": sampling.get("top_p", 1.0)})
    else:
        options["temperature"] = 0.0      # greedy, as the transformers path is
    body = _json.dumps({"model": OLLAMA_MODEL, "messages": messages,
                        "stream": False, "think": False,
                        "options": options}).encode("utf-8")
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        payload = _json.loads(response.read())
    return (payload.get("message", {}).get("content") or "").strip()


def _reply(messages: List[dict], max_new_tokens: int,
           sampling: Optional[dict] = None) -> str:
    """One reply, from whichever backend is configured."""
    if BACKEND == "ollama":
        return _ollama_reply(messages, max_new_tokens, sampling)
    tokenizer, model = _get_model()
    inputs = _prompt_inputs(tokenizer, model, messages)
    settings = dict(sampling) if sampling else {"do_sample": False}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **settings)
    return _decode_reply(tokenizer, inputs, outputs)


def _get_model():
    return _load(MODEL_NAME)


def _get_chat_model():
    return _load(CHAT_MODEL_NAME)


def _prompt_inputs(tokenizer, model, messages: List[dict]):
    """
    Messages in the model's chat format, ready to generate from. A seq2seq
    model has no chat template, so its instructions are flattened into the
    plain prompt it was trained on.
    """
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    else:
        text = "\n\n".join(m["content"] for m in messages)
    device = "cpu" if _model_is_ov else getattr(model, "device", "cpu")
    return tokenizer([text], return_tensors="pt",
                     truncation=True, max_length=MAX_PROMPT_TOKENS).to(device)


def _decode_reply(tokenizer, inputs, output) -> str:
    """
    The generated text alone. A decoder-only model's output repeats the
    prompt; a seq2seq model's does not.
    """
    tokens = output[0]
    if tokens.shape[-1] > inputs["input_ids"].shape[1] and _looks_like_prompt(
            tokens, inputs["input_ids"][0]):
        tokens = tokens[inputs["input_ids"].shape[1]:]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def _looks_like_prompt(output_tokens, prompt_tokens) -> bool:
    n = prompt_tokens.shape[0]
    return bool((output_tokens[:n] == prompt_tokens).all())


def _chat_messages(query: str, context_chunks: List[str]) -> List[dict]:
    """The question and its passages, as the chat view asks them."""
    if isinstance(context_chunks, str):
        context_chunks = [context_chunks]
    passages = "\n\n".join(f"[{i + 1}] {chunk}"
                           for i, chunk in enumerate(context_chunks))
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {query}"},
    ]


def _chat_inputs(query: str, context_chunks: List[str]):
    """The same messages, tokenised for the transformers backend."""
    tokenizer, model = _get_chat_model()
    return _prompt_inputs(tokenizer, model,
                          _chat_messages(query, context_chunks))


def _chat_settings() -> dict:
    return {"max_new_tokens": CHAT_MAX_TOKENS, "do_sample": True,
            "temperature": CHAT_TEMPERATURE, "top_p": CHAT_TOP_P}


def explain(query: str, context_chunks: List[str]) -> str:
    """A short paragraph answering the question from the retrieved passages."""
    if BACKEND == "ollama":
        return _fix_number_spacing(
            _ollama_reply(_chat_messages(query, context_chunks),
                          CHAT_MAX_TOKENS, _chat_settings()))
    tokenizer, model = _get_chat_model()
    inputs = _chat_inputs(query, context_chunks)
    output = model.generate(**inputs, **_chat_settings())
    return _fix_number_spacing(_decode_reply(tokenizer, inputs, output))


def explain_stream(query: str, context_chunks: List[str]) -> Iterator[str]:
    """explain(), piece by piece, for the web interface."""
    if BACKEND == "ollama":
        # Ollama can stream, but the comparison runs that use this backend do
        # not, so the answer arrives in one piece rather than adding a second
        # streaming path to keep correct.
        yield explain(query, context_chunks)
        return

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


# The budget shared out among the retrieved passages. Qwen2.5 could take far
# more, but keeping the figure means the trimming behaviour Chapter 5
# describes is unchanged, and a short prompt is a fast prompt on a laptop.
MAX_INPUT_TOKENS = 1024
# A hard ceiling on the whole prompt, including the system message and the
# chat template around it.
MAX_PROMPT_TOKENS = 2048
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
    A sentencepiece tokenizer splits digits into subword pieces, and decoding
    those back can leave stray spaces around punctuation inside numbers and
    times (e.g. "0. 28", "11 : 39 a. m.", "$ 975. 00"). Collapse spacing
    immediately around '.', ',', and ':' when both sides are digits. Qwen's
    tokenizer does not do this, but chunk text decoded elsewhere in the
    pipeline can still reach an answer through the passages.
    """
    text = re.sub(r'(\d)\s*([.,:])\s*(\d)', r'\1\2\3', text)
    text = re.sub(r'([$#])\s+(\d)', r'\1\2', text)
    return text


def complete(prompt: str, max_new_tokens: int = 128,
             system: str = INSTRUCTION_SYSTEM) -> str:
    """
    Run the model on an arbitrary instruction (greedy decoding) and return
    what it replied. Shared by answer_short() and the quiz layer, which
    prompts the same model to write questions. Prompts longer than the input
    limit are truncated from the end, so callers should keep their passage
    short.
    """
    return _reply([{"role": "system", "content": system},
                   {"role": "user", "content": prompt}], max_new_tokens)


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
    inputs = _short_inputs(tokenizer, model, query, context_chunks)
    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=not _is_seq2seq(MODEL_NAME),
                                    skip_special_tokens=True)
    failure = []

    def run():
        try:
            model.generate(**inputs, max_new_tokens=SHORT_MAX_TOKENS,
                           do_sample=False, streamer=streamer)
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


def _short_messages(query: str, context_chunks: List[str]) -> List[dict]:
    """
    The extractive prompt: the same "Question / Context / Answer" wording
    flan-t5 was given, now carried as the user turn of a chat prompt under
    SHORT_SYSTEM, which is what keeps an instruction-tuned model from
    answering in a sentence.
    """
    context = _budget_context(_budget_tokenizer(), query, context_chunks)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    system = {"boolean": BOOLEAN_SYSTEM,
              "sentence": SENTENCE_SYSTEM}.get(question_shape(query),
                                               SHORT_SYSTEM)
    if ABSTAIN == "firm":
        system += FIRM_CLAUSE
    return [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]


def _short_inputs(tokenizer, model, query: str, context_chunks: List[str]):
    """The same prompt, tokenised for the transformers backend."""
    return _prompt_inputs(tokenizer, model,
                          _short_messages(query, context_chunks))


def passages_answer(query: str, context_chunks: List[str]) -> bool:
    """
    Whether the passages contain the answer, asked as its own yes/no question.

    Used by answer_short under SA_ABSTAIN=check. The model is far readier to
    say "no" here than to abstain while also being asked for an answer, which
    is the whole point of separating the two.
    """
    context = _budget_context(_budget_tokenizer(), query, context_chunks)
    reply = _reply(
        [{"role": "system", "content": SUPPORT_SYSTEM},
         {"role": "user", "content": f"Passages: {context}\n\nQuestion: {query}"}],
        SUPPORT_MAX_TOKENS).strip().lower()
    # Anything that is not a clear "no" is treated as support, so the cost of
    # an unparseable reply is the old behaviour rather than a lost answer.
    return not reply.startswith("no")


def answer_short(query: str, context_chunks: List[str]) -> str:
    """
    The extractive answer: greedy decoding, a span or a few words. Used by the
    quiz layer, which compares a reference answer with a student's, and by the
    evaluation scripts, whose numbers describe it.
    """
    shape = question_shape(query)
    # The support gate is skipped for a yes or no question. It guards against
    # invented facts, and a yes or no is not a fact to invent — it is one bit,
    # read off the passages, and BOOLEAN_SYSTEM carries its own way to
    # decline. Left in, the gate declines almost every boolean question,
    # because a passage says what a model does rather than saying "yes":
    # on QASPER it took boolean answers from 0.50 to 0.00.
    if (ABSTAIN == "check" and shape != "boolean"
            and not passages_answer(query, context_chunks)):
        return ABSTAIN_ANSWER

    budget = SENTENCE_MAX_TOKENS if shape == "sentence" else SHORT_MAX_TOKENS
    answer = _reply(_short_messages(query, context_chunks), budget)
    return _fix_number_spacing(_tidy_short(answer))


def _tidy_short(answer: str) -> str:
    """
    Drop the wrapping an instruction-tuned model adds to a short answer: a
    repeated "Answer:" label, surrounding quotes, a trailing full stop.
    """
    answer = re.sub(r"^\s*(answer|a)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip().strip('"“”')
    if answer.count(".") == 1 and answer.endswith("."):
        answer = answer[:-1]
    return answer.strip()
