"""
backend/pipeline/generator.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 6 of pipeline: GENERATION
Answers a question from the retrieved context with Qwen2.5-1.5B-Instruct,
the same model that writes the quiz questions, so one set of weights serves
both. Runs on the Intel Arc GPU by default, with optional NPU and CPU paths
selected by the SA_DEVICE env var (see backend/pipeline/device.py).

Env vars read here:
  SA_MODEL           the in-process model (SA_CHAT_MODEL for the chat view)
  SA_ANSWER_STYLE    "short" or "explain" (see ANSWER_STYLE)
  SA_BACKEND         "transformers" or "ollama" (see BACKEND)
  SA_ABSTAIN         "off", "firm" or "check" (see ABSTAIN)
  SA_TURN_GATE       "rule" or "model" (see TURN_GATE)
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

# Where the model runs. "transformers" (the default) loads the weights into
# this process. "ollama" sends the same messages to a larger model served on
# this machine by Ollama, which listens on 127.0.0.1, so nothing leaves the
# machine either way.
#
# The server is probed once, on the first answer. If it is not running, or is
# not serving OLLAMA_MODEL, the process switches to the in-process model for
# good (see _use_ollama and _fall_back).
BACKEND = os.environ.get("SA_BACKEND", "transformers").lower()
OLLAMA_URL = os.environ.get("SA_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SA_OLLAMA_MODEL", "qwen3:14b")
OLLAMA_TIMEOUT = 900
OLLAMA_PROBE_TIMEOUT = 3
# The in-process model, which also answers whenever Ollama cannot.
LOCAL_MODEL_NAME = MODEL_NAME
if BACKEND == "ollama":
    MODEL_NAME = OLLAMA_MODEL
# Context is budgeted with this tokenizer whichever backend answers, so both
# receive identical passages. Loading a tokenizer loads no weights.
BUDGET_TOKENIZER = os.environ.get("SA_BUDGET_TOKENIZER",
                                  "Qwen/Qwen2.5-1.5B-Instruct")

# Two answering styles, selected by SA_ANSWER_STYLE. Both run the same
# weights under different instructions and decoding settings.
#
#   "short"   — extractive: a span, a number or a few words, decoded greedily.
#               The quiz compares a student's answer against this.
#   "explain" — a short paragraph, sampled, for the chat view.
#
# The library default is "short"; backend/service.py sets "explain" for the
# application. SA_MODEL=google/flan-t5-large selects a seq2seq model instead,
# which loads through a different class (see _is_seq2seq).
ANSWER_STYLE = os.environ.get("SA_ANSWER_STYLE", "short").lower()
CHAT_MODEL_NAME = os.environ.get("SA_CHAT_MODEL", MODEL_NAME)
LOCAL_CHAT_MODEL_NAME = os.environ.get("SA_CHAT_MODEL", LOCAL_MODEL_NAME)

# A short answer takes the shape of its question. question_shape() sorts the
# question into "span", "boolean" or "sentence" in code, and each shape gets
# its own system prompt below, so no single prompt has to choose between the
# three.
SHORT_SYSTEM = (
    "You answer comprehension questions about passages from a student's own "
    "course material. Reply with the words from the passage that answer the "
    "question — a name, a number, a date or a short phrase — and nothing "
    "else: no sentence, no explanation, no label. Give every part the question "
    "asks for, and write numbers and units exactly as the passage does. Answer "
    "unanswerable only when the passages genuinely do not contain the answer."
)
# Used when question_shape() returns "boolean": one word, and its own way to
# decline, so the support gate in answer_short is skipped for these.
BOOLEAN_SYSTEM = (
    "You answer a yes or no question from the passages given. Reply with "
    "exactly one word: Yes, or No, or unanswerable if the passages do not "
    "settle it. Nothing else."
)
# Used when question_shape() returns "sentence".
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
# "How many" and its relatives open like an explanation but want a number, so
# they are matched before _EXPLAINING and answered as a span.
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

# How hard the model is pushed to decline a question its passages do not
# answer, selected by SA_ABSTAIN:
#
#   "off"   — the system prompts above and nothing more (the default). They
#             already allow the reply "unanswerable".
#   "firm"  — the same, plus FIRM_CLAUSE, an explicit instruction not to
#             guess.
#   "check" — a separate yes/no generation first (passages_answer): do these
#             passages contain the answer? The answer is only asked for once
#             that says yes. Two generations instead of one, and skipped for
#             boolean questions (see answer_short).
ABSTAIN = os.environ.get("SA_ABSTAIN", "off").lower()
ABSTAIN_ANSWER = "unanswerable"
FIRM_CLAUSE = (
    " Do not guess and do not answer from your own knowledge: if the answer is "
    "not stated in the passages, the only correct reply is unanswerable."
)
# The gate used by SA_ABSTAIN=check. It asks whether the passages carry the
# information the question is about, not whether they state the answer in so
# many words.
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
# Decoding for the chat view: sampled, unlike the greedy short answer.
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

# Each turn of a conversation is sorted into one of three kinds, which
# decides whether the documents are searched again:
#
#   "new"          stands on its own: retrieve and answer
#   "continuation" about the same material but leaning on what came before:
#                  rewrite it to stand alone (standalone_question), then
#                  retrieve and answer
#   "followup"     about the answer just given rather than the material:
#                  answer from the conversation, retrieve nothing
#
# SA_TURN_GATE picks how the decision is made. "rule" (the default) is the
# test in classify_turn: a turn is a follow-up only when it carries an
# explicit marker about the previous answer (FOLLOWUP_MARKER) and introduces
# no subject matter of its own (introduces_new_subject). "model" asks the
# model to sort the turn instead, under TURN_SYSTEM. Both fall back to
# retrieving when they cannot tell.
TURN_GATE = os.environ.get("SA_TURN_GATE", "rule").lower()
TURN_KINDS = ("new", "continuation", "followup")

# Phrases that talk about the answer rather than about the subject.
FOLLOWUP_MARKER = re.compile(
    r"\b(simpl\w*|rephras\w*|reword\w*|shorter|briefer|summar\w*|again|repeat|"
    r"restate|translat\w*|bullet points?|in (english|arabic|french|spanish)|"
    r"elaborate|clarify)\b|what do you mean|your (last |previous )?answer|"
    r"that answer|(explain|expand on) (that|this|it)\b|"
    r"where did (that|this|it) come from", re.IGNORECASE)
# Words that carry no subject matter, so introduces_new_subject ignores them.
_EMPTY_WORDS = {
    "a", "about", "again", "all", "an", "and", "answer", "any", "are", "as",
    "ask", "at", "be", "briefer", "bullet", "but", "by", "can", "clarify",
    "come", "did", "do", "does", "elaborate", "explain", "expand", "for",
    "from", "get", "give", "how", "i", "in", "is", "it", "its", "just", "know",
    "last", "less", "make", "me", "mean", "more", "my", "of", "on", "one",
    "or", "please", "point", "points", "previous", "put", "question", "repeat",
    "rephrase", "restate", "reword", "say", "sentence", "shorter", "simpler",
    "simply", "so", "some", "summarise", "summarize", "tell", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "translate",
    "up", "us", "was", "way", "we", "were", "what", "when", "where", "which",
    "who", "why", "with", "word", "words", "you", "your",
}


def _content_words(text: str):
    """The subject-matter words of a message, _EMPTY_WORDS removed."""
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", str(text).lower())
            if w not in _EMPTY_WORDS}


def introduces_new_subject(question: str, history: List[dict]) -> bool:
    """
    Whether the message names something the last TURN_HISTORY_MESSAGES
    messages have not mentioned. "say that more simply" does not; "what about
    tournament selection?" does.
    """
    seen = set()
    for message in list(history)[-TURN_HISTORY_MESSAGES:]:
        seen |= _content_words(message.get("content", ""))
    return bool(_content_words(question) - seen)
TURN_SYSTEM = (
    "You sort a student's latest message in a conversation about their course "
    "notes into one of three kinds, and reply with that one word only.\n"
    "new: it asks about a topic the conversation has not been discussing, and "
    "makes sense on its own.\n"
    "continuation: it asks for more about the same topic, and needs the "
    "earlier messages to be understood — a pronoun, or a bare noun phrase.\n"
    "followup: it asks about the assistant's previous answer itself — to "
    "restate it, shorten it, translate it, explain a word in it, or say where "
    "it came from — and needs no new material.\n"
    "Reply with exactly one word: new, continuation, or followup."
)
TURN_MAX_TOKENS = 4
REWRITE_SYSTEM = (
    "You rewrite a student's latest message as a question that stands on its "
    "own, so it can be looked up without the conversation. Replace pronouns "
    "and bare references with what they refer to, keep it to one line, and "
    "change nothing else. Reply with the rewritten question only."
)
REWRITE_MAX_TOKENS = 48
FOLLOWUP_SYSTEM = (
    "You are helping a student revise from their own course material. They "
    "are asking about the answer you just gave, not about new material, so "
    "answer from the conversation. Keep it to two or three sentences. If "
    "answering would need something from their documents that is not in the "
    "conversation, say so in one sentence."
)
# How many recent messages the gate, the rewriter and the follow-up answer
# are given.
TURN_HISTORY_MESSAGES = 4


def _transcript(history: List[dict], limit: int = TURN_HISTORY_MESSAGES) -> str:
    """The last `limit` messages as "Student:" / "Assistant:" lines."""
    lines = []
    for message in list(history)[-limit:]:
        who = "Student" if message.get("role") == "user" else "Assistant"
        body = " ".join(str(message.get("content", "")).split())
        if body:
            lines.append(f"{who}: {body}")
    return "\n".join(lines)


def classify_turn(question: str, history: List[dict]) -> str:
    """
    Whether this message is a new question, a continuation of the topic, or
    a follow-up about the answer just given. Always "new" without a history.

    SA_TURN_GATE selects the rule (the default) or the model; see TURN_GATE.
    """
    if not history:
        return "new"
    if TURN_GATE == "model":
        return _classify_turn_by_model(question, history)
    if FOLLOWUP_MARKER.search(question) and not introduces_new_subject(
            question, history):
        return "followup"
    # Everything else is retrieved for; a continuation is the case where the
    # message has to be rewritten to stand alone first.
    return "continuation" if _leans_on_history(question) else "new"


def _leans_on_history(question: str) -> bool:
    """Does the message depend on what came before to be understood?"""
    return bool(_LEANING.search(question)) or len(question.split()) <= 4


_LEANING = re.compile(
    r"^\s*(and|but|what about|how about|ok|okay|also)\b|"
    r"\b(it|its|that|this|they|them|those|these|there)\b", re.IGNORECASE)


def _classify_turn_by_model(question: str, history: List[dict]) -> str:
    transcript = _transcript(history)
    if not transcript:
        return "new"
    reply = _reply(
        [{"role": "system", "content": TURN_SYSTEM},
         {"role": "user", "content":
             f"Conversation so far:\n{transcript}\n\n"
             f"Latest message: {question}\n\nKind:"}],
        TURN_MAX_TOKENS).strip().lower()
    for kind in TURN_KINDS:
        if reply.startswith(kind):
            return kind
    # An unparseable reply falls back to retrieving.
    return "new"


def standalone_question(question: str, history: List[dict]) -> str:
    """The question rewritten so it can be retrieved on without the history."""
    transcript = _transcript(history)
    if not transcript:
        return question
    rewritten = _reply(
        [{"role": "system", "content": REWRITE_SYSTEM},
         {"role": "user", "content":
             f"Conversation so far:\n{transcript}\n\n"
             f"Latest message: {question}\n\nStandalone question:"}],
        REWRITE_MAX_TOKENS).strip().splitlines()
    first = rewritten[0].strip().strip('"“”') if rewritten else ""
    # A rewrite that is no longer a question, or has run away with it, is
    # discarded in favour of what the student typed.
    if not first.endswith("?") or len(first.split()) > 40:
        return question
    return first


def followup_stream(question: str, history: List[dict]) -> Iterator[str]:
    """Answer about the previous answer, from the conversation alone."""
    messages = [{"role": "system", "content": FOLLOWUP_SYSTEM}]
    for message in list(history)[-TURN_HISTORY_MESSAGES:]:
        role = "user" if message.get("role") == "user" else "assistant"
        body = " ".join(str(message.get("content", "")).split())
        if body:
            messages.append({"role": role, "content": body})
    messages.append({"role": "user", "content": question})
    settings = dict(_chat_settings())
    settings.pop("max_new_tokens", None)
    yield from _stream_reply(messages, CHAT_MAX_TOKENS, settings)


# Loaded models, by name. Short answers, quiz questions and explanations all
# use MODEL_NAME and so share one set of weights, unless SA_CHAT_MODEL points
# the chat view at another.
_loaded = {}
_load_lock = threading.Lock()
_model_is_ov = False


def _is_seq2seq(name: str) -> bool:
    """
    Whether `name` is an encoder-decoder model. These load through
    AutoModelForSeq2SeqLM and have no chat template, so _prompt_inputs
    flattens the messages into a plain prompt for them.
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
        except Exception as e:      # an unusable GPU falls back to the CPU
            log.warning(f"Could not place {name} on {device} ({e}) — using cpu")
            model.to("cpu")
        _loaded[name] = (tokenizer, model)
        return _loaded[name]


def _budget_tokenizer():
    """
    The tokenizer used to measure context against MAX_INPUT_TOKENS. Under the
    transformers backend it is the answering model's own; under Ollama, where
    there is no local tokenizer, BUDGET_TOKENIZER is loaded instead.
    """
    if BACKEND == "ollama":
        global _budget_tok
        if _budget_tok is None:
            _budget_tok = AutoTokenizer.from_pretrained(BUDGET_TOKENIZER)
        return _budget_tok
    return _load(MODEL_NAME)[0]


_budget_tok = None


class OllamaUnavailable(RuntimeError):
    """Ollama could not answer; the caller answers with the in-process model."""


# None until the first answer asks, then True, or False for the rest of the
# process once Ollama has been found missing.
_ollama_up = None


def _use_ollama() -> bool:
    """
    Whether this answer should come from Ollama. The server is checked once,
    on first use: if it is not running, or is running without the model, the
    process switches to the in-process model for good and says so in the log.
    """
    global _ollama_up
    if BACKEND != "ollama" or _ollama_up is False:
        return False
    if _ollama_up is None:
        problem = _ollama_problem()
        if problem:
            _fall_back(problem)
            return False
        _ollama_up = True
    return True


def _ollama_problem() -> Optional[str]:
    """Why Ollama cannot answer, or None when it is serving OLLAMA_MODEL."""
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags",
                                    timeout=OLLAMA_PROBE_TIMEOUT) as response:
            models = _json.loads(response.read()).get("models", [])
    except Exception as exc:
        return f"there is no Ollama server at {OLLAMA_URL} ({exc})"
    names = {m.get("name") for m in models} | {m.get("model") for m in models}
    if not names & {OLLAMA_MODEL, OLLAMA_MODEL + ":latest"}:
        return (f"Ollama is not serving {OLLAMA_MODEL} "
                f"(run: ollama pull {OLLAMA_MODEL})")
    return None


def _fall_back(reason: str) -> None:
    """Answer with the in-process model from now on."""
    global _ollama_up, MODEL_NAME, CHAT_MODEL_NAME
    _ollama_up = False
    MODEL_NAME = LOCAL_MODEL_NAME
    CHAT_MODEL_NAME = LOCAL_CHAT_MODEL_NAME
    log.warning(f"SA_BACKEND=ollama, but {reason}; "
                f"answering with {LOCAL_MODEL_NAME} instead")


def _ollama_open(messages: List[dict], max_new_tokens: int,
                 sampling: Optional[dict], stream: bool):
    """
    Post the same messages the transformers path would build to the model
    Ollama is serving, and return the open response.

    Sampling settings are mapped onto Ollama's options; with none given,
    decoding is greedy, as it is in process. think=False suppresses the
    reasoning block Qwen3 emits by default.
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
                        "stream": stream, "think": False,
                        "options": options}).encode("utf-8")
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", body,
                                     {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT)
    except OSError as exc:     # URLError and HTTPError are both OSErrors
        _fall_back(f"Ollama stopped answering ({exc})")
        raise OllamaUnavailable(str(exc)) from exc


def _ollama_reply(messages: List[dict], max_new_tokens: int,
                  sampling: Optional[dict] = None) -> str:
    """One whole reply from Ollama."""
    import json as _json

    with _ollama_open(messages, max_new_tokens, sampling, stream=False) as response:
        payload = _json.loads(response.read())
    return (payload.get("message", {}).get("content") or "").strip()


def _ollama_stream(messages: List[dict], max_new_tokens: int,
                   sampling: Optional[dict] = None) -> Iterator[str]:
    """
    A reply from Ollama piece by piece. With streaming on, Ollama sends one
    JSON object per line, each carrying the next piece of the message, and a
    final one marked done.
    """
    import json as _json

    with _ollama_open(messages, max_new_tokens, sampling, stream=True) as response:
        try:
            for line in response:
                if not line.strip():
                    continue
                event = _json.loads(line)
                if event.get("error"):
                    raise RuntimeError(f"Ollama: {event['error']}")
                piece = event.get("message", {}).get("content") or ""
                if piece:
                    yield piece
                if event.get("done"):
                    return
        except OSError as exc:
            _fall_back(f"Ollama stopped answering ({exc})")
            raise OllamaUnavailable(str(exc)) from exc


def _stream_reply(messages: List[dict], max_new_tokens: int,
                  sampling: Optional[dict] = None) -> Iterator[str]:
    """
    A reply piece by piece, from Ollama when it is in use and otherwise from
    the in-process model. If Ollama fails before the first piece, the
    in-process model answers instead; once part of an answer has been shown,
    the failure is reported rather than a second answer started under it.
    """
    if _use_ollama():
        started = False
        try:
            for piece in _ollama_stream(messages, max_new_tokens, sampling):
                started = True
                yield piece
            return
        except OllamaUnavailable:
            if started:
                raise
    yield from _stream_messages(messages, max_new_tokens, sampling)


def _reply(messages: List[dict], max_new_tokens: int,
           sampling: Optional[dict] = None) -> str:
    """One reply, from Ollama when it is in use and otherwise in process."""
    if _use_ollama():
        try:
            return _ollama_reply(messages, max_new_tokens, sampling)
        except OllamaUnavailable:
            pass
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
    if _use_ollama():
        try:
            return _fix_number_spacing(
                _ollama_reply(_chat_messages(query, context_chunks),
                              CHAT_MAX_TOKENS, _chat_settings()))
        except OllamaUnavailable:
            pass
    tokenizer, model = _get_chat_model()
    inputs = _chat_inputs(query, context_chunks)
    output = model.generate(**inputs, **_chat_settings())
    return _fix_number_spacing(_decode_reply(tokenizer, inputs, output))


def _stream_messages(messages: List[dict], max_new_tokens: int,
                     sampling: Optional[dict] = None) -> Iterator[str]:
    """
    A reply to these messages, piece by piece, from the transformers backend.
    The generation runs on its own thread; joining it in `finally` means the
    model is free again even when the caller stops early, which happens
    whenever a browser tab is closed mid-answer.
    """
    from transformers import TextIteratorStreamer

    tokenizer, model = _get_chat_model()
    inputs = _prompt_inputs(tokenizer, model, messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    settings = dict(sampling) if sampling else {"do_sample": False}
    failure = []

    def run():
        try:
            model.generate(**inputs, max_new_tokens=max_new_tokens,
                           streamer=streamer, **settings)
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


def explain_stream(query: str, context_chunks: List[str]) -> Iterator[str]:
    """explain(), piece by piece, for the web interface."""
    settings = dict(_chat_settings())
    settings.pop("max_new_tokens", None)
    yield from _stream_reply(_chat_messages(query, context_chunks),
                             CHAT_MAX_TOKENS, settings)


# The token budget shared out among the retrieved passages.
MAX_INPUT_TOKENS = 1024
# A ceiling on the whole prompt, including the system message and the chat
# template around it; _prompt_inputs truncates to it.
MAX_PROMPT_TOKENS = 2048
# Slack left in the budget for tokens that merge across a join (see
# _budget_context).
_CONTEXT_SAFETY_MARGIN = 10

# How much of the budget each rank gets relative to the one above it.
_RANK_DECAY = 0.85


def _allocate_budget(lengths: List[int], budget: int) -> List[int]:
    """
    Split `budget` tokens across chunks whose lengths are given in retrieval
    order, most similar first.

    Each chunk's share is weighted by _RANK_DECAY ** rank, so a higher-ranked
    chunk keeps more of its text, and the surplus from a chunk shorter than
    its share is redistributed to the rest (water-filling). Every chunk gets
    a share, so none is dropped outright for coming last. Returns one token
    count per chunk, in the same order.
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
            # Every share has rounded down to zero: hand what is left to the
            # highest-ranked chunks still short of their full length.
            for i in active[:remaining]:
                allocation[i] += 1
            break

    return allocation


def _budget_context(tokenizer, query: str, context_chunks: List[str]) -> str:
    """
    Fit the retrieved chunks into the token budget left over after the
    prompt template, and return them as one context string.

    When everything fits, the chunks are joined unchanged. When it does not,
    the budget is shared across the chunks by _allocate_budget and each is
    trimmed to its share, so every retrieved chunk is still represented
    rather than the last ones being cut away entirely.
    """
    if isinstance(context_chunks, str):
        context_chunks = [context_chunks]

    # What the fixed parts of the template cost: the question and the tags
    shell = f"Question: {query}\nContext: \nAnswer:"
    shell_tokens = len(tokenizer.encode(shell, add_special_tokens=True))
    budget = MAX_INPUT_TOKENS - shell_tokens - _CONTEXT_SAFETY_MARGIN
    if budget <= 0:
        return ""

    chunk_ids = [tokenizer.encode(c, add_special_tokens=False) for c in context_chunks]
    lengths = [len(ids) for ids in chunk_ids]
    # One token reserved for each space joining two chunks.
    separator_cost = max(0, len(chunk_ids) - 1)

    if sum(lengths) + separator_cost <= budget:
        # Nothing to trim. The encode/decode round trip normalises some OCR
        # artefacts, so it is applied to a context that fits as well.
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

    # Sentencepiece can merge tokens across a join, so the reassembled string
    # may re-tokenize a few tokens longer than the sum of its parts. Measure
    # it again and cut to the budget if it did.
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(context_ids) > budget:
        context = tokenizer.decode(context_ids[:budget], skip_special_tokens=True)

    return context


def _fix_number_spacing(text: str) -> str:
    """
    Close up the spaces a sentencepiece decode leaves inside numbers and
    times ("0. 28", "11 : 39 a. m.", "$ 975. 00").

    Spacing around '.', ',' and ':' is removed when both sides are digits,
    and after '$' or '#' before one. Qwen's tokenizer does not produce this,
    but chunk text decoded elsewhere in the pipeline can carry it into an
    answer through the passages.
    """
    text = re.sub(r'(\d)\s*([.,:])\s*(\d)', r'\1\2\3', text)
    text = re.sub(r'([$#])\s+(\d)', r'\1\2', text)
    return text


# Chunk text is wordpiece-decoded, which puts spaces around punctuation:
# "ilur . am", "bleu - 4", "pubmed + pmc", "vendor lock - in". A span copied
# out of a passage carries that spacing into the answer.
#
# fix_decoded_spacing is applied to short answers only. The dot rule requires
# a lower-case letter or digit after the dot — the shape of a decoded name
# rather than a sentence boundary — so it cannot run two sentences together.
_DECODED_HYPHEN = re.compile(r"(?<=[A-Za-z0-9])\s+-\s+(?=[A-Za-z0-9])")
_DECODED_DOT = re.compile(r"(?<=[A-Za-z0-9])\s*\.\s*(?=[a-z0-9])")
_DECODED_JOINER = re.compile(r"(?<=[A-Za-z0-9])\s*([+/_])\s*(?=[A-Za-z0-9])")
_DECODED_OPEN = re.compile(r"\(\s+")
_DECODED_CLOSE = re.compile(r"\s+\)")


def fix_decoded_spacing(text: str) -> str:
    """Undo the spacing a wordpiece decode leaves around punctuation."""
    out = _DECODED_HYPHEN.sub("-", str(text))
    out = _DECODED_DOT.sub(".", out)
    out = _DECODED_JOINER.sub(r"\1", out)
    out = _DECODED_CLOSE.sub(")", _DECODED_OPEN.sub("(", out))
    return re.sub(r"\s{2,}", " ", out).strip()


def complete(prompt: str, max_new_tokens: int = 128,
             system: str = INSTRUCTION_SYSTEM) -> str:
    """
    Run the model on an arbitrary instruction, decoded greedily, and return
    the reply. Used by the quiz layer to write questions. A prompt longer
    than MAX_PROMPT_TOKENS is truncated from the end, so a caller should keep
    its passage short.
    """
    return _reply([{"role": "system", "content": system},
                   {"role": "user", "content": prompt}], max_new_tokens)


def stream(query: str, context_chunks: List[str]) -> Iterator[str]:
    """
    generate(), yielded piece by piece as the model produces it, for the web
    interface. Joining the pieces and passing them through
    _fix_number_spacing gives generate()'s output.
    """
    if ANSWER_STYLE == "explain":
        yield from explain_stream(query, context_chunks)
        return
    if _use_ollama():
        # A short answer is a few words, so it is yielded whole.
        yield answer_short(query, context_chunks)
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
            streamer.end()   # or the loop below waits forever

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        yield from streamer
    finally:
        # Also reached when the caller stops early, as it does when a browser
        # tab closes mid-answer: wait for the model before releasing it.
        worker.join()
    if failure:
        raise failure[0]


def generate(query: str, context_chunks: List[str]) -> str:
    """
    Answer the query from the retrieved chunks, in whichever style
    SA_ANSWER_STYLE selects (see ANSWER_STYLE).
    """
    if ANSWER_STYLE == "explain":
        return explain(query, context_chunks)
    return answer_short(query, context_chunks)


def _short_messages(query: str, context_chunks: List[str]) -> List[dict]:
    """
    The extractive prompt: "Question / Context / Answer" as the user turn,
    under the system prompt question_shape() selects, with the passages
    trimmed to the token budget by _budget_context.
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
    Whether the passages contain the answer, asked as its own yes/no
    question under SUPPORT_SYSTEM. Used by answer_short when
    SA_ABSTAIN=check.
    """
    context = _budget_context(_budget_tokenizer(), query, context_chunks)
    reply = _reply(
        [{"role": "system", "content": SUPPORT_SYSTEM},
         {"role": "user", "content": f"Passages: {context}\n\nQuestion: {query}"}],
        SUPPORT_MAX_TOKENS).strip().lower()
    # Anything but a clear "no" counts as support, so an unparseable reply
    # leaves the question to be answered normally.
    return not reply.startswith("no")


def answer_short(query: str, context_chunks: List[str]) -> str:
    """
    The extractive answer: greedy decoding, a span or a few words. Used by
    the quiz layer, which compares it with what the student typed.

    The answer's length and system prompt follow question_shape(), and the
    text is tidied of the label, quotes and decoding spacing a model leaves
    on it.
    """
    shape = question_shape(query)
    # The support gate is skipped for a yes or no question: BOOLEAN_SYSTEM
    # carries its own way to decline, and a passage states what a model does
    # rather than stating "yes", so the gate reads almost every boolean
    # question as unsupported.
    if (ABSTAIN == "check" and shape != "boolean"
            and not passages_answer(query, context_chunks)):
        return ABSTAIN_ANSWER

    budget = SENTENCE_MAX_TOKENS if shape == "sentence" else SHORT_MAX_TOKENS
    answer = _reply(_short_messages(query, context_chunks), budget)
    return fix_decoded_spacing(_fix_number_spacing(_tidy_short(answer)))


def _tidy_short(answer: str) -> str:
    """
    Drop the wrapping an instruction-tuned model puts around a short answer:
    an "Answer:" label, surrounding quotes, and a lone trailing full stop.
    """
    answer = re.sub(r"^\s*(answer|a)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip().strip('"“”')
    if answer.count(".") == 1 and answer.endswith("."):
        answer = answer[:-1]
    return answer.strip()
