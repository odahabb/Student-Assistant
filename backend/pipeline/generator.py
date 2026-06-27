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


def _budget_context(tokenizer, query: str, raw_context: str) -> str:
    """Truncate raw_context so the full prompt fits within MAX_INPUT_TOKENS."""
    # Tokens consumed by the fixed parts of the template (question + answer tag)
    shell = f"Question: {query}\nContext: \nAnswer:"
    shell_tokens = len(tokenizer.encode(shell, add_special_tokens=True))
    budget = MAX_INPUT_TOKENS - shell_tokens - _CONTEXT_SAFETY_MARGIN
    if budget <= 0:
        return ""
    context_ids = tokenizer.encode(raw_context, add_special_tokens=False)
    if len(context_ids) > budget:
        context_ids = context_ids[:budget]
    return tokenizer.decode(context_ids, skip_special_tokens=True)


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

    raw_context = " ".join(context_chunks)
    context = _budget_context(tokenizer, query, raw_context)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    device = "cpu" if _model_is_ov else get_torch_device()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, max_length=MAX_INPUT_TOKENS).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _fix_number_spacing(answer)
