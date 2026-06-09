"""
backend/pipeline/generator.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 6 of pipeline: GENERATION
Generates an answer from retrieved context using google/flan-t5-large (CPU only).
"""

from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-large"

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to("cpu")
    return _tokenizer, _model


def generate(query: str, context_chunks: List[str]) -> str:
    """
    Build a prompt from the query and context chunks, run inference on CPU
    using flan-t5-large, and return the generated answer string.
    """
    tokenizer, model = _get_model()

    context = " ".join(context_chunks)
    prompt = f"Context: {context} Question: {query} Answer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=128)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
