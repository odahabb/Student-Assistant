# Multimodal RAG Student Assistant
**Student:** Omar Dahab — 23100704

## What this is

A study assistant that answers questions about a student's own course material —
lecture PDFs, slide images, recorded audio — using retrieval-augmented generation
(RAG) instead of a model's general knowledge. The goal is to ground every answer
in the material the student actually provides, across whichever modality it's in.

## How it works

The pipeline turns raw input into a small, queryable knowledge base, then answers
questions against it:

1. **Load** (`loader.py`) — Pull text out of whatever modality came in:
   PyMuPDF for PDFs, Qwen2-VL-2B-Instruct for images (transcription + chart/table
   description, falling back to EasyOCR + BLIP if Qwen2-VL fails to load/run),
   Whisper for audio, plain passthrough for text.
2. **Preprocess** (`preprocessor.py`) — Clean the extracted text and split it into
   overlapping chunks sized for the embedding model.
3. **Embed** (`embedder.py`) — Encode each chunk into a dense vector with
   all-MiniLM-L6-v2.
4. **Store** (`vector_store.py`) — Build/save/load a FAISS `IndexFlatL2` index of
   chunk vectors plus the chunk texts themselves.
5. **Retrieve** (`retriever.py`) — Embed the user's question and pull the top-k
   most similar chunks from the FAISS index.
6. **Generate** (`generator.py`) — Feed the question and retrieved chunks to
   google/flan-t5-large to produce a grounded answer.

Steps 1–6 are implemented and demonstrated end-to-end in
`notebooks/rag_pipeline_demo.ipynb`. What's left is wrapping the pipeline in a
FastAPI service and building the frontend on top of it — there's no app to run
yet, just the pipeline.

## Project Structure

```
Student Assistant/
├── backend/
│   ├── main.py                       ← TODO — FastAPI app, not started
│   └── pipeline/
│       ├── loader.py                 ← DONE — input gathering (PDF/image/audio/text)
│       ├── device.py                 ← DONE — torch device selection (xpu/cpu/npu)
│       ├── preprocessor.py           ← DONE — text cleaning + chunking
│       ├── embedder.py               ← DONE — MiniLM embeddings
│       ├── vector_store.py           ← DONE — FAISS index build/save/load
│       ├── retriever.py              ← DONE — top-k chunk retrieval
│       └── generator.py              ← DONE — flan-t5-large answer generation
├── Frontend/                         ← TODO — not started
├── data/
│   ├── eda/                          ← Saved PNGs/CSV from the DocVQA EDA notebook
│   ├── eval/                         ← EasyOCR+BLIP vs Qwen2-VL eval results
│   ├── raw/docvqa_eval25/            ← 25-sample DocVQA eval set used for that comparison
│   └── Prototype/                    ← Sample PDF/image/audio for manual pipeline testing
├── notebooks/
│   ├── docvqa_eda.ipynb              ← DONE — DocVQA dataset exploration
│   ├── easyocr_blip_vs_qwen2vl_eval.ipynb  ← DONE — image-extraction method comparison
│   ├── rag_pipeline_demo.ipynb       ← DONE — full pipeline walkthrough
│   └── data/processed/               ← FAISS index + chunks produced by the demo notebook
└── requirements.txt
```

