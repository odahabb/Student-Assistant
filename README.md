# Multimodal RAG Student Assistant
**Student:** Omar Dahab — 23100704
**Module:** CM3020 Artificial Intelligence — Project Idea 1, "Orchestrating AI models to achieve a goal"

## What this is

A study assistant that works only from a student's own course material —
lecture PDFs, slide or page images, recorded audio, plain text. It answers
questions with retrieval-augmented generation (RAG), so answers come from the
uploaded material rather than a model's general knowledge, and it quizzes the
student on that material, tracks mastery per section and recommends what to
revise next. Everything runs locally; nothing is sent to an external API.

## Features

The Streamlit app (`app.py`) organises material into **subjects**. Each subject
has three views over one shared index:

- **Ask** — chat with the subject's documents. Every answer lists its sources
  (file, page and section).
- **Quiz** — short questions generated from a chosen or recommended section.
  Difficulty adapts to past answers: multiple choice → short answer with a hint
  → short answer.
- **Progress** — estimated mastery per section, the three sections to revise
  next, and a button to practise each.

## How it works

| Step | Module | What it does | Model |
|---|---|---|---|
| 1. Load | `loader.py` | Extracts text per modality. PDFs keep page numbers and are tagged with their section (from the PDF outline, numbered/"Lecture N" headings, or page groups). | PyMuPDF · **Qwen2-VL-2B-Instruct** for images (falls back to **EasyOCR + BLIP**) · **Whisper** (base) for audio |
| 2. Preprocess | `preprocessor.py` | Cleans text, strips page-1 author/affiliation lines, and splits each page into 220-token windows with 40-token overlap (a sentence-aware mode is also available). | bert-base-uncased tokenizer |
| 3. Embed | `embedder.py` | Encodes chunks as 384-d unit vectors. | **bge-small-en-v1.5** in the app; **all-MiniLM-L6-v2** is the original model and the library default |
| 4. Store | `vector_store.py` | FAISS `IndexFlatL2` plus chunk text and metadata. | — |
| 5. Retrieve | `retriever.py` | Top-k (k = 3) nearest chunks for a question. | same embedding model |
| 6. Generate | `generator.py` | Answers from the retrieved chunks, sharing the 1,024-token input budget across them by rank. | **FLAN-T5-Large** |
| 7. Quiz | `quiz.py` | Groups chunks into topics (sections), writes questions, keeps only those whose answer survives a round-trip retrieval check, and grades answers. | FLAN-T5-Large; all-MiniLM-L6-v2 for grading |
| 8. Recommend | `recommender.py` | Per-topic ability estimate (online Rasch / Elo update), next-question difficulty, and revision ranking. | — |

`chunk.py` defines `Chunk`, a `str` carrying `source_file`, `page` and
`section`; `device.py` picks the torch device.

## Running it

Requires Python 3.11+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a subject in the sidebar, upload documents (PDF, PNG/JPG/TIFF/BMP,
MP3/MP4/WAV/M4A, TXT), then ask questions or open the Quiz view. The first
answer and the first quiz question on a topic are slow because models load
and questions are written on demand. Quiz questions and progress are saved in
`data/projects/<subject>/_study/`.

**Embedding model.** `SA_EMBEDDER` selects `bge-small` (the app's default,
chosen in the embedding-model comparison below) or `minilm`.

**Device.** `SA_DEVICE` selects `gpu` (Intel Arc via PyTorch XPU), `cpu` or
`npu` (OpenVINO, generator and embedder only). The app defaults to `cpu`;
`SA_DEVICE=gpu streamlit run app.py` is much faster when an XPU build of
PyTorch is installed. Any unavailable device falls back to CPU.

Optional acceleration packages (not in `requirements.txt`): the PyTorch XPU
wheel, and `optimum[openvino]` / `openvino` for the NPU path.

## Tests

```bash
python -m unittest discover -s tests -t .
```

85 unit tests cover loading and section detection, chunking (both modes),
boilerplate stripping, context budgeting, storage and retrieval, device
fallback, quiz generation and grading, and the recommender. They stub out the
models, so they run in under a second without downloading anything.

## Evaluation

Scripts live in `backend/scripts/`; results are committed in `data/eval/`.

| Question | Script | Output |
|---|---|---|
| Image extraction: Qwen2-VL-2B vs EasyOCR+BLIP on 25 DocVQA questions | `notebooks/easyocr_blip_vs_qwen2vl_eval.ipynb` | `easyocr_blip_vs_qwen2vl_results.csv` |
| Both evaluation datasets described; image results re-scored with strict EM and ANLS | `dataset_and_metrics.py` | `dataset_and_metrics.json`, `published_baselines.json` |
| Retrieval Recall@1/3/5 on 25 hand-labelled questions | `eval_recall.py` | `recall_results.json` |
| MRR and similarity-gap diagnostics | `rank_diagnostics.py` | `rank_diagnostics.json` |
| Is the corpus mix to blame? (one index per document) | `ablation_single_doc.py` | `recall_ablation_single_doc.json` |
| What outranks the correct chunk? | `competitor_analysis.py` | `competitor_analysis.json` |
| Chunking and section-context variants | `retrieval_variants.py` | `retrieval_variants.json` |
| Embedding models: MiniLM, multi-qa-MiniLM, bge-small, mpnet | `embedder_comparison.py` | `embedder_comparison.json` |
| Retrieval failures vs generation failures at k = 3 | `[SA_EMBEDDER=bge-small] generation_analysis.py [--chunking sentence]` | `generation_analysis*.json`, `generation_manual_review.json` |
| Quiz answer grader calibration | `eval_grader.py` | `grader_calibration.json`, `grader_decisions.csv` |
| Quiz question generation (plus blind rating sheet) | `eval_quiz_generation.py [score]` | `quiz_generation.json`, `quiz_rating_sheet.csv` |
| Recommender, on simulated students | `eval_recommender.py` | `recommender_simulation.json` |
| Vector/metadata alignment self-check | `selfcheck_alignment.py` | `selfcheck_alignment_results.json` |
| Report figures, drawn from the results above | `make_report_figures.py` | `figures/*.png` |

`easyocr_blip_vs_qwen2vl_results_cpu_partial.csv` is an earlier, partial CPU
run kept for reference; the reported numbers come from
`easyocr_blip_vs_qwen2vl_results.csv`.

### Data

Evaluation inputs are not redistributed and live in the git-ignored
`data/raw/`:

- `data/raw/docvqa_eval25/` — the first 25 questions of the
  [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
  validation split (`eval_set.csv` + images), exported by the evaluation
  notebook.
- The four retrieval-evaluation papers, saved under these names:
  `Whisper.pdf` (Radford et al., 2022, arXiv:2212.04356),
  `Flant5pdf.pdf` (Chung et al., 2022, arXiv:2210.11416),
  `embedding.pdf` (Sajja, Sermet and Demir, 2024,
  github.com/uihilab/educational-qa-embeddings) and
  `Hallucinations_in_Large_Language_Models_LLMs.pdf`
  (Reddy, Kumar and Prakash, 2024, IEEE eStream).

`data/Prototype/` holds small sample inputs (a six-page lecture-notes PDF,
an image and an audio clip) that are in the repository.

## Project structure

```
Student Assistant/
├── app.py                        Streamlit app (Ask / Quiz / Progress)
├── backend/
│   ├── pipeline/
│   │   ├── loader.py             input loading + PDF section detection
│   │   ├── device.py             torch device selection (gpu / cpu / npu)
│   │   ├── chunk.py              Chunk type (text + file, page, section)
│   │   ├── preprocessor.py       cleaning, boilerplate stripping, chunking
│   │   ├── embedder.py           bge-small / MiniLM embeddings
│   │   ├── vector_store.py       FAISS index save / load
│   │   ├── retriever.py          top-k retrieval
│   │   ├── generator.py          FLAN-T5-Large answering
│   │   ├── quiz.py               topics, question generation, grading
│   │   └── recommender.py        mastery, difficulty, revision ranking
│   └── scripts/                  evaluation scripts (see above)
├── tests/                        unittest suite
├── data/
│   ├── eval/                     evaluation results
│   ├── eda/                      DocVQA exploration outputs
│   ├── Prototype/                sample inputs
│   ├── raw/                      evaluation inputs (git-ignored)
│   └── projects/                 subjects created in the app (git-ignored)
├── notebooks/
│   ├── docvqa_eda.ipynb                     DocVQA exploration
│   ├── easyocr_blip_vs_qwen2vl_eval.ipynb   image-extraction comparison
│   ├── rag_pipeline_demo.ipynb              pipeline walkthrough
│   └── ask_question_demo.ipynb              question-answering demo
└── requirements.txt
```

## Known limitations

- Retrieval is still the main weakness. With bge-small the app answers 18 of
  the 25 evaluation questions correctly (12 with the original MiniLM), and the
  correct page is missing from the top 5 for 6 of them. The configuration was
  chosen on those same 25 questions, so these figures are optimistic.
- Qwen2-VL-2B takes about a minute per image on the Arc GPU; exact chart
  values cannot be read reliably by either image method.
- Fewer than a third of generated quiz questions pass the round-trip check on
  research papers (far more on lecture-style notes), so some sections of a
  dense paper may get no questions.
- The recommender has been evaluated on simulated students only.
