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

The web app organises material into **subjects**, laid out like a project
workspace: a grid of subjects, then one subject at a time with a composer to
start a conversation, a list of recent conversations, and a rail of what that
subject holds (materials, quiz, progress, index). Each subject has three
views over one shared index:

- **Ask** — chat with the subject's documents. Answers appear as they are
  written, and each one lists the three passages it was written from — a page,
  a range of slides, or a moment in a recording ("12:03-15:40"). Opening one
  shows the passage with the answer highlighted, says when the text was read
  out of a picture, and links to that page of the document.
- **Quiz** — questions written from a chosen or recommended section, shown as
  index cards. Difficulty adapts to past answers: multiple choice → short
  answer with a hint → short answer. Keyboard: `N` new question, `1`–`4`
  choose, `Enter` check.
- **Progress** — answers so far, estimated mastery per section, the three
  sections to revise next, and a button to practise each.

The interface is a FastAPI server (`backend/api.py`) over a service layer
(`backend/service.py`), with a hand-written HTML/CSS/JavaScript page in
`frontend/` (no build step, no third-party scripts or fonts). The page is
addressed by the location hash — `#/` for the grid, `#/s/<subject>`,
`#/s/<subject>/c/<conversation>`, `#/s/<subject>/quiz|progress` — so any
screen can be linked to or reloaded. The server
listens on 127.0.0.1 only, so documents and questions stay on the machine.
Indexing runs in the background in two passes: text first, which takes
seconds and makes the subject answerable, then the pictures on pages with
thin or missing text, which take minutes on a deck of diagrams. The picture
pass holds the model lock one page at a time, so a question asked meanwhile
waits for a page rather than for the whole document. Answers are streamed as
server-sent events.

**Reading pictures.** A slide that is one big diagram has no text layer, so
it is rendered at 150 dpi and read: EasyOCR first (about 7 s a page), and
Qwen2-VL only when OCR finds almost nothing and the picture fills the page
(about a minute a page, capped at 8 pages per document). Results are cached
by page image, so re-indexing never repeats the work, and chunks built from
them are marked as read from a picture — the vision model describes a chart
rather than reading its values exactly.

## How it works

| Step | Module | What it does | Model |
|---|---|---|---|
| 1. Load | `loader.py` | Extracts text per modality and records what structure each one has. PDFs keep page numbers and sections (PDF outline, numbered/"Lecture N" headings, or page groups); slide decks are detected and each slide's title read from its largest type; pages whose text layer is thin or empty are rendered and read as pictures; audio keeps Whisper's segments and timestamps. | PyMuPDF · **Qwen2-VL-2B-Instruct** for images (falls back to **EasyOCR + BLIP**) · **Whisper** (base) for audio |
| 2. Preprocess | `preprocessor.py` | Cleans text and produces chunks of at most 220 tokens. A page of prose is *split*: fixed windows with 40-token overlap (the library default), sentence-aware (the app's choice), heading-aware, or semantic. A slide or a spoken segment is smaller than a chunk, so it is *packed* instead — slides up to the limit, breaking at a new title, with title-only slides becoming the section label; segments up to the limit, breaking at pauses of two seconds or more. | bert-base-uncased tokenizer; MiniLM for semantic breaks |
| 3. Embed | `embedder.py` | Encodes chunks as 384-d unit vectors. | **bge-small-en-v1.5** in the app; **all-MiniLM-L6-v2** is the original model and the library default |
| 4. Store | `vector_store.py` | FAISS `IndexFlatL2` plus chunk text and metadata. | — |
| 5. Retrieve | `retriever.py`, `sparse.py` | Top-k (k = 3) chunks for a question. The app uses hybrid retrieval: embedding similarity and BM25 keyword scores, each min-max scaled and mixed 0.4 / 0.6. | same embedding model |
| 6. Generate | `generator.py` | Two answering styles. *Explain* (the chat view): three to five sentences of prose from the retrieved passages, sampled at temperature 0.6. *Short* (the quiz and every evaluation script): an extractive span, sharing the 1,024-token input budget across chunks by rank. | **Qwen2.5-1.5B-Instruct** for explanations; **FLAN-T5-Large** for short answers |
| 7. Quiz | `quiz.py` | Groups chunks into topics — one per section, plus one per whole file — writes questions, keeps only those whose answer survives a round-trip retrieval check, and grades answers. | FLAN-T5-Large; all-MiniLM-L6-v2 for grading |
| 8. Recommend | `recommender.py` | Per-topic ability estimate (online Rasch / Elo update), next-question difficulty, and revision ranking. | — |

`chunk.py` defines `Chunk`, a `str` carrying `source_file`, `page` and
`section`; `device.py` picks the torch device.

## Running it

Requires Python 3.11+.

```bash
pip install -r requirements.txt
python -m backend.api
```

Then open http://127.0.0.1:8000 (`SA_PORT` changes the port). Create a
subject on the Subjects page, upload documents (PDF, PNG/JPG/TIFF/BMP,
MP3/MP4/WAV/M4A, TXT), then ask questions or open the Quiz view. The first
answer and the first quiz question on a topic are slow because models load
and questions are written on demand. Everything a subject remembers is saved
in `data/projects/<subject>/_study/`: conversations in `chats/`, quiz
questions in `quiz_pool.json`, answers in `progress.json`.

**Embedding model.** `SA_EMBEDDER` selects `bge-small` (the app's default,
chosen in the embedding-model comparison below), `multi-qa` or `minilm`.

**Answer style.** `SA_ANSWER_STYLE` selects `short` (the library default:
FLAN-T5-Large, extractive, what the evaluation scripts measure) or `explain`
(Qwen2.5-1.5B-Instruct writing a short paragraph, which the app turns on for
the chat view). The quiz always uses short answers, because it compares a
reference answer with what the student types. FLAN-T5 cannot produce the
paragraph: asked for three to four sentences it returns one of ten words, and
sampling does not change that. `SA_CHAT_MODEL` overrides the model used.

**Device.** `SA_DEVICE` selects `gpu` (Intel Arc via PyTorch XPU), `cpu` or
`npu` (OpenVINO, generator and embedder only). The app defaults to `cpu`;
`SA_DEVICE=gpu python -m backend.api` is much faster when an XPU build of
PyTorch is installed. Any unavailable device falls back to CPU.

Optional acceleration packages (not in `requirements.txt`): the PyTorch XPU
wheel, and `optimum[openvino]` / `openvino` for the NPU path.

## Tests

```bash
python -m unittest discover -s tests -t .
```

182 tests cover loading and section detection, slide detection and titles,
picture reading and its cache, Whisper segments, all four chunking modes,
slide and audio packing, boilerplate stripping, context budgeting, storage,
dense and hybrid retrieval, device fallback, answer styles, quiz generation
and grading, the recommender, the web server (subjects and what the grid
shows, uploads, two-pass indexing, streamed answers, conversations and their
titles, deleting conversations and subjects, quiz and progress endpoints),
and what happens when an upload is empty, corrupt, password-protected, binary
or not what its extension claims. They stub out the models, so they run in a
few seconds without downloading anything.

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
| Embedding models (MiniLM, multi-qa-MiniLM, bge-small, mpnet) × chunking modes (window, sentence, heading, semantic) | `embedder_comparison.py` | `embedder_comparison.json` |
| Retrieval failures vs generation failures at k = 3, per configuration | `[SA_EMBEDDER=...] generation_analysis.py [--chunking ...] [--retrieval dense\|hybrid\|keyword]` | `generation_analysis[_chunking][_embedder][_hybrid\|_keyword].json`, `generation_manual_review.json` |
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
├── backend/
│   ├── api.py                    FastAPI server for the web app
│   ├── service.py                subjects, indexing, asking, quiz, progress
│   ├── pipeline/
│   │   ├── loader.py             input loading, slide/section detection, picture reading
│   │   ├── device.py             torch device selection (gpu / cpu / npu)
│   │   ├── chunk.py              Chunk type (text + file, page, section)
│   │   ├── preprocessor.py       cleaning, chunking, slide/audio packing
│   │   ├── embedder.py           bge-small / MiniLM embeddings
│   │   ├── vector_store.py       FAISS index save / load
│   │   ├── retriever.py          top-k retrieval (dense or hybrid)
│   │   ├── sparse.py             BM25 keyword index
│   │   ├── generator.py          FLAN-T5-Large answering
│   │   ├── quiz.py               topics, question generation, grading
│   │   └── recommender.py        mastery, difficulty, revision ranking
│   └── scripts/                  evaluation scripts (see above)
├── frontend/                     web page: index.html, styles.css, app.js
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

- Retrieval is still the main weakness. The app's configuration (bge-small,
  sentence chunks, hybrid retrieval) answers 20 of the 25 evaluation questions
  correctly, against 12 for the original MiniLM with fixed windows. It was
  chosen from many configurations on those same 25 questions, so the figure is
  optimistic, and the questions were written with the papers' wording, which
  favours keyword matching.
- Qwen2-VL-2B takes about a minute per image on the Arc GPU; exact chart
  values cannot be read reliably by either image method.
- Fewer than a third of generated quiz questions pass the round-trip check on
  research papers (far more on lecture-style notes), so some sections of a
  dense paper may get no questions.
- The recommender has been evaluated on simulated students only.
