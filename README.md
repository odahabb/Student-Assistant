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
| 6. Generate | `generator.py` | Two answering styles. *Explain* (the chat view): three to five sentences of prose from the retrieved passages, sampled at temperature 0.6. *Short* (the quiz and every evaluation script): an extractive span, sharing the 1,024-token input budget across chunks by rank. | **Qwen2.5-1.5B-Instruct** for both |
| 7. Quiz | `quiz.py` | Groups chunks into topics — one per section, plus one per whole file — writes questions, keeps only those whose answer survives a round-trip retrieval check, and grades answers. | **Qwen2.5-1.5B-Instruct**; all-MiniLM-L6-v2 for grading |
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
extractive, decoded greedily, what the evaluation scripts measure) or
`explain` (a short paragraph, sampled at temperature 0.6, which the app turns
on for the chat view). The quiz always uses short answers, because it compares
a reference answer with what the student types.

**Declining to answer.** `SA_ABSTAIN` selects `off` (the default), `check` (a
separate yes/no generation asks whether the passages contain the answer before
the answer is requested) or `firm` (an instruction not to guess). `check` is
off by default because it refuses too much: across the QASPER dev set it
declined 351 questions and only 73 deserved it, costing 8 points of extractive
and 7 of abstractive answer F1 to buy 64 on the unanswerable class — a wash
overall, for a second generation on every question. It earns its place only
where a large share of the questions have no answer in the documents at all,
which is not what a student asks of their own notes.

**Language model.** One model answers, explains and writes the quiz questions:
`SA_MODEL` (default `Qwen/Qwen2.5-1.5B-Instruct`), with `SA_CHAT_MODEL`
overriding it for the chat view alone. Until 21 September 2026 the short
answers and the quiz ran on `google/flan-t5-large`, which is what the recorded
evaluation numbers in `data/eval/` without a model suffix describe;
`SA_MODEL=google/flan-t5-large` reproduces them. On the 25 ground-truth
questions flan-t5-large answers 20 correctly and Qwen2.5-1.5B-Instruct 18, so
the move to one model costs two answers and saves loading a second model.

**Quality mode (optional).** `SA_BACKEND=ollama` sends the same prompts to a
larger model served locally by [Ollama](https://ollama.com), `qwen3:14b` by
default (`SA_OLLAMA_MODEL`). Retrieval, chunks and prompts are unchanged; only
the answering model differs. On the QASPER dev set it scores 0.339 answer F1
against the 1.5B's 0.217, and on the 25 ground-truth questions 20 against 18.
The cost is speed: a paragraph is written at about 7 tokens a second against
48, so answers stream in over fifteen seconds or so rather than one or two.
It needs Ollama installed and `ollama pull qwen3:14b` (9.3 GB). It cannot be
loaded in process instead: in 16-bit the model needs about 30 GB, and Ollama
serves a 4-bit copy that fits on the Arc in 9.6 GB. Nothing leaves the
machine, because Ollama listens on 127.0.0.1. If Ollama is not running, or is
not serving the model, the app logs why and answers with the 1.5B instead.

```bash
SA_BACKEND=ollama python -m backend.api
```

**Device.** `SA_DEVICE` selects `gpu` (Intel Arc via PyTorch XPU, the app's
default), `cpu` or `npu` (OpenVINO, generator and embedder only). Any
unavailable device falls back to CPU, so `gpu` is safe on a machine with
neither the XPU wheel nor the Arc driver. The `npu` path was tested on this
laptop's Intel AI Boost NPU and does not load: both models are exported with
dynamic input shapes, which the NPU compiler rejects, so it falls back. Run
through OpenVINO GenAI with fixed shapes, the 1.5B does answer on the NPU, but
at about 19 tokens a second and 2 s to the first word, slower than the Arc GPU. The GPU is several times faster on
every stage that runs a model (`latency_gpu.json` against `latency_cpu.json`):
a short answer 0.45s against 3.38s, a paragraph 1.45s against 9.36s, a quiz
question 1.53s against 7.93s.

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

Every evaluation is a notebook in `notebooks/`; results are committed in
`data/eval/`. Each notebook opens with the evaluation's own code and a
`RUN = False` switch: as committed it only loads and shows the saved results,
and `RUN = True` measures again and overwrites them. Run them with the Python
environment that has the project's requirements installed; `notebooks/eval_common.py`
holds the answer-grading rule they share.

| Notebook | What it measures | Results |
|---|---|---|
| `01_eval_pdf_text.ipynb` | Prose PDFs: QASPER (answer and evidence F1 against the published baselines, abstention on its unanswerable questions), and the 25 hand-labelled questions split into retrieval and generation failures | `qasper_*.json`, `generation_analysis_*.json` |
| `02_eval_images.ipynb` | Document images: the DocVQA subset described and the image reader's answers scored with strict EM and ANLS | `dataset_and_metrics.json` |
| `03_eval_slides.ipynb` | Slide decks: packing vs one chunk per slide, answers on the student's own decks, reading slides as pictures | `slide_chunking.json`, `own_material.json`, `figure_reading.json` |
| `04_eval_audio.ipynb` | Recordings: SLUE-SQA-5, Whisper against the reference transcript, split by whether the clip really answers the question | `spoken_qa_300q*.json`, `spoken_qa_support_labels.json` |
| `05_choice_image_reader.ipynb` | Qwen2-VL-2B vs EasyOCR+BLIP on 25 DocVQA questions | `easyocr_blip_vs_qwen2vl_results.csv` |
| `06_choice_embedder.ipynb` | Embedding model × chunking mode, and the hybrid weight on both question sets | `embedder_comparison.json`, `hybrid_weight_sweep[_slides].json` |
| `07_choice_answer_model.ipynb` | flan-t5-large vs Qwen2.5-1.5B vs qwen3:14b, from the saved runs of 01 and 09 | (reads the files above) |
| `08_choice_whisper.ipynb` | Whisper base vs small vs turbo, from the saved runs of 04 | (reads `spoken_qa_300q*.json`) |
| `09_eval_quiz.ipynb` | Quiz question generation and the answer grader's calibration | `quiz_generation*.json`, `grader_calibration*.json` |
| `10_eval_recommender.ipynb` | The recommender, on simulated students | `recommender_simulation.json` |
| `11_eval_system.ipynb` | Time per pipeline stage on GPU and CPU; whether retrieval does the work | `latency_*.json`, `no_retrieval.json` |

Results from evaluations that were retired are kept in `data/eval/` for the
record: the retrieval diagnostics (`rank_diagnostics.json`,
`recall_ablation_single_doc.json`, `competitor_analysis.json`,
`retrieval_variants.json`, `selfcheck_alignment_results.json`,
`recall_results*.json`).
`easyocr_blip_vs_qwen2vl_results_cpu_partial.csv` is an earlier, partial CPU
run; the reported numbers come from `easyocr_blip_vs_qwen2vl_results.csv`.

`backend/scripts/` keeps the tools that are not evaluations:
`build_slide_ground_truth.py` (writes `slide_ground_truth.json`),
`draft_candidate_ground_truth.py` and `make_report_figures.py` (draws
`data/eval/figures/*.png` from the results). `backend/scripts/vendor/` holds
QASPER's official evaluator, copied unchanged so those numbers mean what they
mean in the paper.

### Data

Evaluation inputs are not redistributed and live in the git-ignored
`data/raw/`:

- `data/raw/docvqa_eval25/` — the first 25 questions of the
  [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
  validation split (`eval_set.csv` + images), exported by
  `notebooks/05_choice_image_reader.ipynb`.
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
│   │   ├── generator.py          Qwen2.5-1.5B-Instruct answering
│   │   ├── quiz.py               topics, question generation, grading
│   │   └── recommender.py        mastery, difficulty, revision ranking
│   └── scripts/                  ground-truth and figure tools
├── frontend/                     web page: index.html, styles.css, app.js
├── tests/                        unittest suite
├── data/
│   ├── eval/                     evaluation results
│   ├── eda/                      DocVQA exploration outputs
│   ├── Prototype/                sample inputs
│   ├── raw/                      evaluation inputs (git-ignored)
│   └── projects/                 subjects created in the app (git-ignored)
├── notebooks/
│   ├── 01_eval_pdf_text.ipynb … 11_eval_system.ipynb   evaluations (see above)
│   ├── eval_common.py                       helpers the evaluations share
│   ├── docvqa_eda.ipynb                     DocVQA exploration
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
