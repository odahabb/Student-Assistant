# Multimodal RAG Student Assistant
**Student:** Omar Dahab — 23100704

## Project Structure

multimodal_rag/
├── backend/
│   ├── main.py                  ← FastAPI app (TODO)
│   └── pipeline/
│       ├── loader.py            ←  DONE — input gathering (PDF/image/audio/text)
│       ├── preprocessor.py      ← TODO
│       ├── embedder.py          ← TODO
│       ├── vector_store.py      ← TODO
│       ├── retriever.py         ← TODO
│       └── generator.py         ← TODO
├── frontend/                    ← TODO (React)
├── data/
│   ├── eda/                     ← Saved PNGs from Notebook
│   └── sample/                  ← Has Sample Data for Demo
├── notebooks/
│   └── docvqa_eda.ipynb         ←  DONE
└── requirements.txt

