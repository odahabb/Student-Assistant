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

## Intel Arc GPU / NPU acceleration

The pipeline uses the Intel Arc GPU (torch `xpu` backend) **by default**, controlled by
the `SA_DEVICE` env var (`gpu` | `cpu` | `npu`, default `gpu`). If the XPU torch wheel
isn't installed or no Arc driver is present, it silently falls back to CPU — never hard-fails.

Covers: embedder/retriever (MiniLM), generator (flan-t5-large), loader (Qwen2-VL,
falling back to EasyOCR + BLIP if Qwen2-VL fails to load/run), and the eval notebook/scripts.

### Arc GPU setup (one-time)

The XPU torch wheel is **not** installed by `pip install -r requirements.txt` — it
replaces the default CPU-only torch wheel and pulls from a different package index:

    pip uninstall torch
    pip install torch --index-url https://download.pytorch.org/whl/xpu

Once installed, GPU is used automatically — no env var needed. To force CPU instead
(e.g. for debugging or on a machine without an Arc GPU):

    SA_DEVICE=cpu python ...

**Gotcha:** `requirements.txt` lists plain `torch` (needed so a fresh clone gets a working
CPU install with zero extra steps). Re-running `pip install -r requirements.txt` after
installing the XPU wheel will silently reinstall the CPU-only build, undoing GPU support
with no error. If that happens, just reinstall the XPU wheel again with the command above.

### NPU (OpenVINO — generator + embedder/retriever only, opt-in)

    pip install "optimum[openvino]" openvino

Then run with:

    SA_DEVICE=npu python ...

NPU support via OpenVINO/optimum-intel is solid for `flan-t5-large` and
`all-MiniLM-L6-v2`, but unreliable for BLIP/EasyOCR/Qwen2-VL — those stay GPU/CPU-only
regardless of `SA_DEVICE=npu`. First run exports each model to OpenVINO IR format
(one-time delay); subsequent runs use the cached export from the HuggingFace cache dir.

