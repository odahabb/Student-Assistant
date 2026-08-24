"""
backend/api.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

FastAPI server for the web interface in frontend/.

Serves the static page and a small JSON API over backend/service.py. Answers
are streamed as server-sent events so the student sees the answer being
written instead of waiting for the whole of it. The server listens on
127.0.0.1 only and the page loads nothing from the internet: documents and
questions never leave the machine.

Run from the repo root:
    python -m backend.api            (then open http://127.0.0.1:8000)
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend import service

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Study Assistant", docs_url="/api/docs", redoc_url=None)


@app.middleware("http")
async def no_caching(request: Request, call_next):
    # Subjects and progress change all the time; never let the browser reuse
    # an earlier API response.
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(service.NotFound)
async def not_found(_: Request, exc: service.NotFound):
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(ValueError)
async def bad_request(_: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(service.IndexNotReady)
async def not_ready(_: Request, exc: service.IndexNotReady):
    detail = ("None of this subject's documents could be read."
              if exc.status.get("state") == "unreadable"
              else "The subject's documents are still being read.")
    return JSONResponse({"detail": detail, "status": exc.status}, status_code=409)


class NewSubject(BaseModel):
    name: str


class Question(BaseModel):
    question: str


class QuizRequest(BaseModel):
    topic: Optional[str] = None


class QuizAnswer(BaseModel):
    id: str
    answer: str


def subject_summary(name: str) -> dict:
    return {"name": name,
            "documents": [{"name": p.name, "size": p.stat().st_size,
                           "type": p.suffix.lower().lstrip(".")}
                          for p in service.documents(name)]}


# Subjects and documents

@app.get("/api/settings")
def get_settings():
    return service.settings()


@app.get("/api/subjects")
def list_subjects():
    return [subject_summary(n) for n in service.subject_names()]


@app.post("/api/subjects", status_code=201)
def create_subject(body: NewSubject):
    return subject_summary(service.create_subject(body.name))


@app.get("/api/subjects/{name}")
def get_subject(name: str):
    return {**subject_summary(name), "index": service.index_status(name)}


@app.get("/api/subjects/{name}/status")
def get_status(name: str):
    return service.index_status(name)


@app.post("/api/subjects/{name}/documents")
async def upload(name: str, files: List[UploadFile] = File(...)):
    service.subject_path(name)
    added, skipped = [], []
    for f in files:
        data = await f.read()
        if await run_in_threadpool(service.add_document, name, f.filename, data):
            added.append(f.filename)
        else:
            skipped.append(f.filename)
    return {"added": added, "skipped": skipped,
            "index": service.index_status(name)}


@app.post("/api/subjects/{name}/samples")
def add_samples(name: str):
    return {"added": service.add_samples(name), "index": service.index_status(name)}


@app.get("/api/subjects/{name}/documents/{filename}")
def open_document(name: str, filename: str):
    path = service.document_path(name, filename)
    return FileResponse(path, content_disposition_type="inline", filename=path.name)


@app.delete("/api/subjects/{name}/documents/{filename}")
def delete_document(name: str, filename: str):
    service.remove_document(name, filename)
    return {"index": service.index_status(name)}


# Asking

@app.get("/api/subjects/{name}/chat")
def get_chat(name: str):
    return service.chat_history(name)


@app.delete("/api/subjects/{name}/chat")
def clear_chat(name: str):
    service.clear_chat(name)
    return {"ok": True}


def _event(payload: dict) -> str:
    return f"event: {payload['type']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/subjects/{name}/ask")
async def ask(name: str, body: Question):
    events = service.ask(name, body.question)
    # Run retrieval before answering, so a missing subject or an index that
    # isn't ready yet comes back as a normal error rather than a broken stream.
    first = await run_in_threadpool(next, events)

    async def stream():
        # The generator holds the model lock while it runs. Advancing it in
        # the thread pool keeps the server responsive, and closing it in
        # `finally` releases the lock even when the browser disconnects.
        try:
            yield _event(first)
            while True:
                payload = await run_in_threadpool(next, events, None)
                if payload is None:
                    break
                yield _event(payload)
        except Exception as exc:
            yield _event({"type": "error", "detail": str(exc)})
        finally:
            await asyncio.shield(run_in_threadpool(events.close))

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# Quiz and progress

@app.get("/api/subjects/{name}/topics")
def get_topics(name: str):
    return service.topics(name)


@app.post("/api/subjects/{name}/quiz")
def next_question(name: str, body: QuizRequest):
    question = service.new_question(name, body.topic)
    if question is None:
        raise HTTPException(422, "Couldn't write a usable question for that topic "
                                 "— try another one.")
    return question


@app.post("/api/subjects/{name}/quiz/answer")
def check_answer(name: str, body: QuizAnswer):
    return service.answer_question(name, body.id, body.answer)


@app.get("/api/subjects/{name}/progress")
def get_progress(name: str):
    return service.progress_summary(name)


@app.delete("/api/subjects/{name}/progress")
def reset_progress(name: str):
    service.reset_progress(name)
    return {"ok": True}


# The page itself

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("SA_PORT", "8000"))
    print(f"\n  Study Assistant: http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port)
