"""
backend/api.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

FastAPI server for the web interface in frontend/.

Serves the static page and a JSON API over backend/service.py. Answers are
streamed as server-sent events, so the page can show the answer as it is
written. The server listens on 127.0.0.1 only and the page loads nothing
from the internet, so documents and questions never leave the machine.

service.py raises the errors; the exception handlers below turn each into
its status code: NotFound into 404, ValueError into 400, IndexNotReady into
409 and RuntimeError into 500.

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
    """Keep the browser from reusing an API response, and have it revalidate
    the page and its script on every load."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    else:
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(service.NotFound)
async def not_found(_: Request, exc: service.NotFound):
    """A subject, document, conversation or question that does not exist."""
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(ValueError)
async def bad_request(_: Request, exc: ValueError):
    """A request the student can correct: an empty question, a taken name."""
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(RuntimeError)
async def server_problem(_: Request, exc: RuntimeError):
    """A failure the student is told about in words, not as a 500 page."""
    return JSONResponse({"detail": str(exc)}, status_code=500)


@app.exception_handler(service.IndexNotReady)
async def not_ready(_: Request, exc: service.IndexNotReady):
    """The subject was asked something before its documents were read."""
    detail = ("None of this subject's documents could be read."
              if exc.status.get("state") == "unreadable"
              else "The subject's documents are still being read.")
    return JSONResponse({"detail": detail, "status": exc.status}, status_code=409)


class NewSubject(BaseModel):
    """The name for a new subject, or the new name of an existing one."""

    name: str


class Question(BaseModel):
    """A question, and the conversation to add the exchange to."""

    question: str
    chat: Optional[str] = None      # a new conversation when absent


class QuizRequest(BaseModel):
    """Which topic to be quizzed on; the recommender chooses when absent."""

    topic: Optional[str] = None


class QuizAnswer(BaseModel):
    """An answer to a question that was handed out earlier, by its id."""

    id: str
    answer: str


def subject_summary(name: str) -> dict:
    """A subject and its documents, as every subject response carries them."""
    return {"name": name,
            "documents": [{"name": p.name, "size": p.stat().st_size,
                           "type": p.suffix.lower().lstrip(".")}
                          for p in service.documents(name)]}


# Subjects and documents

@app.get("/api/settings")
def get_settings():
    """The configuration the page displays and what it may offer."""
    return service.settings()


@app.get("/api/subjects")
def list_subjects():
    """Every subject, with what the subjects grid shows for each."""
    return [service.subject_overview(n) for n in service.subject_names()]


@app.get("/api/chats")
def recent_chats(limit: int = 12):
    """Recent conversations across all subjects, for the sidebar."""
    return service.recent_chats(limit)


@app.post("/api/subjects", status_code=201)
def create_subject(body: NewSubject):
    """Create a subject."""
    return subject_summary(service.create_subject(body.name))


@app.get("/api/subjects/{name}")
def get_subject(name: str):
    """One subject and its index status, starting a build if one is due."""
    return {**subject_summary(name), "index": service.index_status(name)}


@app.patch("/api/subjects/{name}")
def rename_subject(name: str, body: NewSubject):
    """Rename a subject, keeping its documents, conversations, quiz and
    progress."""
    renamed = service.rename_subject(name, body.name)
    return {**subject_summary(renamed), "index": service.index_status(renamed)}


@app.delete("/api/subjects/{name}")
def delete_subject(name: str):
    """Delete a subject and everything in it; the page confirms first."""
    service.delete_subject(name)
    return {"ok": True}


@app.get("/api/subjects/{name}/status")
def get_status(name: str):
    """Where a subject's index stands, which the page polls while it builds."""
    return service.index_status(name)


@app.post("/api/subjects/{name}/documents")
async def upload(name: str, files: List[UploadFile] = File(...)):
    """Add documents to a subject, skipping names it already holds."""
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
    """Copy the sample documents into a subject."""
    return {"added": service.add_samples(name), "index": service.index_status(name)}


@app.get("/api/subjects/{name}/documents/{filename}")
def open_document(name: str, filename: str):
    """Serve a document for the browser to display in place."""
    path = service.document_path(name, filename)
    return FileResponse(path, content_disposition_type="inline", filename=path.name)


@app.delete("/api/subjects/{name}/documents/{filename}")
def delete_document(name: str, filename: str):
    """Remove one document from a subject."""
    service.remove_document(name, filename)
    return {"index": service.index_status(name)}


# Asking

@app.get("/api/subjects/{name}/chats")
def list_chats(name: str):
    """A subject's conversations, most recently used first."""
    return service.chat_list(name)


@app.post("/api/subjects/{name}/chats", status_code=201)
def new_chat(name: str):
    """Start an empty conversation."""
    return service.create_chat(name)


@app.get("/api/subjects/{name}/chats/{chat_id}")
def get_chat(name: str, chat_id: str):
    """One conversation, with its messages."""
    return service.chat(name, chat_id)


@app.delete("/api/subjects/{name}/chats/{chat_id}")
def delete_chat(name: str, chat_id: str):
    """Delete one conversation."""
    service.delete_chat(name, chat_id)
    return {"ok": True}


@app.delete("/api/subjects/{name}/chats")
def clear_chats(name: str):
    """Delete every conversation in a subject."""
    service.clear_chat(name)
    return {"ok": True}


def _event(payload: dict) -> str:
    """One service.ask event as a server-sent event frame."""
    return f"event: {payload['type']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/subjects/{name}/ask")
async def ask(name: str, body: Question):
    """Answer a question, streamed as server-sent events (see service.ask)."""
    events = service.ask(name, body.question, body.chat)
    # The first event is taken before the response begins, so a missing
    # subject or an index that is not ready is a normal error response rather
    # than a broken stream.
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
    """The topics a subject can be quizzed on."""
    return service.topics(name)


@app.post("/api/subjects/{name}/quiz")
def next_question(name: str, body: QuizRequest):
    """Hand out the next quiz question, on a given topic or a recommended one."""
    question = service.new_question(name, body.topic)
    if question is None:
        raise HTTPException(422, "Couldn't write a usable question for that topic "
                                 "— try another one.")
    return question


@app.post("/api/subjects/{name}/quiz/answer")
def check_answer(name: str, body: QuizAnswer):
    """Grade an answer and record it against the topic's mastery."""
    return service.answer_question(name, body.id, body.answer)


@app.get("/api/subjects/{name}/progress")
def get_progress(name: str):
    """Everything the progress screen shows for a subject."""
    return service.progress_summary(name)


@app.delete("/api/subjects/{name}/progress")
def reset_progress(name: str):
    """Clear a subject's quiz history, keeping its questions."""
    service.reset_progress(name)
    return {"ok": True}


# The page itself

@app.get("/", include_in_schema=False)
def index():
    """The page itself; everything beside it is served as a static file."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("SA_PORT", "8000"))
    print(f"\n  Study Assistant: http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port)
