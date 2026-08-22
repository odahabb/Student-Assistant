"""
Tests for the web interface's server side (backend/service.py and
backend/api.py). The pipeline is replaced with small fakes, so no model is
loaded and nothing is downloaded.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

# The service sets its default device and embedding model in os.environ on
# import; undo that so other test modules see the environment they expect.
with mock.patch.dict(os.environ):
    from backend import api, service
from backend.pipeline import quiz
from backend.pipeline.chunk import Chunk

PASSAGE = ("Whisper was trained on 680,000 hours of multilingual and multitask "
           "supervised data collected from the web, which the authors show leads "
           "to robustness to accents, background noise and technical language. "
           "The model is an encoder-decoder Transformer that takes thirty seconds "
           "of audio at a time and predicts the transcript one token after another.")


def fake_preprocess(loaded, source_file=None, chunking=None):
    return [Chunk(f"{PASSAGE} ({source_file} part {i})", source_file, i + 1,
                  f"Section {i + 1}") for i in range(2)]


def fake_embed(chunks, **kwargs):
    return np.eye(len(chunks), 8, dtype=np.float32)


def fake_retrieve(query, index, chunks, k=3, sparse=None):
    return list(chunks[:k])


def fake_stream(question, context):
    yield from ["680", ",", "000 hours"]


def fake_item(chunk, retrieve_fn=None, **kwargs):
    n = int(str(chunk).rsplit("part ", 1)[1].rstrip(")"))
    return quiz.QuizItem(topic_id="", question=f"How many hours were used, version {n}?",
                         answer=f"{681 + n} hours in {chunk.source_file[0]}", source_file=chunk.source_file,
                         page=chunk.page, section=chunk.section,
                         passage=str(chunk)), None


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.projects = Path(self.tmp.name)
        patches = [
            mock.patch.object(service, "PROJECTS_DIR", self.projects),
            mock.patch.object(service, "load_file", return_value=[]),
            mock.patch.object(service, "preprocess", side_effect=fake_preprocess),
            mock.patch.object(service, "embed", side_effect=fake_embed),
            mock.patch.object(service, "retrieve", side_effect=fake_retrieve),
            mock.patch.object(service, "_stream_answer", side_effect=fake_stream),
            mock.patch.object(service, "_tidy", side_effect=lambda s: s.strip()),
            mock.patch.object(quiz, "generate_item", side_effect=fake_item),
            mock.patch.object(quiz, "grade", side_effect=lambda a, r: quiz.Grade(
                float(a == r), quiz.normalize(a) == quiz.normalize(r), "test")),
            mock.patch.dict(service._indexes, clear=True),
            mock.patch.dict(service._building, clear=True),
            mock.patch.dict(service._issued, clear=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(api.app)

    def make_subject(self, name="Speech", files=("whisper.pdf",)):
        folder = self.projects / name
        folder.mkdir()
        for f in files:
            (folder / f).write_bytes(b"%PDF-1.4 test")
        return name

    def wait_ready(self, name):
        for _ in range(200):
            status = self.client.get(f"/api/subjects/{name}/status").json()
            if status["state"] != "indexing":
                return status
            time.sleep(0.01)
        self.fail("index never finished")


class SubjectTests(ServiceTestCase):
    def test_create_and_list_subjects(self):
        created = self.client.post("/api/subjects", json={"name": "  Statistics: 101 "})
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["name"], "Statistics 101")
        self.client.post("/api/subjects", json={"name": "Biology"})
        names = [s["name"] for s in self.client.get("/api/subjects").json()]
        self.assertEqual(names, ["Biology", "Statistics 101"])

    def test_unusable_name_is_rejected(self):
        response = self.client.post("/api/subjects", json={"name": "///"})
        self.assertEqual(response.status_code, 400)

    def test_unknown_subject_is_404(self):
        self.assertEqual(self.client.get("/api/subjects/Nope").status_code, 404)
        self.assertEqual(self.client.get("/api/subjects/..").status_code, 404)

    def test_upload_skips_existing_and_rejects_unknown_types(self):
        name = self.make_subject()
        response = self.client.post(f"/api/subjects/{name}/documents", files=[
            ("files", ("whisper.pdf", b"x", "application/pdf")),
            ("files", ("notes.txt", b"some notes", "text/plain")),
        ])
        self.assertEqual(response.json()["added"], ["notes.txt"])
        self.assertEqual(response.json()["skipped"], ["whisper.pdf"])
        bad = self.client.post(f"/api/subjects/{name}/documents",
                               files=[("files", ("run.exe", b"x", "application/octet-stream"))])
        self.assertEqual(bad.status_code, 400)
        self.assertFalse((self.projects / name / "run.exe").exists())

    def test_upload_cannot_escape_the_subject_folder(self):
        name = self.make_subject()
        self.client.post(f"/api/subjects/{name}/documents",
                         files=[("files", ("../../escape.txt", b"x", "text/plain"))])
        self.assertTrue((self.projects / name / "escape.txt").exists())
        self.assertFalse((self.projects.parent / "escape.txt").exists())

    def test_documents_can_be_opened_and_removed(self):
        name = self.make_subject()
        opened = self.client.get(f"/api/subjects/{name}/documents/whisper.pdf")
        self.assertEqual(opened.content, b"%PDF-1.4 test")
        self.assertEqual(self.client.get(f"/api/subjects/{name}/documents/other.pdf").status_code, 404)
        self.client.delete(f"/api/subjects/{name}/documents/whisper.pdf")
        self.assertEqual(self.client.get("/api/subjects").json()[0]["documents"], [])

    def test_page_is_served(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Study Assistant", page.text)
        self.assertEqual(self.client.get("/app.js").status_code, 200)


class IndexTests(ServiceTestCase):
    def test_empty_subject_has_nothing_to_index(self):
        name = self.make_subject(files=())
        self.assertEqual(self.client.get(f"/api/subjects/{name}/status").json(),
                         {"state": "empty"})

    def test_index_builds_in_the_background(self):
        name = self.make_subject(files=("a.pdf", "b.pdf"))
        status = self.wait_ready(name)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["chunks"], 4)
        self.assertEqual([d["name"] for d in status["documents"]], ["a.pdf", "b.pdf"])

    def test_changing_the_documents_rebuilds(self):
        name = self.make_subject()
        self.wait_ready(name)
        (self.projects / name / "more.pdf").write_bytes(b"x")
        self.assertEqual(service.index_status(name)["state"], "indexing")
        self.assertEqual(self.wait_ready(name)["chunks"], 4)

    def test_unreadable_document_is_reported_not_fatal(self):
        name = self.make_subject(files=("good.pdf", "bad.pdf"))
        with mock.patch.object(service, "load_file",
                               side_effect=lambda p: (_ for _ in ()).throw(
                                   RuntimeError("broken")) if p.endswith("bad.pdf") else []):
            status = self.wait_ready(name)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["failures"], [{"name": "bad.pdf", "error": "broken"}])

    def test_asking_before_the_index_is_ready_is_409(self):
        name = self.make_subject()
        gate = mock.patch.object(service, "embed", side_effect=lambda c, **k: time.sleep(0.3) or fake_embed(c))
        with gate:
            response = self.client.post(f"/api/subjects/{name}/ask", json={"question": "hi?"})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["status"]["state"], "indexing")
            self.wait_ready(name)


def read_events(response):
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append(json.loads(lines["data"]))
    return events


class AskTests(ServiceTestCase):
    def test_answer_streams_and_is_saved(self):
        name = self.make_subject()
        self.wait_ready(name)
        response = self.client.post(f"/api/subjects/{name}/ask",
                                    json={"question": "How many hours?"})
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        events = read_events(response)
        self.assertEqual([e["type"] for e in events],
                         ["sources", "token", "token", "token", "done"])
        self.assertEqual(events[0]["sources"][0]["file"], "whisper.pdf")
        self.assertEqual(events[0]["sources"][0]["section"], "Section 1")
        self.assertEqual(events[-1]["answer"], "680,000 hours")

        history = self.client.get(f"/api/subjects/{name}/chat").json()
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[1]["content"], "680,000 hours")
        self.client.delete(f"/api/subjects/{name}/chat")
        self.assertEqual(self.client.get(f"/api/subjects/{name}/chat").json(), [])

    def test_empty_answer_says_so(self):
        name = self.make_subject()
        self.wait_ready(name)
        with mock.patch.object(service, "_stream_answer", side_effect=lambda q, c: iter([" "])):
            events = read_events(self.client.post(f"/api/subjects/{name}/ask",
                                                  json={"question": "Why?"}))
        self.assertEqual(events[-1]["answer"], service.NO_ANSWER)

    def test_blank_question_is_rejected(self):
        name = self.make_subject()
        self.wait_ready(name)
        self.assertEqual(self.client.post(f"/api/subjects/{name}/ask",
                                          json={"question": "  "}).status_code, 400)

    def test_generation_error_ends_the_stream_and_frees_the_model(self):
        name = self.make_subject()
        self.wait_ready(name)

        def broken(q, c):
            yield "6"
            raise RuntimeError("device lost")

        with mock.patch.object(service, "_stream_answer", side_effect=broken):
            events = read_events(self.client.post(f"/api/subjects/{name}/ask",
                                                  json={"question": "Q?"}))
        self.assertEqual(events[-1], {"type": "error", "detail": "device lost"})
        self.assertFalse(service.MODEL_LOCK.locked())
        self.assertEqual(service.chat_history(name), [])


class QuizTests(ServiceTestCase):
    def setUp(self):
        super().setUp()
        self.name = self.make_subject(files=("a.pdf", "b.pdf"))
        self.wait_ready(self.name)

    def ask_question(self, topic=None):
        response = self.client.post(f"/api/subjects/{self.name}/quiz", json={"topic": topic})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def answer(self, question, text):
        return self.client.post(f"/api/subjects/{self.name}/quiz/answer",
                                json={"id": question["id"], "answer": text})

    def test_topics_are_document_sections(self):
        topics = self.client.get(f"/api/subjects/{self.name}/topics").json()
        self.assertEqual([t["id"] for t in topics],
                         ["a.pdf › Section 1", "a.pdf › Section 2",
                          "b.pdf › Section 1", "b.pdf › Section 2"])

    def test_first_question_is_multiple_choice_without_the_answer(self):
        question = self.ask_question("a.pdf › Section 2")
        self.assertEqual(question["level"], 1)
        self.assertEqual(question["topic"], "a.pdf › Section 2")
        self.assertIn("682 hours in a", question["options"])
        self.assertGreaterEqual(len(question["options"]), 3)
        self.assertNotIn("answer", question)

    def test_correct_answer_is_recorded_and_raises_mastery(self):
        question = self.ask_question("a.pdf › Section 2")
        result = self.answer(question, "682 hours in a").json()
        self.assertTrue(result["correct"])
        self.assertGreater(result["mastery"], result["mastery_before"])
        self.assertEqual(result["source"]["file"], "a.pdf")
        progress = self.client.get(f"/api/subjects/{self.name}/progress").json()
        self.assertEqual((progress["answered"], progress["correct"], progress["practised"]),
                         (1, 1, 1))
        row = next(r for r in progress["topics"] if r["id"] == "a.pdf › Section 2")
        self.assertEqual(row["answered"], 1)
        # The recommender now puts unpractised topics ahead of this one.
        self.assertNotEqual(progress["recommendations"][0]["topic"], "a.pdf › Section 2")

    def test_a_question_can_only_be_answered_once(self):
        question = self.ask_question()
        self.assertEqual(self.answer(question, "wrong").status_code, 200)
        self.assertEqual(self.answer(question, "wrong").status_code, 404)

    def test_blank_answer_keeps_the_question_open(self):
        question = self.ask_question()
        self.assertEqual(self.answer(question, "   ").status_code, 400)
        self.assertEqual(self.answer(question, "x").status_code, 200)

    def test_unknown_topic_is_404(self):
        response = self.client.post(f"/api/subjects/{self.name}/quiz", json={"topic": "nope"})
        self.assertEqual(response.status_code, 404)

    def test_pool_is_saved_and_reused(self):
        first = self.ask_question("a.pdf › Section 1")
        calls = quiz.generate_item.call_count
        pool = service.load_pool(self.name, service.signature(self.name))
        self.assertEqual([i.question for i in pool["items"]["a.pdf › Section 1"]],
                         [first["question"]])
        # Not answered yet, so asking again reuses it instead of writing more.
        self.assertEqual(self.ask_question("a.pdf › Section 1")["question"],
                         first["question"])
        self.assertEqual(quiz.generate_item.call_count, calls)

    def test_no_usable_question_is_422(self):
        with mock.patch.object(quiz, "generate_item", return_value=(None, "bad")):
            response = self.client.post(f"/api/subjects/{self.name}/quiz",
                                        json={"topic": "b.pdf › Section 1"})
        self.assertEqual(response.status_code, 422)

    def test_reset_progress(self):
        self.answer(self.ask_question(), "x")
        self.client.delete(f"/api/subjects/{self.name}/progress")
        self.assertEqual(self.client.get(f"/api/subjects/{self.name}/progress").json()["answered"], 0)


class PickItemTests(unittest.TestCase):
    def item(self, q):
        return quiz.QuizItem("t", q, "a", None, None, None, "")

    def test_avoids_repeating_the_last_question(self):
        progress = service.Progress()
        progress.record("t", 1, True, question="one?")
        for _ in range(20):
            self.assertEqual(service.pick_item([self.item("one?"), self.item("two?")],
                                               progress).question, "two?")

    def test_prefers_questions_asked_least(self):
        progress = service.Progress()
        for q in ["one?", "two?", "one?", "three?"]:
            progress.record("t", 1, True, question=q)
        picks = {service.pick_item([self.item(q) for q in ["one?", "two?", "three?"]],
                                   progress).question for _ in range(20)}
        self.assertEqual(picks, {"two?"})


if __name__ == "__main__":
    unittest.main()
