"""
Tests for the web interface's server side (backend/service.py and
backend/api.py). The pipeline is replaced with small fakes, so no model is
loaded and nothing is downloaded.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

# service.py sets its defaults in os.environ when it is imported; they are
# undone here, so the other test modules see the environment they expect.
with mock.patch.dict(os.environ):
    from backend import api, service
from backend.pipeline import generator, quiz
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
    return quiz.QuizItem(topic_id=quiz.topic_id(chunk.source_file, chunk.section),
                         question=f"How many hours were used, version {n}?",
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

    # Indexing runs on a background thread, so the test waits for it. The
    # budget is generous because a loaded machine — a sync client walking the
    # project folder, say — can hold the thread off for seconds.
    INDEX_TIMEOUT = 30.0

    def wait_ready(self, name):
        """Poll a subject's status until its build finishes, and return it."""
        deadline = time.monotonic() + self.INDEX_TIMEOUT
        while time.monotonic() < deadline:
            status = self.client.get(f"/api/subjects/{name}/status").json()
            if status["state"] != "indexing":
                return status
            time.sleep(0.01)
        self.fail(f"index never finished within {self.INDEX_TIMEOUT:g}s")


class SubjectTests(ServiceTestCase):
    def test_create_and_list_subjects(self):
        created = self.client.post("/api/subjects", json={"name": "  Statistics: 101 "})
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["name"], "Statistics 101")
        self.client.post("/api/subjects", json={"name": "Biology"})
        names = [s["name"] for s in self.client.get("/api/subjects").json()]
        self.assertEqual(names, ["Biology", "Statistics 101"])

    def test_the_subject_list_carries_what_the_grid_shows(self):
        name = self.make_subject(files=("a.pdf", "b.pdf"))
        self.wait_ready(name)
        read_events(self.client.post(f"/api/subjects/{name}/ask",
                                     json={"question": "How many hours?"}))
        card = self.client.get("/api/subjects").json()[0]
        self.assertEqual(card["name"], name)
        self.assertEqual(len(card["documents"]), 2)
        self.assertEqual(card["chats"], 1)
        self.assertEqual(card["state"], "ready")
        self.assertIsNotNone(card["updated"])

    def test_listing_subjects_does_not_start_indexing_them(self):
        self.make_subject(files=("a.pdf",))
        with mock.patch.object(service, "build_index") as build:
            card = self.client.get("/api/subjects").json()[0]
        build.assert_not_called()
        self.assertEqual(card["state"], "stale")

    def test_recent_conversations_span_subjects(self):
        for name in ("Speech", "Vision"):
            self.make_subject(name=name)
            self.wait_ready(name)
            read_events(self.client.post(f"/api/subjects/{name}/ask",
                                         json={"question": f"About {name}?"}))
        recent = self.client.get("/api/chats?limit=5").json()
        self.assertEqual([c["subject"] for c in recent], ["Vision", "Speech"])
        self.assertEqual(recent[0]["title"], "About Vision?")

    def test_renaming_a_subject_keeps_its_work(self):
        # A rename moves the folder, so the conversation goes with it, and
        # the in-memory index is carried across rather than rebuilt.
        name = self.make_subject()
        self.wait_ready(name)
        read_events(self.client.post(f"/api/subjects/{name}/ask",
                                     json={"question": "How many hours?"}))
        renamed = self.client.patch(f"/api/subjects/{name}",
                                    json={"name": "Speech Recognition"})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["name"], "Speech Recognition")
        self.assertEqual(renamed.json()["index"]["state"], "ready")
        self.assertEqual([s["name"] for s in self.client.get("/api/subjects").json()],
                         ["Speech Recognition"])
        chats = self.client.get("/api/subjects/Speech Recognition/chats").json()
        self.assertEqual(len(chats), 1)
        self.assertEqual(self.client.get(f"/api/subjects/{name}").status_code, 404)

    def test_renaming_carries_the_index_rather_than_rebuilding_it(self):
        name = self.make_subject()
        self.wait_ready(name)
        with mock.patch.object(service, "build_index") as build:
            self.client.patch(f"/api/subjects/{name}", json={"name": "Audio"})
            state = self.client.get("/api/subjects/Audio/status").json()["state"]
        build.assert_not_called()
        self.assertEqual(state, "ready")

    def test_renaming_onto_an_existing_subject_is_refused(self):
        self.make_subject(name="Speech")
        self.make_subject(name="Vision")
        response = self.client.patch("/api/subjects/Speech", json={"name": "Vision"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(sorted(p.name for p in self.projects.iterdir()),
                         ["Speech", "Vision"])

    def test_renaming_to_an_unusable_name_is_refused(self):
        name = self.make_subject()
        response = self.client.patch(f"/api/subjects/{name}", json={"name": "  ///  "})
        self.assertEqual(response.status_code, 400)
        self.assertTrue((self.projects / name).is_dir())

    def test_renaming_a_subject_being_read_is_refused(self):
        name = self.make_subject()
        self.client.get(f"/api/subjects/{name}/status")      # starts the build
        service._building[name] = {"state": "indexing", "signature": ()}
        response = self.client.patch(f"/api/subjects/{name}", json={"name": "Later"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("still being read", response.json()["detail"])

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

    def test_deleting_a_subject_removes_everything_in_it(self):
        name = self.make_subject(files=("whisper.pdf",))
        self.wait_ready(name)
        service._write_json(service.study_path(name, "progress.json"), {"attempts": []})
        self.assertEqual(self.client.delete(f"/api/subjects/{name}").status_code, 200)
        self.assertFalse((self.projects / name).exists())
        self.assertEqual(self.client.get("/api/subjects").json(), [])
        self.assertNotIn(name, service._indexes)
        self.assertEqual(self.client.get(f"/api/subjects/{name}").status_code, 404)

    def test_a_folder_that_will_not_delete_still_stops_being_a_subject(self):
        # Windows can hold a handle on a folder inside OneDrive for a moment
        # after its files go; the subject disappears from the list anyway.
        name = self.make_subject()
        with mock.patch.object(service.shutil, "rmtree",
                               side_effect=PermissionError("Access is denied")):
            self.assertEqual(self.client.delete(f"/api/subjects/{name}").status_code, 200)
        self.assertEqual(self.client.get("/api/subjects").json(), [])
        left = [p.name for p in self.projects.iterdir()]
        self.assertEqual(len(left), 1)
        self.assertTrue(left[0].startswith(service.DELETED_PREFIX), left)

    def test_a_leftover_folder_is_swept_up_by_the_next_delete(self):
        leftover = self.projects / f"{service.DELETED_PREFIX}Old-123"
        leftover.mkdir()
        (leftover / "notes.pdf").write_bytes(b"x")
        name = self.make_subject()
        self.client.delete(f"/api/subjects/{name}")
        self.assertEqual(list(self.projects.iterdir()), [])

    def test_deleting_one_subject_leaves_the_others(self):
        self.make_subject(name="Speech")
        self.make_subject(name="Vision")
        self.client.delete("/api/subjects/Speech")
        self.assertEqual([s["name"] for s in self.client.get("/api/subjects").json()],
                         ["Vision"])

    def test_page_is_served_and_never_cached_stale(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Study Assistant", page.text)
        script = self.client.get("/app.js")
        self.assertEqual(script.status_code, 200)
        # The page and its script are revalidated on every load.
        self.assertEqual(script.headers["cache-control"], "no-cache")
        self.assertEqual(self.client.get("/api/subjects").headers["cache-control"],
                         "no-store")


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
        def read(path, **kwargs):
            if path.endswith("bad.pdf"):
                raise RuntimeError("broken")
            return []

        with mock.patch.object(service, "load_file", side_effect=read):
            status = self.wait_ready(name)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["failures"], [{"name": "bad.pdf", "error": "broken"}])

    def test_a_subject_whose_documents_all_fail_is_unreadable_not_ready(self):
        name = self.make_subject(files=("scan.pdf",))

        def broken(path, **kwargs):
            raise ValueError("No readable text in scan.pdf")

        with mock.patch.object(service, "load_file", side_effect=broken):
            status = self.wait_ready(name)
        self.assertEqual(status["state"], "unreadable")
        self.assertEqual(status["chunks"], 0)
        response = self.client.post(f"/api/subjects/{name}/ask",
                                    json={"question": "anything?"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("None of this subject's documents could be read",
                      response.json()["detail"])

    def test_one_broken_document_does_not_stop_the_others(self):
        name = self.make_subject(files=("good.pdf", "broken.pdf"))

        def read(path, **kwargs):
            if path.endswith("broken.pdf"):
                raise ValueError("could not be opened as a PDF")
            return []

        with mock.patch.object(service, "load_file", side_effect=read):
            status = self.wait_ready(name)
        self.assertEqual(status["state"], "ready")
        self.assertEqual([f["name"] for f in status["failures"]], ["broken.pdf"])
        events = read_events(self.client.post(f"/api/subjects/{name}/ask",
                                              json={"question": "How many hours?"}))
        self.assertEqual(events[-1]["answer"], "680,000 hours")

    def test_questions_work_while_the_pictures_are_still_being_read(self):
        name = self.make_subject(files=("deck.pdf",))
        reading = threading.Event()
        release = threading.Event()

        def read(path, figures="off", **kwargs):
            if figures == "auto":          # the slow second pass
                reading.set()
                release.wait(5)
            return []

        with mock.patch.object(service, "load_file", side_effect=read):
            self.client.get(f"/api/subjects/{name}/status")      # starts the build
            self.assertTrue(reading.wait(5), "second pass never started")
            status = self.client.get(f"/api/subjects/{name}/status").json()
            self.assertEqual(status["state"], "ready")
            self.assertIsNotNone(status["enriching"])
            events = read_events(self.client.post(f"/api/subjects/{name}/ask",
                                                  json={"question": "How many hours?"}))
            self.assertEqual(events[-1]["answer"], "680,000 hours")
            release.set()
            self.wait_ready(name)

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


class TurnRoutingTests(ServiceTestCase):
    """
    Not every message in a conversation is a new question. A follow-up about
    the answer just given must not send the student's documents through
    retrieval again, and a continuation has to be rewritten to stand alone
    before it can be looked up at all.
    """

    def ask(self, name, question, chat=None):
        response = self.client.post(f"/api/subjects/{name}/ask",
                                    json={"question": question, "chat": chat})
        self.assertEqual(response.status_code, 200, response.text)
        return read_events(response)

    def test_the_first_question_of_a_conversation_is_new(self):
        name = self.make_subject()
        self.wait_ready(name)
        with mock.patch.object(service, "_route_turn",
                               wraps=service._route_turn) as route:
            events = self.ask(name, "How many hours?")
        route.assert_called_once()
        self.assertEqual(route.call_args[0][1], [])      # no history yet
        self.assertEqual(events[0], {"type": "turn", "kind": "new",
                                     "retrieved_for": None})

    def test_a_follow_up_retrieves_nothing(self):
        name = self.make_subject()
        self.wait_ready(name)
        first = self.ask(name, "How many hours?")
        chat_id = first[-1]["chat"]["id"]
        with mock.patch.object(generator, "classify_turn",
                               return_value="followup"), \
             mock.patch.object(service, "_stream_followup",
                               return_value=iter(["Put simply, "])) as followup:
            events = self.ask(name, "say that more simply", chat_id)
        followup.assert_called_once()
        self.assertEqual([e["type"] for e in events],
                         ["turn", "token", "done"])
        self.assertEqual(events[0]["kind"], "followup")
        history = self.client.get(
            f"/api/subjects/{name}/chats/{chat_id}").json()["messages"]
        self.assertEqual(history[-1]["turn"], "followup")
        self.assertEqual(history[-1]["sources"], [])

    def test_a_continuation_is_rewritten_before_it_is_looked_up(self):
        name = self.make_subject()
        self.wait_ready(name)
        first = self.ask(name, "How many hours of audio?")
        chat_id = first[-1]["chat"]["id"]
        with mock.patch.object(generator, "classify_turn",
                               return_value="continuation"), \
             mock.patch.object(generator, "standalone_question",
                               return_value="How many hours of German audio?"):
            events = self.ask(name, "what about German?", chat_id)
        self.assertEqual(events[0], {"type": "turn", "kind": "continuation",
                                     "retrieved_for":
                                         "How many hours of German audio?"})
        self.assertEqual(events[1]["type"], "sources")

    def test_any_kind_that_is_not_a_follow_up_still_retrieves(self):
        name = self.make_subject()
        self.wait_ready(name)
        first = self.ask(name, "How many hours?")
        chat_id = first[-1]["chat"]["id"]
        with mock.patch.object(generator, "classify_turn",
                               return_value="something else"):
            events = self.ask(name, "and in German?", chat_id)
        self.assertEqual(events[1]["type"], "sources")


class AskTests(ServiceTestCase):
    def test_answer_streams_and_is_saved(self):
        name = self.make_subject()
        self.wait_ready(name)
        response = self.client.post(f"/api/subjects/{name}/ask",
                                    json={"question": "How many hours?"})
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        events = read_events(response)
        self.assertEqual([e["type"] for e in events],
                         ["turn", "sources", "token", "token", "token", "done"])
        self.assertEqual(events[0]["kind"], "new")
        self.assertEqual(events[1]["sources"][0]["file"], "whisper.pdf")
        self.assertEqual(events[1]["sources"][0]["section"], "Section 1")
        self.assertEqual(events[-1]["answer"], "680,000 hours")

        chats = self.client.get(f"/api/subjects/{name}/chats").json()
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["id"], events[-1]["chat"]["id"])
        history = self.client.get(
            f"/api/subjects/{name}/chats/{chats[0]['id']}").json()["messages"]
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[1]["content"], "680,000 hours")
        self.client.delete(f"/api/subjects/{name}/chats")
        self.assertEqual(self.client.get(f"/api/subjects/{name}/chats").json(), [])

    def test_a_conversation_is_named_after_its_first_question(self):
        name = self.make_subject()
        self.wait_ready(name)
        events = read_events(self.client.post(
            f"/api/subjects/{name}/ask",
            json={"question": "How many hours of audio was Whisper trained on?"}))
        chat_id = events[-1]["chat"]["id"]
        self.assertEqual(events[-1]["chat"]["title"],
                         "How many hours of audio was Whisper trained…")
        # Only the first question names a conversation.
        read_events(self.client.post(f"/api/subjects/{name}/ask",
                                     json={"question": "And what about images?",
                                           "chat": chat_id}))
        chats = self.client.get(f"/api/subjects/{name}/chats").json()
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["title"], "How many hours of audio was Whisper trained…")
        self.assertEqual(chats[0]["exchanges"], 2)

    def test_conversations_are_separate_and_deletable(self):
        name = self.make_subject()
        self.wait_ready(name)
        first = read_events(self.client.post(f"/api/subjects/{name}/ask",
                                             json={"question": "About Whisper?"}))
        second = read_events(self.client.post(f"/api/subjects/{name}/ask",
                                              json={"question": "About images?"}))
        one, two = first[-1]["chat"]["id"], second[-1]["chat"]["id"]
        self.assertNotEqual(one, two)
        chats = self.client.get(f"/api/subjects/{name}/chats").json()
        self.assertEqual([c["title"] for c in chats], ["About images?", "About Whisper?"])

        self.assertEqual(self.client.delete(
            f"/api/subjects/{name}/chats/{one}").status_code, 200)
        remaining = self.client.get(f"/api/subjects/{name}/chats").json()
        self.assertEqual([c["id"] for c in remaining], [two])
        self.assertEqual(self.client.get(
            f"/api/subjects/{name}/chats/{one}").status_code, 404)

    def test_an_empty_conversation_can_be_started_and_named_later(self):
        name = self.make_subject()
        self.wait_ready(name)
        created = self.client.post(f"/api/subjects/{name}/chats").json()
        self.assertEqual(created["title"], "New conversation")
        read_events(self.client.post(f"/api/subjects/{name}/ask",
                                     json={"question": "What about audio?",
                                           "chat": created["id"]}))
        chats = self.client.get(f"/api/subjects/{name}/chats").json()
        self.assertEqual([c["title"] for c in chats], ["What about audio?"])

    def test_asking_in_a_conversation_that_is_gone_is_404(self):
        name = self.make_subject()
        self.wait_ready(name)
        response = self.client.post(f"/api/subjects/{name}/ask",
                                    json={"question": "Q?", "chat": "deadbeef"})
        self.assertEqual(response.status_code, 404)

    def test_an_older_single_chat_file_becomes_a_conversation(self):
        name = self.make_subject()
        service._write_json(service.study_path(name, "chat.json"), [
            {"role": "user", "content": "How long was the training set?", "time": 1.0},
            {"role": "assistant", "content": "680,000 hours", "time": 2.0}])
        chats = self.client.get(f"/api/subjects/{name}/chats").json()
        self.assertEqual([c["title"] for c in chats],
                         ["How long was the training set?"])
        self.assertFalse(service.study_path(name, "chat.json").exists())

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

    def test_a_topic_is_worth_questions_in_proportion_to_its_material(self):
        from backend.pipeline.quiz import Topic
        small = Topic("t", "a.pdf", "S", list(range(2)))
        medium = Topic("t", "a.pdf", "S", list(range(12)))
        huge = Topic("t", "a.pdf", "S", list(range(200)))
        self.assertEqual(service.questions_worth(small),
                         service.QUESTIONS_PER_TOPIC)
        self.assertEqual(service.questions_worth(medium), 4)
        self.assertEqual(service.questions_worth(huge),
                         service.MAX_QUESTIONS_PER_TOPIC)

    def test_the_topic_list_says_how_much_each_one_holds(self):
        topics = self.client.get(f"/api/subjects/{self.name}/topics").json()
        whole = topics[0]
        self.assertEqual(whole["scope"], "document")
        self.assertEqual(whole["passages"], 2)
        self.assertEqual(whole["questions"], service.QUESTIONS_PER_TOPIC)

    def test_topics_are_whole_files_and_their_sections(self):
        topics = self.client.get(f"/api/subjects/{self.name}/topics").json()
        self.assertEqual([t["id"] for t in topics],
                         ["a.pdf › Everything in this file",
                          "a.pdf › Section 1", "a.pdf › Section 2",
                          "b.pdf › Everything in this file",
                          "b.pdf › Section 1", "b.pdf › Section 2"])
        self.assertEqual(topics[0]["scope"], "document")
        self.assertEqual(topics[1]["scope"], "section")

    def test_a_whole_file_topic_asks_about_any_section_of_that_file(self):
        seen = set()
        for _ in range(4):
            question = self.ask_question("a.pdf › Everything in this file")
            seen.add(question["topic"])
            self.answer(question, "wrong")
        # A question is filed under the section it came from, never under the
        # whole-file topic, so mastery stays per section.
        self.assertTrue(seen <= {"a.pdf › Section 1", "a.pdf › Section 2"}, seen)
        progress = self.client.get(f"/api/subjects/{self.name}/progress").json()
        self.assertEqual([r["id"] for r in progress["topics"]],
                         ["a.pdf › Section 1", "a.pdf › Section 2",
                          "b.pdf › Section 1", "b.pdf › Section 2"])
        self.assertEqual(sum(r["answered"] for r in progress["topics"]), 4)

    def test_a_whole_file_topic_does_not_repeat_a_section_question(self):
        # The file's two sections hold one question each, and the whole-file
        # topic draws on the same chunks, so it must not repeat one.
        first = self.ask_question("a.pdf › Section 1")
        self.answer(first, "wrong")
        second = self.ask_question("a.pdf › Everything in this file")
        self.assertNotEqual(second["question"], first["question"])
        self.assertEqual(second["topic"], "a.pdf › Section 2")

    def test_recommendations_never_point_at_a_whole_file(self):
        question = self.ask_question()
        self.assertNotIn("Everything in this file", question["topic"])

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
        # An answered topic drops below the unpractised ones.
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
        # Unanswered, so the next request reuses it rather than writing more.
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
