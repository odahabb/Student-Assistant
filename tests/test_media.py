"""
Tests for the parts of the pipeline that adapt to the kind of material being
uploaded: slide decks, pictures inside a PDF, and recordings. No model is
loaded — the vision and speech models are replaced by stubs.
"""

import os
import unittest
from unittest import mock

from backend.pipeline import loader, preprocessor
from tests.helpers import FakeEmbeddingModel, audio_segments, make_deck, make_pdf

WORDS = ("Retrieval augmented generation grounds an answer in the documents a "
         "student uploaded rather than in the weights of a language model, and "
         "that is the whole point of the system under discussion here today.")


class SlideDetectionTests(unittest.TestCase):
    def deck(self, slides):
        path = make_deck(slides)
        self.addCleanup(os.remove, path)
        return loader.load_pdf(path)

    def test_a_deck_is_recognised_and_its_titles_read(self):
        pages = self.deck([("Genetic algorithms", []),
                           ("Selection", ["Pick the fittest", "Then breed them"]),
                           ("Crossover", ["Swap the genes of two parents"]),
                           ("Mutation", ["Flip a gene at random"]),
                           ("Fitness", ["Score each individual"])])
        self.assertEqual({p["kind"] for p in pages}, {"slide"})
        self.assertEqual([p["title"] for p in pages],
                         ["Genetic algorithms", "Selection", "Crossover",
                          "Mutation", "Fitness"])

    def test_a_title_only_slide_is_marked_as_a_divider(self):
        pages = self.deck([("Genetic algorithms", []),
                           ("Selection", ["Pick the fittest of the population"]),
                           ("Crossover", ["Swap genes between two parents"]),
                           ("Mutation", ["Flip one gene at random"])])
        self.assertEqual([p["divider"] for p in pages], [True, False, False, False])

    def test_a_wrapped_title_is_joined_rather_than_truncated(self):
        # The title is long enough to wrap onto a second line on the slide.
        pages = self.deck([("Bio-inspired computing and artificial life", []),
                           ("Selection", ["Pick the fittest of the population"]),
                           ("Crossover", ["Swap genes between two parents"]),
                           ("Mutation", ["Flip one gene at random"])])
        self.assertEqual(pages[0]["title"],
                         "Bio-inspired computing and artificial life")

    def test_a_text_document_is_not_treated_as_a_deck(self):
        path = make_pdf([[WORDS] * 6, [WORDS] * 6, [WORDS] * 6, [WORDS] * 6])
        self.addCleanup(os.remove, path)
        pages = loader.load_pdf(path)
        self.assertEqual({p["kind"] for p in pages}, {"page"})
        self.assertEqual({p["title"] for p in pages}, {None})


class PictureReadingTests(unittest.TestCase):
    def blank_page_deck(self):
        # Slide 2 has no text at all, standing in for a slide that is one big
        # picture; the loader should offer it to the vision chain.
        path = make_deck([("Selection", ["Pick the fittest of the population"]),
                          ("", []),
                          ("Crossover", ["Swap genes between two parents"]),
                          ("Mutation", ["Flip one gene at random"])])
        self.addCleanup(os.remove, path)
        return path

    def test_pictures_are_left_alone_by_default(self):
        with mock.patch.object(loader, "_read_page_picture") as reader:
            pages = loader.load_pdf(self.blank_page_deck())
        reader.assert_not_called()
        self.assertEqual([p["page"] for p in pages], [1, 3, 4])

    def test_a_picture_only_slide_is_read_and_marked(self):
        with mock.patch.object(loader, "_read_page_picture",
                               return_value=("A fitness landscape with two peaks", "ocr")):
            pages = loader.load_pdf(self.blank_page_deck(), figures="auto")
        self.assertEqual([p["page"] for p in pages], [1, 2, 3, 4])
        picture_page = pages[1]
        self.assertIn("fitness landscape", picture_page["text"])
        self.assertTrue(picture_page["from_image"])
        self.assertFalse(pages[0]["from_image"])

    def test_progress_is_reported_while_reading_pictures(self):
        seen = []
        with mock.patch.object(loader, "_read_page_picture",
                               return_value=("some words from the picture", "ocr")):
            loader.load_pdf(self.blank_page_deck(), figures="auto",
                            report=lambda done, total, page: seen.append((done, total)))
        self.assertEqual(seen, [(0, 1), (1, 1)])

    def test_the_lock_is_released_between_pages(self):
        import threading
        lock = threading.Lock()
        held_during_read = []

        def read(page, number, allow_vision=True):
            held_during_read.append(lock.locked())
            return "words read from this picture", "ocr"

        with mock.patch.object(loader, "_read_page_picture", side_effect=read):
            loader.load_pdf(self.blank_page_deck(), figures="auto", lock=lock)
        self.assertEqual(held_during_read, [True])      # held while reading
        self.assertFalse(lock.locked())                 # released afterwards

    def test_a_failing_vision_model_does_not_lose_the_document(self):
        with mock.patch.object(loader, "_read_page_picture",
                               side_effect=RuntimeError("out of memory")):
            pages = loader.load_pdf(self.blank_page_deck(), figures="auto")
        self.assertEqual([p["page"] for p in pages], [1, 3, 4])

    def test_ocr_is_tried_before_the_vision_model(self):
        page = mock.Mock()
        page.get_pixmap.return_value.tobytes.return_value = b"fake png bytes"
        long_enough = "one two three four five six seven eight nine ten more words"
        with mock.patch.object(loader, "_cached_figure_text", return_value=None), \
             mock.patch.object(loader, "_cache_figure_text"), \
             mock.patch.object(loader, "_ocr_page_image", return_value=long_enough) as ocr, \
             mock.patch.object(loader, "_run_qwen_extraction") as vision:
            text, method = loader._read_page_picture(page, 1)
        ocr.assert_called_once()
        vision.assert_not_called()
        self.assertEqual((text, method), (long_enough, "ocr"))

    def test_the_vision_model_handles_what_ocr_cannot_read(self):
        page = mock.Mock()
        page.get_pixmap.return_value.tobytes.return_value = b"fake png bytes"
        with mock.patch.object(loader, "_cached_figure_text", return_value=None), \
             mock.patch.object(loader, "_cache_figure_text"), \
             mock.patch.object(loader, "_ocr_page_image", return_value="x y"), \
             mock.patch.object(loader, "_run_qwen_extraction",
                               return_value="A bar chart comparing four algorithms by score"):
            text, method = loader._read_page_picture(page, 1)
        self.assertEqual(method, "vision")
        self.assertIn("bar chart", text)

    def test_a_cached_page_is_not_read_again(self):
        page = mock.Mock()
        page.get_pixmap.return_value.tobytes.return_value = b"fake png bytes"
        with mock.patch.object(loader, "_cached_figure_text", return_value="from cache"), \
             mock.patch.object(loader, "_ocr_page_image") as ocr, \
             mock.patch.object(loader, "_run_qwen_extraction") as vision:
            self.assertEqual(loader._read_page_picture(page, 1), ("from cache", "cache"))
        ocr.assert_not_called()
        vision.assert_not_called()


class AudioSegmentTests(unittest.TestCase):
    def transcribe(self, result):
        whisper = mock.Mock()
        whisper.load_model.return_value.transcribe.return_value = result
        path = make_pdf([["x"]])          # any existing, non-empty file
        self.addCleanup(os.remove, path)
        with mock.patch.dict("sys.modules", {"whisper": whisper}):
            return loader.load_audio_segments(path)

    def test_segments_keep_their_timestamps(self):
        segments = self.transcribe({"text": "a b", "segments": [
            {"text": " Today we look at selection.", "start": 0.0, "end": 4.2},
            {"text": " Then at crossover.", "start": 7.9, "end": 10.1}]})
        self.assertEqual([(s["start"], s["end"]) for s in segments],
                         [(0.0, 4.2), (7.9, 10.1)])
        self.assertEqual({s["kind"] for s in segments}, {"audio"})

    def test_a_transcript_without_segments_still_loads(self):
        segments = self.transcribe({"text": "one long transcript", "segments": []})
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "one long transcript")

    def test_the_plain_text_loader_still_returns_a_string(self):
        whisper = mock.Mock()
        whisper.load_model.return_value.transcribe.return_value = {
            "text": "", "segments": [{"text": " Hello.", "start": 0, "end": 1},
                                     {"text": " Goodbye.", "start": 1, "end": 2}]}
        path = make_pdf([["x"]])
        self.addCleanup(os.remove, path)
        with mock.patch.dict("sys.modules", {"whisper": whisper}):
            self.assertEqual(loader.load_audio(path), "Hello. Goodbye.")


class SlidePackingTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(preprocessor, "_get_model",
                                    return_value=FakeEmbeddingModel())
        patcher.start()
        self.addCleanup(patcher.stop)

    def slides(self, *specs):
        """specs: (page, title, text, divider)"""
        return [{"source_file": "deck.pdf", "page": page, "section": "Pages 1-5",
                 "text": text, "kind": "slide", "title": title,
                 "divider": divider, "start": None, "end": None,
                 "from_image": False}
                for page, title, text, divider in specs]

    def test_consecutive_slides_are_packed_up_to_the_limit(self):
        # Slides continuing one topic, as a deck's "Selection (2)" slides do.
        units = self.slides(*[(i, "Selection", "Selection " + " ".join(
            f"word{j}" for j in range(8)), False) for i in range(1, 7)])
        chunks = preprocessor.preprocess(units, chunk_tokens=30, overlap=5,
                                         chunking="sentence")
        self.assertLess(len(chunks), len(units))
        for chunk in chunks:
            self.assertLessEqual(len(str(chunk).split()), 30)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[0].page_end, 3)
        self.assertEqual(chunks[0].pages, "1-3")

    def test_a_divider_labels_the_slides_after_it_and_is_not_a_chunk(self):
        units = self.slides(
            (1, "Selection", "Selection", True),
            (2, "How it works", "How it works we pick the fittest individuals", False),
            (3, "Tournament", "Tournament selection picks the best of a sample", False))
        chunks = preprocessor.preprocess(units, chunk_tokens=40, overlap=5,
                                         chunking="sentence")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Selection")
        self.assertNotIn("Selection Selection", str(chunks[0]))
        self.assertEqual(chunks[0].page, 2)

    def test_a_new_title_ends_a_chunk_that_is_already_half_full(self):
        body = " ".join(f"word{j}" for j in range(24))
        units = self.slides((1, "Selection", f"Selection {body}", False),
                            (2, "Crossover", f"Crossover {body}", False))
        chunks = preprocessor.preprocess(units, chunk_tokens=40, overlap=5,
                                         chunking="sentence")
        self.assertEqual([c.section for c in chunks], ["Selection", "Crossover"])

    def test_a_slide_longer_than_a_chunk_is_split(self):
        long_text = " ".join(f"word{j}" for j in range(80))
        units = self.slides((1, "Details", f"Details. {long_text}.", False))
        chunks = preprocessor.preprocess(units, chunk_tokens=30, overlap=5,
                                         chunking="sentence")
        self.assertGreater(len(chunks), 1)
        self.assertEqual({c.page for c in chunks}, {1})
        self.assertEqual({c.section for c in chunks}, {"Details"})

    def test_picture_text_marks_the_chunk_it_lands_in(self):
        units = self.slides((1, "Fitness", "Fitness landscape with two peaks", False))
        units[0]["from_image"] = True
        chunks = preprocessor.preprocess(units, chunk_tokens=40, overlap=5,
                                         chunking="sentence")
        self.assertTrue(chunks[0].from_image)

    def test_symbol_font_bullets_are_stripped(self):
        units = self.slides((1, "Summary", "Summary  first point  second point",
                             False))
        chunks = preprocessor.preprocess(units, chunk_tokens=40, overlap=5,
                                         chunking="sentence")
        self.assertNotIn("", str(chunks[0]))
        self.assertIn("first point", str(chunks[0]))


class AudioPackingTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(preprocessor, "_get_model",
                                    return_value=FakeEmbeddingModel())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_segments_are_packed_and_keep_their_time(self):
        units = audio_segments(*[(f"segment {i} " + " ".join(f"w{j}" for j in range(8)),
                                  i * 10.0, i * 10.0 + 9.0) for i in range(6)])
        chunks = preprocessor.preprocess(units, chunk_tokens=40, overlap=5,
                                         chunking="sentence")
        self.assertLess(len(chunks), len(units))
        self.assertEqual(chunks[0].start, 0.0)
        self.assertIsNotNone(chunks[0].timecode)
        self.assertIsNone(chunks[0].page)
        self.assertTrue(chunks[0].section.startswith("Part 1 (0:00-"))

    def test_a_long_pause_ends_a_chunk(self):
        body = " ".join(f"w{j}" for j in range(35))
        units = audio_segments((f"first {body}", 0.0, 20.0),      # then a 5 s pause
                               (f"second {body}", 25.0, 45.0))
        chunks = preprocessor.preprocess(units, chunk_tokens=60, overlap=5,
                                         chunking="sentence")
        self.assertEqual(len(chunks), 2)
        self.assertIn("first", str(chunks[0]))
        self.assertIn("second", str(chunks[1]))

    def test_a_short_gap_does_not(self):
        body = " ".join(f"w{j}" for j in range(20))
        units = audio_segments((f"first {body}", 0.0, 20.0),
                               (f"second {body}", 20.4, 40.0))
        chunks = preprocessor.preprocess(units, chunk_tokens=60, overlap=5,
                                         chunking="sentence")
        self.assertEqual(len(chunks), 1)

    def test_timecodes_are_minutes_and_seconds(self):
        units = audio_segments(("and that is how a genetic algorithm searches "
                                "the space of possible solutions", 723.0, 940.0))
        chunks = preprocessor.preprocess(units, chunk_tokens=60, overlap=5,
                                         chunking="sentence")
        self.assertEqual(chunks[0].timecode, "12:03-15:40")
        self.assertEqual(chunks[0].section, "Part 1 (12:03-15:40)")


class BadUploadTests(unittest.TestCase):
    """Whatever is uploaded, the loader must fail with a sentence, not a stack."""

    def file_with(self, data, suffix=".pdf"):
        import tempfile
        handle, path = tempfile.mkstemp(suffix=suffix)
        os.write(handle, data)
        os.close(handle)
        self.addCleanup(os.remove, path)
        return path

    def test_an_empty_pdf_file(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            loader.load_pdf(self.file_with(b""))

    def test_a_file_that_is_not_a_pdf(self):
        with self.assertRaisesRegex(ValueError, "could not be opened as a PDF"):
            loader.load_pdf(self.file_with(b"this is just some text, not a PDF"))

    def test_a_password_protected_pdf(self):
        import fitz
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "secret")
        path = self.file_with(b"")
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="letmein")
        doc.close()
        with self.assertRaisesRegex(ValueError, "password-protected"):
            loader.load_pdf(path)

    def test_a_file_of_null_bytes_is_not_text(self):
        with self.assertRaisesRegex(ValueError, "no readable text"):
            loader.load_text("\x00" * 500)

    def test_a_corrupt_image_says_so_in_plain_words(self):
        path = self.file_with(b"\x89PNG\r\n\x1a\n" + b"junk" * 50, suffix=".png")
        with self.assertRaisesRegex(ValueError, "could not be opened as an image"):
            loader.load_image(path)

    def test_an_image_too_small_to_hold_anything(self):
        import io

        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
        path = self.file_with(buffer.getvalue(), suffix=".png")
        with self.assertRaisesRegex(ValueError, "too small"):
            loader.load_image(path)

    def test_an_unsupported_extension(self):
        with self.assertRaisesRegex(ValueError, "Cannot auto-detect"):
            loader.load_file("notes.docx")

    def test_an_empty_audio_file(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            loader.load_audio_segments(self.file_with(b"", suffix=".mp3"))

    def test_audio_that_whisper_cannot_open(self):
        whisper = mock.Mock()
        whisper.load_model.return_value.transcribe.side_effect = RuntimeError("no ffmpeg")
        with mock.patch.dict("sys.modules", {"whisper": whisper}):
            with self.assertRaisesRegex(ValueError, "could not be transcribed"):
                loader.load_audio_segments(self.file_with(b"noise", suffix=".mp3"))


if __name__ == "__main__":
    unittest.main()
