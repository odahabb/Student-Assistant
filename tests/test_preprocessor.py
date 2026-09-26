"""
Tests for backend/pipeline/preprocessor.py: the four chunking modes, the page
boundaries chunks stay inside, and the page-1 boilerplate filter. The
tokenizer and the embedding model are fakes, so no model is loaded.
"""

import unittest
from unittest import mock

import numpy as np

from backend.pipeline import preprocessor
from backend.pipeline.chunk import Chunk
from tests.helpers import FakeEmbeddingModel


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(preprocessor, "_get_model",
                                    return_value=FakeEmbeddingModel())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_windows_overlap_and_cover_the_text(self):
        chunks = preprocessor.preprocess(words(25), chunk_tokens=10, overlap=4)
        self.assertEqual([c.split()[0] for c in chunks], ["w0", "w6", "w12", "w18"])
        self.assertEqual(chunks[-1].split()[-1], "w24")
        self.assertTrue(all(len(c.split()) <= 10 for c in chunks))

    def test_chunks_never_cross_page_boundaries(self):
        pages = [
            {"source_file": "notes.pdf", "page": 1, "section": "Intro",
             "text": words(15, "a")},
            {"source_file": "notes.pdf", "page": 2, "section": "Methods",
             "text": words(15, "b")},
        ]
        chunks = preprocessor.preprocess(pages, chunk_tokens=10, overlap=2,
                                         strip_page1_boilerplate=False)
        for chunk in chunks:
            prefixes = {w[0] for w in chunk.split()}
            self.assertEqual(len(prefixes), 1, chunk)
            self.assertEqual(chunk.page, 1 if prefixes == {"a"} else 2)
            self.assertEqual(chunk.section, "Intro" if chunk.page == 1 else "Methods")
            self.assertEqual(chunk.source_file, "notes.pdf")

    def test_plain_string_has_no_page_or_section(self):
        chunks = preprocessor.preprocess(words(30), source_file="talk.m4a")
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
        self.assertEqual({(c.source_file, c.page, c.section) for c in chunks},
                         {("talk.m4a", None, None)})

    def test_whitespace_is_collapsed(self):
        chunks = preprocessor.preprocess("alpha   beta\n\n gamma " + words(10))
        self.assertTrue(chunks[0].startswith("alpha beta gamma"))

    def test_sentence_chunking_keeps_sentences_whole(self):
        sentences = [f"Sentence {n} has exactly six words." for n in range(10)]
        chunks = preprocessor.preprocess(" ".join(sentences), chunk_tokens=20,
                                         overlap=6, chunking="sentence")
        for chunk in chunks:
            self.assertTrue(chunk.startswith("Sentence"), chunk)
            self.assertTrue(chunk.endswith("words."), chunk)
            self.assertLessEqual(len(chunk.split()), 20)
        # the trailing sentence (6 tokens) is repeated as the overlap
        self.assertEqual(chunks[1].split()[:2], ["Sentence", "2"])
        self.assertIn("Sentence 9", chunks[-1])

    def test_sentence_chunking_splits_an_overlong_sentence(self):
        text = "Short one here. " + words(50) + " end. Another short sentence."
        chunks = preprocessor.preprocess(text, chunk_tokens=20, overlap=4,
                                         chunking="sentence")
        self.assertTrue(all(len(c.split()) <= 20 for c in chunks))
        self.assertIn("w49", " ".join(chunks))
        self.assertTrue(chunks[-1].endswith("Another short sentence."))

    def test_unknown_chunking_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocessor.preprocess(words(30), chunking="topic")

    def test_heading_chunking_never_mixes_subsections(self):
        page = "\n".join([
            "2.1. Data Processing",
            "Audio is split into thirty second segments. Transcripts are normalised.",
            "2.2. Model",
            "The model is an encoder decoder transformer. It uses log mel features.",
            "3",
            "Experiments",
            "Results are reported as word error rate.",
        ])
        chunks = preprocessor.preprocess(
            [{"source_file": "w.pdf", "page": 2, "section": "Approach", "text": page}],
            chunk_tokens=50, chunking="heading", strip_page1_boilerplate=False)
        self.assertEqual([c.split()[0] for c in chunks], ["2.1.", "2.2.", "3"])
        self.assertTrue(all(c.page == 2 and c.section == "Approach" for c in chunks))
        self.assertNotIn("encoder", chunks[0])

    def test_heading_detection_ignores_sentences_and_long_lines(self):
        self.assertTrue(preprocessor._is_heading("II. CAUSES OF HALLUCINATIONS"))
        self.assertTrue(preprocessor._is_heading("Lecture 4 - Overfitting"))
        self.assertTrue(preprocessor._is_heading("A. Evaluation Datasets"))
        self.assertFalse(preprocessor._is_heading("3 Models were trained for five days."))
        self.assertFalse(preprocessor._is_heading("3 data sources and performed manual"))
        self.assertFalse(preprocessor._is_heading(
            "2 This line is far too long to be a heading because it keeps going on"))

    def test_semantic_chunking_breaks_where_topics_change(self):
        text = ("Cats purr loudly. Cats sleep often. Cats chase mice. "
                "Stocks rose today. Markets closed higher. Bonds fell slightly.")
        topic = {"Cats": [1.0, 0.0], "Stocks": [0.0, 1.0], "Markets": [0.1, 0.99],
                 "Bonds": [0.0, 1.0]}

        def fake_vectors(sentences):
            return np.array([topic[s.split()[0]] for s in sentences])

        with mock.patch.object(preprocessor, "_sentence_vectors", fake_vectors), \
             mock.patch.object(preprocessor, "MIN_SEGMENT_TOKENS", 1):
            chunks = preprocessor.preprocess(text, chunk_tokens=50, chunking="semantic")
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Cats") and "Stocks" not in chunks[0])
        self.assertTrue(chunks[1].startswith("Stocks"))

    def test_semantic_chunking_merges_tiny_segments_and_respects_size(self):
        text = " ".join(f"Sentence {i} about topic {i}." for i in range(12))

        def alternating(sentences):
            return np.array([[1.0, 0.0] if i % 2 else [0.0, 1.0]
                             for i in range(len(sentences))])

        with mock.patch.object(preprocessor, "_sentence_vectors", alternating):
            chunks = preprocessor.preprocess(text, chunk_tokens=12, chunking="semantic")
        self.assertTrue(all(len(c.split()) <= 12 for c in chunks))
        self.assertIn("Sentence 11", " ".join(chunks))

    def test_too_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocessor.preprocess("tiny")

    def test_bad_input_types_are_rejected(self):
        with self.assertRaises(TypeError):
            preprocessor.preprocess(42)
        with self.assertRaises(TypeError):
            preprocessor.preprocess([{"page": 1}])


class BoilerplateTests(unittest.TestCase):
    def test_author_and_affiliation_lines_are_removed(self):
        page = "\n".join([
            "Scaling Instruction-Finetuned Language Models",
            "Hyung Won Chung*",
            "Le Hou*",
            "Shayne Longpre*",
            "Department of Computer Science, University of Iowa",
            "jane.doe@example.edu",
            "Abstract",
            "Finetuning language models on a collection of datasets improves "
            "performance and generalization to unseen tasks.",
        ])
        kept = preprocessor._strip_page1_boilerplate(page)
        self.assertIn("Scaling Instruction-Finetuned Language Models", kept)
        self.assertIn("Finetuning language models on a collection", kept)
        for removed in ("Hyung Won Chung", "University of Iowa", "@example.edu"):
            self.assertNotIn(removed, kept)

    def test_prose_mentioning_a_university_is_kept(self):
        page = ("The study was conducted on 28 university course syllabi from the "
                "University of Iowa and evaluated with expert reviewers.")
        self.assertEqual(preprocessor._strip_page1_boilerplate(page), page)

    def test_page_left_alone_when_rules_would_remove_most_of_it(self):
        # The title line is always kept, so the rules would remove 5 of 7
        # lines (71%), which is over _MAX_REMOVAL_FRACTION.
        emails = [f"person{i}@example.org" for i in range(5)]
        page = "\n".join(["A Title", *emails, "Some real text here."])
        self.assertEqual(preprocessor._strip_page1_boilerplate(page), page)


if __name__ == "__main__":
    unittest.main()
