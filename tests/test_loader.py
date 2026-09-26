"""
Tests for backend/pipeline/loader.py: page and section extraction from PDFs,
the OCR reading-order pass, and the errors a bad file raises. The PDFs are
written by tests/helpers.py; no model is loaded.
"""

import os
import tempfile
import unittest

from backend.pipeline import loader
from tests.helpers import make_pdf

BODY = "This paragraph is ordinary body text about the topic of the page."


class LoadPdfTests(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "paths", []):
            os.remove(path)

    def pdf(self, pages, toc=None):
        path = make_pdf(pages, toc)
        self.paths = getattr(self, "paths", []) + [path]
        return path

    def test_pages_keep_numbers_and_skip_blank_pages(self):
        pages = loader.load_pdf(self.pdf([[BODY], [], [BODY]]))
        self.assertEqual([p["page"] for p in pages], [1, 3])
        self.assertTrue(all(p["source_file"].endswith(".pdf") for p in pages))

    def test_outline_defines_sections(self):
        path = self.pdf([[BODY]] * 4,
                        toc=[[1, "Intro", 1], [2, "Detail", 2], [1, "Methods", 3]])
        sections = [p["section"] for p in loader.load_pdf(path)]
        # level-2 outline entries are ignored, and a page carries the last
        # level-1 section to have started
        self.assertEqual(sections, ["Intro", "Intro", "Methods", "Methods"])

    def test_lecture_headings_define_sections(self):
        path = self.pdf([["Lecture 3 - Supervised Learning", BODY],
                         [BODY],
                         ["Lecture 4 - Overfitting", BODY]])
        sections = [p["section"] for p in loader.load_pdf(path)]
        self.assertEqual(sections, ["Lecture 3 - Supervised Learning",
                                    "Lecture 3 - Supervised Learning",
                                    "Lecture 4 - Overfitting"])

    def test_numbered_headings_must_count_up_from_one(self):
        path = self.pdf([["Title of the paper", "4 Example Institute, Somewhere", BODY],
                         ["1. Introduction", BODY],
                         ["II. NOT A HEADING", "2. Method", BODY],
                         ["References", BODY],
                         ["3 Appendix table row", BODY]])
        sections = [p["section"] for p in loader.load_pdf(path)]
        self.assertEqual(sections, ["Overview", "Introduction", "Method",
                                    "References", "References"])

    def test_roman_numerals_and_upper_case_titles(self):
        path = self.pdf([["I. INTRODUCTION", BODY], ["II. CAUSES OF ERRORS", BODY]])
        sections = [p["section"] for p in loader.load_pdf(path)]
        self.assertEqual(sections, ["Introduction", "Causes Of Errors"])

    def test_bare_number_line_followed_by_title(self):
        path = self.pdf([["1.", "Introduction", BODY], ["2.", "Results", BODY]])
        sections = [p["section"] for p in loader.load_pdf(path)]
        self.assertEqual(sections, ["Introduction", "Results"])

    def test_page_groups_when_no_headings(self):
        path = self.pdf([[BODY]] * 7)
        sections = [p["section"] for p in loader.load_pdf(path)]
        self.assertEqual(sections, ["Pages 1–5"] * 5 + ["Pages 6–7"] * 2)


class LoadFileTests(unittest.TestCase):
    def test_text_file_returns_contents_not_path(self):
        handle, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write("Lecture notes about gradient descent.")
        try:
            self.assertEqual(loader.load_file(path),
                             "Lecture notes about gradient descent.")
        finally:
            os.remove(path)

    def test_unknown_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            loader.load_file("notes.docx")

    def test_unknown_input_type_is_rejected(self):
        with self.assertRaises(ValueError):
            loader.load_input("x", "video")

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            loader.load_text("   ")


class DegenerateExtractionTests(unittest.TestCase):
    def test_bounding_box_only_output(self):
        self.assertTrue(loader._is_degenerate_extraction("(10,7),(984,990)"))

    def test_empty_markdown_table(self):
        self.assertTrue(loader._is_degenerate_extraction("| a | b |\n|---|---|\n| | |\n| | |"))

    def test_blank_output(self):
        self.assertTrue(loader._is_degenerate_extraction("   \n"))

    def test_real_transcription_is_kept(self):
        self.assertFalse(loader._is_degenerate_extraction(
            "Annual report 1975. Revenue increased by 12 percent."))

    def test_short_label_is_kept(self):
        self.assertFalse(loader._is_degenerate_extraction("Figure 3: accuracy"))


class OcrLayoutTests(unittest.TestCase):
    @staticmethod
    def box(x, y):
        return [[x, y], [x + 40, y], [x + 40, y + 10], [x, y + 10]]

    def test_rows_read_top_to_bottom_left_to_right(self):
        detections = [
            (self.box(200, 52), "450", 0.9),
            (self.box(10, 100), "Costs", 0.9),
            (self.box(10, 50), "Revenue", 0.9),
            (self.box(200, 101), "300", 0.9),
            (self.box(100, 48), "2023", 0.9),
        ]
        self.assertEqual(loader._reorder_ocr_by_layout(detections),
                         "Revenue 2023 450\nCosts 300")

    def test_no_detections(self):
        self.assertEqual(loader._reorder_ocr_by_layout([]), "")


if __name__ == "__main__":
    unittest.main()
