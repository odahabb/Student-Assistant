import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class NewSubjectTests(unittest.TestCase):
    """Drives the real Streamlit script; no documents, so no model is loaded."""

    def run_app(self, projects):
        from streamlit.testing.v1 import AppTest
        with mock.patch.dict(os.environ, {"SA_PROJECTS_DIR": projects}):
            at = AppTest.from_file(APP, default_timeout=120)
            at.run()
        return at

    def test_creating_a_subject_when_others_exist_selects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Existing").mkdir()
            at = self.run_app(tmp)
            with mock.patch.dict(os.environ, {"SA_PROJECTS_DIR": tmp}):
                at.text_input[0].input("Statistics")
                [b for b in at.button if "New subject" in b.label][0].click().run()
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
            self.assertTrue((Path(tmp) / "Statistics").is_dir())
            self.assertEqual(at.radio[0].value, "Statistics")

    def test_first_subject_can_be_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            at = self.run_app(tmp)
            with mock.patch.dict(os.environ, {"SA_PROJECTS_DIR": tmp}):
                at.text_input[0].input("Biology")
                [b for b in at.button if "New subject" in b.label][0].click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.radio[0].value, "Biology")


if __name__ == "__main__":
    unittest.main()
