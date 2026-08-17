import json
import os
import tempfile
import unittest
from unittest import mock

import faiss
import numpy as np

from backend.pipeline import device, retriever, vector_store
from backend.pipeline.chunk import Chunk


class ChunkTests(unittest.TestCase):
    def test_behaves_as_a_string_with_metadata(self):
        chunk = Chunk("some text", source_file="a.pdf", page=3, section="Intro")
        self.assertEqual(chunk, "some text")
        self.assertEqual(chunk.upper(), "SOME TEXT")
        self.assertEqual((chunk.source_file, chunk.page, chunk.section),
                         ("a.pdf", 3, "Intro"))

    def test_record_round_trip(self):
        chunk = Chunk("x", source_file="a.pdf", page=2, section="Methods")
        again = Chunk.from_record(chunk.to_record())
        self.assertEqual(again.to_record(), chunk.to_record())

    def test_older_records_still_load(self):
        self.assertIsNone(Chunk.from_record("bare string").page)
        old = Chunk.from_record({"text": "t", "source_file": "a.pdf", "page": 1})
        self.assertIsNone(old.section)


class SectionContextTests(unittest.TestCase):
    def test_section_title_is_prefixed_only_when_known(self):
        from backend.pipeline.embedder import with_section_context
        self.assertEqual(with_section_context(Chunk("body", "a.pdf", 2, "Model")),
                         "Model. body")
        self.assertEqual(with_section_context(Chunk("body")), "body")
        self.assertEqual(with_section_context("plain"), "plain")


class VectorStoreTests(unittest.TestCase):
    def test_save_and_load_keep_vectors_order_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vector_store, "INDEX_PATH", os.path.join(tmp, "i.faiss")), \
                 mock.patch.object(vector_store, "CHUNKS_PATH", os.path.join(tmp, "c.json")):
                vectors = np.eye(3, 4, dtype=np.float32)
                chunks = [Chunk("zero", "a.pdf", 1, "S1"), "plain", Chunk("two", "b.pdf", 5)]
                vector_store.build_and_save(vectors, chunks)
                index, loaded = vector_store.load()

                self.assertEqual(index.ntotal, 3)
                self.assertEqual(list(map(str, loaded)), ["zero", "plain", "two"])
                self.assertEqual((loaded[0].page, loaded[0].section), (1, "S1"))
                self.assertIsNone(loaded[1].source_file)
                _, nearest = index.search(vectors[2:3], 1)
                self.assertEqual(loaded[nearest[0][0]], "two")
                with open(vector_store.CHUNKS_PATH, encoding="utf-8") as f:
                    self.assertEqual(len(json.load(f)), 3)

    def test_missing_store_is_reported(self):
        with mock.patch.object(vector_store, "INDEX_PATH", "missing/i.faiss"):
            with self.assertRaises(FileNotFoundError):
                vector_store.load()


class FakeEncoder:
    """Maps known strings to fixed vectors."""
    vectors = {"q-first": [1, 0, 0], "q-second": [0, 1, 0]}

    def encode(self, texts, **kwargs):
        return np.array([self.vectors[t] for t in texts], dtype=np.float32)


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.index = faiss.IndexFlatL2(3)
        self.index.add(np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32))
        self.chunks = ["first", "second", "near-first"]
        patcher = mock.patch.object(retriever, "_get_model", return_value=FakeEncoder())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_nearest_chunks_in_order(self):
        self.assertEqual(retriever.retrieve("q-first", self.index, self.chunks, k=2),
                         ["first", "near-first"])

    def test_k_larger_than_index_returns_what_exists(self):
        self.assertEqual(len(retriever.retrieve("q-second", self.index, self.chunks, k=10)), 3)


class DeviceTests(unittest.TestCase):
    def test_cpu_and_npu_requests_give_a_cpu_torch_device(self):
        for requested in ("cpu", "npu", "CPU"):
            with mock.patch.dict(os.environ, {"SA_DEVICE": requested}):
                self.assertEqual(device.get_torch_device(), "cpu")

    def test_npu_flag(self):
        with mock.patch.dict(os.environ, {"SA_DEVICE": "npu"}):
            self.assertTrue(device.should_use_npu())
        with mock.patch.dict(os.environ, {"SA_DEVICE": "gpu"}):
            self.assertFalse(device.should_use_npu())

    def test_gpu_request_falls_back_when_xpu_is_unavailable(self):
        import torch
        with mock.patch.dict(os.environ, {"SA_DEVICE": "gpu"}), \
             mock.patch.object(torch.xpu, "is_available", return_value=False):
            self.assertEqual(device.get_torch_device(), "cpu")

    def test_gpu_request_falls_back_when_the_check_raises(self):
        import torch
        with mock.patch.dict(os.environ, {"SA_DEVICE": "gpu"}), \
             mock.patch.object(torch.xpu, "is_available", side_effect=RuntimeError("no driver")):
            self.assertEqual(device.get_torch_device(), "cpu")

    def test_easyocr_gets_false_on_cpu(self):
        with mock.patch.dict(os.environ, {"SA_DEVICE": "cpu"}):
            self.assertIs(device.get_easyocr_device(), False)


if __name__ == "__main__":
    unittest.main()
