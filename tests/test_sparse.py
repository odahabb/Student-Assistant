import unittest
from unittest import mock

import faiss
import numpy as np

from backend.pipeline import retriever, sparse
from backend.pipeline.chunk import Chunk


class TokenizeTests(unittest.TestCase):
    def test_lowercases_splits_and_strips_plurals(self):
        self.assertEqual(sparse.tokenize("Stakeholders' F-P-CN-01 models, 680,000 hours!"),
                         ["stakeholder", "f", "p", "cn", "01", "model", "680", "000", "hour"])

    def test_keeps_words_that_only_look_plural(self):
        self.assertEqual(sparse.tokenize("process status analysis bus"),
                         ["process", "status", "analysis", "bus"])


class BM25Tests(unittest.TestCase):
    docs = [
        "whisper was trained on 680000 hours of audio",
        "the model uses an encoder and a decoder",
        "audio is split into thirty second segments of audio",
        "training data and evaluation data",
    ]

    def test_exact_term_ranks_its_document_first(self):
        scores = sparse.BM25(self.docs).scores("How many hours was Whisper trained on?")
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_rare_terms_weigh_more_than_common_ones(self):
        bm25 = sparse.BM25(self.docs)
        self.assertGreater(bm25.idf["encoder"], bm25.idf["audio"])

    def test_repeated_terms_saturate(self):
        once = sparse.BM25(["cat dog", "cat " * 1 + "x"]).scores("cat")[1]
        many = sparse.BM25(["cat dog", "cat " * 20 + "x"]).scores("cat")[1]
        self.assertLess(many, once * 3)

    def test_unknown_query_scores_zero(self):
        self.assertEqual(sparse.BM25(self.docs).scores("zebra"), [0.0] * 4)

    def test_section_title_is_indexed(self):
        chunk = Chunk("list of parties", "a.pdf", 2, "Stakeholder")
        self.assertEqual(sparse.index_text(chunk), "Stakeholder\nlist of parties")
        bm25 = sparse.build_index([chunk, Chunk("other text here", "a.pdf", 3, "Scope")])
        self.assertGreater(bm25.scores("stakeholders")[0], 0)


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class HybridRetrieveTests(unittest.TestCase):
    def setUp(self):
        # chunk 0 is closest in embedding space; chunk 2 holds the exact term
        vectors = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
        self.index = faiss.IndexFlatL2(2)
        self.index.add(vectors)
        self.chunks = ["general overview of speech models",
                       "notes on decoders",
                       "whisper large has 1550M parameters"]
        patcher = mock.patch.object(retriever, "_get_model", return_value=FakeEncoder())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dense_only_without_a_sparse_index(self):
        self.assertEqual(retriever.retrieve("whisper parameters", self.index, self.chunks, k=1),
                         ["general overview of speech models"])

    def test_keyword_match_can_outrank_the_nearest_embedding(self):
        bm25 = sparse.BM25(self.chunks)
        top = retriever.retrieve("how many parameters does whisper large have",
                                 self.index, self.chunks, k=3, sparse=bm25)
        self.assertEqual(top[0], "whisper large has 1550M parameters")
        self.assertEqual(len(top), 3)

    def test_dense_weight_one_reproduces_dense_ranking(self):
        bm25 = sparse.BM25(self.chunks)
        top = retriever.retrieve("whisper parameters", self.index, self.chunks, k=3,
                                 sparse=bm25, dense_weight=1.0)
        self.assertEqual(top, retriever.retrieve("whisper parameters", self.index,
                                                 self.chunks, k=3))

    def test_mismatched_index_is_rejected(self):
        with self.assertRaises(ValueError):
            retriever.retrieve("q", self.index, self.chunks, k=1,
                               sparse=sparse.BM25(self.chunks[:2]))


if __name__ == "__main__":
    unittest.main()
