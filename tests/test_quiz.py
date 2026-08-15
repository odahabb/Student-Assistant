import random
import unittest

from backend.pipeline import quiz
from backend.pipeline.chunk import Chunk

LONG = " ".join(["gradient descent updates the weights using the loss gradient"] * 6)
REFERENCES = ("smith , j . et al . ( 2021 ) . a study . in proceedings of the "
              "conference . arxiv preprint https : / / doi . org / 10 . 1 ( 2020 ) ") * 2


def no_similarity(a, b):
    return 0.0


class TopicTests(unittest.TestCase):
    def test_chunks_group_by_document_and_section_in_order(self):
        chunks = [
            Chunk(LONG, "a.pdf", 1, "Intro"),
            Chunk(LONG, "a.pdf", 2, "Methods"),
            Chunk(LONG, "a.pdf", 3, "Intro"),
            Chunk(LONG, "talk.m4a"),
        ]
        topics = quiz.build_topics(chunks)
        self.assertEqual([t.id for t in topics], [
            "a.pdf › Intro", "a.pdf › Methods", "talk.m4a › Whole document"])
        self.assertEqual(topics[0].chunk_indices, [0, 2])

    def test_reference_sections_short_chunks_and_bibliographies_are_skipped(self):
        chunks = [
            Chunk(LONG, "a.pdf", 9, "References"),
            Chunk("too short to quiz on", "a.pdf", 1, "Intro"),
            Chunk(REFERENCES, "a.pdf", 2, "Intro"),
            Chunk(LONG, "a.pdf", 3, "Results"),
        ]
        self.assertEqual([t.id for t in quiz.build_topics(chunks)], ["a.pdf › Results"])

    def test_reference_signals(self):
        self.assertTrue(quiz.looks_like_reference_list(REFERENCES))
        self.assertFalse(quiz.looks_like_reference_list(LONG))


class GradeTests(unittest.TestCase):
    def test_exact_and_contained_answers_are_correct(self):
        for answer in ("F1 score", "the f1-score", "It is the F1 score."):
            result = quiz.grade(answer, "f1 score", similarity=no_similarity)
            self.assertTrue(result.correct, answer)
            self.assertEqual(result.method, "containment")

    def test_a_fragment_of_a_long_reference_is_not_enough(self):
        result = quiz.grade("error", "widening gap between training error and "
                            "validation error", similarity=no_similarity)
        self.assertFalse(result.correct)

    def test_partial_overlap_uses_token_f1(self):
        result = quiz.grade("gap between training and validation error",
                            "widening gap between training error and validation error",
                            similarity=no_similarity)
        self.assertTrue(result.correct)
        self.assertEqual(result.method, "token_f1")

    def test_semantic_similarity_can_accept_a_paraphrase(self):
        result = quiz.grade("the model memorises its data", "overfitting",
                            similarity=lambda a, b: 0.82)
        self.assertTrue(result.correct)
        self.assertEqual(result.method, "cosine")

    def test_wrong_and_empty_answers(self):
        self.assertFalse(quiz.grade("bias", "variance", similarity=no_similarity).correct)
        self.assertEqual(quiz.grade("  ", "variance").score, 0.0)

    def test_numbers_in_the_reference_must_match(self):
        high = lambda a, b: 0.95  # noqa: E731
        self.assertFalse(quiz.grade("861 hours", "680,000 hours", similarity=high).correct)
        self.assertEqual(quiz.grade("2e-5", "1e-5", similarity=high).method,
                         "number_mismatch")
        self.assertTrue(quiz.grade("about 680 000 hours of audio", "680,000 hours",
                                   similarity=no_similarity).correct)
        self.assertTrue(quiz.grade("three thousand one hundred ninety-seven pairs",
                                   "sentence pairs", similarity=high).correct)

    def test_threshold_is_respected(self):
        result = quiz.grade("x", "y", similarity=lambda a, b: 0.69)
        self.assertFalse(result.correct)
        result = quiz.grade("x", "y", threshold=0.6, similarity=lambda a, b: 0.69)
        self.assertTrue(result.correct)


class GenerationTests(unittest.TestCase):
    chunk = Chunk(LONG, "notes.pdf", 4, "Optimisation")

    def test_well_formed_rules(self):
        self.assertIsNone(quiz.well_formed("What does gradient descent update?", "the weights"))
        self.assertIsNotNone(quiz.well_formed("What does it update", "the weights"))
        self.assertIsNotNone(quiz.well_formed("Why?", "because"))
        self.assertIsNotNone(quiz.well_formed("What does it update?", "unanswerable"))
        self.assertIsNotNone(quiz.well_formed("What updates the weights?", "the weights"))
        self.assertIsNotNone(quiz.well_formed("What does it update?", " ".join(["w"] * 20)))

    def test_item_carries_its_source(self):
        item, reason = quiz.generate_item(
            self.chunk,
            question_fn=lambda prompt: "What does gradient descent update?",
            answer_fn=lambda q, chunks: "the weights")
        self.assertIsNone(reason)
        self.assertEqual((item.topic_id, item.page, item.section),
                         ("notes.pdf › Optimisation", 4, "Optimisation"))
        self.assertIn(LONG, quiz.QUESTION_PROMPT.format(passage=item.passage))
        self.assertEqual(quiz.QuizItem.from_record(item.to_record()), item)

    def test_round_trip_disagreement_rejects_the_item(self):
        answers = iter(["the weights", "the learning rate"])
        item, reason = quiz.generate_item(
            self.chunk,
            question_fn=lambda prompt: "What does gradient descent update?",
            answer_fn=lambda q, chunks: next(answers),
            retrieve_fn=lambda q: ["other chunk"],
            similarity=no_similarity)
        self.assertIsNone(item)
        self.assertEqual(reason, "failed round-trip check")

    def test_round_trip_agreement_keeps_the_item(self):
        seen = []
        item, _ = quiz.generate_item(
            self.chunk,
            question_fn=lambda prompt: "What does gradient descent update?",
            answer_fn=lambda q, chunks: seen.append(chunks) or "the weights",
            retrieve_fn=lambda q: ["retrieved chunk"],
            similarity=no_similarity)
        self.assertEqual(item.roundtrip_score, 1.0)
        self.assertEqual(seen, [[self.chunk], ["retrieved chunk"]])

    def test_pool_skips_duplicates_and_respects_per_topic(self):
        chunks = [Chunk(LONG, "a.pdf", p, "S") for p in range(1, 5)]
        topics = quiz.build_topics(chunks)
        questions = iter(["What is item one?", "What is item one?",
                          "What is item two?", "What is item three?"])
        pool = quiz.build_pool(
            chunks, topics, per_topic=2,
            question_fn=lambda prompt: next(questions),
            answer_fn=lambda q, c: "an answer")
        self.assertEqual([i.question for i in pool],
                         ["What is item one?", "What is item two?"])


class MultipleChoiceTests(unittest.TestCase):
    @staticmethod
    def item(topic, answer):
        return quiz.QuizItem(topic, f"Q about {answer}?", answer, "a.pdf", 1, "S", "p")

    def test_options_are_unique_include_the_answer_and_prefer_the_same_topic(self):
        target = self.item("T1", "accuracy")
        pool = [target, self.item("T1", "precision"), self.item("T1", "Accuracy"),
                self.item("T1", "recall"), self.item("T2", "bias"), self.item("T2", "variance")]
        options = quiz.multiple_choice(target, pool, random.Random(1))
        self.assertEqual(len(options), 4)
        self.assertIn("accuracy", options)
        self.assertEqual(len({quiz.normalize(o) for o in options}), 4)
        self.assertTrue({"precision", "recall"} <= set(options))

    def test_small_pool_gives_fewer_options(self):
        target = self.item("T1", "accuracy")
        options = quiz.multiple_choice(target, [target, self.item("T2", "bias")])
        self.assertEqual(sorted(options), ["accuracy", "bias"])


if __name__ == "__main__":
    unittest.main()
