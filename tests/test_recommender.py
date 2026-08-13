import os
import tempfile
import unittest

from backend.pipeline import recommender
from backend.pipeline.recommender import Progress


class ModelTests(unittest.TestCase):
    def test_harder_levels_are_less_likely_to_be_answered(self):
        p = [recommender.success_probability(0.0, level) for level in (1, 2, 3)]
        self.assertEqual(p, sorted(p, reverse=True))
        self.assertAlmostEqual(p[1], 0.5)

    def test_new_topic_starts_at_the_prior(self):
        progress = Progress()
        self.assertAlmostEqual(progress.mastery("T"), 0.5)
        self.assertEqual(progress.attempts_on("T"), 0)
        self.assertIsNone(progress.accuracy("T"))

    def test_correct_answers_raise_and_wrong_answers_lower_mastery(self):
        progress = Progress()
        progress.record("up", 2, True)
        progress.record("down", 2, False)
        self.assertGreater(progress.mastery("up"), 0.5)
        self.assertLess(progress.mastery("down"), 0.5)

    def test_updates_shrink_as_evidence_accumulates(self):
        progress = Progress()
        steps, before = [], progress.ability.get("T", 0.0)
        for _ in range(4):
            after = progress.record("T", 2, True)
            steps.append(after - before)
            before = after
        self.assertEqual(steps, sorted(steps, reverse=True))

    def test_surprising_answers_move_the_estimate_more(self):
        easy_wrong, hard_wrong = Progress(), Progress()
        easy_wrong.record("T", 1, False)
        hard_wrong.record("T", 3, False)
        self.assertLess(easy_wrong.mastery("T"), hard_wrong.mastery("T"))

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            Progress().record("T", 7, True)


class NextLevelTests(unittest.TestCase):
    def test_new_topic_starts_easy(self):
        self.assertEqual(Progress().next_level("T"), 1)

    def test_difficulty_rises_with_success_and_falls_with_failure(self):
        progress = Progress()
        levels = []
        for _ in range(6):
            levels.append(progress.next_level("T"))
            progress.record("T", levels[-1], True)
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(levels[-1], 3)

        for _ in range(6):
            progress.record("T", progress.next_level("T"), False)
        self.assertEqual(progress.next_level("T"), 1)


class RecommendTests(unittest.TestCase):
    def test_weak_before_unseen_before_strong(self):
        progress = Progress()
        for _ in range(3):
            progress.record("strong", 2, True)
            progress.record("weak", 2, False)
        ranked = [r.topic_id for r in progress.recommend(["strong", "unseen", "weak"], n=3)]
        self.assertEqual(ranked, ["weak", "unseen", "strong"])

    def test_reasons_and_limit(self):
        progress = Progress()
        progress.record("A", 2, False)
        progress.record("A", 2, True)
        recs = progress.recommend(["A", "B", "C"], n=2)
        self.assertEqual(len(recs), 2)
        reasons = {r.topic_id: r.reason for r in recs}
        self.assertIn("1/2 correct", reasons.get("A", ""))
        self.assertIn("Not practised yet", reasons.values())


class PersistenceTests(unittest.TestCase):
    def test_saved_progress_replays_to_the_same_state(self):
        progress = Progress()
        for topic, level, correct in [("A", 1, True), ("A", 2, False), ("B", 3, True)]:
            progress.record(topic, level, correct, question="q", answer="a")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "progress.json")
            progress.save(path)
            loaded = Progress.load(path)
        self.assertEqual(loaded.ability, progress.ability)
        self.assertEqual(loaded.count, progress.count)
        self.assertEqual(loaded.attempts[0]["question"], "q")

    def test_missing_file_gives_empty_progress(self):
        self.assertEqual(Progress.load("does/not/exist.json").attempts, [])


if __name__ == "__main__":
    unittest.main()
