import unittest
from unittest import mock

from backend.pipeline import generator
from tests.helpers import WhitespaceTokenizer


class AllocateBudgetTests(unittest.TestCase):
    def test_never_exceeds_budget_or_chunk_length(self):
        for lengths, budget in [([500, 400, 300], 700), ([50, 900], 400),
                                ([10] * 8, 37), ([1000] * 8, 990)]:
            allocation = generator._allocate_budget(lengths, budget)
            self.assertLessEqual(sum(allocation), budget)
            self.assertTrue(all(a <= n for a, n in zip(allocation, lengths)))

    def test_rank_decay_favours_the_top_chunk(self):
        # Worked example used in the report.
        self.assertEqual(generator._allocate_budget([500, 400, 300], 700),
                         [273, 231, 196])

    def test_surplus_from_short_chunks_is_redistributed(self):
        allocation = generator._allocate_budget([20, 500, 500], 600)
        self.assertEqual(allocation[0], 20)
        self.assertEqual(sum(allocation), 600)

    def test_every_chunk_keeps_a_share(self):
        allocation = generator._allocate_budget([1000] * 8, 990)
        self.assertTrue(all(a > 0 for a in allocation))
        self.assertEqual(allocation, sorted(allocation, reverse=True))

    def test_everything_fits(self):
        self.assertEqual(generator._allocate_budget([10, 20], 100), [10, 20])


class BudgetContextTests(unittest.TestCase):
    def test_context_that_fits_is_unchanged(self):
        chunks = ["alpha beta", "gamma delta"]
        context = generator._budget_context(WhitespaceTokenizer(), "q?", chunks)
        self.assertEqual(context, "alpha beta gamma delta")

    def test_long_context_is_trimmed_to_budget_keeping_every_chunk(self):
        tokenizer = WhitespaceTokenizer()
        chunks = [" ".join(f"{p}{i}" for i in range(600)) for p in "abc"]
        context = generator._budget_context(tokenizer, "what?", chunks)
        tokens = context.split()
        self.assertLessEqual(len(tokens), generator.MAX_INPUT_TOKENS)
        self.assertEqual({t[0] for t in tokens}, {"a", "b", "c"})


class AnswerStyleTests(unittest.TestCase):
    """
    Which model answers depends on where the answer is going: a paragraph for
    the chat view, a short span for the quiz and the evaluation scripts.
    """

    def test_short_is_the_default_so_measurements_keep_meaning(self):
        self.assertEqual(generator.ANSWER_STYLE, "short")

    def test_explain_style_answers_in_prose(self):
        with mock.patch.object(generator, "ANSWER_STYLE", "explain"), \
             mock.patch.object(generator, "explain",
                               return_value="A fitness function scores a solution.") as explain, \
             mock.patch.object(generator, "answer_short") as short:
            answer = generator.generate("What is a fitness function?", ["a passage"])
        self.assertEqual(answer, "A fitness function scores a solution.")
        explain.assert_called_once()
        short.assert_not_called()

    def test_short_style_uses_flan(self):
        with mock.patch.object(generator, "ANSWER_STYLE", "short"), \
             mock.patch.object(generator, "answer_short", return_value="a score") as short, \
             mock.patch.object(generator, "explain") as explain:
            self.assertEqual(generator.generate("q?", ["a passage"]), "a score")
        short.assert_called_once()
        explain.assert_not_called()

    def test_streaming_follows_the_same_style(self):
        with mock.patch.object(generator, "ANSWER_STYLE", "explain"), \
             mock.patch.object(generator, "explain_stream",
                               return_value=iter(["A ", "paragraph."])) as streamer:
            self.assertEqual(list(generator.stream("q?", ["a passage"])),
                             ["A ", "paragraph."])
        streamer.assert_called_once()

    def test_the_quiz_asks_for_a_short_answer_whatever_the_chat_style_is(self):
        from backend.pipeline import quiz
        with mock.patch.object(generator, "ANSWER_STYLE", "explain"), \
             mock.patch.object(generator, "answer_short",
                               return_value="680,000 hours") as short, \
             mock.patch.object(generator, "complete",
                               return_value="How many hours of audio were used?"), \
             mock.patch.object(generator, "explain") as explain:
            item, reason = quiz.generate_item(
                "Whisper was trained on 680,000 hours of audio collected from "
                "the web, which is far more than earlier systems used.")
        self.assertIsNotNone(item, reason)
        self.assertEqual(item.answer, "680,000 hours")
        short.assert_called()
        explain.assert_not_called()


class NumberSpacingTests(unittest.TestCase):
    def test_decoded_number_artefacts_are_repaired(self):
        self.assertEqual(generator._fix_number_spacing("0. 28"), "0.28")
        self.assertEqual(generator._fix_number_spacing("11 : 39 a.m."), "11:39 a.m.")
        self.assertEqual(generator._fix_number_spacing("$ 975. 00"), "$975.00")
        self.assertEqual(generator._fix_number_spacing("3, 197 pairs"), "3,197 pairs")

    def test_prose_is_untouched(self):
        text = "Whisper was trained on audio. It works well."
        self.assertEqual(generator._fix_number_spacing(text), text)


if __name__ == "__main__":
    unittest.main()
