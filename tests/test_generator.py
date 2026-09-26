"""
Tests for backend/pipeline/generator.py: the context budget, the question
shapes, the conversation turn gate, the text tidying, and the Ollama backend.
No model is loaded — generation is replaced by stubs.
"""

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
        # The top-ranked chunk keeps the largest share of the budget.
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
    SA_ANSWER_STYLE decides the shape of an answer: a paragraph for the chat
    view, a short span for the quiz.
    """

    def test_short_is_the_default(self):
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

    def test_short_style_answers_extractively(self):
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


class ShortAnswerTidyingTests(unittest.TestCase):
    """
    An instruction-tuned model wraps a short answer in the shape of a reply;
    the quiz compares the answer itself with what the student typed.
    """

    def test_labels_quotes_and_a_lone_full_stop_are_removed(self):
        for raw, clean in [("Answer: 680,000 hours", "680,000 hours"),
                           ('"the roulette wheel"', "the roulette wheel"),
                           ("A: Turing", "Turing"),
                           ("a fitness function.", "a fitness function")]:
            self.assertEqual(generator._tidy_short(raw), clean)

    def test_a_real_sentence_keeps_its_punctuation(self):
        text = "It scores a solution. It is used for selection."
        self.assertEqual(generator._tidy_short(text), text)

    def test_the_model_is_used_for_both_styles_by_default(self):
        self.assertEqual(generator.CHAT_MODEL_NAME, generator.MODEL_NAME)
        self.assertIn("Qwen", generator.MODEL_NAME)

    def test_only_t5_loads_as_an_encoder_decoder(self):
        self.assertTrue(generator._is_seq2seq("google/flan-t5-large"))
        self.assertFalse(generator._is_seq2seq("Qwen/Qwen2.5-1.5B-Instruct"))


class QuestionShapeTests(unittest.TestCase):
    """
    A short answer takes the shape of its question. The classification is done
    in code because one prompt asking a 1.5B model to choose between three
    shapes answered "No" to "how many TPUs were used?".
    """

    def test_yes_or_no_questions(self):
        for question in ("Does BERT use absolute position embeddings?",
                         "Is the model trained from scratch?",
                         "Are the weights released?",
                         "Can the method run on one GPU?",
                         "Have they compared with BM25?"):
            self.assertEqual(generator.question_shape(question), "boolean",
                             question)

    def test_how_and_why_questions_want_a_sentence(self):
        for question in ("How does BERT represent position?",
                         "Why is the loss weighted?",
                         "In what way does it differ from BM25?"):
            self.assertEqual(generator.question_shape(question), "sentence",
                             question)

    def test_counting_questions_want_a_span_not_a_sentence(self):
        for question in ("How many TPUs were used?",
                         "How much data was collected?",
                         "How long did training take?"):
            self.assertEqual(generator.question_shape(question), "span",
                             question)

    def test_everything_else_is_a_span(self):
        for question in ("What is a fitness function?",
                         "Which datasets were used?",
                         "Who wrote the paper?",
                         ""):
            self.assertEqual(generator.question_shape(question), "span",
                             question)

    def test_each_shape_gets_its_own_instruction(self):
        seen = set()
        for question in ("Is it open source?", "Why does it work?",
                         "What is the batch size?"):
            messages = generator._short_messages(question, ["a passage"])
            seen.add(messages[0]["content"])
        self.assertEqual(len(seen), 3)


class TurnRoutingTests(unittest.TestCase):
    """
    Deciding what a message in a conversation is. The shipped gate is a rule;
    SA_TURN_GATE=model asks the model instead, which is how the two were
    compared. On fifteen hand-written turns the rule was right fourteen times
    and the model eight, and every one of the model's mistakes skipped
    retrieval on a question that needed it.
    """

    HISTORY = [{"role": "user", "content": "What is a fitness function?"},
               {"role": "assistant", "content":
                   "A fitness function assigns a score to a candidate "
                   "solution based on how well it meets the objective."}]

    def test_a_marker_about_the_previous_answer_is_a_follow_up(self):
        for message in ("can you say that more simply?",
                        "explain that again in one sentence",
                        "what do you mean by 'objective'?",
                        "summarise your answer",
                        "where did that come from?",
                        "put it in bullet points"):
            self.assertEqual(generator.classify_turn(message, self.HISTORY),
                             "followup", message)

    def test_a_message_naming_new_subject_matter_is_never_a_follow_up(self):
        # A message naming something the conversation has not covered is
        # retrieved for, whatever markers it carries.
        for message in ("what about tournament selection?",
                        "and roulette wheel selection?",
                        "how is it used in breeding?",
                        "What is a phenotype?",
                        "summarise the mutation slides"):
            self.assertNotEqual(generator.classify_turn(message, self.HISTORY),
                                "followup", message)

    def test_new_subject_matter_is_what_the_conversation_has_not_mentioned(self):
        self.assertFalse(generator.introduces_new_subject(
            "can you say that more simply?", self.HISTORY))
        self.assertTrue(generator.introduces_new_subject(
            "what about tournament selection?", self.HISTORY))

    def test_the_rule_needs_no_model(self):
        with mock.patch.object(generator, "_reply") as reply:
            generator.classify_turn("say that more simply", self.HISTORY)
            generator.classify_turn("what about crossover?", self.HISTORY)
        reply.assert_not_called()

    def test_the_first_message_of_a_conversation_needs_no_model(self):
        with mock.patch.object(generator, "_reply") as reply:
            self.assertEqual(generator.classify_turn("What is it?", []), "new")
            self.assertEqual(generator.standalone_question("What is it?", []),
                             "What is it?")
        reply.assert_not_called()

    def test_a_rewrite_that_is_not_a_question_is_discarded(self):
        for reply in ("The student wants tournament selection.", "",
                      " ".join(["word"] * 50) + "?"):
            with mock.patch.object(generator, "_reply", return_value=reply):
                self.assertEqual(
                    generator.standalone_question("what about it?", self.HISTORY),
                    "what about it?")

    def test_a_good_rewrite_is_used(self):
        with mock.patch.object(generator, "_reply",
                               return_value='"What is tournament selection?"'):
            self.assertEqual(
                generator.standalone_question("what about that?", self.HISTORY),
                "What is tournament selection?")


class ModelTurnGateTests(unittest.TestCase):
    """The optional model gate, and how it reads a one-word verdict."""

    HISTORY = TurnRoutingTests.HISTORY

    def classify(self, reply):
        with mock.patch.object(generator, "TURN_GATE", "model"), \
             mock.patch.object(generator, "_reply", return_value=reply):
            return generator.classify_turn("say that again simply", self.HISTORY)

    def test_each_kind_is_recognised(self):
        for reply in ("new", "continuation", "followup"):
            self.assertEqual(self.classify(reply), reply)

    def test_the_word_is_taken_from_a_longer_reply(self):
        self.assertEqual(self.classify("followup - it asks about the answer"),
                         "followup")

    def test_an_unparseable_reply_is_treated_as_a_new_question(self):
        for reply in ("", "I think this is about genetic algorithms", "42"):
            self.assertEqual(self.classify(reply), "new")


class DecodedSpacingTests(unittest.TestCase):
    """
    Chunk text is wordpiece-decoded, so punctuation carries spaces around it
    and a span copied out of a passage arrives with that spacing.
    """

    def test_decoded_punctuation_is_rejoined(self):
        for damaged, clean in [
                ("ilur . am", "ilur.am"),
                ("bleu - 4, nist - 4", "bleu-4, nist-4"),
                ("pubmed + pmc", "pubmed+pmc"),
                ("vendor lock - in", "vendor lock-in"),
                ("all - minilm - l6 - v2", "all-minilm-l6-v2"),
                ("multilingual nmt ( mnmt )", "multilingual nmt (mnmt)"),
                ("presence / absence", "presence/absence")]:
            self.assertEqual(generator.fix_decoded_spacing(damaged), clean)

    def test_a_sentence_boundary_is_not_closed_up(self):
        # A full stop between two sentences is left alone; only one inside
        # a decoded name is closed up.
        for prose in ("It scores a solution. It guides selection.",
                      "The model is small. Training took four days.",
                      "See Table 2. Results follow."):
            self.assertEqual(generator.fix_decoded_spacing(prose), prose)

    def test_a_short_answer_is_repaired(self):
        with mock.patch.object(generator, "ABSTAIN", "off"), \
             mock.patch.object(generator, "_reply", return_value="bleu - 4"), \
             mock.patch.object(generator, "_budget_context",
                               return_value="context"):
            self.assertEqual(generator.answer_short("Which metric?", ["p"]),
                             "bleu-4")


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


class _FakeResponse:
    """What urllib.request.urlopen returns: readable whole, or line by line."""

    def __init__(self, payload=None, lines=()):
        self.payload = payload
        self.lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        import json
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        import json
        return iter(json.dumps(line).encode("utf-8") + b"\n" for line in self.lines)


class OllamaBackendTests(unittest.TestCase):
    """The optional quality mode, and falling back when Ollama is missing."""

    def setUp(self):
        # _fall_back rebinds these module globals, so each test restores them.
        for name, value in [("BACKEND", "ollama"), ("_ollama_up", None),
                            ("MODEL_NAME", generator.OLLAMA_MODEL),
                            ("CHAT_MODEL_NAME", generator.OLLAMA_MODEL)]:
            patcher = mock.patch.object(generator, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def serving(self, *names):
        return _FakeResponse({"models": [{"name": n, "model": n} for n in names]})

    def test_the_in_process_model_is_the_default(self):
        with mock.patch.object(generator, "BACKEND", "transformers"), \
             mock.patch("urllib.request.urlopen") as urlopen:
            self.assertFalse(generator._use_ollama())
        urlopen.assert_not_called()

    def test_a_missing_server_falls_back_to_the_in_process_model(self):
        import urllib.error
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("refused")), \
             self.assertLogs(generator.log, "WARNING"):
            self.assertFalse(generator._use_ollama())
        self.assertEqual(generator.MODEL_NAME, generator.LOCAL_MODEL_NAME)
        self.assertEqual(generator.CHAT_MODEL_NAME, generator.LOCAL_CHAT_MODEL_NAME)

    def test_a_server_without_the_model_falls_back(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=self.serving("llama3.2:latest")), \
             self.assertLogs(generator.log, "WARNING") as logs:
            self.assertFalse(generator._use_ollama())
        self.assertIn("ollama pull", logs.output[0])

    def test_the_server_is_checked_once(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=self.serving(generator.OLLAMA_MODEL)) as urlopen:
            self.assertTrue(generator._use_ollama())
            self.assertTrue(generator._use_ollama())
        self.assertEqual(urlopen.call_count, 1)

    def test_a_paragraph_streams_piece_by_piece(self):
        lines = [{"message": {"content": "Glycolysis "}, "done": False},
                 {"message": {"content": "happens in the cytoplasm."}, "done": False},
                 {"message": {"content": ""}, "done": True}]
        with mock.patch.object(generator, "_ollama_up", True), \
             mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse(lines=lines)) as urlopen:
            pieces = list(generator.explain_stream("Where?", ["passage"]))
        self.assertEqual(pieces, ["Glycolysis ", "happens in the cytoplasm."])
        import json
        body = json.loads(urlopen.call_args[0][0].data)
        self.assertTrue(body["stream"])
        self.assertFalse(body["think"])

    def test_ollama_failing_before_the_first_piece_answers_in_process(self):
        import urllib.error
        with mock.patch.object(generator, "_ollama_up", True), \
             mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("gone")), \
             mock.patch.object(generator, "_stream_messages",
                               return_value=iter(["from ", "the 1.5B"])) as local, \
             self.assertLogs(generator.log, "WARNING"):
            pieces = list(generator.explain_stream("Where?", ["passage"]))
        self.assertEqual(pieces, ["from ", "the 1.5B"])
        local.assert_called_once()
        self.assertFalse(generator._ollama_up)

    def test_a_short_answer_falls_back_too(self):
        import urllib.error
        with mock.patch.object(generator, "_ollama_up", True), \
             mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("gone")), \
             mock.patch.object(generator, "_get_model",
                               side_effect=RuntimeError("in process")) as local, \
             self.assertLogs(generator.log, "WARNING"), \
             self.assertRaisesRegex(RuntimeError, "in process"):
            generator._reply([{"role": "user", "content": "q"}], 8)
        local.assert_called_once()

    def test_an_error_mid_answer_is_reported_not_restarted(self):
        class Broken(_FakeResponse):
            def __iter__(self):
                import json
                yield json.dumps({"message": {"content": "Half an "}}).encode()
                raise ConnectionResetError("dropped")

        with mock.patch.object(generator, "_ollama_up", True), \
             mock.patch("urllib.request.urlopen", return_value=Broken()), \
             mock.patch.object(generator, "_stream_messages") as local, \
             self.assertLogs(generator.log, "WARNING"):
            stream = generator.explain_stream("Where?", ["passage"])
            self.assertEqual(next(stream), "Half an ")
            with self.assertRaises(generator.OllamaUnavailable):
                next(stream)
        local.assert_not_called()
