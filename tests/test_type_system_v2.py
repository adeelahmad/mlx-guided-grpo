"""
Comprehensive Tests for Type System V2
========================================

Tests:
1. EventBus: pub/sub, handler isolation, unsubscribe
2. TypeComponentMeta: auto-registration
3. MCQReward: answer matching, gaming detection, format scoring
4. GeneralQNAReward: correctness, format, thinking quality
5. ToolCallReward: contamination detection, function scoring
6. MCQDatasetLoader: validation
7. GeneralQNADatasetLoader: validation
8. MCQRolloutGenerator: config, curriculum, completion check
9. GeneralQNARolloutGenerator: config, curriculum
10. TypeCoordinator: registration, retrieval, detection, normalization
11. Bridge: adapter, type normalization, coordinator factory
"""

import unittest
from unittest.mock import MagicMock

from mlx_grpo.trainer.type_system_v2.events import (
    Event, EventBus,
    REWARD_COMPUTED, REWARD_INVALID, TYPE_REGISTERED,
)
from mlx_grpo.trainer.type_system_v2.coordinator import (
    TypeCoordinator, normalize_type,
)
from mlx_grpo.trainer.type_system_v2.rewards.mcq import MCQReward
from mlx_grpo.trainer.type_system_v2.rewards.general_qna import GeneralQNAReward
from mlx_grpo.trainer.type_system_v2.rewards.tool_call import ToolCallReward
from mlx_grpo.trainer.type_system_v2.loaders.mcq import MCQDatasetLoader
from mlx_grpo.trainer.type_system_v2.loaders.general_qna import GeneralQNADatasetLoader
from mlx_grpo.trainer.type_system_v2.generators.mcq import MCQRolloutGenerator
from mlx_grpo.trainer.type_system_v2.generators.general_qna import GeneralQNARolloutGenerator
from mlx_grpo.trainer.type_system_v2.bridge import (
    v2_reward_adapter, v2_type_normalizer,
)


# =============================================================================
# 1. EventBus Tests
# =============================================================================

class TestEventBus(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()

    def test_subscribe_and_publish(self):
        received = []
        self.bus.subscribe("test.event", lambda e: received.append(e))
        self.bus.publish(Event(name="test.event", data={"val": 42}))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data["val"], 42)

    def test_unsubscribe(self):
        received = []
        handler = lambda e: received.append(e)
        self.bus.subscribe("test.event", handler)
        self.bus.unsubscribe("test.event", handler)
        self.bus.publish(Event(name="test.event"))
        self.assertEqual(len(received), 0)

    def test_handler_isolation(self):
        """Handler errors should not propagate."""
        received = []

        def bad_handler(e):
            raise RuntimeError("boom")

        def good_handler(e):
            received.append(e)

        self.bus.subscribe("test.event", bad_handler)
        self.bus.subscribe("test.event", good_handler)
        self.bus.publish(Event(name="test.event"))
        self.assertEqual(len(received), 1)  # good_handler still runs

    def test_no_duplicate_subscribers(self):
        count = []
        handler = lambda e: count.append(1)
        self.bus.subscribe("test.event", handler)
        self.bus.subscribe("test.event", handler)  # duplicate
        self.bus.publish(Event(name="test.event"))
        self.assertEqual(len(count), 1)

    def test_clear(self):
        self.bus.subscribe("a", lambda e: None)
        self.bus.subscribe("b", lambda e: None)
        self.bus.clear()
        self.assertEqual(self.bus.subscriber_count("a"), 0)
        self.assertEqual(self.bus.subscriber_count("b"), 0)

    def test_event_immutability(self):
        event = Event(name="test", data={"key": "value"})
        self.assertEqual(event.name, "test")
        with self.assertRaises(AttributeError):
            event.name = "changed"


# =============================================================================
# 2. Type Normalization Tests
# =============================================================================

class TestTypeNormalization(unittest.TestCase):
    def test_tool_call_aliases(self):
        for alias in ["tool", "function", "function_calling", "func", "tool_use", "api_call"]:
            self.assertEqual(normalize_type(alias), "tool_call", f"Failed for {alias}")

    def test_mcq_aliases(self):
        for alias in ["exam", "aime", "math500", "olympiad", "multiple_choice"]:
            self.assertEqual(normalize_type(alias), "mcq", f"Failed for {alias}")

    def test_general_qna_aliases(self):
        for alias in ["math", "thinking", "reasoning", "code", "default"]:
            self.assertEqual(normalize_type(alias), "general_qna", f"Failed for {alias}")

    def test_none_returns_default(self):
        self.assertEqual(normalize_type(None), "general_qna")

    def test_dict_input(self):
        self.assertEqual(normalize_type({"type": "tool"}), "tool_call")
        self.assertEqual(normalize_type({"type": "exam"}), "mcq")
        self.assertEqual(normalize_type({}), "general_qna")

    def test_case_insensitive(self):
        self.assertEqual(normalize_type("TOOL"), "tool_call")
        self.assertEqual(normalize_type("Exam"), "mcq")

    def test_unknown_type_passthrough(self):
        self.assertEqual(normalize_type("custom_type"), "custom_type")

    def test_v2_type_normalizer_bridge(self):
        self.assertEqual(v2_type_normalizer("tool"), "tool_call")
        self.assertEqual(v2_type_normalizer(None), "general_qna")


# =============================================================================
# 3. MCQ Reward Tests
# =============================================================================

class TestMCQReward(unittest.TestCase):
    def setUp(self):
        self.reward = MCQReward()

    def test_exact_letter_match(self):
        scores = self.reward.compute(
            prompts=["What is 2+2?"],
            completions=["<think>2+2=4</think>B"],
            answers=["B"],
        )
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.3)  # Should get good score

    def test_boxed_answer_match(self):
        scores = self.reward.compute(
            prompts=["Solve x=7"],
            completions=["<think>x=7</think>\\boxed{7}"],
            answers=["7"],
        )
        self.assertGreater(scores[0], 0.3)

    def test_empty_completion_zero(self):
        scores = self.reward.compute(
            prompts=["Q"], completions=[""], answers=["A"],
        )
        self.assertEqual(scores[0], 0.0)

    def test_gaming_detection(self):
        """Hedging should be penalized."""
        # Clean answer
        clean_scores = self.reward.compute(
            prompts=["Q"], completions=["<think>...</think>B"], answers=["B"],
        )
        # Hedging answer
        hedge_scores = self.reward.compute(
            prompts=["Q"],
            completions=["<think>...</think>A or B"],
            answers=["B"],
        )
        # Clean should score higher (or at least not lower)
        self.assertGreaterEqual(clean_scores[0], hedge_scores[0])

    def test_format_scoring(self):
        """Proper think/answer structure should score higher on format."""
        reward = MCQReward()
        # With proper format
        result_good = reward.compute_single(
            "Q", "<think>\nStep 1\n</think>\nB", "B"
        )
        # Without format
        result_bad = reward.compute_single("Q", "B", "B")

        self.assertGreater(
            result_good.component_scores["format_quality"],
            result_bad.component_scores["format_quality"],
        )

    def test_type_info_ground_truth(self):
        """type_info with ground_truth should be used."""
        scores = self.reward.compute(
            prompts=["Q"],
            completions=["<think>...</think>\\boxed{42}"],
            answers=["wrong"],
            types=[{"ground_truth": "42"}],
        )
        self.assertGreater(scores[0], 0.3)

    def test_batch_processing(self):
        scores = self.reward.compute(
            prompts=["Q1", "Q2"],
            completions=[
                "<think>think</think>A",
                "<think>think</think>B",
            ],
            answers=["A", "B"],
        )
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# =============================================================================
# 4. GeneralQNA Reward Tests
# =============================================================================

class TestGeneralQNAReward(unittest.TestCase):
    def setUp(self):
        self.reward = GeneralQNAReward()

    def test_exact_match(self):
        scores = self.reward.compute(
            prompts=["What is 2+2?"],
            completions=["<think>2+2 is 4</think>4"],
            answers=["4"],
        )
        self.assertGreater(scores[0], 0.5)

    def test_boxed_match(self):
        scores = self.reward.compute(
            prompts=["Solve"],
            completions=["<think>Working...</think>\\boxed{42}"],
            answers=["\\boxed{42}"],
        )
        self.assertGreater(scores[0], 0.5)

    def test_no_thinking_lower_score(self):
        """Completions without thinking should score lower on format."""
        with_think = self.reward.compute(
            prompts=["Q"],
            completions=["<think>Step by step</think>answer"],
            answers=["answer"],
        )
        without_think = self.reward.compute(
            prompts=["Q"],
            completions=["answer"],
            answers=["answer"],
        )
        self.assertGreater(with_think[0], without_think[0])

    def test_wrong_answer_low_score(self):
        scores = self.reward.compute(
            prompts=["Q"],
            completions=["<think>...</think>wrong_answer"],
            answers=["correct_answer"],
        )
        self.assertLess(scores[0], 0.5)

    def test_empty_completion(self):
        scores = self.reward.compute(
            prompts=["Q"], completions=[""], answers=["A"],
        )
        self.assertEqual(scores[0], 0.0)

    def test_validate_completion(self):
        valid, reason = self.reward.validate_completion("some text")
        self.assertTrue(valid)

        valid, reason = self.reward.validate_completion("")
        self.assertFalse(valid)

    def test_fuzzy_match(self):
        """Partial word overlap should get some credit."""
        scores = self.reward.compute(
            prompts=["Q"],
            completions=["<think>...</think>The capital of France is Paris"],
            answers=["Paris is the capital of France"],
        )
        self.assertGreater(scores[0], 0.2)


# =============================================================================
# 5. ToolCall Reward Tests
# =============================================================================

class TestToolCallReward(unittest.TestCase):
    def setUp(self):
        self.reward = ToolCallReward(strict=True)

    def test_exact_function_match(self):
        scores = self.reward.compute(
            prompts=["Call add"],
            completions=["add(a=5, b=3)"],
            answers=["add(a=5, b=3)"],
        )
        self.assertGreater(scores[0], 0.8)

    def test_thinking_contamination_zero(self):
        scores = self.reward.compute(
            prompts=["Call add"],
            completions=["<think>thinking</think>add(a=5, b=3)"],
            answers=["add(a=5, b=3)"],
        )
        self.assertEqual(scores[0], 0.0)

    def test_no_function_call_zero(self):
        scores = self.reward.compute(
            prompts=["Call add"],
            completions=["The answer is 8"],
            answers=["add(a=5, b=3)"],
        )
        self.assertEqual(scores[0], 0.0)


# =============================================================================
# 6. MCQ Dataset Loader Tests
# =============================================================================

class TestMCQDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.loader = MCQDatasetLoader(self.tokenizer)

    def test_validate_letter_answer(self):
        valid, _ = self.loader.validate_sample({
            "prompt": "What is 2+2?", "answer": "B",
        })
        self.assertTrue(valid)

    def test_validate_exam_type(self):
        valid, _ = self.loader.validate_sample({
            "prompt": "Solve", "answer": "42", "type": "exam",
        })
        self.assertTrue(valid)

    def test_validate_ground_truth(self):
        valid, _ = self.loader.validate_sample({
            "prompt": "Q", "answer": "42", "ground_truth": "42",
        })
        self.assertTrue(valid)

    def test_reject_non_mcq(self):
        valid, reason = self.loader.validate_sample({
            "prompt": "Q", "answer": "a long answer",
        })
        self.assertFalse(valid)

    def test_reject_missing_fields(self):
        valid, _ = self.loader.validate_sample({"prompt": "Q"})
        self.assertFalse(valid)

    def test_preprocess_adds_type_info(self):
        sample = {"prompt": "Q", "answer": "B", "type": "mcq"}
        processed = self.loader.preprocess_sample(sample)
        self.assertEqual(processed["type_info"]["type"], "mcq")
        self.assertTrue(processed["type_info"]["is_exam"])


# =============================================================================
# 7. GeneralQNA Dataset Loader Tests
# =============================================================================

class TestGeneralQNADatasetLoader(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.loader = GeneralQNADatasetLoader(self.tokenizer)

    def test_validate_basic(self):
        valid, _ = self.loader.validate_sample({
            "prompt": "What is Python?",
            "answer": "A programming language",
        })
        self.assertTrue(valid)

    def test_reject_short_prompt(self):
        valid, _ = self.loader.validate_sample({"prompt": "Q", "answer": "A"})
        self.assertFalse(valid)

    def test_reject_missing_answer(self):
        valid, _ = self.loader.validate_sample({"prompt": "What is Python?"})
        self.assertFalse(valid)

    def test_require_think_tags(self):
        strict_loader = GeneralQNADatasetLoader(
            self.tokenizer, require_think_tags=True,
        )
        valid, _ = strict_loader.validate_sample({
            "prompt": "What is 2+2?", "answer": "4",
        })
        self.assertFalse(valid)

        valid, _ = strict_loader.validate_sample({
            "prompt": "What is 2+2?", "answer": "<think>...</think>4",
        })
        self.assertTrue(valid)


# =============================================================================
# 8. MCQ Rollout Generator Tests
# =============================================================================

class TestMCQRolloutGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = MCQRolloutGenerator()

    def test_generation_config(self):
        config = self.gen.get_generation_config()
        self.assertEqual(config.max_length, 1536)
        self.assertEqual(config.temperature, 0.85)
        self.assertTrue(config.two_phase)
        self.assertTrue(config.enforce_thinking)

    def test_curriculum_full(self):
        scaffold = self.gen.apply_curriculum("<think>step1</think>B", ratio=1.0)
        self.assertEqual(scaffold, "<think>step1</think>B")

    def test_curriculum_none(self):
        scaffold = self.gen.apply_curriculum("<think>step1</think>B", ratio=0.0)
        self.assertEqual(scaffold, "")

    def test_curriculum_partial(self):
        scaffold = self.gen.apply_curriculum(
            "<think>step one step two step three</think>B", ratio=0.5
        )
        self.assertTrue(scaffold.startswith("<think>"))
        self.assertLess(len(scaffold), len("<think>step one step two step three</think>B"))

    def test_is_complete_with_answer(self):
        complete, reason = self.gen.is_generation_complete(
            "<think>thinking</think>\nThe answer is B", phase=1
        )
        self.assertTrue(complete)

    def test_is_incomplete_no_close(self):
        complete, _ = self.gen.is_generation_complete(
            "<think>thinking...", phase=1
        )
        self.assertFalse(complete)


# =============================================================================
# 9. GeneralQNA Rollout Generator Tests
# =============================================================================

class TestGeneralQNARolloutGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = GeneralQNARolloutGenerator()

    def test_generation_config(self):
        config = self.gen.get_generation_config()
        self.assertEqual(config.max_length, 1024)
        self.assertEqual(config.temperature, 0.8)
        self.assertTrue(config.two_phase)

    def test_curriculum_with_think_tags(self):
        scaffold = self.gen.apply_curriculum(
            "<think>step one two three four five</think>answer", ratio=0.5
        )
        self.assertTrue(scaffold.startswith("<think>"))

    def test_curriculum_plain_text(self):
        scaffold = self.gen.apply_curriculum(
            "word1 word2 word3 word4", ratio=0.5
        )
        self.assertTrue(scaffold.startswith("<think>"))


# =============================================================================
# 10. TypeCoordinator Tests
# =============================================================================

class TestTypeCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = TypeCoordinator()
        self.reward = MCQReward()
        self.loader = MagicMock()
        self.loader.validate_sample = MagicMock(return_value=(True, None))
        self.generator = MagicMock()

    def test_register_and_retrieve(self):
        self.coordinator.register_type(
            "test_type", self.reward, self.loader, self.generator,
        )
        retrieved = self.coordinator.get_reward("test_type")
        self.assertIs(retrieved, self.reward)

    def test_default_fallback(self):
        self.coordinator.register_type(
            "default_type", self.reward, self.loader, self.generator,
            set_as_default=True,
        )
        # Unknown type should fall back to default
        components = self.coordinator.get_or_default("unknown")
        self.assertIs(components.reward, self.reward)

    def test_list_types(self):
        self.coordinator.register_type(
            "type_a", self.reward, self.loader, self.generator,
        )
        self.coordinator.register_type(
            "type_b", self.reward, self.loader, self.generator,
        )
        types = self.coordinator.list_types()
        self.assertEqual(types, ["type_a", "type_b"])

    def test_unregister(self):
        self.coordinator.register_type(
            "temp", self.reward, self.loader, self.generator,
        )
        self.coordinator.unregister_type("temp")
        self.assertNotIn("temp", self.coordinator.list_types())

    def test_event_on_register(self):
        events = []
        self.coordinator.event_bus.subscribe(
            TYPE_REGISTERED, lambda e: events.append(e)
        )
        self.coordinator.register_type(
            "test_type", self.reward, self.loader, self.generator,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["type_name"], "test_type")

    def test_detect_type_explicit(self):
        self.coordinator.register_type(
            "mcq", self.reward, self.loader, self.generator,
        )
        detected = self.coordinator.detect_type({"type": "mcq", "prompt": "Q"})
        self.assertEqual(detected, "mcq")

    def test_get_type_info(self):
        self.coordinator.register_type(
            "test", self.reward, self.loader, self.generator,
            metadata={"desc": "test type"},
        )
        info = self.coordinator.get_type_info("test")
        self.assertEqual(info["type_name"], "test")
        self.assertEqual(info["reward"], "MCQReward")

    def test_missing_type_raises(self):
        with self.assertRaises(KeyError):
            self.coordinator.get_reward("nonexistent")


# =============================================================================
# 11. Bridge Tests
# =============================================================================

class TestBridge(unittest.TestCase):
    def setUp(self):
        self.coordinator = TypeCoordinator()
        # Register with real rewards
        self.coordinator.register_type(
            "tool_call",
            ToolCallReward(strict=True),
            MagicMock(), MagicMock(),
        )
        self.coordinator.register_type(
            "mcq",
            MCQReward(),
            MagicMock(), MagicMock(),
        )
        self.coordinator.register_type(
            "general_qna",
            GeneralQNAReward(),
            MagicMock(), MagicMock(),
            set_as_default=True,
        )

    def test_reward_adapter_dispatches_by_type(self):
        adapter = v2_reward_adapter(self.coordinator)

        # Tool call sample
        scores = adapter(
            prompts=["Call add"],
            completions=["add(a=1, b=2)"],
            answers=["add(a=1, b=2)"],
            types=[{"type": "tool_call"}],
        )
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.5)

    def test_reward_adapter_mixed_types(self):
        adapter = v2_reward_adapter(self.coordinator)

        scores = adapter(
            prompts=["Call", "What is 2+2?"],
            completions=[
                "add(a=1, b=2)",
                "<think>2+2=4</think>4",
            ],
            answers=[
                "add(a=1, b=2)",
                "4",
            ],
            types=[
                {"type": "tool_call"},
                {"type": "math"},  # normalizes to general_qna
            ],
        )
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_reward_adapter_no_types(self):
        """Without types, should use default (general_qna)."""
        adapter = v2_reward_adapter(self.coordinator)
        scores = adapter(
            prompts=["Q"],
            completions=["<think>think</think>answer"],
            answers=["answer"],
        )
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.0)


# =============================================================================
# 12. Event Bus Integration Tests
# =============================================================================

class TestEventBusIntegration(unittest.TestCase):
    def test_reward_publishes_events(self):
        bus = EventBus()
        events = []
        bus.subscribe(REWARD_COMPUTED, lambda e: events.append(e))

        reward = GeneralQNAReward(event_bus=bus)
        reward.compute(["Q"], ["<think>t</think>ans"], ["ans"])

        self.assertEqual(len(events), 1)
        self.assertIn("mean_score", events[0].data)

    def test_reward_publishes_invalid_event(self):
        bus = EventBus()
        events = []
        bus.subscribe(REWARD_INVALID, lambda e: events.append(e))

        reward = GeneralQNAReward(event_bus=bus)
        reward.compute(["Q"], [""], ["ans"])  # Empty = invalid

        self.assertEqual(len(events), 1)


# =============================================================================
# 13. ThinkingBasedGenerator Tests
# =============================================================================

class TestThinkingBasedGenerator(unittest.TestCase):
    """Tests for the ThinkingBasedGenerator intermediate class."""

    def setUp(self):
        from mlx_grpo.trainer.type_system_v2.generators.thinking_based import (
            ThinkingBasedGenerator,
        )
        # Use GeneralQNA as concrete subclass
        self.gen = GeneralQNARolloutGenerator()
        self.mcq_gen = MCQRolloutGenerator()
        self.ThinkingBased = ThinkingBasedGenerator

    def test_inheritance(self):
        """MCQ and GeneralQNA should extend ThinkingBasedGenerator."""
        self.assertIsInstance(self.gen, self.ThinkingBased)
        self.assertIsInstance(self.mcq_gen, self.ThinkingBased)

    def test_shared_curriculum(self):
        """Both generators should share curriculum logic."""
        answer = "<think>step one step two step three step four</think>42"
        gen_scaffold = self.gen.apply_curriculum(answer, ratio=0.5)
        mcq_scaffold = self.mcq_gen.apply_curriculum(answer, ratio=0.5)
        # Same algorithm, same result
        self.assertEqual(gen_scaffold, mcq_scaffold)

    def test_shared_completeness_check(self):
        """Both generators should share completeness checking."""
        text_complete = "<think>thinking</think>\nThe answer is 42"
        text_incomplete = "<think>thinking..."

        gen_ok, _ = self.gen.is_generation_complete(text_complete, phase=1)
        mcq_ok, _ = self.mcq_gen.is_generation_complete(text_complete, phase=1)
        self.assertTrue(gen_ok)
        self.assertTrue(mcq_ok)

        gen_nok, _ = self.gen.is_generation_complete(text_incomplete, phase=1)
        mcq_nok, _ = self.mcq_gen.is_generation_complete(text_incomplete, phase=1)
        self.assertFalse(gen_nok)
        self.assertFalse(mcq_nok)

    def test_needs_phase_recovery(self):
        """Both thinking-based generators need phase recovery."""
        self.assertTrue(self.gen.needs_phase_recovery())
        self.assertTrue(self.mcq_gen.needs_phase_recovery())


# =============================================================================
# 14. Phase Recovery Tests (check_incomplete)
# =============================================================================

class TestCheckIncomplete(unittest.TestCase):
    """Tests for type-dispatched phase recovery via check_incomplete()."""

    def setUp(self):
        from mlx_grpo.trainer.type_system_v2.generators.tool_call import (
            ToolCallRolloutGenerator,
        )
        self.gen = GeneralQNARolloutGenerator()
        self.mcq_gen = MCQRolloutGenerator(exam_phase_recovery_ratio=1.0)
        self.tc_gen = ToolCallRolloutGenerator()
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

    def test_tool_call_no_recovery(self):
        """Tool calls should never trigger phase recovery."""
        self.assertFalse(self.tc_gen.needs_phase_recovery())
        is_incomplete, prefix, count = self.tc_gen.check_incomplete(
            text="add(a=1, b=2)",
            scaffold_ratio=0.0,
            target="add(a=1, b=2)",
            type_info=None,
            tokenizer=self.tokenizer,
        )
        self.assertFalse(is_incomplete)

    def test_general_complete_no_recovery(self):
        """Complete thinking output should not trigger recovery."""
        text = "<think>Let me solve this. 2+2=4</think>\n\\boxed{4}"
        is_incomplete, prefix, count = self.gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target="<think>2+2=4</think>\\boxed{4}",
            type_info=None,
            tokenizer=self.tokenizer,
        )
        self.assertFalse(is_incomplete)

    def test_general_missing_think_end_triggers_recovery(self):
        """Missing </think> should trigger recovery."""
        text = "<think>Let me think about this for a while..."
        is_incomplete, prefix, count = self.gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target="<think>thinking</think>\\boxed{42}",
            type_info=None,
            tokenizer=self.tokenizer,
        )
        self.assertTrue(is_incomplete)
        self.assertIn("</think>", prefix)
        self.assertIn("\\boxed{", prefix)

    def test_general_think_end_no_answer_triggers_recovery(self):
        """Has </think> but no answer should trigger recovery."""
        text = "<think>thinking content here</think>"
        is_incomplete, prefix, count = self.gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target=None,
            type_info=None,
            tokenizer=self.tokenizer,
        )
        self.assertTrue(is_incomplete)
        self.assertIn("\\boxed{", prefix)

    def test_mcq_exam_recovery_missing_boxed(self):
        """MCQ: has </think> but no \\boxed{ should inject it."""
        text = "<think>Analyzing options A through D...</think>\nI think the answer is B"
        is_incomplete, prefix, count = self.mcq_gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target="B",
            type_info={"is_exam": True},
            tokenizer=self.tokenizer,
        )
        self.assertTrue(is_incomplete)
        self.assertIn("\\boxed{", prefix)

    def test_mcq_exam_recovery_missing_think_end(self):
        """MCQ: has <think> but no </think> should truncate and inject."""
        text = "<think>Let me analyze this exam question very carefully..."
        is_incomplete, prefix, count = self.mcq_gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target="B",
            type_info={"is_exam": True},
            tokenizer=self.tokenizer,
        )
        self.assertTrue(is_incomplete)
        self.assertIn("</think>", prefix)
        self.assertIn("\\boxed{", prefix)

    def test_mcq_complete_with_boxed(self):
        """MCQ: complete with both </think> and \\boxed{ should not recover."""
        text = "<think>Analysis done</think>\n\\boxed{B}"
        is_incomplete, prefix, count = self.mcq_gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target="B",
            type_info={"is_exam": True},
            tokenizer=self.tokenizer,
        )
        self.assertFalse(is_incomplete)

    def test_scaffold_token_count_returned(self):
        """Injected tokens should be counted for loss masking."""
        text = "<think>thinking..."
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        is_incomplete, prefix, count = self.gen.check_incomplete(
            text=text,
            scaffold_ratio=0.0,
            target=None,
            type_info=None,
            tokenizer=self.tokenizer,
        )
        self.assertTrue(is_incomplete)
        self.assertEqual(count, 5)  # Mock returns 5 tokens

    def test_full_scaffold_skips_recovery(self):
        """Full scaffold (ratio=1.0) should skip recovery."""
        text = "<think>thinking..."
        is_incomplete, prefix, count = self.gen.check_incomplete(
            text=text,
            scaffold_ratio=1.0,
            target=None,
            type_info=None,
            tokenizer=self.tokenizer,
        )
        self.assertFalse(is_incomplete)


# =============================================================================
# 15. V2 Adapter Marker Tests
# =============================================================================

class TestV2AdapterMarker(unittest.TestCase):
    """Test that v2 adapter functions are properly marked."""

    def test_adapter_has_marker(self):
        coordinator = TypeCoordinator()
        coordinator.register_type(
            "general_qna",
            GeneralQNAReward(),
            MagicMock(), MagicMock(),
            set_as_default=True,
        )
        adapter = v2_reward_adapter(coordinator)
        self.assertTrue(getattr(adapter, "_is_v2_adapter", False))
        self.assertEqual(adapter.__name__, "v2_type_dispatched_reward")


if __name__ == "__main__":
    unittest.main()
