"""CPU-only regression checks for graph capacity routing; no checkpoint or CUDA required."""
import importlib.util
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen_tts" / "inference" / "cuda_graph.py"
SPEC = importlib.util.spec_from_file_location("cuda_graph_capacity_review", MODULE_PATH)
CUDA_GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUDA_GRAPH)


class CapacityRoutingTests(unittest.TestCase):
    def setUp(self):
        # Bypass model construction; only the real public routing method is exercised.
        self.engine = object.__new__(CUDA_GRAPH.BatchedEngine)
        self.engine.max_seq_len = 1024
        self.engine.eager = Mock(return_value="eager")
        self.engine._generate = Mock(return_value="graph")

    def request(self, prompt_length, **kwargs):
        return dict(inputs_embeds=SimpleNamespace(shape=(3, prompt_length, 64)), **kwargs)

    def assert_graph(self, request):
        self.assertEqual(self.engine.generate(**request), "graph")
        self.engine._generate.assert_called_once_with(**request)
        self.engine.eager.assert_not_called()

    def test_default_wrapper_budget_does_not_truncate(self):
        request = self.request(400, max_new_tokens=2048, do_sample=False,
                               trailing_text_hidden=object(), attention_mask=object())
        self.assert_graph(request)

    def test_missing_budget_keeps_engine_default(self):
        self.assert_graph(self.request(100))

    def test_one_token_over_old_capacity_still_uses_graph(self):
        self.assert_graph(self.request(400, max_new_tokens=625))

    def test_exact_capacity_uses_graph(self):
        request = self.request(400, max_new_tokens=624)
        self.assertEqual(self.engine.generate(**request), "graph")
        self.engine._generate.assert_called_once_with(**request)
        self.engine.eager.assert_not_called()

    def test_short_budget_uses_graph(self):
        request = self.request(400, max_new_tokens=8)
        self.assertEqual(self.engine.generate(**request), "graph")
        self.engine.eager.assert_not_called()

    def test_long_prompt_still_uses_graph(self):
        self.assert_graph(self.request(513, max_new_tokens=8))

    def test_prompt_exceeding_old_cache_still_uses_graph(self):
        self.assert_graph(self.request(1100, max_new_tokens=8))

    def test_single_token_and_invalid_budgets_delegate_to_eager(self):
        for budget in (1, 0, -1):
            with self.subTest(budget=budget):
                self.engine.eager.reset_mock()
                self.engine._generate.reset_mock()
                request = self.request(100, max_new_tokens=budget)
                self.assertEqual(self.engine.generate(**request), "eager")
                self.engine.eager.assert_called_once_with(**request)
                self.engine._generate.assert_not_called()
                self.assertEqual(self.engine.last["reason"], "token_budget")


class GraphBucketTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(CUDA_GRAPH.BatchedEngine)
        self.engine.max_seq_len = 1024
        self.engine.max_resident = 2
        self.engine.resident = OrderedDict()
        self.engine.talker = SimpleNamespace(model=object(), code_predictor=object())
        self.engine.config = SimpleNamespace(hidden_size=64, num_code_groups=2)
        self.engine.device = "cuda"
        self.engine.dtype = "bfloat16"
        self.sampling = dict(do_sample=False, top_k=50, top_p=1., temperature=.9)
        self.talker_constructor = Mock(side_effect=lambda *args: SimpleNamespace(max_seq_len=args[3]))
        self.predictor_constructor = Mock(side_effect=lambda *args: object())
        for target, replacement in (("BatchedTalkerGraph", self.talker_constructor),
                                    ("BatchedPredictorGraph", self.predictor_constructor)):
            context = patch.object(CUDA_GRAPH, target, replacement)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(CUDA_GRAPH.torch.cuda, "empty_cache")
        context.start()
        self.addCleanup(context.stop)

    def test_default_budget_allocates_sufficient_cache(self):
        batch, _, graph = self.engine.graphs(3, self.sampling, 400 + 2048 - 1)
        self.assertEqual(batch, 3)
        self.assertEqual(graph.max_seq_len, 3072)
        self.assertIn((3, 3072), self.engine.resident)

    def test_cache_bucket_boundary(self):
        self.engine.graphs(1, self.sampling, 1024)
        self.engine.graphs(1, self.sampling, 1025)
        self.assertEqual(list(self.engine.resident), [(1, 1024), (1, 2048)])

    def test_same_batch_smaller_budget_reuses_larger_cache(self):
        original = self.engine.graphs(3, self.sampling, 2500)
        reused = self.engine.graphs(3, self.sampling, 500)
        self.assertEqual(original, reused)
        self.talker_constructor.assert_called_once()
        self.predictor_constructor.assert_called_once()

    def test_smaller_batch_does_not_reuse_different_shape(self):
        original = self.engine.graphs(4, self.sampling, 2500)
        smaller = self.engine.graphs(3, self.sampling, 500)
        self.assertEqual(smaller[0], 3)
        self.assertIsNot(original[2], smaller[2])

    def test_larger_budget_cannot_reuse_short_cache(self):
        short = self.engine.graphs(4, self.sampling, 500)
        long = self.engine.graphs(3, self.sampling, 2500)
        self.assertIsNot(short[2], long[2])
        self.assertEqual(long[2].max_seq_len, 3072)

    def test_cache_eviction_tracks_batch_and_length(self):
        self.engine.graphs(1, self.sampling, 500)
        self.engine.graphs(1, self.sampling, 1500)
        self.engine.graphs(1, self.sampling, 500)  # touch the smaller bucket
        self.engine.graphs(1, self.sampling, 2500)
        self.assertEqual(list(self.engine.resident), [(1, 1024), (1, 3072)])

    def test_same_batch_larger_prompt_allocates_larger_cache(self):
        self.engine.graphs(3, self.sampling, 1100 + 2048 - 1)
        self.assertIn((3, 4096), self.engine.resident)


if __name__ == "__main__":
    unittest.main()
