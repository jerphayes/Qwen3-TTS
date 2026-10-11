"""Check sampler distributions against pinned Transformers, with an optional CUDA capture check."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen_tts" / "inference" / "cuda_graph.py"
SPEC = importlib.util.spec_from_file_location("cuda_graph_sampling_review", MODULE_PATH)
CUDA_GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUDA_GRAPH)


def hf_probabilities(logits, temperature=1.0, top_k=0, top_p=1.0, suppress_mask=None):
    scores = logits.float().clone()
    if suppress_mask is not None:
        scores[..., suppress_mask] = float("-inf")
    ids = torch.empty(logits.shape[0], 0, dtype=torch.long, device=logits.device)
    scores = TemperatureLogitsWarper(temperature)(ids, scores)
    if top_k > 0:
        scores = TopKLogitsWarper(top_k)(ids, scores)
    if top_p < 1.0:
        scores = TopPLogitsWarper(top_p)(ids, scores)
    return scores.softmax(dim=-1)


class SamplingTests(unittest.TestCase):
    def assert_distribution_matches(self, logits, **sampling):
        expected = hf_probabilities(logits, **sampling)
        # Inspect the actual multinomial input, avoiding seed/graph RNG differences.
        with patch.object(torch, "multinomial", return_value=torch.zeros(logits.shape[0], 1, dtype=torch.long)) as draw:
            CUDA_GRAPH.sample_logits(logits, do_sample=True, **sampling)
        actual = draw.call_args.args[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)

    def test_threshold_crossing_retains_boundary_token(self):
        logits = torch.tensor([[0.6, 0.3, 0.1]]).log()
        self.assert_distribution_matches(logits, temperature=1.0, top_k=0, top_p=0.7)
        expected = hf_probabilities(logits, top_p=0.7)
        torch.testing.assert_close(expected, torch.tensor([[2 / 3, 1 / 3, 0.0]]))

    def test_ties_extreme_logits_and_thresholds(self):
        logits = torch.tensor([[0., 0., 0., 0.], [1000., 0., -1000., -2000.],
                               [4., 3., 2., 1.], [0., float("-inf"), -1., -2.]])
        for top_p in (0.01, 0.25, 0.5, 0.7, 0.9, 1.0):
            with self.subTest(top_p=top_p):
                self.assert_distribution_matches(logits, temperature=1.0, top_k=0, top_p=top_p)

    def test_batched_warper_combinations_and_low_precision(self):
        logits = torch.randn(4, 64, generator=torch.Generator().manual_seed(123))
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for top_k in (0, 1, 7, 100):
                for top_p in (0.1, 0.7, 1.0):
                    with self.subTest(dtype=dtype, top_k=top_k, top_p=top_p):
                        self.assert_distribution_matches(logits.to(dtype), temperature=0.9,
                                                         top_k=top_k, top_p=top_p)

    def test_suppression_precedes_warpers(self):
        logits = torch.tensor([[6., 5., 4., 3.], [3., 4., 5., 6.]])
        mask = torch.tensor([True, False, False, False])
        self.assert_distribution_matches(logits, temperature=0.7, top_k=2, top_p=0.8, suppress_mask=mask)

    def test_greedy_selection_is_unchanged(self):
        logits = torch.tensor([[3., 2., 1.], [1., 3., 2.]], dtype=torch.bfloat16)
        original = logits.clone()
        result = CUDA_GRAPH.sample_logits(logits, temperature=0.9, top_k=1, top_p=0.1,
                                          do_sample=False, suppress_mask=torch.tensor([False, True, False]))
        torch.testing.assert_close(result, torch.tensor([0, 2]))
        torch.testing.assert_close(logits, original)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_sampling_captures_and_replays_on_cuda(self):
        logits = torch.tensor([[3., 2., 1., 0.], [0., 1., 2., 3.]], device="cuda", dtype=torch.bfloat16)
        sampling = dict(temperature=0.9, top_k=3, top_p=0.7, do_sample=True)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                CUDA_GRAPH.sample_logits(logits, **sampling)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            output = CUDA_GRAPH.sample_logits(logits, **sampling)
        support = hf_probabilities(logits, temperature=0.9, top_k=3, top_p=0.7) > 0
        for _ in range(8):
            graph.replay()
            self.assertEqual(tuple(output.shape), (2,))
            self.assertTrue(bool(support.gather(1, output[:, None]).all()))


if __name__ == "__main__":
    unittest.main()
