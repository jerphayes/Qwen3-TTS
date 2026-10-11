"""Exercise the real decode loop with deterministic CPU doubles for model kernels and CUDA timing."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen_tts" / "inference" / "cuda_graph.py"
SPEC = importlib.util.spec_from_file_location("cuda_graph_budget_review", MODULE_PATH)
CUDA_GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUDA_GRAPH)
EOS = 7


class DecodeFixture:
    def __init__(self, schedule):
        self.schedule = torch.tensor(schedule)
        rows = self.schedule.shape[0]
        self.position = 0
        self.decode_positions = []
        embedding = torch.nn.Embedding(8, 4)
        with torch.no_grad():
            embedding.weight.copy_(torch.arange(8)[:, None].expand(-1, 4))

        def logits(hidden):
            result = torch.full((rows, 8), -100.)
            index = min(self.position, self.schedule.shape[1] - 1)
            result.scatter_(1, self.schedule[:, index:index + 1], 100.)
            return result

        def prefill(**kwargs):
            return SimpleNamespace(logits=logits(None)[:, None], past_key_values=None,
                                   past_hidden=torch.zeros(rows, 1, 4))

        def decode(embeds, position):
            self.decode_positions.append(position)
            self.position += 1
            return torch.zeros(rows, 1, 4)

        # Second-codebook values depend on the actual selected first-codebook token.
        self.predictor = SimpleNamespace(run=Mock(side_effect=lambda x: (x[:, -1, :1].long() + 1) % 7))
        self.graph = SimpleNamespace(prefill=Mock(return_value=5), run=Mock(side_effect=decode), max_seq_len=1024)
        self.engine = object.__new__(CUDA_GRAPH.BatchedEngine)
        self.engine.config = SimpleNamespace(vocab_size=8, num_code_groups=2, codec_eos_token_id=EOS)
        self.engine.device = torch.device("cpu")
        self.engine.talker = SimpleNamespace(
            forward=prefill, codec_head=logits, get_input_embeddings=lambda: embedding,
            code_predictor=SimpleNamespace(get_input_embeddings=lambda: [embedding]), rope_deltas=None)
        self.engine.graphs = Mock(return_value=(rows, self.predictor, self.graph))
        self.kwargs = dict(inputs_embeds=torch.zeros(rows, 5, 4), attention_mask=torch.ones(rows, 5),
                           trailing_text_hidden=torch.zeros(rows, 16, 4), tts_pad_embed=torch.zeros(1, 1, 4),
                           do_sample=False, subtalker_dosample=False, repetition_penalty=1., min_new_tokens=0)

    def generate(self, budget):
        event = SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 0.)
        with patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "Event", return_value=event):
            result = self.engine._generate(**self.kwargs, max_new_tokens=budget)
        return torch.stack([entry[-1] for entry in result.hidden_states if entry[-1] is not None], dim=1)


def trim_eos(codes):
    # This is the public wrapper's contract: exclude EOS and everything following it per row.
    result = []
    for row in codes:
        stops = (row[:, 0] == EOS).nonzero()
        length = int(stops[0, 0]) if len(stops) else len(row)
        result.append(row[:length])
    return result


class TokenBudgetTests(unittest.TestCase):
    def test_no_eos_returns_only_forwarded_tokens(self):
        for budget in (2, 5, 8):
            with self.subTest(budget=budget):
                fixture = DecodeFixture([[1, 2, 3, 4, 5, 6, 1, 2]])
                codes = fixture.generate(budget)
                self.assertEqual(tuple(codes.shape), (1, budget - 1, 2))
                torch.testing.assert_close(codes[0, :, 0], fixture.schedule[0, :budget - 1])
                torch.testing.assert_close(codes[0, :, 1], (fixture.schedule[0, :budget - 1] + 1) % 7)
                self.assertEqual(fixture.predictor.run.call_count, budget - 1)
                self.assertEqual(fixture.decode_positions, list(range(5, 5 + budget - 2)))
                self.assertEqual(fixture.engine.graphs.call_args.args[2], 5 + budget - 1)

    def test_final_selected_eos_has_no_codec_record(self):
        fixture = DecodeFixture([[1, 2, 3, 4, EOS]])
        codes = fixture.generate(5)
        torch.testing.assert_close(trim_eos(codes)[0][:, 0], torch.tensor([1, 2, 3, 4]))
        self.assertEqual(codes.shape[1], 4)

    def test_mixed_completion_and_minimum_token_suppression(self):
        fixture = DecodeFixture([[1, 2, EOS, 5, 6], [3, 4, 5, 6, 1]])
        fixture.kwargs["min_new_tokens"] = 2
        codes = fixture.generate(5)
        rows = trim_eos(codes)
        torch.testing.assert_close(rows[0][:, 0], torch.tensor([1, 2]))
        torch.testing.assert_close(rows[1][:, 0], torch.tensor([3, 4, 5, 6]))
        self.assertTrue(bool((codes[0, 2:, 0] == EOS).all()))

    def test_all_finished_rows_stop_at_periodic_check(self):
        fixture = DecodeFixture([[1, 2, EOS], [3, 4, EOS]])
        codes = fixture.generate(20)
        self.assertEqual(codes.shape[1], CUDA_GRAPH.EOS_CHECK_EVERY)
        rows = trim_eos(codes)
        torch.testing.assert_close(rows[0][:, 0], torch.tensor([1, 2]))
        torch.testing.assert_close(rows[1][:, 0], torch.tensor([3, 4]))


if __name__ == "__main__":
    unittest.main()
