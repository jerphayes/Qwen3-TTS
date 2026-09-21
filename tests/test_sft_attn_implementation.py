"""Regression tests: --attn_implementation in finetuning/sft_12hz.py.

Proves: (1) default stays flash_attention_2; (2) sdpa selectable;
(3) the selected value reaches Qwen3TTSModel.from_pretrained;
(4) parser stays backward compatible (the pre-patch argument set,
    whose real names are init_model_path, output_model_path,
    train_jsonl [required], batch_size, lr, num_epochs, speaker_name,
    still parses with no new flags).
GPU-free: from_pretrained is intercepted before any weight load.

Adaptation vs. the plan draft: upstream's parser marks --train_jsonl
as required=True, so this module always supplies a dummy
--train_jsonl /tmp/dummy.jsonl (the file is never opened because
from_pretrained raises _StopAfterLoad first).
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))
import sft_12hz  # noqa: E402

_DUMMY_TRAIN_JSONL = "/tmp/dummy.jsonl"


class _StopAfterLoad(Exception):
    pass


def _parse(argv):
    # --train_jsonl is required=True upstream; satisfy it for every parse.
    argv = ["--train_jsonl", _DUMMY_TRAIN_JSONL, *argv]
    with mock.patch.object(sys, "argv", ["sft_12hz.py", *argv]):
        return sft_12hz.build_arg_parser().parse_args()


class AttnImplementationTests(unittest.TestCase):
    def test_default_is_flash_attention_2(self):
        self.assertEqual(_parse([]).attn_implementation, "flash_attention_2")

    def test_explicit_sdpa_selectable(self):
        self.assertEqual(
            _parse(["--attn_implementation", "sdpa"]).attn_implementation, "sdpa"
        )

    def test_value_reaches_model_loader(self):
        captured = {}

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            raise _StopAfterLoad()

        with mock.patch.object(sft_12hz, "Accelerator"), \
                mock.patch.object(sft_12hz, "Qwen3TTSModel") as q:
            q.from_pretrained.side_effect = fake_from_pretrained
            with mock.patch.object(
                sys,
                "argv",
                ["sft_12hz.py", "--train_jsonl", _DUMMY_TRAIN_JSONL,
                 "--attn_implementation", "sdpa"],
            ):
                with self.assertRaises(_StopAfterLoad):
                    sft_12hz.train()
        self.assertEqual(captured["attn_implementation"], "sdpa")

    def test_default_value_reaches_model_loader(self):
        captured = {}

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            raise _StopAfterLoad()

        with mock.patch.object(sft_12hz, "Accelerator"), \
                mock.patch.object(sft_12hz, "Qwen3TTSModel") as q:
            q.from_pretrained.side_effect = fake_from_pretrained
            with mock.patch.object(
                sys, "argv", ["sft_12hz.py", "--train_jsonl", _DUMMY_TRAIN_JSONL]
            ):
                with self.assertRaises(_StopAfterLoad):
                    sft_12hz.train()
        self.assertEqual(captured["attn_implementation"], "flash_attention_2")

    def test_backward_compatible_parse(self):
        # Upstream's actual pre-patch argument set parses identically with
        # no new flags. train_jsonl (required) is supplied by _parse.
        args = _parse([
            "--init_model_path", "/tmp/init-model",
            "--output_model_path", "/tmp/out",
            "--batch_size", "4",
            "--lr", "1e-4",
            "--num_epochs", "1",
            "--speaker_name", "speaker_test",
        ])
        self.assertEqual(args.init_model_path, "/tmp/init-model")
        self.assertEqual(args.output_model_path, "/tmp/out")
        self.assertEqual(args.train_jsonl, _DUMMY_TRAIN_JSONL)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.num_epochs, 1)
        self.assertEqual(args.speaker_name, "speaker_test")


if __name__ == "__main__":
    unittest.main()
