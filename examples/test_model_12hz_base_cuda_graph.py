# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Voice clone with use_cuda_graph=True: check that it matches the default decode, then time both.

1. Greedy decoding, same batch, bounded token budget: compare audio with and without CUDA graphs.
2. Default sampling, a batch of BATCH texts: wall time and real-time factor of both paths.
"""
import os
import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base/"
OUT_DIR = "qwen3_tts_test_cuda_graph_output_wav"
BATCH = 4  # validated with the checkpoint's original token budget; increase only if VRAM permits
PARITY_MAX_NEW_TOKENS = 256  # bound greedy checks that reach the token limit; timing keeps original defaults

REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
TEXTS = [
    "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "The meeting has been moved to Thursday afternoon, so please bring the printed slides.",
    "雨下了一整夜，街道上的积水映着昏黄的路灯。",
]
LANGUAGES = ["English", "Chinese", "English", "Chinese"]


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cuda:0", dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    prompt = tts.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)

    # 1. parity under greedy decoding
    greedy = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=PARITY_MAX_NEW_TOKENS)
    base, _ = tts.generate_voice_clone(text=TEXTS, language=LANGUAGES, voice_clone_prompt=prompt, **greedy)
    graph, _ = tts.generate_voice_clone(text=TEXTS, language=LANGUAGES, voice_clone_prompt=prompt,
                                        use_cuda_graph=True, **greedy)
    if tts._cuda_graph_engine.last.get("path") != "cuda_graph":
        raise RuntimeError("Parity check used eager fallback rather than CUDA graphs.")
    same = all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(base, graph))
    print(f"[parity] greedy, batch {len(TEXTS)}, token budget {PARITY_MAX_NEW_TOKENS}: "
          f"identical audio with and without CUDA graphs: {same}")
    if not same:
        raise AssertionError("Greedy waveform parity failed; skip speed comparisons until it is resolved.")

    # 2. speed under default sampling
    texts = (TEXTS * BATCH)[:BATCH]
    languages = (LANGUAGES * BATCH)[:BATCH]
    for use_cuda_graph in (False, True):
        tts.generate_voice_clone(text=texts, language=languages, voice_clone_prompt=prompt,
                                 use_cuda_graph=use_cuda_graph)  # warmup with the same budget/cache bucket
        (wavs, sr), seconds = timed(lambda: tts.generate_voice_clone(
            text=texts, language=languages, voice_clone_prompt=prompt, use_cuda_graph=use_cuda_graph))
        path = tts._cuda_graph_engine.last.get("path") if use_cuda_graph else "eager"
        audio = sum(len(w) for w in wavs) / sr
        print(f"[speed] use_cuda_graph={use_cuda_graph}, actual_path={path}: batch {BATCH}, {seconds:.2f}s for {audio:.1f}s audio "
              f"({audio / seconds:.1f}x real time)")
        for i, w in enumerate(wavs[:4]):
            sf.write(os.path.join(OUT_DIR, f"cuda_graph_{use_cuda_graph}_{i}.wav"), w, sr)


if __name__ == "__main__":
    main()
