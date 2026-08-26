# Third-party notices and asset boundaries

## Hugging Face Transformers

`src/recirculation/gemma3_recirculation.py` subclasses and calls the Gemma 3
implementation from Hugging Face Transformers 5.0.0. Its layer/attention
structure follows that upstream implementation. Transformers is Apache-2.0;
the upstream Google/Hugging Face copyright and change notice are retained in
the file and `NOTICE`.

- Repository: <https://github.com/huggingface/transformers>
- License: <https://github.com/huggingface/transformers/blob/main/LICENSE>

## Author reference implementation

Shoaib Ahmed Siddiqui's `recurrent_gemma3.py` gist was used as a behavioral
reference for recurrence, mask, and KV-cache semantics. The gist states that it
was adapted from Transformers but does not itself state a license. Therefore,
no gist source is included in the public tree. The compact public adapter was
written independently against the published algorithm and official
Transformers APIs.

Before removing the local reference-derived file, the public adapter was
regressed against it on the pinned Apple Silicon environment. First-64-token
logits and the full per-target NLL vector for a 1,024-token window were bitwise
identical. Hashes and exact checks are in
[`results/implementation_equivalence.json`](results/implementation_equivalence.json).

- Gist: <https://gist.github.com/shoaibahmed/10702acc01cc5a169fdbc1719932438f>
- Pinned gist commit: `d88d797f9ae0e88073ce219a41887c48408e5bf4`

## Gemma 3 weights

No model weights, tokenizer files, or Hugging Face cache contents are included.
Users download `google/gemma-3-1b-pt` directly after accepting Google's Gemma
terms. The repository's Apache-2.0 license does not relicense Gemma.

- Model page and terms gate: <https://huggingface.co/google/gemma-3-1b-pt>

## PG-19 data

The full dataset and parquet mirror are not included. Committed manifests
contain only the selected evaluation token IDs, record metadata, offsets, and
hashes needed to audit the locked comparisons. PG-19 is built from pre-1919
Project Gutenberg books; its original benchmark repository declares
Apache-2.0. The pinned Hugging Face mirror does not declare separate license
metadata.

- Original benchmark: <https://github.com/google-deepmind/pg19>
- Pinned mirror: <https://huggingface.co/datasets/emozilla/pg19-test>

## Dependencies

Runtime and development dependencies remain under their own licenses. Notably,
PyTorch is BSD-3-Clause, Transformers/Datasets/Hugging Face Hub are Apache-2.0,
and this repository does not redistribute those packages.

## Capability evaluation harness and benchmarks

The Gemma 3 1B IT capability milestone uses EleutherAI's
`lm-evaluation-harness` at commit
`dd417662e5bda6a247489ec28d2ff46a45d1c42c` as an installed dependency. The
harness is MIT-licensed; no harness source is copied into this repository.

The committed capability result artifacts contain the selected task documents,
canonical rendered prompts, targets, and model responses emitted by the harness.
They are retained for paired auditability under their upstream terms:

- MMLU-Pro: Apache-2.0, TIGER-AI-Lab. Dataset revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
  <https://github.com/TIGER-AI-Lab/MMLU-Pro>
- GSM8K: MIT, Copyright (c) 2021 OpenAI. Dataset revision
  `740312add88f781978c0658806c59bc2815b9866`.
  <https://github.com/openai/grade-school-math>
- IFEval: Apache-2.0, Google Research Authors. Dataset revision
  `966cd89545d6b6acfd7638bc708b98261ca58e84`.
  <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
- HellaSwag: MIT, Copyright (c) 2019 Rowan Zellers. Dataset revision
  `218ec52e09a7e7462a5400043bb9a69a41d06b76`.
  <https://github.com/rowanz/hellaswag>

The repository license applies only to repository-authored work and does not
relicense benchmark material embedded in the result records.

### MIT notices

Copyright (c) 2020 EleutherAI

Copyright (c) 2021 OpenAI

Copyright (c) 2019 Rowan Zellers

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notices and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
