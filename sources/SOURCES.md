# Primary sources and pinned artifacts

## Paper

- Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer, Sunny Sanyal, and
  Rosanne Liu, *Recirculation*, arXiv:2608.17981v1, 18 August 2026.
- PDF: <https://arxiv.org/pdf/2608.17981>
- Downloaded source SHA-256:
  `0a50d36ca19425e82d50417e2fa942ed047c7fec3485d09259e7ee2093db3c34`
- The downloaded PDF is intentionally gitignored.

Relevant locations: Equations 1–2 (PDF p. 4), Sections 4.1–4.3 and Table 1
(pp. 6–8), Appendix B.1–B.3 and Tables B.1–B.2 (pp. 24–27).

## Author reference implementation

- Shoaib Ahmed Siddiqui, `recurrent_gemma3.py`:
  <https://gist.github.com/shoaibahmed/10702acc01cc5a169fdbc1719932438f>
- Sole gist commit:
  `d88d797f9ae0e88073ce219a41887c48408e5bf4` (24 January 2026)
- Unmodified upstream raw-file SHA-256:
  `c21ea064c29c04c8c4cf93c91f134b7719dbbdb9960759c191790c41850990b1`
- The paper does not formally link the gist. It is treated as author-provided
  reference code, not an arXiv ancillary artifact.
- The gist does not state a license. It was consulted for behavioral semantics,
  but no gist source is included in the public distribution.

## Public implementation provenance

- `src/recirculation/gemma3_recirculation.py` is a compact implementation of
  the published method over the official Transformers Gemma 3 classes.
- It subclasses and follows layer/attention structure from Hugging Face
  Transformers 5.0.0, licensed Apache-2.0, and retains the relevant Google and
  Hugging Face copyright notice.
- The public implementation was compared before publication with the local
  reference-derived implementation used for the experiments. On the pinned
  environment, 64-token logits and a 1,024-token per-target NLL vector were
  bitwise identical. See `results/implementation_equivalence.json`.
- Repository-authored code and documentation are released under Apache-2.0.
  Model and data assets retain their separate terms.

## Model and data

- Model: <https://huggingface.co/google/gemma-3-1b-pt>, revision
  `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`
- Weight-file SHA-256:
  `ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27`
- Dataset mirror: <https://huggingface.co/datasets/emozilla/pg19-test>, revision
  `c5e39bf32e33f9111323aa68d7d9000d22722035`
- Original PG-19 project: <https://github.com/deepmind/pg19>

No model weights, Hugging Face caches, or full PG-19 dataset files are committed.
The Gemma checkpoint is governed by Google's Gemma terms. The original PG-19
benchmark repository declares Apache-2.0; the pinned mirror declares no separate
license metadata. See `THIRD_PARTY_NOTICES.md` for the release boundary.

## Downstream capability milestone

- Instruction-tuned model:
  <https://huggingface.co/google/gemma-3-1b-it>, revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Weight-file SHA-256:
  `3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6`
- EleutherAI `lm-evaluation-harness`:
  <https://github.com/EleutherAI/lm-evaluation-harness>, commit
  `dd417662e5bda6a247489ec28d2ff46a45d1c42c`
- MMLU-Pro: <https://github.com/TIGER-AI-Lab/MMLU-Pro>
- GSM8K: <https://github.com/openai/grade-school-math>
- IFEval:
  <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
- HellaSwag: <https://github.com/rowanz/hellaswag>

Exact task versions, Hub dataset commits, selected document identifiers/hashes,
chat-template hash, seeds, and canonical harness settings are frozen in
`experiments/capability_locked_manifest.json`. The five-example-per-task smoke
artifacts are plumbing evidence only and are kept separate under
`results/capability_smoke/`.
