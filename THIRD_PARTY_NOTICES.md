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
