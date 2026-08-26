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
- Vendored and adapted at
  `src/recirculation/author_recurrent_gemma3.py`; adaptations are marked
  `REPRODUCTION`.

The paper does not formally link the gist. It is treated as author-provided
reference code, not an arXiv ancillary artifact. The gist does not state a
license; this repository is a local research reproduction and makes no broader
redistribution claim.

## Model and data

- Model: <https://huggingface.co/google/gemma-3-1b-pt>, revision
  `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`
- Weight-file SHA-256:
  `ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27`
- Dataset mirror: <https://huggingface.co/datasets/emozilla/pg19-test>, revision
  `c5e39bf32e33f9111323aa68d7d9000d22722035`
- Original PG-19 project: <https://github.com/deepmind/pg19>
