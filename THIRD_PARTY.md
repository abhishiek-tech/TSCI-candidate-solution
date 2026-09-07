# Third-Party Components and Licenses

| Component | Used for | License | Commercial use |
|---|---|---|---|
| [OpenCV](https://opencv.org/) (`opencv-python`) | video I/O, background subtraction, morphology, contours, drawing | Apache-2.0 | Permitted |
| [NumPy](https://numpy.org/) | array and colour-distance maths | BSD-3-Clause | Permitted |
| [SciPy](https://scipy.org/) | `scipy.optimize.linear_sum_assignment` (Hungarian assignment) in the tracker's fallback path | BSD-3-Clause | Permitted |

All three are permissively licensed, with no source-disclosure or
copyleft obligations and no restrictions on commercial use.

**No pretrained models or model weights are used or redistributed.** Object
identity is derived at runtime from the footage itself, so there is nothing to
download and no AGPL/GPL-licensed model dependency to account for. The
pipeline runs entirely offline — no network access at install time beyond
`pip`, and none at all at run time.

## AI development assistance

| Tool | Use | Status |
|---|---|---|
| Claude Code (Anthropic) | Reading the challenge documents, writing source and tests, and running the instrumented measurements that drove the design decisions | Permitted by the challenge rules; disclosed here and in REPORT.md |

All design decisions and reported numbers were verified against actual runs of
the challenge's own `tools/evaluate_events.py`, not asserted from
documentation.

## Data

The four public benchmark clips are synthetic, generated locally by the
challenge repository's own `tools/generate_synthetic_dataset.py` from its
scenario definitions. No third-party video or image dataset is used,
redistributed, or required. No customer or personally identifying footage was
used at any point.
