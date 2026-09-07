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

## Optional review UI (`ui/`, not part of the graded pipeline)

| Component | Used for | License |
|---|---|---|
| [Gradio](https://www.gradio.app/) | local web interface | Apache-2.0 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | bundled ffmpeg binary, used only to transcode the annotated video into a browser-playable copy for preview | BSD-2-Clause for the wrapper; the bundled FFmpeg build carries its own LGPL/GPL terms and is invoked as an unmodified external executable, never linked into this codebase |

Neither is imported by `src/`. The graded pipeline runs with only OpenCV,
NumPy and SciPy installed, and the UI is entirely optional.

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
