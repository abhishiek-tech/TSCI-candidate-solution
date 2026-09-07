# NOP Vision Intelligence Challenge — Candidate Solution

Video-intelligence pipeline for the NOP AI Developer Challenge 2026:
detect → track → zone/dwell events → structured evidence.

The detector is the distinguishing piece. Rather than detecting a generic
foreground blob and then guessing which object it belongs to, it learns each
object's stable appearance signature on the fly and treats every signature as
its own detection channel — so two overlapping objects stay separable instead
of becoming one ambiguous blob. See REPORT.md for why, and for the assumption
this rests on.

## Tested environment

- **OS**: Windows 11 (10.0.26200), developed and tested via Git Bash
- **Language/runtime**: Python 3.10.11
- **Hardware**: CPU only — no GPU used or required

## Setup

```bash
cd TSCI-candidate-solution
python -m venv .venv
# Windows (Git Bash):     source .venv/Scripts/activate
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Linux/macOS:            source .venv/bin/activate

pip install -r requirements.txt
```

Three dependencies, all permissively licensed: OpenCV, NumPy, SciPy. No model
weights to download, no network access needed at any point.

## Input format

MP4, MKV, AVI — anything OpenCV's `VideoCapture` opens, at any resolution or
frame rate. Zone coordinates in a config are pixel coordinates for that
camera's view; see "Using this pipeline on your own footage" below.

## Run

Generate the public benchmark clips from the challenge repository (or point
`--input` at your own video):

```bash
python tools/generate_synthetic_dataset.py --all --output-dir data/generated
```

Run one scenario:

```bash
cd src
python main.py \
  --input D:/TSCI/data/generated/S01_BASIC_GOODS.mp4 \
  --output-dir ../output/S01 \
  --config ../config/scenario_S01.json
```

Reproduce all four public benchmark results:

```bash
cd src
for s in S01_BASIC_GOODS S02_OCCLUSION_REVERSAL S03_DENSE_CROSSING S04_DWELL_QUEUE; do
  id=$(echo $s | cut -d_ -f1)
  python main.py --input "D:/TSCI/data/generated/${s}.mp4" \
                 --output-dir "../output/${id}" \
                 --config "../config/scenario_${id}.json"
done
```

Then score them with the challenge's own evaluator:

```bash
cd ..
{ head -1 output/S01/events.csv; for d in S01 S02 S03 S04; do tail -n +2 output/$d/events.csv; done; } > output/combined_events.csv
python D:/TSCI/tools/evaluate_events.py \
  --candidate output/combined_events.csv \
  --truth D:/TSCI/data/ground_truth/public_event_truth.csv
```

Optional flags: `--display` for a live preview window, `--max-frames N` to
stop early while debugging, `--scenario-id ID` to override the evidence
`scenario_id`.

## Configuration

One JSON file per camera view. `config/default.json` is a generic two-zone
template to copy; `config/scenario_S01.json` … `scenario_S04.json` cover the
four public benchmarks.

| Key | Meaning |
|---|---|
| `zones` | Named regions, as `[x1, y1, x2, y2]` or an explicit polygon |
| `detector_params.signature_match_threshold` | How close two colours must be to count as the same object (0–1) |
| `detector_params.signature_confirm_hits` | Observations required before a new object identity is registered |
| `detector_params.signature_min_dominant_fraction` | How uniform a blob must be to be trusted as a single object |
| `tracker.max_missed_frames` | How long a track survives while unmatched (occlusion tolerance) |
| `zone_flow.stability_frames` | Frames a zone must be held before a crossing counts |
| `zone_flow.cooldown_seconds` | Suppression window for duplicate counts of one crossing |
| `dwell_queue` | Optional: queue zone name, dwell threshold, occupancy threshold |

## Outputs (per run, into `--output-dir`)

- `annotated.mp4` — zones, boxes, track ids, dwell timers, occupancy HUD
- `events.jsonl` / `events.json` — one structured evidence record per event
- `events.csv` — flat per-event rows for independent evaluation
- `counts.csv` — aggregate counts per event type
- `queue_occupancy.csv` — occupancy time series (when `dwell_queue` is configured)
- `evidence/*.jpg` — a snapshot frame per event
- `performance.json` — frames, wall-clock, processing FPS, identities learned

`output/S01` … `output/S04` in this repository hold one full run per public
scenario, plus `output/combined_events.csv` for the evaluator.

## Results on the public benchmark

Measured with the challenge's own `tools/evaluate_events.py`:

| Scenario | Expected | Matched | Identities learned / real objects |
|---|---:|---:|---|
| S01_BASIC_GOODS | 3 | **3** | 3 / 3 |
| S02_OCCLUSION_REVERSAL | 3 | **3** | 3 / 3 |
| S03_DENSE_CROSSING | 5 | **5** | 5 / 5 |
| S04_DWELL_QUEUE | 3 | **3** | 3 / 3 |

**Route events: 14 matched, 0 missed, 0 false positives — precision 1.00,
recall 1.00, F1 1.00.**

Scoring the full event stream reports 5 additional "false positives"; these
are the `DWELL_THRESHOLD` and `QUEUE_THRESHOLD` extension events, which the
base route ground truth does not model at all (it knows only `A_TO_B` and
`B_TO_A`). They are not incorrect route counts. See REPORT.md.

## Tests

```bash
python -m unittest discover -s tests -v
```

38 unit tests covering appearance-signature learning, per-signature blob
splitting, tracker identity and occlusion behaviour, zone-transition logic,
dwell/occupancy analytics, and evidence-output integrity. No video files
required.

## Using this pipeline on your own footage

```bash
cd src
python main.py --input your_video.mp4 --output-dir ../output/real \
  --config ../config/default.json
```

Edit that config's `zones` to your camera's pixel geometry, and retune
`detector_params` (`min_area`, `var_threshold`, `learning_rate`) for your
scene's scale and lighting.

Read the "known limitations" below first — the detector's strongest mechanism
assumes objects have distinguishable appearance signatures, which holds well
for uniformly-coloured objects and weakly for crowds.

## Known limitations

- **Appearance separation assumes distinguishable objects.** Identity is
  anchored to each object's colour signature. Where objects genuinely share a
  colour — a crowd in similar clothing, a car park of grey cars — the detector
  falls back to geometric blob splitting, which is the conventional approach
  and no better than a standard pipeline at a dense crossing. REPORT.md states
  this trade-off in full.
- **Class-agnostic.** Every object is reported as `moving_object`, not a
  semantic label like `person` or `car`. It answers "how many crossed, when,
  and here is the proof", not "what kind of thing was it".
- **Assumes a mostly-static camera.** Background subtraction underpins
  detection, so a panning or shaking camera needs stabilisation first.
- **Zone geometry is per-camera.** Zones are pixel coordinates; a new camera
  or resolution needs its own config.
- **Detector parameters are tuned for the benchmark's scale and contrast**
  and would need re-tuning for a very different scene.
