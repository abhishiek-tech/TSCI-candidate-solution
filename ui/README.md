# Review UI (optional — not part of the graded pipeline)

A small local web interface over `src/main.py`: pick a video, run it, watch the
annotated result, inspect the events, and download everything as a zip.

It shells out to the same `main.py` the CLI uses, so results shown here are
identical to a command-line run by construction — no detection, tracking or
evidence logic is duplicated in the UI.

## Setup

```bash
pip install -r ui/requirements-ui.txt
```

## Run

```bash
python ui/app.py
```

Then open the printed `http://127.0.0.1:7860`.

## Notes

- **Zones are per-camera.** Each configuration carries pixel coordinates for
  one camera view. The `scenario_S01`–`S04` configs match the four benchmark
  clips; for your own footage start from `default.json` and edit its `zones`.
- **Each run is self-contained.** Output goes to
  `output/ui_runs/<timestamp>/`, with a matching `results_<timestamp>.zip`
  beside it, so repeated runs never overwrite each other. That directory is
  git-ignored — it is scratch space, not the benchmark results in `output/S0N`.
- **The preview video is a transcode.** `annotated.mp4` is written with the
  `mp4v` codec, which browsers cannot decode, so the UI transcodes an H.264
  copy purely for playback. The original is what ships in the download.
