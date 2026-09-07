"""Local review UI for the pipeline: upload footage, run it, inspect results.

Not part of the graded pipeline. This is a thin wrapper that shells out to
``src/main.py`` -- the same entry point the CLI uses -- so results shown here
are identical to a command-line run by construction, and no detection,
tracking or evidence logic is duplicated.

Run:  python ui/app.py     then open http://127.0.0.1:7860
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time

import gradio as gr
import imageio_ffmpeg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
RUNS_DIR = os.path.join(REPO_ROOT, "output", "ui_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

CONFIG_CHOICES = sorted(f for f in os.listdir(CONFIG_DIR) if f.endswith(".json"))
DEFAULT_CONFIG = "default.json" if "default.json" in CONFIG_CHOICES else CONFIG_CHOICES[0]

EMPTY_METRICS = "_Run a video to see results._"


def read_csv_rows(path, limit=500):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:limit + 1]


def transcode_for_browser(source_path, run_dir):
    """Produce an H.264 copy of the annotated video for in-browser playback.

    OpenCV writes ``annotated.mp4`` with the ``mp4v`` fourcc, which no browser
    can decode -- the file is perfectly valid for VLC, ffprobe or OpenCV, but a
    <video> element shows only "video not playable". This transcodes a copy for
    preview using the ffmpeg binary bundled with imageio-ffmpeg. The original is
    left untouched, since that is the artefact the pipeline actually produced.
    """
    if not os.path.exists(source_path):
        return None
    preview_path = os.path.join(run_dir, "annotated_preview.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", source_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-crf", "23", "-movflags", "+faststart", preview_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except Exception:
        return None
    if result.returncode != 0 or not os.path.exists(preview_path):
        return None
    return preview_path


def format_metrics(run_dir):
    performance_path = os.path.join(run_dir, "performance.json")
    if not os.path.exists(performance_path):
        return "_Run finished without writing performance.json._"
    with open(performance_path, "r", encoding="utf-8") as handle:
        performance = json.load(handle)

    _, count_rows = read_csv_rows(os.path.join(run_dir, "counts.csv"))
    total_events = sum(int(row[1]) for row in count_rows if len(row) > 1)
    breakdown = ", ".join(f"{row[0]} {row[1]}" for row in count_rows) or "no events"

    return (
        f"### {performance.get('scenario_id')}\n"
        f"| | |\n|---|---|\n"
        f"| **Events** | {total_events} — {breakdown} |\n"
        f"| **Identities learned** | {performance.get('appearance_signatures_learned')} |\n"
        f"| **Frames** | {performance.get('frames_processed')} |\n"
        f"| **Processing speed** | {performance.get('processing_fps')} fps "
        f"(source {performance.get('video_fps')} fps) |\n"
        f"| **Wall clock** | {performance.get('wall_clock_seconds')} s |\n"
    )


def run_pipeline(video_path, config_name, max_frames, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Choose a video file first.")
    if not config_name:
        raise gr.Error("Choose a configuration.")

    progress(0.05, desc="Preparing run")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(os.path.join(CONFIG_DIR, config_name), os.path.join(run_dir, "config.json"))

    command = [
        sys.executable, os.path.join(SRC_DIR, "main.py"),
        "--input", video_path,
        "--output-dir", run_dir,
        "--config", os.path.join(CONFIG_DIR, config_name),
    ]
    if max_frames and int(max_frames) > 0:
        command += ["--max-frames", str(int(max_frames))]

    progress(0.15, desc="Detecting, tracking, resolving zone events")
    result = subprocess.run(command, cwd=SRC_DIR, capture_output=True, text=True, timeout=3600)
    log = ((result.stdout or "") + (result.stderr or "")).strip()
    if result.returncode != 0:
        raise gr.Error(f"Pipeline failed (exit {result.returncode}).\n\n{log[-2000:]}")

    progress(0.8, desc="Preparing preview")
    annotated = os.path.join(run_dir, "annotated.mp4")
    preview = transcode_for_browser(annotated, run_dir)
    metrics = format_metrics(run_dir)
    if preview is None and os.path.exists(annotated):
        metrics += ("\n_Browser preview unavailable; `annotated.mp4` is in the "
                    "download and plays in any desktop player._")

    counts_header, counts_rows = read_csv_rows(os.path.join(run_dir, "counts.csv"))
    events_header, events_rows = read_csv_rows(os.path.join(run_dir, "events.csv"))

    progress(0.95, desc="Packaging results")
    archive = shutil.make_archive(os.path.join(RUNS_DIR, f"results_{run_id}"), "zip", run_dir)

    return (
        preview or (annotated if os.path.exists(annotated) else None),
        metrics,
        gr.Dataframe(headers=counts_header or ["event_type", "count"], value=counts_rows),
        gr.Dataframe(headers=events_header or ["scenario_id", "event_type", "track_id",
                                                "object_type", "confidence",
                                                "video_time_seconds"], value=events_rows),
        archive,
        log or "(no console output)",
    )


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    block_radius="*radius_lg",
    block_title_text_weight="600",
    # Soft tints every field label with the primary hue, which reads as a wall
    # of coloured pills once a form has more than a couple of fields. Labels are
    # flattened to a quiet neutral so the accent is spent only on the action.
    block_label_background_fill="*neutral_100",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_color="*neutral_600",
    block_label_text_color_dark="*neutral_300",
    block_label_border_width="0",
)

CSS = """
.header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 2px; }
.header h1 { font-size: 1.45rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
.header .tag {
    font-family: var(--font-mono); font-size: 0.74rem; letter-spacing: 0.02em;
    color: var(--body-text-color-subdued);
}
.section {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--body-text-color-subdued);
    margin: 6px 0 2px 2px;
}
#run-button { height: 44px; font-size: 1rem; font-weight: 600; }
footer { display: none !important; }
"""

with gr.Blocks(title="NOP Vision Intelligence") as demo:
    gr.HTML(
        '<div class="header"><h1>NOP Vision Intelligence</h1>'
        '<span class="tag">detect &rarr; track &rarr; zones &rarr; evidence</span></div>'
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            gr.Markdown("Input", elem_classes="section")
            with gr.Group():
                # A plain file picker, and file_types is deliberately extensions
                # rather than the "video" category: the category makes Gradio
                # mount its video-preview widget, which tries to decode the file
                # in-browser and fails loudly on mp4v-encoded footage that the
                # pipeline reads without any trouble.
                video_input = gr.File(
                    label="Video file",
                    file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm"],
                    type="filepath",
                )
                config_input = gr.Dropdown(CONFIG_CHOICES, value=DEFAULT_CONFIG,
                                           label="Configuration")
                frames_input = gr.Number(value=0, precision=0,
                                         label="Frame limit (0 = whole video)")
                run_button = gr.Button("Run pipeline", variant="primary",
                                       elem_id="run-button")

            gr.Markdown("Summary", elem_classes="section")
            metrics_output = gr.Markdown(EMPTY_METRICS)
            download_output = gr.File(label="Download all results (.zip)")

        with gr.Column(scale=6):
            gr.Markdown("Annotated result", elem_classes="section")
            video_output = gr.Video(label=None, show_label=False, height=420)

            gr.Markdown("Event data", elem_classes="section")
            with gr.Tab("Events"):
                events_output = gr.Dataframe(headers=["scenario_id", "event_type", "track_id",
                                                       "object_type", "confidence",
                                                       "video_time_seconds"],
                                              wrap=True, show_label=False)
            with gr.Tab("Counts"):
                counts_output = gr.Dataframe(headers=["event_type", "count"],
                                              wrap=True, show_label=False)

    with gr.Accordion("Console output", open=False):
        log_output = gr.Textbox(show_label=False, lines=6, max_lines=14)

    run_button.click(
        fn=run_pipeline,
        inputs=[video_input, config_input, frames_input],
        outputs=[video_output, metrics_output, counts_output, events_output,
                 download_output, log_output],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
