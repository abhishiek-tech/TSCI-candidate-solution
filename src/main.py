"""Pipeline entry point: video in, annotated video and structured evidence out.

    video -> detector -> tracker -> zone logic -> evidence

Run one scenario:

    cd src
    python main.py --input ../data/generated/S01_BASIC_GOODS.mp4 \
                   --output-dir ../output/S01 \
                   --config ../config/scenario_S01.json
"""

import argparse
import json
import os
import time

import cv2

from detector import build_detector
from evidence import EvidenceWriter
from geometry import draw_zones, normalize_zone, normalize_zones
from tracker import build_tracker
from zones import DwellQueueAnalyzer, ZoneFlow

TRACK_COLOR = (80, 220, 120)
HUD_COLOR = (240, 240, 240)


def parse_args():
    parser = argparse.ArgumentParser(description="NOP vision-intelligence pipeline")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output-dir", default="output", help="Directory for outputs")
    parser.add_argument("--config", default="../config/default.json", help="JSON configuration")
    parser.add_argument("--scenario-id", default=None,
                        help="Override the evidence scenario_id (defaults to config, then filename)")
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (debugging)")
    return parser.parse_args()


def derive_scenario_id(config, args):
    if args.scenario_id:
        return args.scenario_id
    if config.get("scenario_id"):
        return config["scenario_id"]
    return os.path.splitext(os.path.basename(args.input))[0]


def draw_track(frame, track, zone, dwell_seconds=None):
    x, y, w, h = track.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), TRACK_COLOR, 2)
    label = f"id {track.track_id}"
    if zone:
        label += f" [{zone}]"
    cv2.putText(frame, label, (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TRACK_COLOR, 2)
    if dwell_seconds:
        cv2.putText(frame, f"dwell {dwell_seconds:.1f}s", (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)


def draw_hud(frame, frame_index, processing_fps, signature_count,
             occupancy=None, queue_zone=None):
    text = f"frame {frame_index} | {processing_fps:.1f} fps | signatures {signature_count}"
    if occupancy is not None and queue_zone:
        text += f" | {queue_zone} occupancy {occupancy}"
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    scenario_id = derive_scenario_id(config, args)

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(os.path.join(args.output_dir, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = build_detector(config)
    tracker = build_tracker(config)
    model_name = f"{config.get('detector', 'appearance')}-{type(detector).__name__}"

    all_zones = normalize_zones(config["zones"])
    dwell_config = config.get("dwell_queue")
    dwell = None
    if dwell_config:
        dwell = DwellQueueAnalyzer(
            zone_polygon=normalize_zone(config["zones"][dwell_config["queue_zone"]]),
            queue_zone=dwell_config["queue_zone"],
            dwell_threshold_seconds=dwell_config.get("dwell_threshold_seconds", 5.0),
            occupancy_threshold=dwell_config.get("occupancy_threshold"),
            grace_frames=dwell_config.get("grace_frames", 5),
        )

    # A waiting area is somewhere an object passes *through* on the way from A
    # to B. Including it in route classification would split one real journey
    # into A-to-QUEUE plus QUEUE-to-B, which is not the crossing the ground
    # truth counts, so it is excluded here and handled by the dwell analyzer.
    route_zones = {name: polygon for name, polygon in all_zones.items()
                   if not dwell_config or name != dwell_config["queue_zone"]}
    flow_config = config.get("zone_flow", {})
    flow = ZoneFlow(route_zones,
                    stability_frames=flow_config.get("stability_frames", 3),
                    cooldown_seconds=flow_config.get("cooldown_seconds", 1.5))

    sample_interval = (dwell_config or {}).get("sample_interval_seconds", 1.0)
    last_sample_time = -sample_interval

    evidence = EvidenceWriter(args.output_dir, config.get("camera_id", "candidate_cam_01"),
                              scenario_id, model_name)

    frame_index = 0
    started_at = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if args.max_frames and frame_index > args.max_frames:
            break

        video_time = frame_index / fps
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        draw_zones(frame, all_zones)

        occupancy = None
        dwell_lookup = {}
        if dwell is not None:
            centroids = {tid: track.centroid for tid, track in tracks.items()}
            occupancy, dwell_events, occupancy_event = dwell.update(centroids, video_time)

            for event in dwell_events:
                track = tracks.get(event["track_id"])
                evidence.write_dwell_event(
                    frame=frame, track_id=event["track_id"],
                    object_type=track.object_type if track else "unknown",
                    confidence=track.confidence if track else 0.5,
                    event=event, video_time_seconds=video_time)

            if occupancy_event:
                evidence.write_occupancy_event(frame, occupancy_event, video_time)

            if video_time - last_sample_time >= sample_interval:
                evidence.write_occupancy_sample(video_time, dwell.queue_zone, occupancy)
                last_sample_time = video_time

            for track_id, track in tracks.items():
                if dwell.locate(track.centroid):
                    dwell_lookup[track_id] = dwell.current_dwell(track_id, video_time)

        for track_id, track in tracks.items():
            zone, event = flow.update(track_id, track.centroid, video_time)
            draw_track(frame, track, zone, dwell_lookup.get(track_id))
            if event:
                evidence.write_route_event(
                    frame=frame, track_id=track_id, object_type=track.object_type,
                    confidence=track.confidence, event=event, video_time_seconds=video_time)

        elapsed = time.perf_counter() - started_at
        processing_fps = frame_index / elapsed if elapsed > 0 else 0.0
        draw_hud(frame, frame_index, processing_fps, detector.signature_count,
                 occupancy, dwell.queue_zone if dwell else None)

        writer.write(frame)
        if args.display:
            cv2.imshow("NOP Vision Intelligence", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    performance = {
        "scenario_id": scenario_id,
        "frames_processed": frame_index,
        "video_fps": fps,
        "wall_clock_seconds": round(elapsed, 3),
        "processing_fps": round(frame_index / elapsed, 2) if elapsed > 0 else 0.0,
        "detector": config.get("detector", "appearance"),
        "appearance_signatures_learned": detector.signature_count,
    }
    with open(os.path.join(args.output_dir, "performance.json"), "w", encoding="utf-8") as handle:
        json.dump(performance, handle, indent=2)

    print(f"Done. {frame_index} frames in {elapsed:.2f}s "
          f"({performance['processing_fps']:.1f} fps), "
          f"{detector.signature_count} appearance signatures. "
          f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
