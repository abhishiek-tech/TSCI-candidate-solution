"""Structured evidence output: JSONL/JSON records, CSV summaries, snapshots.

Follows the NOP evidence contract (``nop_reference/evidence_contract.json``):
``observation_id, camera_id, observed_at, object_type, event_type, track_id,
confidence, attributes, evidence.image_path, model.name/version``, plus
``scenario_id`` and ``video_time_seconds`` as shown in the candidate guide's
example record, so one record satisfies both.

Every promised output file is created up front, at run start, rather than
lazily on the first event. A video that legitimately produces no events must
still leave behind a valid (empty) ``counts.csv`` and ``events.json`` -- those
are required outputs, not side effects of something having happened.
"""

import csv
import json
import os
import uuid
from datetime import datetime, timezone

import cv2

MODEL_VERSION = "1.0"

EVENTS_CSV_HEADER = ["scenario_id", "event_type", "track_id", "object_type",
                     "confidence", "video_time_seconds"]
COUNTS_CSV_HEADER = ["event_type", "count"]
OCCUPANCY_CSV_HEADER = ["video_time_seconds", "zone", "occupancy_count"]


class EvidenceWriter:
    def __init__(self, output_dir, camera_id, scenario_id, model_name):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.scenario_id = scenario_id
        self.model_name = model_name

        self.evidence_dir = os.path.join(output_dir, "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)
        # Snapshots are named by a fresh uuid per event, so re-running into the
        # same directory would pile new files on top of the previous run's
        # instead of replacing them -- leaving orphaned images that belong to no
        # record in events.jsonl. Every other output file is truncated at start;
        # this one has to be cleared for the same reason.
        for stale in os.listdir(self.evidence_dir):
            if stale.lower().endswith(".jpg"):
                os.remove(os.path.join(self.evidence_dir, stale))

        self.jsonl_path = os.path.join(output_dir, "events.jsonl")
        self.json_path = os.path.join(output_dir, "events.json")
        self.events_csv_path = os.path.join(output_dir, "events.csv")
        self.counts_csv_path = os.path.join(output_dir, "counts.csv")
        self.occupancy_csv_path = os.path.join(output_dir, "queue_occupancy.csv")

        self.records = []
        self.counts = {}

        open(self.jsonl_path, "w", encoding="utf-8").close()
        self._write_json()
        self._write_csv(self.events_csv_path, EVENTS_CSV_HEADER, [])
        self._write_counts()
        self._write_csv(self.occupancy_csv_path, OCCUPANCY_CSV_HEADER, [])

    @staticmethod
    def _write_csv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def _write_json(self):
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2)

    def _write_counts(self):
        rows = [[event_type, count] for event_type, count in sorted(self.counts.items())]
        self._write_csv(self.counts_csv_path, COUNTS_CSV_HEADER, rows)

    def _build_record(self, track_id, object_type, confidence, event_type,
                      video_time_seconds, attributes, frame=None):
        observation_id = str(uuid.uuid4())
        image_path = ""
        if frame is not None:
            filename = f"{observation_id}.jpg"
            absolute = os.path.join(self.evidence_dir, filename)
            cv2.imwrite(absolute, frame)
            image_path = os.path.join("evidence", filename).replace("\\", "/")

        return {
            "observation_id": observation_id,
            "scenario_id": self.scenario_id,
            "camera_id": self.camera_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "observed_at_seconds": round(video_time_seconds, 3),
            "video_time_seconds": round(video_time_seconds, 3),
            "object_type": object_type,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": float(confidence),
            "attributes": attributes,
            "evidence": {"image_path": image_path, "clip_path": ""},
            "model": {"name": self.model_name, "version": MODEL_VERSION},
        }

    def _persist(self, record):
        self.records.append(record)
        self.counts[record["event_type"]] = self.counts.get(record["event_type"], 0) + 1

        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._write_json()

        with open(self.events_csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                self.scenario_id, record["event_type"], record["track_id"],
                record["object_type"], record["confidence"], record["video_time_seconds"],
            ])
        self._write_counts()
        return record

    def write_route_event(self, frame, track_id, object_type, confidence, event,
                          video_time_seconds):
        attributes = {
            "source_zone": event.get("source_zone"),
            "destination_zone": event.get("destination_zone"),
        }
        return self._persist(self._build_record(
            track_id, object_type, confidence, event["event_type"],
            video_time_seconds, attributes, frame))

    def write_dwell_event(self, frame, track_id, object_type, confidence, event,
                          video_time_seconds):
        attributes = {"zone": event.get("zone"), "dwell_seconds": event.get("dwell_seconds")}
        return self._persist(self._build_record(
            track_id, object_type, confidence, event["event_type"],
            video_time_seconds, attributes, frame))

    def write_occupancy_event(self, frame, event, video_time_seconds):
        attributes = {"zone": event.get("zone"), "occupancy": event.get("occupancy")}
        return self._persist(self._build_record(
            "occupancy", "queue", 1.0, event["event_type"],
            video_time_seconds, attributes, frame))

    def write_occupancy_sample(self, video_time_seconds, zone, occupancy_count):
        with open(self.occupancy_csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([round(video_time_seconds, 3), zone, occupancy_count])
