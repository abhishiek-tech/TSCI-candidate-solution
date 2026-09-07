import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evidence import EvidenceWriter


class TestEvidenceWriter(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.frame = np.zeros((40, 40, 3), dtype=np.uint8)

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _writer(self):
        return EvidenceWriter(self.output_dir, "cam_01", "S_TEST", "test-model")

    def _route_event(self):
        return {"event_type": "A_TO_B", "source_zone": "A", "destination_zone": "B"}

    def test_all_required_outputs_exist_even_with_zero_events(self):
        # A video that legitimately produces no events must still leave valid
        # (empty) outputs behind -- they are required deliverables, not side
        # effects of something having happened.
        self._writer()
        for name in ("events.jsonl", "events.json", "events.csv",
                     "counts.csv", "queue_occupancy.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, name)), name)

        with open(os.path.join(self.output_dir, "events.json")) as handle:
            self.assertEqual(json.load(handle), [])

    def test_record_satisfies_the_evidence_contract(self):
        writer = self._writer()
        record = writer.write_route_event(self.frame, 7, "moving_object", 0.9,
                                          self._route_event(), 12.5)

        for field in ("observation_id", "camera_id", "observed_at", "object_type",
                      "event_type", "track_id", "confidence", "attributes",
                      "evidence", "model", "scenario_id", "video_time_seconds"):
            self.assertIn(field, record)
        self.assertIn("image_path", record["evidence"])
        self.assertIn("name", record["model"])
        self.assertIn("version", record["model"])
        self.assertEqual(record["attributes"]["source_zone"], "A")

    def test_every_event_has_a_snapshot_on_disk(self):
        writer = self._writer()
        record = writer.write_route_event(self.frame, 1, "moving_object", 0.9,
                                          self._route_event(), 1.0)
        image_path = os.path.join(self.output_dir, record["evidence"]["image_path"])
        self.assertTrue(os.path.exists(image_path))

    def test_rerun_does_not_leave_orphaned_snapshots(self):
        # Snapshots are uuid-named, so re-running into the same directory would
        # accumulate images belonging to no record in the current events file.
        first = self._writer()
        for i in range(3):
            first.write_route_event(self.frame, i, "moving_object", 0.9,
                                    self._route_event(), float(i))

        second = self._writer()
        second.write_route_event(self.frame, 0, "moving_object", 0.9,
                                 self._route_event(), 0.0)

        evidence_dir = os.path.join(self.output_dir, "evidence")
        snapshots = [f for f in os.listdir(evidence_dir) if f.endswith(".jpg")]
        with open(os.path.join(self.output_dir, "events.jsonl")) as handle:
            records = [line for line in handle if line.strip()]
        self.assertEqual(len(snapshots), len(records))

    def test_counts_track_event_totals(self):
        writer = self._writer()
        for i in range(2):
            writer.write_route_event(self.frame, i, "moving_object", 0.9,
                                     self._route_event(), float(i))
        writer.write_dwell_event(self.frame, 5, "moving_object", 0.8,
                                 {"event_type": "DWELL_THRESHOLD", "zone": "QUEUE",
                                  "dwell_seconds": 4.2}, 9.0)

        with open(os.path.join(self.output_dir, "counts.csv")) as handle:
            counts = {row[0]: row[1] for row in list(csv.reader(handle))[1:]}
        self.assertEqual(counts["A_TO_B"], "2")
        self.assertEqual(counts["DWELL_THRESHOLD"], "1")

    def test_events_csv_row_per_event(self):
        writer = self._writer()
        writer.write_route_event(self.frame, 3, "moving_object", 0.75,
                                 self._route_event(), 4.25)
        with open(os.path.join(self.output_dir, "events.csv")) as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], "S_TEST")
        self.assertEqual(rows[1][1], "A_TO_B")


if __name__ == "__main__":
    unittest.main()
