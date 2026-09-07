import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import normalize_zone
from zones import DwellQueueAnalyzer, ZoneFlow

ZONES = {
    "A": normalize_zone([0, 0, 100, 100]),
    "B": normalize_zone([200, 0, 300, 100]),
}


class TestZoneFlow(unittest.TestCase):
    def setUp(self):
        self.flow = ZoneFlow(ZONES, stability_frames=3, cooldown_seconds=1.5)

    def _settle(self, track_id, point, start_time=0.0, frames=3):
        event = None
        for i in range(frames):
            _, event = self.flow.update(track_id, point, start_time + i * 0.04)
        return event

    def test_first_zone_entry_is_not_an_event(self):
        # Where an object is first seen is its origin, not a crossing.
        self.assertIsNone(self._settle(1, (50, 50)))

    def test_completed_crossing_emits_event(self):
        self._settle(1, (50, 50), start_time=0.0)
        event = self._settle(1, (250, 50), start_time=1.0)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "A_TO_B")
        self.assertEqual(event["source_zone"], "A")
        self.assertEqual(event["destination_zone"], "B")

    def test_zone_memory_survives_the_gap_between_zones(self):
        # The heart of route detection: an object spends most of a crossing
        # inside neither zone. If leaving A wiped the remembered origin, the
        # arrival at B could never be recognised as an A-to-B crossing.
        self._settle(1, (50, 50), start_time=0.0)
        for i in range(50):
            self.flow.update(1, (150, 50), 1.0 + i * 0.04)  # open space between zones
        event = self._settle(1, (250, 50), start_time=5.0)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "A_TO_B")

    def test_boundary_jitter_does_not_emit_an_event(self):
        self._settle(1, (50, 50), start_time=0.0)
        # One frame barely inside B, then back out: not a stable crossing.
        self.flow.update(1, (250, 50), 1.0)
        self.flow.update(1, (150, 50), 1.04)
        _, event = self.flow.update(1, (150, 50), 1.08)
        self.assertIsNone(event)

    def test_duplicate_crossing_suppressed_within_cooldown(self):
        self._settle(1, (50, 50), start_time=0.0)
        first = self._settle(1, (250, 50), start_time=1.0)
        self.assertIsNotNone(first)
        self._settle(1, (50, 50), start_time=1.2)
        duplicate = self._settle(1, (250, 50), start_time=1.4)
        self.assertIsNone(duplicate)

    def test_genuine_repeat_crossing_allowed_after_cooldown(self):
        # Someone walking back out the way they came in is a real second
        # event, so deduplication must be time-based rather than permanent.
        self._settle(1, (50, 50), start_time=0.0)
        self.assertIsNotNone(self._settle(1, (250, 50), start_time=1.0))
        self._settle(1, (50, 50), start_time=5.0)
        self.assertIsNotNone(self._settle(1, (250, 50), start_time=10.0))

    def test_tracks_are_independent(self):
        self._settle(1, (50, 50), start_time=0.0)
        self.assertIsNone(self._settle(2, (250, 50), start_time=1.0))


class TestDwellQueueAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DwellQueueAnalyzer(
            zone_polygon=normalize_zone([0, 0, 100, 100]), queue_zone="QUEUE",
            dwell_threshold_seconds=2.0, occupancy_threshold=2, grace_frames=5)

    def test_occupancy_counts_only_objects_inside(self):
        occupancy, _, _ = self.analyzer.update({1: (50, 50), 2: (500, 500)}, 0.0)
        self.assertEqual(occupancy, 1)

    def test_dwell_threshold_fires_once(self):
        self.analyzer.update({1: (50, 50)}, 0.0)
        _, events, _ = self.analyzer.update({1: (50, 50)}, 2.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "DWELL_THRESHOLD")
        _, again, _ = self.analyzer.update({1: (50, 50)}, 3.0)
        self.assertEqual(again, [])

    def test_brief_absence_does_not_reset_the_dwell_clock(self):
        # A single missed detection must not discard several seconds of
        # accumulated waiting time.
        self.analyzer.update({1: (50, 50)}, 0.0)
        self.analyzer.update({}, 0.5)
        _, events, _ = self.analyzer.update({1: (50, 50)}, 2.5)
        self.assertEqual(len(events), 1)

    def test_occupancy_event_fires_on_rising_edge_only(self):
        _, _, event = self.analyzer.update({1: (50, 50)}, 0.0)
        self.assertIsNone(event)
        _, _, event = self.analyzer.update({1: (50, 50), 2: (60, 60)}, 1.0)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "QUEUE_THRESHOLD")
        _, _, event = self.analyzer.update({1: (50, 50), 2: (60, 60)}, 2.0)
        self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
