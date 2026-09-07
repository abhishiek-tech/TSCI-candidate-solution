import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import make_detection
from tracker import SignatureTracker


def detection(x, y, signature_id=None, color=(100, 100, 100), size=40):
    return make_detection(x, y, size, size, 0.9, "moving_object",
                          mean_color=list(color), signature_id=signature_id)


class TestSignatureIdentity(unittest.TestCase):
    def test_same_signature_keeps_the_same_track_id(self):
        tracker = SignatureTracker(min_hits=1)
        first = tracker.update([detection(10, 10, signature_id=0)])
        track_id = next(iter(first))
        second = tracker.update([detection(60, 10, signature_id=0)])
        self.assertEqual(list(second), [track_id])

    def test_identity_survives_a_large_positional_jump(self):
        # Signature identity is evidence from pixels, not proximity, so it
        # holds even when the object reappears far from where it vanished --
        # a jump a distance-gated tracker would refuse to accept.
        tracker = SignatureTracker(min_hits=1, max_match_distance=50)
        first = tracker.update([detection(10, 10, signature_id=0)])
        track_id = next(iter(first))
        second = tracker.update([detection(900, 600, signature_id=0)])
        self.assertEqual(list(second), [track_id])

    def test_different_signatures_never_share_a_track(self):
        # Two objects at nearly the same place stay distinct because their
        # signatures differ -- the case that defeats position-based matching.
        tracker = SignatureTracker(min_hits=1)
        tracks = tracker.update([
            detection(100, 100, signature_id=0),
            detection(104, 102, signature_id=1),
        ])
        self.assertEqual(len(tracks), 2)
        self.assertEqual({t.signature_id for t in tracks.values()}, {0, 1})

    def test_track_survives_long_occlusion_and_resumes_its_identity(self):
        # An object buried under others in a dense crossing can be genuinely
        # invisible for a second or more; waiting is safe here because a
        # signature can only ever re-acquire its own object.
        tracker = SignatureTracker(min_hits=1, max_missed_frames=100)
        first = tracker.update([detection(10, 10, signature_id=0)])
        track_id = next(iter(first))
        for _ in range(50):
            tracker.update([])
        resumed = tracker.update([detection(400, 300, signature_id=0)])
        self.assertEqual(list(resumed), [track_id])

    def test_track_dropped_after_exceeding_missed_budget(self):
        tracker = SignatureTracker(min_hits=1, max_missed_frames=5)
        tracker.update([detection(10, 10, signature_id=0)])
        for _ in range(7):
            tracker.update([])
        self.assertEqual(tracker.tracks, {})

    def test_fragmented_signature_keeps_the_nearest_fragment(self):
        # Segmentation can briefly yield two pieces of one object; only one
        # can be the object, so the nearer one to the prediction wins.
        tracker = SignatureTracker(min_hits=1)
        tracker.update([detection(100, 100, signature_id=0)])
        tracks = tracker.update([
            detection(700, 700, signature_id=0),
            detection(110, 100, signature_id=0),
        ])
        self.assertEqual(len(tracks), 1)
        self.assertLess(next(iter(tracks.values())).centroid[0], 400)


class TestIdentityContinuity(unittest.TestCase):
    def test_track_adopts_its_signature_instead_of_duplicating(self):
        # A signature takes a few observations to confirm, so an object is
        # tracked without one at first. When the signature arrives it must
        # attach to the track already following that object, not spawn a
        # second identity for something already being tracked.
        tracker = SignatureTracker(min_hits=1, max_match_distance=100)
        first = tracker.update([detection(100, 100)])
        original_id = next(iter(first))

        second = tracker.update([detection(108, 100, signature_id=0)])

        self.assertEqual(list(second), [original_id])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[original_id].signature_id, 0)

    def test_distant_signature_does_not_hijack_an_unrelated_track(self):
        # Adoption is for the same object continuing, so it must not reach
        # across the frame and swallow a different object's track.
        tracker = SignatureTracker(min_hits=1, max_match_distance=50)
        tracker.update([detection(100, 100)])
        tracker.update([detection(900, 700, signature_id=0)])
        self.assertEqual(len(tracker.tracks), 2)


class TestFallbackAssignment(unittest.TestCase):
    def test_unsignatured_detections_track_by_position(self):
        # Graceful degradation: without signatures this behaves like a
        # conventional tracker rather than failing.
        tracker = SignatureTracker(min_hits=1, max_match_distance=100)
        first = tracker.update([detection(100, 100)])
        track_id = next(iter(first))
        second = tracker.update([detection(120, 100)])
        self.assertEqual(list(second), [track_id])

    def test_distant_unsignatured_detection_starts_a_new_track(self):
        tracker = SignatureTracker(min_hits=1, max_match_distance=50)
        tracker.update([detection(100, 100)])
        tracker.update([detection(900, 700)])
        self.assertEqual(len(tracker.tracks), 2)

    def test_new_track_is_not_reported_before_min_hits(self):
        # Suppresses one-frame noise blobs from ever generating an event.
        tracker = SignatureTracker(min_hits=2)
        self.assertEqual(tracker.update([detection(10, 10, signature_id=0)]), {})
        self.assertEqual(len(tracker.update([detection(14, 10, signature_id=0)])), 1)


if __name__ == "__main__":
    unittest.main()
