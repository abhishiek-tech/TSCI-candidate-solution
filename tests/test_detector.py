import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import AppearanceChannelDetector, AppearanceSignatureBank, color_distance

_BACKGROUND = 30


def _blob_contour(frame):
    # Per-channel "differs from the flat background" test rather than a
    # luminance threshold: a pure-blue fill has very low grayscale luminance
    # and a luminance threshold can miss it entirely.
    mask = np.any(frame != _BACKGROUND, axis=2).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1, f"fixture should produce one contour, got {len(contours)}"
    return contours[0]


class TestSignatureBank(unittest.TestCase):
    def test_signature_registers_only_after_repeated_evidence(self):
        bank = AppearanceSignatureBank(confirm_hits=3, min_confirm_area=100)
        self.assertIsNone(bank.observe([10, 200, 40], area=5000))
        self.assertIsNone(bank.observe([10, 200, 40], area=5000))
        self.assertEqual(bank.observe([10, 200, 40], area=5000), 0)
        self.assertEqual(len(bank), 1)

    def test_one_off_blend_colour_never_registers(self):
        # A colour seen once at an overlap boundary is not a real object.
        bank = AppearanceSignatureBank(confirm_hits=3, min_confirm_area=100)
        bank.observe([120, 120, 120], area=5000)
        self.assertEqual(len(bank), 0)

    def test_tiny_blobs_cannot_seed_a_signature(self):
        bank = AppearanceSignatureBank(confirm_hits=2, min_confirm_area=1000)
        for _ in range(5):
            bank.observe([10, 200, 40], area=50)
        self.assertEqual(len(bank), 0)

    def test_confirmed_signature_is_frozen_not_averaged(self):
        # A signature is a fixed property of an object. If it drifted toward
        # whatever it last matched, it would migrate toward a neighbouring
        # object during an overlap and silently merge two identities.
        bank = AppearanceSignatureBank(match_threshold=0.2, confirm_hits=1,
                                       min_confirm_area=100)
        signature_id = bank.observe([100, 100, 100], area=5000)
        original = list(bank.signatures[signature_id])
        for _ in range(20):
            bank.observe([115, 115, 115], area=5000)
        self.assertEqual(bank.signatures[signature_id], original)

    def test_distinct_colours_get_distinct_signatures(self):
        bank = AppearanceSignatureBank(match_threshold=0.07, confirm_hits=1,
                                       min_confirm_area=100)
        first = bank.observe([200, 40, 40], area=5000)
        second = bank.observe([40, 200, 40], area=5000)
        self.assertNotEqual(first, second)
        self.assertEqual(len(bank), 2)


class TestAppearanceChannelSplit(unittest.TestCase):
    def setUp(self):
        self.detector = AppearanceChannelDetector(min_area=100, min_channel_area=50)
        # Two known signatures, as if already learned from clean frames.
        self.detector.signatures.signatures = [[255.0, 0.0, 0.0], [0.0, 0.0, 255.0]]

    def test_overlapping_objects_split_by_signature(self):
        # Two differently-coloured rectangles overlapping into one contour --
        # exactly the case where geometry alone cannot say which pixel belongs
        # to which object, but colour can.
        frame = np.full((250, 250, 3), _BACKGROUND, dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.rectangle(frame, (120, 60), (220, 160), (0, 0, 255), -1)
        contour = _blob_contour(frame)

        detections = self.detector._split_by_signature(frame, contour)

        self.assertEqual(len(detections), 2)
        self.assertEqual({d["signature_id"] for d in detections}, {0, 1})
        centroids = [d["centroid"] for d in detections]
        separation = abs(centroids[0][0] - centroids[1][0])
        self.assertGreater(separation, 20, "split detections should be spatially distinct")

    def test_absent_signature_produces_no_channel(self):
        # Only one of the two known signatures is present in the frame, so the
        # other must not conjure a detection out of nothing.
        frame = np.full((250, 250, 3), _BACKGROUND, dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 0, 0), -1)
        contour = _blob_contour(frame)

        detections = self.detector._split_by_signature(frame, contour)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["signature_id"], 0)

    def test_fully_occluded_signature_yields_no_detection(self):
        # When one object is completely hidden behind another, its channel is
        # genuinely empty. Reporting a position anyway would be fabrication;
        # the correct behaviour is to report nothing and let the tracker treat
        # it as an occlusion.
        frame = np.full((250, 250, 3), _BACKGROUND, dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), -1)
        contour = _blob_contour(frame)

        detections = self.detector._split_by_signature(frame, contour)

        self.assertEqual([d["signature_id"] for d in detections], [1])


class TestColorDistance(unittest.TestCase):
    def test_identical_colours_have_zero_distance(self):
        self.assertEqual(color_distance([10, 20, 30], [10, 20, 30]), 0.0)

    def test_distance_is_normalised_to_unit_range(self):
        self.assertAlmostEqual(color_distance([0, 0, 0], [255, 255, 255]), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
