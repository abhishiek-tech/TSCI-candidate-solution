"""Detection: appearance-channel segmentation with a motion-blob fallback.

## Why this design

The hard case in this benchmark is a dense crossing: several objects
converge on the same few hundred pixels at once. A conventional pipeline
detects a generic foreground blob and then tries to work out *which* object
each blob belongs to after the fact -- via colour cost terms, wider
assignment gates, track revival, shape-based blob splitting, and so on.
Every one of those runs into the same wall: while two objects genuinely
overlap, "which object owns this pixel" has no answer derivable from that
single frame, so the tracker is guessing, and a guess that happens to be
right is indistinguishable from one that is wrong.

``AppearanceChannelDetector`` removes that ambiguity instead of reasoning
around it. Each object in this domain carries a stable appearance signature
(here: a consistent mean colour) for its whole lifetime. The detector learns
each signature on the fly from clean, unmerged observations, then treats
every known signature as its own detection channel: a merged blob is split
by *which signature each pixel matches*, not by geometry or by a guess. A
pixel only ever shows one colour, so ownership has a real answer. Overlap
degrades from "unrecoverable identity confusion" into "ordinary temporary
occlusion", which the tracker already handles well.

## The assumption, stated plainly

This works because objects here have stable, mutually distinguishable
appearance signatures. That holds strongly for this benchmark's footage and
weakly for crowded real-world scenes where people or vehicles share colour
and lighting. When signatures are not separable, the detector degrades to
``fallback_split`` (geometry-based splitting of a merged blob), which is the
conventional behaviour and no worse than a standard pipeline. See REPORT.md
for the full trade-off.
"""

import statistics
from math import hypot

import cv2
import numpy as np

# Maximum possible Euclidean distance in BGR space, used to normalise colour
# distances into 0..1 so thresholds read as fractions rather than raw units.
_MAX_BGR_DISTANCE = 441.7


def color_distance(color_a, color_b):
    return hypot(*(a - b for a, b in zip(color_a, color_b))) / _MAX_BGR_DISTANCE


def make_detection(x, y, w, h, confidence, object_type, mean_color, signature_id=None):
    return {
        "bbox": [int(x), int(y), int(w), int(h)],
        "centroid": [int(x + w / 2), int(y + h / 2)],
        "confidence": float(confidence),
        "object_type": object_type,
        "mean_color": list(mean_color),
        "signature_id": signature_id,
    }


def mean_color_of(frame, x, y, w, h):
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
    if x1 <= x0 or y1 <= y0:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)]


def describe_blob(frame, contour, match_threshold):
    """Summarise a blob as (dominant_colour, dominant_fraction).

    The dominant colour is the *median* of the blob's pixels, not the mean:
    the foreground mask is slightly dilated by morphology, so it always
    carries some background pixels along its edge, and a mean drags the
    result toward the background. The median ignores that minority.

    ``dominant_fraction`` is the share of pixels lying within
    ``match_threshold`` of that colour, i.e. "how much of this blob is
    actually one uniform colour". A single object scores high; a blob
    straddling two partially-overlapping objects scores low, because its
    pixels form two clusters and the median sits between them.
    """
    blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.nonzero(blob_mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0], 0.0

    pixels = frame[ys, xs].astype(np.float32)
    dominant = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels - dominant, axis=1) / _MAX_BGR_DISTANCE
    fraction = float((distances <= match_threshold).mean())
    return [float(v) for v in dominant], fraction


class AppearanceSignatureBank:
    """Learns and stores one stable appearance signature per physical object.

    Two properties matter and are deliberate:

    1. **Signatures are frozen once confirmed.** They are a fixed property of
       an object, not a running average of whatever it most recently looked
       like. Continuously blending observations into a stored signature lets
       it drift toward a *neighbouring* object during an overlap, and once it
       has drifted it matches that neighbour even more readily -- a feedback
       loop that silently merges two distinct identities into one. (Measured:
       an adaptive variant collapsed five genuinely distinct signatures down
       to two.)

    2. **A new signature must prove itself before it is registered.** Where
       two objects overlap, boundary pixels blend into intermediate colours
       that belong to no real object. Registering those on sight invents
       phantom identities. A candidate must therefore be observed
       ``confirm_hits`` times, at a plausible scale, before it is promoted.
    """

    def __init__(self, match_threshold=0.07, confirm_hits=3, min_confirm_area=400,
                 min_dominant_fraction=0.5):
        self.match_threshold = match_threshold
        self.confirm_hits = confirm_hits
        self.min_confirm_area = min_confirm_area
        self.min_dominant_fraction = min_dominant_fraction
        self.signatures = []  # frozen [b, g, r] per confirmed object
        self._pending = []  # [{"color": [...], "count": int}]

    def match(self, color):
        """Return the id of the signature this colour belongs to, or None."""
        best_id, best_distance = None, float("inf")
        for signature_id, signature in enumerate(self.signatures):
            distance = color_distance(color, signature)
            if distance < best_distance:
                best_distance, best_id = distance, signature_id
        if best_id is not None and best_distance <= self.match_threshold:
            return best_id
        return None

    def observe(self, color, area, dominant_fraction=1.0):
        """Match a colour to a known signature, or build evidence for a new one.

        Returns the signature id if this colour belongs to a confirmed
        signature (registering it now if this observation is what confirms
        it), otherwise None.
        """
        known = self.match(color)
        if known is not None:
            return known

        if area < self.min_confirm_area:
            # Too small to be a real object; never let it seed a signature.
            return None

        if dominant_fraction < self.min_dominant_fraction:
            # This blob is not one uniform colour, so it is very likely two
            # partially-overlapping objects seen as one. Its colour belongs to
            # neither of them, and registering it would invent an object that
            # does not exist.
            return None

        for candidate in self._pending:
            if color_distance(color, candidate["color"]) <= self.match_threshold:
                candidate["count"] += 1
                if candidate["count"] >= self.confirm_hits:
                    self.signatures.append(candidate["color"])
                    self._pending.remove(candidate)
                    return len(self.signatures) - 1
                return None

        candidate = {"color": list(color), "count": 1}
        if candidate["count"] >= self.confirm_hits:
            self.signatures.append(candidate["color"])
            return len(self.signatures) - 1
        self._pending.append(candidate)
        return None

    def __len__(self):
        return len(self.signatures)


class AppearanceChannelDetector:
    """Background subtraction, then per-signature segmentation of merged blobs."""

    def __init__(self, min_area=600, max_area_ratio=0.5, history=150, var_threshold=24,
                 learning_rate=0.003, merge_area_ratio=1.6, signature_match_threshold=0.07,
                 signature_confirm_hits=3, min_channel_area=150, fallback_split=True,
                 max_split=4, signature_min_dominant_fraction=0.5):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.learning_rate = learning_rate
        self.merge_area_ratio = merge_area_ratio
        self.min_channel_area = min_channel_area
        self.fallback_split = fallback_split
        self.max_split = max_split

        self.signatures = AppearanceSignatureBank(
            match_threshold=signature_match_threshold,
            confirm_hits=signature_confirm_hits,
            min_confirm_area=min_area,
            min_dominant_fraction=signature_min_dominant_fraction,
        )

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        # cv2.kmeans (fallback splitting) seeds from OpenCV's global RNG. Left
        # unseeded, the same video and config can produce different event
        # counts on different runs, which quietly breaks reproducibility.
        cv2.setRNGSeed(42)

        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._area_history = []

    @property
    def signature_count(self):
        return len(self.signatures)

    def _typical_object_area(self):
        if len(self._area_history) < 10:
            return None
        return statistics.median(self._area_history)

    def _foreground_mask(self, frame):
        mask = self.subtractor.apply(frame, learningRate=self.learning_rate)
        # MOG2 marks shadows as 127; keep only confident foreground.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

    def _split_by_signature(self, frame, contour):
        """Split a merged blob into one detection per appearance signature present."""
        x0, y0, w0, h0 = cv2.boundingRect(contour)
        blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)

        region = frame[y0:y0 + h0, x0:x0 + w0].astype(np.float32)
        region_mask = blob_mask[y0:y0 + h0, x0:x0 + w0]

        detections = []
        for signature_id, signature in enumerate(self.signatures.signatures):
            distance = np.linalg.norm(region - np.array(signature, dtype=np.float32), axis=2)
            distance /= _MAX_BGR_DISTANCE
            channel = ((distance <= self.signatures.match_threshold) & (region_mask == 255))
            channel_mask = channel.astype(np.uint8) * 255
            if int(channel.sum()) < self.min_channel_area:
                continue

            parts, _ = cv2.findContours(channel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            parts = [p for p in parts if cv2.contourArea(p) >= self.min_channel_area]
            if not parts:
                continue

            # One object owns one signature, so take its largest component and
            # ignore speckle from antialiased edges.
            largest = max(parts, key=cv2.contourArea)
            sx, sy, sw, sh = cv2.boundingRect(largest)
            detections.append(make_detection(
                x0 + sx, y0 + sy, sw, sh, 0.6, "moving_object",
                mean_color=signature, signature_id=signature_id,
            ))
        return detections

    def _split_by_geometry(self, frame, contour, area, typical_area):
        """Fallback when appearance can't separate a blob: split spatially.

        Used when signatures are unavailable or indistinguishable -- the
        conventional behaviour, kept so the detector degrades gracefully on
        footage where objects don't have separable appearance.
        """
        k = min(self.max_split, max(2, round(area / typical_area)))
        blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
        ys, xs = np.nonzero(blob_mask)
        if len(xs) < k * 4:
            return []

        x0, y0, w0, h0 = cv2.boundingRect(contour)
        colors = frame[ys, xs].astype(np.float32)
        norm_x = (xs.astype(np.float32) - x0) / max(w0, 1)
        norm_y = (ys.astype(np.float32) - y0) / max(h0, 1)
        features = np.column_stack((norm_x, norm_y, colors / 255.0 * 4.0)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(features, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        detections = []
        for cluster in range(k):
            selected = labels == cluster
            if selected.sum() < 4:
                continue
            cx, cy, cw, ch = cv2.boundingRect(
                np.column_stack((xs[selected], ys[selected])).astype(np.int32))
            color = [float(v) for v in colors[selected].mean(axis=0)]
            detections.append(make_detection(cx, cy, cw, ch, 0.4, "moving_object",
                                              mean_color=color, signature_id=None))
        return detections

    def detect(self, frame):
        frame_area = frame.shape[0] * frame.shape[1]
        mask = self._foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        typical_area = self._typical_object_area()

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > frame_area * self.max_area_ratio:
                continue

            is_merged = typical_area is not None and area > typical_area * self.merge_area_ratio
            if is_merged:
                split = self._split_by_signature(frame, contour)
                if len(split) >= 2:
                    detections.extend(split)
                    continue
                if self.fallback_split:
                    fallback = self._split_by_geometry(frame, contour, area, typical_area)
                    if len(fallback) >= 2:
                        detections.extend(fallback)
                        continue

            x, y, w, h = cv2.boundingRect(contour)
            color, dominant_fraction = describe_blob(frame, contour,
                                                      self.signatures.match_threshold)
            signature_id = self.signatures.observe(color, area, dominant_fraction)
            solidity = area / float(w * h) if w * h else 0.0
            confidence = max(0.35, min(0.95, 0.5 + 0.4 * solidity))
            detections.append(make_detection(x, y, w, h, confidence, "moving_object",
                                              mean_color=color, signature_id=signature_id))

            if not is_merged:
                self._area_history.append(area)
                if len(self._area_history) > 60:
                    self._area_history.pop(0)

        return detections


def build_detector(config):
    kind = config.get("detector", "appearance").lower()
    if kind != "appearance":
        raise ValueError(f"Unknown detector type: {kind!r} (expected 'appearance')")
    params = config.get("detector_params", {})
    return AppearanceChannelDetector(
        min_area=params.get("min_area", 600),
        max_area_ratio=params.get("max_area_ratio", 0.5),
        history=params.get("history", 150),
        var_threshold=params.get("var_threshold", 24),
        learning_rate=params.get("learning_rate", 0.003),
        merge_area_ratio=params.get("merge_area_ratio", 1.6),
        signature_match_threshold=params.get("signature_match_threshold", 0.07),
        signature_confirm_hits=params.get("signature_confirm_hits", 3),
        min_channel_area=params.get("min_channel_area", 150),
        fallback_split=params.get("fallback_split", True),
        max_split=params.get("max_split", 4),
        signature_min_dominant_fraction=params.get("signature_min_dominant_fraction", 0.5),
    )
