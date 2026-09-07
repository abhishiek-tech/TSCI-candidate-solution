"""Multi-object tracking: signature identity first, spatial assignment second.

Most trackers spend their complexity budget deciding *which* detection
continues *which* track. When the detector can already say "this blob is
object #3" -- because it segmented that blob by object #3's own appearance
signature -- that decision is already made, and made from pixel evidence
rather than from a proximity heuristic. Those detections are bound directly
to their track.

Detections without a signature (the fallback path, when appearance can't
separate objects) still need conventional association, so they go through
optimal assignment (Hungarian) over a distance/IoU/appearance cost, which is
what a standard tracker does throughout.

One consequence worth noting: because a signature-identified track can only
ever re-acquire *its own* signature, waiting a long time for it to reappear
is safe. A conventional tracker has to cap its patience, since a track that
has been missing a while will happily snap onto whatever unrelated detection
drifts into its (widening) search radius. That risk doesn't exist here, so
``max_missed_frames`` can be generous enough to survive a long occlusion --
which matters, because an object buried under several others in a dense
crossing can be genuinely invisible for a second or more.
"""

from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import color_distance
from geometry import bbox_iou

_UNREACHABLE = 1e6


class Track:
    def __init__(self, track_id, detection, signature_id=None):
        self.track_id = track_id
        self.signature_id = signature_id
        self.bbox = detection["bbox"]
        self.centroid = detection["centroid"]
        self.mean_color = detection["mean_color"]
        self.object_type = detection["object_type"]
        self.confidence = detection["confidence"]
        self.velocity = [0.0, 0.0]
        self.missed = 0
        self.hits = 1
        self.age = 1

    def predicted_centroid(self, max_extrapolation_steps=30):
        """Where this track is expected to be, extrapolated from its velocity.

        Extrapolation is capped: a track kept alive through a long occlusion
        would otherwise fly off screen on residual velocity, and a nearly
        stationary object needs no extrapolation at all to be re-acquired.
        """
        steps = min(1 + self.missed, max_extrapolation_steps)
        return (self.centroid[0] + self.velocity[0] * steps,
                self.centroid[1] + self.velocity[1] * steps)

    def update_from(self, detection, velocity_damping):
        old_x, old_y = self.centroid
        new_x, new_y = detection["centroid"]
        steps = 1 + self.missed
        vx = (new_x - old_x) / steps
        vy = (new_y - old_y) / steps
        self.velocity = [
            velocity_damping * self.velocity[0] + (1 - velocity_damping) * vx,
            velocity_damping * self.velocity[1] + (1 - velocity_damping) * vy,
        ]
        self.bbox = detection["bbox"]
        self.centroid = detection["centroid"]
        self.object_type = detection["object_type"]
        self.confidence = detection["confidence"]
        if self.signature_id is None:
            # Only unsignatured tracks adapt their stored colour. A signature
            # is a fixed property of the object; blending observations into it
            # lets it drift toward a neighbour during an overlap.
            self.mean_color = detection["mean_color"]
        self.missed = 0
        self.hits += 1
        self.age += 1

    def mark_missed(self):
        self.missed += 1
        self.age += 1


class SignatureTracker:
    def __init__(self, max_match_distance=90, max_missed_frames=300, min_hits=2,
                 velocity_damping=0.8, appearance_weight=1.0):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.velocity_damping = velocity_damping
        self.appearance_weight = appearance_weight
        self.next_id = 1
        self.tracks = {}
        self._track_by_signature = {}

    def _create_track(self, detection):
        track = Track(self.next_id, detection, detection.get("signature_id"))
        self.tracks[self.next_id] = track
        if track.signature_id is not None:
            self._track_by_signature[track.signature_id] = track.track_id
        self.next_id += 1
        return track

    def _assign_by_signature(self, detections):
        """Bind signature-carrying detections straight to their own track."""
        matched_detection_indices = set()
        matched_track_ids = set()

        best_by_signature = {}
        for index, detection in enumerate(detections):
            signature_id = detection.get("signature_id")
            if signature_id is None:
                continue
            # A signature belongs to one object, so at most one detection per
            # signature survives; if segmentation produced several fragments,
            # keep the one nearest where that track is expected to be.
            previous = best_by_signature.get(signature_id)
            if previous is None:
                best_by_signature[signature_id] = index
                continue
            track_id = self._track_by_signature.get(signature_id)
            if track_id is None or track_id not in self.tracks:
                if detection["confidence"] > detections[previous]["confidence"]:
                    best_by_signature[signature_id] = index
                continue
            px, py = self.tracks[track_id].predicted_centroid()
            if (hypot(detection["centroid"][0] - px, detection["centroid"][1] - py) <
                    hypot(detections[previous]["centroid"][0] - px,
                          detections[previous]["centroid"][1] - py)):
                best_by_signature[signature_id] = index

        for signature_id, index in best_by_signature.items():
            detection = detections[index]
            track_id = self._track_by_signature.get(signature_id)
            if track_id is not None and track_id in self.tracks:
                self.tracks[track_id].update_from(detection, self.velocity_damping)
                matched_track_ids.add(track_id)
            else:
                adopted = self._adopt_unsignatured_track(detection, signature_id,
                                                          matched_track_ids)
                if adopted is not None:
                    matched_track_ids.add(adopted)
                else:
                    track = self._create_track(detection)
                    matched_track_ids.add(track.track_id)
            matched_detection_indices.add(index)

        return matched_detection_indices, matched_track_ids

    def _adopt_unsignatured_track(self, detection, signature_id, already_matched):
        """Give a newly confirmed signature to the track already following it.

        A signature needs a few observations before it is confirmed, so an
        object is tracked without one for its first frames. Creating a fresh
        track the moment its signature confirms would leave two identities
        for one object -- the original still drifting along beside the new
        one. Instead the existing track adopts the signature and keeps its id,
        so an object holds one identity from first sighting onward.
        """
        best_id, best_distance = None, float("inf")
        for track_id, track in self.tracks.items():
            if track.signature_id is not None or track_id in already_matched:
                continue
            px, py = track.predicted_centroid()
            distance = hypot(detection["centroid"][0] - px, detection["centroid"][1] - py)
            if distance < best_distance:
                best_distance, best_id = distance, track_id

        if best_id is None or best_distance > self.max_match_distance:
            return None

        track = self.tracks[best_id]
        track.signature_id = signature_id
        self._track_by_signature[signature_id] = best_id
        track.update_from(detection, self.velocity_damping)
        return best_id

    def _assign_by_cost(self, detections, detection_indices, candidate_track_ids):
        """Conventional optimal assignment for detections without a signature."""
        matched_detection_indices = set()
        matched_track_ids = set()
        track_ids = list(candidate_track_ids)
        detection_indices = list(detection_indices)
        if not track_ids or not detection_indices:
            return matched_detection_indices, matched_track_ids

        cost = np.full((len(track_ids), len(detection_indices)), _UNREACHABLE)
        for row, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            px, py = track.predicted_centroid()
            # The search radius grows while a track is unmatched but plateaus,
            # so a long-missing track can't reach across the frame.
            gate = self.max_match_distance * (1.0 + 0.5 * min(track.missed, 10))
            for column, index in enumerate(detection_indices):
                detection = detections[index]
                distance = hypot(detection["centroid"][0] - px, detection["centroid"][1] - py)
                if distance > gate:
                    continue
                iou = bbox_iou(track.bbox, detection["bbox"])
                appearance = color_distance(track.mean_color, detection["mean_color"])
                cost[row, column] = (distance * (1.0 - 0.3 * iou)
                                     + self.appearance_weight * appearance * self.max_match_distance)

        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if cost[row, column] >= _UNREACHABLE:
                continue
            track_id = track_ids[row]
            index = detection_indices[column]
            self.tracks[track_id].update_from(detections[index], self.velocity_damping)
            matched_track_ids.add(track_id)
            matched_detection_indices.add(index)

        return matched_detection_indices, matched_track_ids

    def update(self, detections):
        matched_detections, matched_tracks = self._assign_by_signature(detections)

        unsignatured = [i for i in range(len(detections)) if i not in matched_detections]
        available_tracks = [tid for tid, track in self.tracks.items()
                            if tid not in matched_tracks and track.signature_id is None]
        cost_matched_detections, cost_matched_tracks = self._assign_by_cost(
            detections, unsignatured, available_tracks)
        matched_detections |= cost_matched_detections
        matched_tracks |= cost_matched_tracks

        for track_id, track in list(self.tracks.items()):
            if track_id in matched_tracks:
                continue
            track.mark_missed()
            if track.missed > self.max_missed_frames:
                if track.signature_id is not None:
                    self._track_by_signature.pop(track.signature_id, None)
                del self.tracks[track_id]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            signature_id = detection.get("signature_id")
            if signature_id is not None and signature_id in self._track_by_signature:
                # A leftover piece of an object whose signature already has a
                # track: segmentation briefly split one object into fragments.
                # Spawning a track for it would invent a second identity for
                # something we already know is one object.
                continue
            self._create_track(detection)

        return {track_id: track for track_id, track in self.tracks.items()
                if track.missed == 0 and track.hits >= self.min_hits}


def build_tracker(config):
    params = config.get("tracker", {})
    return SignatureTracker(
        max_match_distance=params.get("max_match_distance", 90),
        max_missed_frames=params.get("max_missed_frames", 300),
        min_hits=params.get("min_hits", 2),
        velocity_damping=params.get("velocity_damping", 0.8),
        appearance_weight=params.get("appearance_weight", 1.0),
    )
