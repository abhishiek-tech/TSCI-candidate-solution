"""Zone logic: route events (A <-> B) and the dwell/queue analytics extension."""

from geometry import locate_point


class ZoneFlow:
    """Turns tracked movement into deduplicated route events.

    Three properties keep the event stream honest:

    **Zone memory survives the gap between zones.** An object crossing from A
    to B spends most of its journey inside neither zone. If leaving A resets
    the remembered zone, then by the time the object reaches B the origin has
    been forgotten and the only observable transitions are "nothing -> A" and
    "A -> nothing" -- an A-to-B crossing can never be seen at all. So the
    remembered zone updates only when the object settles inside a *named*
    zone; open space between zones is transit, not a destination.

    **A zone must be held, not merely touched.** A bounding box jittering on a
    zone boundary would otherwise emit a burst of spurious transitions, so a
    candidate zone must persist for ``stability_frames`` before it counts.

    **Deduplication is time-based, not permanent.** Recording each
    (track, event_type) pair once forever also discards the second
    *legitimate* crossing of the same kind -- someone walking back out the way
    they came in. A cooldown suppresses duplicates from one physical crossing
    while still allowing a genuine repeat later.
    """

    def __init__(self, zones, stability_frames=3, cooldown_seconds=1.5):
        self.zones = zones
        self.stability_frames = stability_frames
        self.cooldown_seconds = cooldown_seconds
        self._state = {}

    def _state_for(self, track_id):
        if track_id not in self._state:
            self._state[track_id] = {
                "zone": None,  # last CONFIRMED named zone; persists through transit
                "candidate": None,
                "candidate_frames": 0,
                "last_event_at": {},
            }
        return self._state[track_id]

    def update(self, track_id, centroid, video_time_seconds):
        """Feed one tracked position; returns (current_zone, event or None)."""
        state = self._state_for(track_id)
        located = locate_point(centroid, self.zones)

        if located is None:
            # In transit between zones: reset the stability candidate but keep
            # the remembered origin zone intact.
            state["candidate"] = None
            state["candidate_frames"] = 0
            return state["zone"], None

        if located == state["candidate"]:
            state["candidate_frames"] += 1
        else:
            state["candidate"] = located
            state["candidate_frames"] = 1

        if state["candidate_frames"] < self.stability_frames:
            return state["zone"], None
        if located == state["zone"]:
            return state["zone"], None

        previous_zone = state["zone"]
        state["zone"] = located
        if previous_zone is None:
            # First zone an object is seen in is its origin, not a crossing.
            return located, None

        event_type = f"{previous_zone}_TO_{located}"
        last_at = state["last_event_at"].get(event_type)
        if last_at is not None and video_time_seconds - last_at < self.cooldown_seconds:
            return located, None

        state["last_event_at"][event_type] = video_time_seconds
        return located, {
            "event_type": event_type,
            "source_zone": previous_zone,
            "destination_zone": located,
        }

    def current_zone(self, track_id):
        return self._state_for(track_id)["zone"]


class DwellQueueAnalyzer:
    """Per-track dwell timing and live occupancy for one named zone.

    Deliberately does *not* reuse ``ZoneFlow``'s route classification. Route
    events describe a journey (A to B) and must not be split into
    A-to-QUEUE plus QUEUE-to-B just because the object passed through a
    waiting area on the way. So the queue zone is excluded from route
    classification and handled here instead, testing membership directly and
    frame-by-frame.

    ``grace_frames`` tolerates brief detection gaps so a single missed frame
    doesn't reset a dwell clock that has been running for seconds.
    """

    def __init__(self, zone_polygon, queue_zone, dwell_threshold_seconds=5.0,
                 occupancy_threshold=None, grace_frames=5):
        self.zone_polygon = zone_polygon
        self.queue_zone = queue_zone
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.occupancy_threshold = occupancy_threshold
        self.grace_frames = grace_frames
        self._dwell = {}
        self._occupancy_active = False

    def locate(self, centroid):
        return locate_point(centroid, {self.queue_zone: self.zone_polygon}) is not None

    def current_dwell(self, track_id, video_time_seconds):
        entry = self._dwell.get(track_id)
        if entry is None:
            return 0.0
        return max(0.0, video_time_seconds - entry["entered_at"])

    def update(self, track_centroids, video_time_seconds):
        """Returns (occupancy_count, dwell_events, occupancy_event or None)."""
        inside_now = set()
        for track_id, centroid in track_centroids.items():
            if self.locate(centroid):
                inside_now.add(track_id)

        dwell_events = []
        for track_id in inside_now:
            entry = self._dwell.get(track_id)
            if entry is None:
                self._dwell[track_id] = {
                    "entered_at": video_time_seconds,
                    "absent_frames": 0,
                    "fired": False,
                }
                continue
            entry["absent_frames"] = 0
            elapsed = video_time_seconds - entry["entered_at"]
            if not entry["fired"] and elapsed >= self.dwell_threshold_seconds:
                entry["fired"] = True
                dwell_events.append({
                    "event_type": "DWELL_THRESHOLD",
                    "track_id": track_id,
                    "zone": self.queue_zone,
                    "dwell_seconds": round(elapsed, 2),
                })

        for track_id in list(self._dwell):
            if track_id in inside_now:
                continue
            entry = self._dwell[track_id]
            entry["absent_frames"] += 1
            if entry["absent_frames"] > self.grace_frames:
                del self._dwell[track_id]

        occupancy = len(inside_now)
        occupancy_event = None
        if self.occupancy_threshold is not None:
            above = occupancy >= self.occupancy_threshold
            if above and not self._occupancy_active:
                # Rising edge only: one event per breach, not one per frame.
                occupancy_event = {
                    "event_type": "QUEUE_THRESHOLD",
                    "zone": self.queue_zone,
                    "occupancy": occupancy,
                }
            self._occupancy_active = above

        return occupancy, dwell_events, occupancy_event
