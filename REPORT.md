# Technical Report

## Problem selected

The mandatory contract (goods A/B flow across configurable zones, with
deduplicated events and traceable evidence), plus the extension:
**queue/dwell analytics** — per-object dwell-time thresholds and live
occupancy counting for a named waiting zone, validated against
`S04_DWELL_QUEUE`.

## Architecture

```
video → AppearanceChannelDetector.detect(frame) → detections (+ signature identity)
      → SignatureTracker.update(detections)      → confirmed tracks
      → ZoneFlow.update(track, centroid, t)      → route events (A ↔ B), deduplicated
      → DwellQueueAnalyzer.update(centroids, t)  → dwell + occupancy events
      → EvidenceWriter                           → jsonl / json / csv / counts /
                                                     snapshots / occupancy series
```

Each stage exchanges plain dictionaries (`bbox`, `centroid`, `confidence`,
`object_type`, `mean_color`, `signature_id`), so the tracker, zone logic and
evidence writer never need to know how a detection was produced. Swapping the
detector is a config change, not a rewrite.

## The core problem, and the decision that shaped this design

The hard scenario is `S03_DENSE_CROSSING`: five objects converge on the same
few hundred pixels simultaneously. Background subtraction returns one merged
contour, and the pipeline must decide which object each blob belongs to.

The conventional answer is to detect a generic blob and disambiguate
afterwards. I built exactly that first, and pushed it hard: an appearance
(colour) term in the assignment cost, a wider matching gate, track "revival"
from a graveyard of recently-lost tracks, distance-transform + watershed blob
splitting, optical-flow motion splitting, and finally a global offline pass
that stitched tracklets across the whole video. Each was measured against the
challenge's own evaluator rather than assumed.

**Every one of them hit the same wall, and the measurements are what made the
wall visible.** Optical-flow splitting worked mechanically — instrumentation
showed it correctly split every blob it was asked to, 7 out of 7 — and moved
the score not at all. The global tracklet-stitching pass, which is the
textbook-correct approach for batch tracking, actually scored *worse* than the
greedy tracker it replaced. Tracing that told me why: the greedy tracker's
apparent win on one event came from latching onto a wrong-coloured object that
happened to continue toward the right zone. The principled method refused to
make that leap, because the evidence genuinely didn't support it — and lost an
event by being right.

That is the real finding: **while two objects fully overlap, "which object owns
this pixel" has no answer derivable from that frame.** Every technique above
was a more elaborate way of guessing. A guess that happens to be correct is
indistinguishable from one that is wrong, which is why more sophistication
produced no improvement.

So this solution does not disambiguate better — it avoids creating the
ambiguity. Each object here carries a stable appearance signature for its
whole lifetime. The detector learns those signatures from clean observations,
then treats each one as its own detection channel: a merged blob is split by
*which signature each pixel matches*. A pixel only ever shows one colour, so
ownership has a real answer. Overlap stops being an identity problem and
becomes ordinary temporary occlusion, which a tracker already handles.

Result: **S03 went from 3 of 5 route events to 5 of 5**, and the other three
scenarios stayed exact.

## The assumption this rests on, stated plainly

This works because objects in this domain have stable, mutually
distinguishable appearance signatures. Nothing reads ground truth and nothing
is hard-coded — signatures are discovered at runtime from the video itself,
and the same code runs unchanged on all four scenarios. But the mechanism is
only as strong as that assumption.

Where it holds (uniformly-coloured goods, packages, containers, vehicles in
distinct liveries) it is very strong. Where objects genuinely share an
appearance — a crowd in similar clothing — it degrades to `fallback_split`,
geometric splitting of a merged blob, which is the conventional approach and
no better than a standard pipeline at a dense crossing. That fallback is why
the pipeline still functions on such footage rather than failing; it just
loses the advantage.

I would rather state that limit clearly than let a benchmark number imply a
generality the method doesn't have.

## Detector

`AppearanceChannelDetector` (`src/detector.py`). MOG2 background subtraction
with explicit shadow rejection and open/close morphology produces a foreground
mask. A contour meaningfully larger than the running median single-object area
is treated as merged and split per appearance channel; otherwise it is one
object, and its colour is offered to the signature bank.

Three properties of the signature bank were each forced by a measured failure:

**Signatures are frozen once confirmed.** My first implementation blended each
new observation into the stored signature. During an overlap a signature
drifts toward its neighbour, and a drifted signature then matches that
neighbour even more readily — a feedback loop that silently merges two
identities. Measured: five genuinely distinct signatures collapsed into two. A
signature is a fixed property of an object, so it is now stored once and never
updated.

**A new signature must prove itself.** Where two objects overlap, boundary
pixels blend into intermediate colours belonging to no real object. A
candidate must be observed `signature_confirm_hits` times before registering.

**A signature may only be learned from a blob that looks like one object.**
Confirmation alone was not enough: a *partially* occluded pair sits below the
merge-area threshold, so it was treated as a single object and its blended
colour registered as a phantom identity. S02 learned 8 identities for 3 real
objects. The detector now measures each blob's *dominant-colour fraction* —
the share of pixels within `signature_match_threshold` of the blob's median
colour — and refuses to learn from a blob below `signature_min_dominant_fraction`.
Measured separation was clean and wide: real objects scored 0.62–0.75, blended
blobs 0.05–0.37, with the threshold at 0.50. Identity counts became exact:
3/3, 3/3, 5/5, 3/3.

That last fix also motivated using the **median** rather than the mean colour.
Morphological closing dilates the mask slightly, so every blob carries
background pixels along its edge and a mean drags toward the background — for
one object, mean `(195,146,98)` versus median `(227,148,79)`, which is the
object's actual colour.

## Tracker

`SignatureTracker` (`src/tracker.py`). Detections carrying a signature bind
directly to that signature's track: the identity decision was already made
from pixel evidence, so no proximity heuristic is involved. Detections without
one — the fallback path — go through optimal (Hungarian) assignment over a
distance/IoU/appearance cost, which is conventional behaviour.

Two consequences are worth calling out:

**Occlusion tolerance can be generous here.** A conventional tracker must cap
how long it waits for a missing track, because a long-missing track will
happily snap onto whatever unrelated detection drifts into its widening search
radius. A signature-bound track can only ever re-acquire *its own* object, so
that risk doesn't exist and `max_missed_frames` can be large. This matters
concretely: at 45 frames, objects buried under others mid-crossing were being
deleted and recreated as fresh tracks, losing the zone history needed to
recognise their crossing.

**An object keeps one identity from its first sighting.** A signature needs a
few frames to confirm, so an object is briefly tracked without one. Creating a
new track the moment its signature confirmed left two identities for one
object — visible in S04 as 4 dwell events for 3 objects. A newly confirmed
signature is now adopted by the track already following that object.

## Zone logic and the dwell/queue extension

`ZoneFlow` (`src/zones.py`) turns tracked movement into deduplicated route
events, with three deliberate properties:

**Zone memory survives the gap between zones.** An object crossing A to B
spends most of its journey inside neither. My first implementation reset the
remembered zone on leaving A, so by the time the object reached B the origin
was forgotten and the only observable transitions were "nothing → A" and
"A → nothing" — an A-to-B crossing could never be detected at all, and the
pipeline reported zero events despite tracking every object correctly. The
remembered zone now updates only when an object settles inside a *named* zone;
open space is transit, not a destination.

**A zone must be held, not merely touched** (`stability_frames`), so a box
jittering on a boundary can't emit a burst of spurious crossings.

**Deduplication is time-based, not permanent.** Recording each
(track, event type) pair once forever also discards the second *legitimate*
crossing of the same kind — someone walking back out the way they came in. A
cooldown suppresses duplicates from one physical crossing while allowing a
genuine repeat later.

`DwellQueueAnalyzer` deliberately does not reuse route classification. A route
event describes a journey, and an object pausing in a waiting area on its way
from A to B must still count as one `A_TO_B` — not `A_TO_QUEUE` plus
`QUEUE_TO_B`. The queue zone is therefore excluded from route classification
and handled separately, testing membership directly. `grace_frames` tolerates
brief detection gaps so one missed frame doesn't reset a dwell clock that has
been running for seconds.

## Evidence output

Follows `nop_reference/evidence_contract.json` — `observation_id`,
`camera_id`, `observed_at`, `object_type`, `event_type`, `track_id`,
`confidence`, `attributes`, `evidence.image_path`, `model.name/version` — plus
`scenario_id` and `video_time_seconds` as shown in the candidate guide's
example, so one record satisfies both.

Every event produces a JSONL line, an entry in a cumulative `events.json`, a
row in `events.csv`, an updated `counts.csv`, and a JPEG snapshot.
`queue_occupancy.csv` is a periodic occupancy series, independent of threshold
events, for the "occupancy over time" view a dashboard would plot.

All output files are created at run start rather than on first write. A video
that legitimately produces no events must still leave a valid, empty
`counts.csv` and `events.json` — those are required outputs, not side effects
of something having happened.

## Accuracy

Measured with the challenge's own `tools/evaluate_events.py` against
`data/ground_truth/public_event_truth.csv`:

| Scenario | Expected | Matched | Identities / real objects |
|---|---:|---:|---|
| S01_BASIC_GOODS | 3 | 3 | 3 / 3 |
| S02_OCCLUSION_REVERSAL | 3 | 3 | 3 / 3 |
| S03_DENSE_CROSSING | 5 | 5 | 5 / 5 |
| S04_DWELL_QUEUE | 3 | 3 | 3 / 3 |

**Route events: `matched=14, missed=0, false_positive=0` — precision 1.00,
recall 1.00, F1 1.00.**

Scoring the complete event stream instead reports 5 additional false
positives. Those are the `DWELL_THRESHOLD` (3) and `QUEUE_THRESHOLD` (2)
extension events, which the base ground truth does not model — it knows only
`A_TO_B` and `B_TO_A`. They are extension output, not incorrect route counts.

**Identity quality**: exactly one learned identity per real object in every
scenario. Since track fragmentation was the mechanism behind the original
S03 failure, this is the diagnostic worth reporting alongside event counts.

**Reproducibility**: `cv2.setRNGSeed(42)` is set because `cv2.kmeans` (used by
the geometric fallback) otherwise seeds from OpenCV's unseeded global RNG,
which can silently change event counts between runs of the same input.
Verified: repeated full runs produce byte-identical `events.csv`.

## Runtime performance (CPU only, 1280×720)

| Scenario | Frames | Wall time | Processing FPS |
|---|---:|---:|---:|
| S01 | 600 | ~21s | ~28 |
| S02 | 750 | ~16s | ~46 |
| S03 | 700 | ~17s | ~41 |
| S04 | 800 | ~19s | ~41 |

Measured on a shared development machine, not a dedicated benchmark rig.
Throughput is load-dependent and moved between roughly 28 and 58 fps across
runs of identical code and input, so treat these as indicative rather than a
tight figure — and re-measure before quoting them. The relative ordering
between scenarios is not stable either, for the same reason. Every run stayed
above the 25 fps source rate.

The dominant-colour computation scans every blob pixel and costs meaningful
throughput versus a plain mean; that was a deliberate trade for exact identity
counts, and could be recovered by sampling. Exact figures for the current
output are in each `output/S0N/performance.json`.

## Known failure cases and limits

**Objects that share an appearance.** The central assumption. Two objects with
genuinely indistinguishable colour fall back to geometric splitting, which is
where the earlier approach already plateaued at 3/5 on a dense crossing. The
method removes the ambiguity when appearance separates objects; it cannot
manufacture a distinction that isn't in the pixels.

**Sustained full occlusion.** A completely hidden object produces no
detection for that signature — correctly, since reporting a position for
something invisible would be fabrication. The tracker carries it on predicted
position, but a very long occlusion combined with a direction change will
place it wrongly on reappearance. `max_missed_frames` bounds this.

**Stationary objects fade into the background.** MOG2 adapts a genuinely
still object into its background model — by design, since that is how it
handles lighting drift — which breaks "pause in place" behaviour. Mitigated
with an explicit low `learning_rate` (0.003) so a multi-second dwell survives,
at the cost of slower adaptation to real background change. This is why S04's
queue dwell and S02's stop-and-resume work.

**Camera motion.** Background subtraction assumes a mostly-static camera.

## What I would improve with more time

- **A learned appearance embedding** (a small CNN feature vector) instead of a
  colour signature. The architecture already treats identity as "a stable
  signature that segments a merged blob"; a learned descriptor would slot into
  the same interface and extend the approach to textured real-world objects
  that colour alone cannot separate.
- **A Kalman filter** instead of damped-average velocity, for better
  prediction through long occlusions and a principled gate radius.
- **Recover the throughput** lost to the dominant-colour scan by sampling
  pixels rather than scanning all of them.
- **An end-to-end regression test** asserting each scenario's ground-truth
  event count, so an accuracy regression fails CI rather than being noticed by
  eye. The current tests cover units well but stop short of the full pipeline.
- **A semantic classifier** for deployments needing `person`/`car` labels
  rather than tracked motion; it would sit behind the same detector interface.

Explicitly ruled out: reading trajectories from `data/scenarios/*.json`. It
would trivially "solve" all four public videos and score zero on unseen
footage, which is exactly the hard-coding the challenge rules call out.

## AI development tools used

Built with **Claude Code** (Anthropic), used for: reading the challenge
documents, writing the source and tests, and — most importantly — running the
measurements that drove the design. The central architectural decision came
directly from instrumented negative results: watching optical-flow splitting
succeed mechanically while changing nothing, and watching a textbook-correct
global stitching pass score worse than the greedy tracker it replaced, is what
established that the bottleneck was ambiguity itself rather than the quality
of the disambiguation. Every claim in this report was verified against actual
runs of the challenge's own evaluator.

## Third-party components and licenses

See `THIRD_PARTY.md`.
