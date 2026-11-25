"""Validate the camera pulses of a session and save one time for each video frame.

This is a read-only audit of the frame signal of the GPIO-synchronized camera
pair (the `frames` event of the session: NIDQ channel 1 for GRB006, OneBox
`io3` for later animals). A frame time is the falling edge of its pulse. The
pulses are accepted only when their count is equal to the decoded frame count
of cam0 after these repairs, each of which is written to the output:

- Glitch: a pulse with a width below 1% of the median is removed (GRB006).
- Preview: select a continuous run with exactly the video's frame count and
  agreement within one frame period of the independently Bpod-mapped camera
  times. Search all pauses, including long previews and acquisition restarts.
- One pulse short: when one saved frame at an end has no pulse, take the end
  whose alignment gives the smaller median offset to the mapped camera times
  (1.5-2 ms in every session; the other end gives one frame period more),
  require at most a quarter period, and place the missing frame one median
  interval beyond the pulses (GRB061 20261109, GRB068 20261110: first frame).
- Pulses before the recording: when a gap of `START_GAP_S` or more is in the
  first `START_SEARCH` pulses, the pulses before it are removed (the camera
  runs for a short time before it saves frames; GRB064: 215 pulses in
  20260916, 1302 pulses in 20261008). After this, one saved frame without a
  pulse is repaired as above (GRB064 20260921: the camera ran 983 pulses,
  restarted after 1.3 s, and cam0 saved one frame more than the pulses).
- Irregular pulses: where two regular pulses are more than `IRREGULAR` frame
  intervals apart or closer than the inverse, the pulses between the regular
  ones on each side are replaced with evenly spaced times (GRB064 has one such
  place, where three pulses are missing and one is extra).

The check against the camera times that the video file stores (mapped through
`thesis.ephys.events.bpod_to_clock`) is in the output. Its end can drift by a
few ms where the video runs past the last Bpod sync pulse (GRB068 20261109). This script does not change
database events or use AVI presentation timestamps.

    uv run python -m thesis.ephys.preprocessing.audit_camera_pulses \
        --subject GRB064 --session 20260916_132939 \
        --output results/glm/GRB064_20260916_132939/camera_pulses.json
    (and --session 20261008_143359 with its own output directory)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

START_GAP_S = 0.5
START_SEARCH = 2000
IRREGULAR = 1.5


def align_frame_times(
    falling: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Select a uniquely supported recording run, retaining legacy repairs."""
    falling = np.asarray(falling, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        reference.ndim != 1
        or len(reference) < 2
        or not np.isfinite(reference).all()
        or np.any(np.diff(reference) <= 0)
    ):
        raise ValueError("Expected increasing, finite camera reference times.")
    if len(falling) == len(reference):
        return falling, {}
    period = float(np.median(np.diff(reference)))
    intervals = np.diff(falling)
    regular = intervals.max() < IRREGULAR * np.median(intervals)
    if len(falling) == len(reference) - 1 and regular:
        # One saved frame without a pulse at an end (GRB061 20261109, GRB068
        # 20261110: the first one). The residual shape is the same for both
        # alignments, so only the median offset (1.5-2 ms in all sessions)
        # can tell which end it is.
        offsets = {
            "first": float(np.median(falling - reference[1:])),
            "last": float(np.median(falling - reference[:-1])),
        }
        end = min(offsets, key=lambda name: abs(offsets[name]))
        if abs(offsets[end]) > period / 4:
            raise ValueError(f"No unambiguous end for the missing pulse: {offsets}.")
        interval = float(np.median(intervals))
        if end == "first":
            frames = np.r_[falling[0] - interval, falling]
        else:
            frames = np.r_[falling, falling[-1] + interval]
        return frames, {
            "frame_without_pulse": end,
            "median_offset_by_missing_end_s": offsets,
            "irregular_places": [],
        }
    boundaries = np.r_[
        0, np.flatnonzero(np.diff(falling) >= START_GAP_S) + 1, len(falling)
    ]
    candidates = [
        (int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if stop - start == len(reference)
        and np.max(abs(falling[start:stop] - reference)) <= period
    ]
    if len(candidates) > 1:
        raise ValueError("Camera reference matches multiple recording runs.")
    if candidates:
        start, stop = candidates[0]
        return falling[start:stop], {
            "pulses_before_the_recording": start,
            "pulses_after_the_recording": len(falling) - stop,
            "irregular_places": [],
        }
    frames, repairs = regular_frame_times(falling)
    if (
        repairs["pulses_before_the_recording"]
        and not repairs["irregular_places"]
        and len(frames) >= 2
        and len(frames) == len(reference) - 1
    ):
        # A restart and then one saved frame without a pulse (GRB064 20260921).
        frames, end_repairs = align_frame_times(frames, reference)
        repairs = {**repairs, **end_repairs}
    if len(frames) != len(reference):
        raise ValueError(
            f"{len(frames)} frame pulses after the repairs, {len(reference)} video frames."
        )
    # Reference validation gates new run selection; keep historical repairs unchanged.
    return frames, repairs


def regular_frame_times(falling: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop the pulses before the recording and re-space irregular pulses."""
    notes = {"pulses_before_the_recording": 0, "irregular_places": []}
    early_gaps = np.flatnonzero(np.diff(falling[:START_SEARCH]) >= START_GAP_S)
    if len(early_gaps):
        notes["pulses_before_the_recording"] = int(early_gaps[-1] + 1)
        falling = falling[early_gaps[-1] + 1 :]
    period = float(np.median(np.diff(falling)))
    ratio = np.diff(falling) / period
    odd = np.flatnonzero((ratio > IRREGULAR) | (ratio < 1 / IRREGULAR))
    pieces, cursor = [], 0
    while len(odd):
        # One place: consecutive irregular intervals between two regular pulses.
        stop = odd[0]
        while stop + 1 in odd:
            stop += 1
        left, right = odd[0], stop + 1  # the regular pulses on each side
        steps = int(round((falling[right] - falling[left]) / period))
        pieces += [
            falling[cursor:left],
            np.linspace(falling[left], falling[right], steps, endpoint=False),
        ]
        notes["irregular_places"].append(
            {
                "after_pulse": int(left),
                "pulses_replaced": int(right - left - 1),
                "frames_placed": steps - 1,
                "span_s": float(falling[right] - falling[left]),
            }
        )
        cursor = right
        odd = odd[odd > stop]
    return np.concatenate([*pieces, falling[cursor:]]), notes


def align_video_frames(subject: str, session: str) -> tuple[np.ndarray, dict, dict]:
    """Return validated cam0 falling-edge times, audit notes and source arrays.

    Times are in the session's hardware clock, one per saved video frame.
    Missing video/sync records or unresolvable pulse counts raise; no raw-edge
    fallback is used. This reads the database without writing audit files.
    """
    from labdata.schema import DatasetEvents, DatasetVideo

    from thesis.ephys.events import _find_sess_ev_sources, bpod_to_clock

    key = {"subject_name": subject, "session_name": session}
    sources = _find_sess_ev_sources(subject, session, allow_incomplete=True)
    if "frames" not in sources:
        raise ValueError(f"No camera frame events for {subject} {session}.")
    source = sources["frames"]
    row = (DatasetEvents.Digital & source).fetch1()
    times = np.asarray(row["event_timestamps"], dtype=float)
    values = np.asarray(row["event_values"])
    if (
        times.ndim != 1
        or times.size < 4
        or not np.isfinite(times).all()
        or times.shape != values.shape
        or not np.isin(values, [0, 1]).all()
    ):
        raise ValueError(
            "Expected finite, timestamp-matched binary camera edges (at least two pulses)."
        )
    if np.any(np.diff(times) <= 0) or not np.all(values[1:] != values[:-1]):
        raise ValueError("Camera edges must alternate in increasing time order.")
    if values[0] != 0 or values[-1] != 1:
        raise ValueError("This audit expects complete low-then-high pulse pairs.")
    falling, rising = times[values == 0], times[values == 1]
    widths = rising - falling
    video = (DatasetVideo & key & {"video_name": "cam0"}).fetch1()
    # Drop sub-percent-width pulses only when the frame count then matches.
    retained = widths >= np.median(widths) / 100
    removed = np.flatnonzero(~retained)
    # Convert only the stored Bpod timestamps for this comparison, never NIDQ.
    reference = bpod_to_clock(subject, session)(
        np.asarray(video["frame_times"], dtype=float)
    )
    if len(reference) != int(video["n_frames"]):
        raise ValueError("Stored camera times do not match the video frame count.")
    frame_times, repairs = align_frame_times(falling[retained], reference)
    residual = frame_times - reference
    result = {
        **key,
        "channel": f"{source['stream_name']}/{source['event_name']}",
        "repairs": repairs,
        "video_name": "cam0",
        "video_frames": int(video["n_frames"]),
        "pulse_pairs": len(widths),
        "median_low_width_s": float(np.median(widths)),
        "removed_short_pulses": [
            {
                "pulse_index": int(i),
                "falling_s": float(falling[i]),
                "rising_s": float(rising[i]),
                "width_s": float(widths[i]),
            }
            for i in removed
        ],
        "frame_count": len(frame_times),
        "frame_interval_range_s": [
            float(x) for x in (np.diff(frame_times).min(), np.diff(frame_times).max())
        ],
        "last_low_width_s": float(widths[-1]),
        "mapping_accepted": True,
        "frame_edge": "falling",
        "first_frame_association": (
            "Bpod-mapped camera times and recording-run boundaries"
            if "pulses_after_the_recording" in repairs
            or "frame_without_pulse" in repairs
            else "first retained pulse"
        ),
        "falling_minus_stored_camera_s_quantiles": np.quantile(
            residual, [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
        "largest_adjacent_residual_change_s": float(np.max(abs(np.diff(residual)))),
    }
    return (
        frame_times,
        result,
        {
            "falling_s": falling,
            "rising_s": rising,
            "retained": retained,
            "stored_camera_neural_s": reference,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    args = parser.parse_args()
    arrays_path = args.output.with_suffix(".npz")
    frame_times_path = args.output.with_name("frame_times.npy")
    if args.output.exists() or arrays_path.exists() or frame_times_path.exists():
        raise FileExistsError(args.output)
    frame_times, result, arrays = align_video_frames(args.subject, args.session)
    result.update(frame_times=str(frame_times_path), candidate_arrays=str(arrays_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    with arrays_path.open("xb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    with frame_times_path.open("xb") as handle:
        np.save(handle, frame_times, allow_pickle=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
