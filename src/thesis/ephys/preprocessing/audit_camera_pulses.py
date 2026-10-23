"""Validate the camera pulses of a session and save one time for each video frame.

This is a read-only audit of the frame signal of the GPIO-synchronized camera
pair (the `frames` event of the session: NIDQ channel 1 for GRB006, OneBox
`io3` for later animals). A frame time is the falling edge of its pulse. The
pulses are accepted only when their count is equal to the decoded frame count
of cam0 after these repairs, each of which is written to the output:

- Glitch: a pulse with a width below 1% of the median is removed (GRB006).
- Pulses before the recording: when a gap of `START_GAP_S` or more is in the
  first `START_SEARCH` pulses, the pulses before it are removed (the camera
  runs for a short time before it saves frames; GRB064).
- Irregular pulses: where two regular pulses are more than `IRREGULAR` frame
  intervals apart or closer than the inverse, the pulses between the regular
  ones on each side are replaced with evenly spaced times (GRB064 has one such
  place, where three pulses are missing and one is extra).

The check against the camera times that the video file stores (mapped through
the Bpod `StreamSync` row) is in the output. This script does not change
database events or use AVI presentation timestamps.

    uv run python -m thesis.ephys.preprocessing.audit_camera_pulses \
        --subject GRB064 --session 20260916_132939 \
        --output results/glm/GRB064_20260916_132939/camera_pulses.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

START_GAP_S = 0.5
START_SEARCH = 1000
IRREGULAR = 1.5


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
    from labdata.schema import DatasetEvents, DatasetVideo, StreamSync

    from thesis.ephys.events import _find_sess_ev_sources

    key = {"subject_name": args.subject, "session_name": args.session}
    source = _find_sess_ev_sources(args.subject, args.session, allow_incomplete=True)[
        "frames"
    ]
    row = (DatasetEvents.Digital & source).fetch1()
    times = np.asarray(row["event_timestamps"], dtype=float)
    values = np.asarray(row["event_values"])
    if times.shape != values.shape or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected timestamp-matched binary camera edges.")
    if np.any(np.diff(times) <= 0) or not np.all(values[1:] != values[:-1]):
        raise ValueError("Camera edges must alternate in increasing time order.")
    if values[0] != 0 or values[-1] != 1:
        raise ValueError("This audit expects complete low-then-high pulse pairs.")
    falling, rising = times[values == 0], times[values == 1]
    widths = rising - falling
    video = (DatasetVideo & key & {"video_name": "cam0"}).fetch1()
    # Drop sub-percent-width pulses only when the frame count then matches.
    retained = widths >= np.median(widths) / 100
    frame_times, repairs = falling[retained], {}
    if len(frame_times) != int(video["n_frames"]):
        frame_times, repairs = regular_frame_times(frame_times)
    if len(frame_times) != int(video["n_frames"]):
        raise ValueError(
            f"{len(frame_times)} frame pulses after the repairs, "
            f"{int(video['n_frames'])} video frames."
        )
    removed = np.flatnonzero(~retained)
    sync = (
        StreamSync()
        & key
        & {"dataset_name": "chipmunk", "stream_name": "bpod", "event_name": "sync"}
    )
    # Convert only the stored Bpod timestamps for this comparison, never NIDQ.
    reference = sync.apply(video["frame_times"], force=True, warn=False)
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
        "first_frame_association": "first retained pulse",
        "frame_times": str(frame_times_path),
        "falling_minus_stored_camera_s_quantiles": np.quantile(
            residual, [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
        "largest_adjacent_residual_change_s": float(np.max(abs(np.diff(residual)))),
        "candidate_arrays": str(arrays_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    with arrays_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            allow_pickle=False,
            falling_s=falling,
            rising_s=rising,
            retained=retained,
            stored_camera_neural_s=reference,
        )
    with frame_times_path.open("xb") as handle:
        np.save(handle, frame_times, allow_pickle=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
