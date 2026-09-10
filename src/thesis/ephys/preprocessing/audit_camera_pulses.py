"""Validate NIDQ camera pulses and save the accepted falling-edge frame mapping.

This is a read-only audit of the GRB006 back-view candidate frame signal.
Gabriel identifies NIDQ channel 1 as frame timing for the GPIO-synchronized
camera pair. Remove the one-sample internal low pulse only when that makes the
retained falling-edge count equal the decoded frame count. This script does not
change database events or use AVI presentation timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def select_falling_edges(
    falling: np.ndarray, rising: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Remove sub-percent-width pulses only when the frame count then matches."""
    widths = rising - falling
    retained = widths >= np.median(widths) / 100
    if int(retained.sum()) != n_frames:
        raise ValueError("Retained falling-edge count does not match video frames.")
    return falling[retained], retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays_path = args.output.with_suffix(".npz")
    frame_times_path = args.output.with_name("frame_times.npy")
    if args.output.exists() or arrays_path.exists() or frame_times_path.exists():
        raise FileExistsError(args.output)
    from labdata.schema import DatasetEvents, DatasetVideo, StreamSync

    key = {"subject_name": "GRB006", "session_name": "20240821_121447"}
    row = (
        DatasetEvents.Digital & key & {"stream_name": "nidq", "event_name": "1"}
    ).fetch1()
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
    frame_times, retained = select_falling_edges(
        falling, rising, int(video["n_frames"])
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
        "channel": "nidq/1",
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
