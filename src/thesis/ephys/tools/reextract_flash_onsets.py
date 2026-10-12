"""Add the flash onsets of early-withdrawal trials from the raw analog channel.

Why: the flash onsets of GRB006 `20240821_121447` in the database
(`DatasetEvents.Digital`, stream `nidq`, event `ai0`) came from a local table
of trials (`trial_ts.pkl`). The old preprocessing detected every flash on the
analog channel, but removed the early-withdrawal trials from that table before
the onsets were written. So the row has the flashes of the completed trials
only. This tool detects the flashes again and adds the ones that are not there.

Method: an onset is a crossing of `AMP_THRESHOLD` on NI analog channel 0 that
comes more than `MIN_GAP_SAMPLES` after the crossing before it (the old
detection, not changed). Its time is the NI sample divided by the NI sampling
rate. The NI is the reference clock of the session: the other `nidq` events
are on it, and `StreamSync` maps the probe and the Bpod to it, so the flashes
need no mapping. (The old preprocessing mapped them to probe time with the
measured probe rate, which agrees with the NI clock to about 0.2 ms.)

Checks before any write:
- The sync pulses that the tool reads on the NI must be within
  `MATCH_TOLERANCE_S` of the `nidq:0` pulses in the database (the same clock).
- Every onset that is in the database now must be within `MATCH_TOLERANCE_S`
  of a detected onset.
The tool prints how many onsets are new and in which trials. Without `--apply`
it only prints. With `--apply` it saves the old onsets to `--backup`, then
writes the row again as the old onsets, not changed, plus the new ones. So the
flashes of the completed trials stay exactly as they are.

Run it where the raw session folder is (the folder with the NI file):

    uv run python -m thesis.ephys.tools.reextract_flash_onsets --session-path <folder>
    uv run python -m thesis.ephys.tools.reextract_flash_onsets --session-path <folder> --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thesis.ephys.trials import build_trial_table

AMP_THRESHOLD = 5000
# The old detection: 0.04 s at the probe rate (30 kHz), counted in NI samples.
MIN_GAP_SAMPLES = 1200
MATCH_TOLERANCE_S = 0.001
ANALOG_CHANNEL = 0
SYNC_PORT = 0


def detect_onsets(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the flash onsets and the sync pulses of the NI, on the NI clock (s)."""
    import spks
    from spks.spikeglx_utils import load_spikeglx_binary

    ni_files = sorted(session_path.glob("**/*.nidq.bin"))
    if not ni_files:
        raise SystemExit(f"No *.nidq.bin file below {session_path}.")
    rate = next(
        float(line.split("=", 1)[1])
        for line in ni_files[0].with_suffix(".meta").read_text().splitlines()
        if line.startswith("niSampRate=")
    )
    print(f"NI sampling rate: {rate} Hz ({ni_files[0].name}).")
    ni_data, _ = load_spikeglx_binary(ni_files[0])
    sync_onsets, _ = spks.sync.unpackbits_gpu(ni_data[:, -1], device="cpu")
    crossings = np.where(np.diff(ni_data[:, ANALOG_CHANNEL] > AMP_THRESHOLD) == 1)[0]
    kept = crossings[np.diff(np.hstack([0, crossings])) > MIN_GAP_SAMPLES]
    return kept / rate, np.asarray(sync_onsets[SYNC_PORT], dtype=float) / rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-path", type=Path, required=True)
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    parser.add_argument("--dataset", default="ephys_g0")
    parser.add_argument("--backup", type=Path, default=Path("flash_onsets_before.npy"))
    parser.add_argument("--apply", action="store_true", help="Write to the database.")
    args = parser.parse_args()

    from labdata.schema import DatasetEvents

    key = {
        "subject_name": args.subject,
        "session_name": args.session,
        "dataset_name": args.dataset,
        "stream_name": "nidq",
        "event_name": "ai0",
    }
    relation = DatasetEvents.Digital() & key
    current = np.sort(np.asarray(relation.fetch1("event_timestamps"), dtype=float))
    detected, ni_sync = detect_onsets(args.session_path)
    detected = np.sort(detected)
    saved_times, saved_values = (
        DatasetEvents.Digital() & {**key, "event_name": str(SYNC_PORT)}
    ).fetch1("event_timestamps", "event_values")
    saved_sync = np.asarray(saved_times, dtype=float)[np.asarray(saved_values) == 1]
    if len(saved_sync) != len(ni_sync):
        raise SystemExit(
            f"Stopped: {len(ni_sync)} sync pulses in the file, {len(saved_sync)} in "
            "the database."
        )
    clock_error = np.abs(ni_sync - saved_sync).max()
    print(
        f"Sync pulses: {len(ni_sync)}; largest distance to the database pulses "
        f"{1000 * clock_error:.4f} ms."
    )
    if clock_error > MATCH_TOLERANCE_S:
        raise SystemExit("Stopped: the file and the database are not on one clock.")
    # The detected onset that is nearest to each database onset.
    right = np.searchsorted(detected, current).clip(1, len(detected) - 1)
    index = np.where(
        np.abs(detected[right - 1] - current) <= np.abs(detected[right] - current),
        right - 1,
        right,
    )
    nearest = np.abs(detected[index] - current)
    matched = nearest <= MATCH_TOLERANCE_S
    print(f"Onsets in the database now: {len(current)}")
    print(f"Onsets detected in the raw channel: {len(detected)}")
    print(
        f"Database onsets with a detected onset within {1000 * MATCH_TOLERANCE_S:.0f} ms: "
        f"{matched.sum()} of {len(current)} (largest distance "
        f"{1000 * nearest.max():.2f} ms)"
    )

    trials = build_trial_table(args.subject, args.session, include_frames=False)
    starts = trials["trial_start_s"].to_numpy(dtype=float)
    early = trials["early_withdrawal"].to_numpy() == 1
    is_new = np.ones(len(detected), dtype=bool)
    is_new[index[matched]] = False
    row = np.searchsorted(starts, detected, side="right") - 1
    inside = row >= 0
    new_rows = row[is_new & inside]
    print(f"New onsets: {int(is_new.sum())}")
    print(f"  in early-withdrawal trials: {int(early[new_rows].sum())}")
    print(f"  in completed trials: {int((~early[new_rows]).sum())}")
    print(f"  before the first trial start: {int((is_new & ~inside).sum())}")
    print(
        "Early-withdrawal trials with one new onset or more: "
        f"{len(np.unique(new_rows[early[new_rows]]))} of {int(early.sum())}"
    )

    # How the two sets of onsets differ, when the completed trials have the same
    # number in both: paired in order, as detected minus database.
    in_completed = np.sort(detected[inside][~early[row[inside]]])
    if len(in_completed) == len(current):
        shift = 1000 * (in_completed - current)
        slope, offset = np.polyfit(current, shift, 1)
        left_over = shift - (offset + slope * current)
        print(
            "Detected minus database, onsets of completed trials paired in order (ms):"
        )
        print(
            "  at 0, 10, ..., 100% of the session: "
            + ", ".join(
                f"{value:.2f}" for value in np.percentile(shift, range(0, 101, 10))
            )
        )
        print(
            "  in time order, every 10% of the onsets: "
            + ", ".join(f"{value:.2f}" for value in shift[:: max(len(shift) // 10, 1)])
        )
        print(
            f"  line fit: {offset:.3f} ms at time 0, {slope * 1000:.3f} ms per 1000 s "
            f"({slope * 1000:.2f} parts per million); what is left has SD "
            f"{left_over.std():.3f} ms, largest {np.abs(left_over).max():.3f} ms"
        )
    else:
        print(
            f"Completed trials: {len(in_completed)} detected onsets against "
            f"{len(current)} in the database; not paired."
        )

    if not args.apply:
        print("No write. Use --apply to replace the database row.")
        return
    if not matched.all():
        raise SystemExit(
            "Stopped: some onsets in the database have no detected onset. "
            "The clock or the method is not the same; nothing was written."
        )
    np.save(args.backup, current)
    print(f"Saved the old onsets to {args.backup}.")
    added = detected[is_new]
    onsets = np.sort(np.concatenate((current, added)))
    relation.delete(force=True)
    DatasetEvents.Digital().insert1(
        {
            **key,
            "event_timestamps": onsets,
            "event_values": np.ones(len(onsets), dtype=np.uint8),
        },
        allow_direct_insert=True,
    )
    print(
        f"Wrote {key['stream_name']}:{key['event_name']}: {len(current)} onsets not "
        f"changed and {len(added)} added ({len(onsets)} in total)."
    )


if __name__ == "__main__":
    main()
