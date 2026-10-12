"""Extract the flash onsets of a session again from its raw analog channel.

Why: the flash onsets of GRB006 `20240821_121447` in the database
(`DatasetEvents.Digital`, stream `nidq`, event `ai0`) came from a local table
of trials (`trial_ts.pkl`). The old preprocessing detected every flash on the
analog channel, but removed the early-withdrawal trials from that table before
the onsets were written. So the row has the flashes of the completed trials
only. This tool detects the flashes with the same method and writes all of
them.

Method (the old one, not changed, so the onsets of completed trials stay the
same): the NI analog channel 0 is mapped to probe time through the sync
pulses (`spks.sync.load_ni_sync_data`), and an onset is a crossing of
`AMP_THRESHOLD` that comes more than `MIN_GAP_S` after the crossing before it.

Checks before any write: every onset that is in the database now must be
within `MATCH_TOLERANCE_S` of a detected onset. The tool prints how many
onsets are new and in which trials (completed, early withdrawal). Without
`--apply` it only prints. With `--apply` it saves the old onsets to
`--backup`, then replaces the database row.

Run it where the raw session folder is (the folder with the NI and probe
files of the recording):

    uv run python -m thesis.ephys.tools.reextract_flash_onsets --session-path <folder>
    uv run python -m thesis.ephys.tools.reextract_flash_onsets --session-path <folder> --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thesis.ephys.trials import build_trial_table

AMP_THRESHOLD = 5000
MIN_GAP_S = 0.04
MATCH_TOLERANCE_S = 0.001
ANALOG_CHANNEL = 0
SYNC_PORT = 0
PROBE_SYNC_KEY = 6


def probe_sync(session_path: Path, session_key: dict) -> tuple[np.ndarray, float]:
    """Return the sync onsets of the probe (samples) and its sampling rate.

    The old preprocessing read them from the `*.metadata.hdf` file that spks
    writes next to the sorted data. When that file is not in the session
    folder, the onsets come from the database (stream `imec0`, event
    `PROBE_SYNC_KEY`) and the rate from the `*.ap.meta` file of the probe.
    """
    import spks

    if list(session_path.glob("**/*.*.metadata.hdf")):
        saved = spks.sync.load_ap_sync_data(str(session_path))[0]
        return saved["file0_sync_onsets"][PROBE_SYNC_KEY], saved["sampling_rate"]
    from labdata.schema import DatasetEvents

    onsets = (
        DatasetEvents.Digital()
        & {**session_key, "stream_name": "imec0", "event_name": str(PROBE_SYNC_KEY)}
    ).fetch1("event_timestamps")
    meta_files = sorted(session_path.glob("**/*.ap.meta"))
    if not meta_files:
        raise SystemExit(f"No *.ap.meta file below {session_path}.")
    rate = next(
        float(line.split("=", 1)[1])
        for line in meta_files[0].read_text().splitlines()
        if line.startswith("imSampRate=")
    )
    print(f"Probe sync from the database; rate {rate} Hz from {meta_files[0].name}.")
    return np.asarray(onsets, dtype=float), rate


def detect_onsets(session_path: Path, session_key: dict) -> np.ndarray:
    """Return the flash onsets in probe time (s), with the old preprocessing method."""
    import spks
    from spks.spikeglx_utils import load_spikeglx_binary

    ni_files = sorted(session_path.glob("**/*.nidq.bin"))
    if not ni_files:
        raise SystemExit(f"No *.nidq.bin file below {session_path}.")
    ni_data, _ = load_spikeglx_binary(ni_files[0])
    ni_onsets, _ = spks.sync.unpackbits_gpu(ni_data[:, -1], device="cpu")
    ni_sync = np.asarray(ni_onsets[SYNC_PORT], dtype=float)
    probe_onsets, rate = probe_sync(session_path, session_key)
    print(f"Sync pulses: {len(ni_sync)} on the NI, {len(probe_onsets)} on the probe.")
    if len(ni_sync) != len(probe_onsets):
        # The check against the database onsets finds a wrong pairing: it gives
        # a shift of one sync period.
        shared = min(len(ni_sync), len(probe_onsets))
        print(f"The counts differ; the first {shared} pulses of each are paired.")
        ni_sync, probe_onsets = ni_sync[:shared], probe_onsets[:shared]
    ni_to_probe = spks.sync.interp1d(ni_sync, probe_onsets, fill_value="extrapolate")
    times = ni_to_probe(np.arange(len(ni_data))) / rate
    crossings = np.where(np.diff(ni_data[:, ANALOG_CHANNEL] > AMP_THRESHOLD) == 1)[0]
    kept = crossings[np.diff(np.hstack([0, crossings])) > MIN_GAP_S * rate]
    return times[kept]


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
    session_key = {k: key[k] for k in ("subject_name", "session_name", "dataset_name")}
    detected = np.sort(detect_onsets(args.session_path, session_key))
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
    relation.delete(force=True)
    DatasetEvents.Digital().insert1(
        {
            **key,
            "event_timestamps": detected,
            "event_values": np.ones(len(detected), dtype=np.uint8),
        },
        allow_direct_insert=True,
    )
    print(
        f"Replaced {key['stream_name']}:{key['event_name']} with {len(detected)} onsets."
    )


if __name__ == "__main__":
    main()
