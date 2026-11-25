"""Extract the UnitMatch inputs of one session: two half-session waveforms per unit.

UnitMatch (van Beest et al. 2024, Nat Methods) tracks units across sessions
that were sorted one by one. It needs, for each unit, the median raw waveform
on every channel, computed twice: from spikes in the first half of the
session and from spikes in the second half. The within-session agreement of
the two halves sets what a match across sessions must look like.

Units: those that pass unit criterion 1 (UnitCount), the same set as
`fetch_unit_table`. Spikes: up to 500 per unit, evenly spaced over the session
(the first 250 make half 1). Each 82-sample snippet (2.7 ms, trough at sample
41, the UnitMatch default) is cut from the raw `.ap.bin`, high-passed at 300 Hz
(NP 2.0 AP data is wideband), median-referenced across channels, and baseline
corrected on its first 20 samples. Values stay in ADC counts: one probe has one
gain in all sessions.

Output: `results/unit_match/<subject>_<session>.npz`. Step 2 runs UnitMatch on
the files of several sessions in its own environment (it needs numpy < 2):

    uv run unit-match-extract GRB064 20261007_141927 20261008_143359
    uv run src/thesis/ephys/preprocessing/unit_match_run.py \
        results/unit_match/GRB064_*.npz --out results/unit_match/GRB064
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

HALF_WIDTH = 41  # samples each side of the spike time
PAD = 150  # extra samples each side, so the filter edges fall outside the snippet
N_SPIKES = 500
OUT_DIR = Path("results/unit_match")


def _raw_file(subject: str, session: str, dataset: str, probe: int) -> Path:
    from labdata import prefs

    root = Path(prefs["local_paths"][0]) / subject / session / dataset
    (path,) = root.glob(f"*imec{probe}/*imec{probe}.ap.bin")
    return path


def _n_saved_channels(bin_path: Path) -> int:
    meta = bin_path.with_suffix(".meta").read_text().splitlines()
    return int(
        next(line for line in meta if line.startswith("nSavedChans=")).split("=")[1]
    )


def unit_waveforms(data, spike_samples, channels, sos) -> np.ndarray:
    """Return (82, n_channels, 2): median waveform of each session half."""
    n_samples = data.shape[0]
    spike_samples = np.sort(spike_samples.astype(np.int64))
    picks = spike_samples[
        np.linspace(
            0, len(spike_samples) - 1, min(N_SPIKES, len(spike_samples))
        ).astype(int)
    ]
    margin = HALF_WIDTH + PAD
    picks = picks[(picks >= margin) & (picks < n_samples - margin)]
    snippets = np.stack(
        [data[t - margin : t + margin, channels] for t in picks]
    ).astype(np.float32)
    snippets = sosfiltfilt(sos, snippets, axis=1)[:, PAD:-PAD]
    snippets -= np.median(snippets, axis=2, keepdims=True)
    snippets = gaussian_filter1d(snippets, 1, axis=1, radius=2)
    snippets -= snippets[:, :20].mean(axis=1, keepdims=True)
    half = len(picks) // 2
    return np.stack(
        [np.median(snippets[:half], axis=0), np.median(snippets[half:], axis=0)],
        axis=-1,
    ).astype(np.float32)


def extract_session(subject: str, session: str, unit_criteria_id: int = 1) -> Path:
    from labdata.schema import EphysRecording, SpikeSorting, UnitCount

    key = {"subject_name": subject, "session_name": session}
    sorting = (SpikeSorting & key).fetch1()
    channels = np.asarray(sorting["sorting_channel_indices"]).astype(int).ravel()
    channel_pos = np.asarray(sorting["sorting_channel_coords"], dtype=float)
    sampling_rate = float(
        (
            EphysRecording.ProbeSetting & key & {"probe_num": sorting["probe_num"]}
        ).fetch1("sampling_rate")
    )
    passing = UnitCount.Unit & key & {"unit_criteria_id": unit_criteria_id, "passes": 1}
    unit_ids, spike_times = (SpikeSorting.Unit & key & passing).fetch(
        "unit_id", "spike_times", order_by="unit_id"
    )

    bin_path = _raw_file(
        subject, session, sorting["dataset_name"], sorting["probe_num"]
    )
    data = np.memmap(bin_path, dtype=np.int16, mode="r").reshape(
        -1, _n_saved_channels(bin_path)
    )
    sos = butter(3, 300, btype="highpass", fs=sampling_rate, output="sos")
    waveforms = np.stack(
        [unit_waveforms(data, times, channels, sos) for times in spike_times]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{subject}_{session}.npz"
    np.savez(
        out,
        waveforms=waveforms,
        unit_ids=unit_ids.astype(int),
        channel_pos=channel_pos,
        subject=subject,
        session=session,
    )
    print(f"{out}: {len(unit_ids)} units")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("subject")
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--unit-criteria-id", type=int, default=1)
    args = parser.parse_args()
    for session in args.sessions:
        extract_session(args.subject, session, args.unit_criteria_id)


if __name__ == "__main__":
    main()
