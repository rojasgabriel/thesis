"""Compare the noise of the raw Neuropixels recordings to find what to fix in the rig.

Question: which recordings are noisy, which kind of noise it is, and does it
follow a rig setting (the probe reference) or the age of the implant?

Recordings: every imec0 AP file (`.ap.bin`, else `.ap.cbin`) of the GRB mice
in the labdata local path. Excluded: AppleDouble `._` files, `*_backup`
copies, and files shorter than 5 min. Sampling unit: the recording.

Samples: `N_WINDOWS` windows of 1 s, evenly spaced from 60 s after the start
to 60 s before the end. Values are in µV (`imAiRangeMax / imMaxInt / gain` of
each file, so the 14-bit NP2010 of GRB006 and the 12-bit NP2014 compare).

Measures (median over windows unless named otherwise):

- floor: the AP band (300-6000 Hz) of each channel, robust SD (MAD / 0.6745),
  after common average reference (CAR, the median over channels). Median over
  channels; `floor_p10` is the 10th percentile over channels, which is closer
  to the electronic floor because multi-unit activity adds to the MAD.
- floor_raw: the same before CAR.
- common: robust SD of the CAR trace itself (the common-mode noise). A high
  value points to the ground or reference path.
- line_60: amplitude (µV RMS) at 60, 120 and 180 Hz, before CAR (`line_raw`)
  and after CAR (`line_car`). If CAR removes it, the pickup is common-mode.
- saturated: share of windows with common-mode SD above `SATURATED_UV`. In
  such windows (GRB068) the common mode is a 60 Hz sine larger than the ADC
  range, and a third of the samples sit at the rail: the ground path floats.
- bad_channels: share of channels with floor < 0.3× or > 2× the recording
  median.
- err_flags: the SpikeGLX `imErrFlags0` counts as written in the meta.

Reference: decoded from `~imroTbl`, not from the imro file name. NP2014
(type 2013): 0 ext, 1 gnd, 2-5 tip of shank 0-3 (all four = joined tips).
NP2010 (type 24): 0 ext, 1-4 tip of shank 0-3. Checked against
`imChan0Ref` where the meta has it.

Reference mode is the same within most mice, so a reference comparison
across mice is also a mouse comparison. Only GRB036 and GRB045 change the
reference within the mouse.

    uv run --with mtscomp python -m thesis.ephys.analyses.recording_noise measure
    uv run python -m thesis.ephys.analyses.recording_noise plot
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

RESULTS = Path("results/recording_noise")
FIGURE = Path("figures/recording_noise.svg")
N_WINDOWS = 40
SATURATED_UV = 100
EDGE_S = 60
MIN_DURATION_S = 300
BAND_HZ = (300, 6000)
LINE_HZ = (60, 120, 180)
LABEL_NUDGE = {"GRB006": -0.3, "GRB060": -0.25}  # keep labels off other lines
REFERENCE_COLORS = {
    "gnd": "#D55E00",
    "joined tips": "#0072B2",
    "tip": "#56B4E9",
    "ext": "#009E73",
}


def read_meta(path: Path) -> dict:
    """Return the SpikeGLX meta file as strings by key (leading `~` removed)."""
    lines = path.read_text().splitlines()
    return dict(line.lstrip("~").split("=", 1) for line in lines if "=" in line)


def reference_mode(meta: dict) -> str:
    """Decode the reference of the recording from the imro table."""
    entries = meta["imroTbl"].strip("()").split(")(")[1:]
    refs = {int(entry.split()[3]) for entry in entries}
    probe = meta["imDatPrb_type"]
    if probe == "2013":
        if refs == {0}:
            return "ext"
        if refs == {1}:
            return "gnd"
        if refs <= {2, 3, 4, 5}:
            return "joined tips" if len(refs) > 1 else "tip"
    if probe == "24":
        if refs == {0}:
            return "ext"
        if refs <= {1, 2, 3, 4}:
            return "tip"
    raise ValueError(f"unknown reference ids {refs} for probe type {probe}")


def find_recordings(root: Path) -> pd.DataFrame:
    """Return one row per imec0 AP file of the GRB mice."""
    rows = []
    for meta_path in sorted(root.glob("GRB*/*/ephys_g*/ephys_g*_imec0/*.ap.meta")):
        if meta_path.name.startswith("._") or "backup" in str(meta_path):
            continue
        data_path = meta_path.with_suffix(".bin")
        if not data_path.exists():
            data_path = meta_path.with_suffix(".cbin")
        if not data_path.exists():
            continue
        meta = read_meta(meta_path)
        duration_s = float(meta["fileTimeSecs"])
        if duration_s < MIN_DURATION_S:
            continue
        rel = meta_path.relative_to(root).parts
        rows.append(
            {
                "subject": rel[0],
                "session": rel[1],
                "run": rel[2],
                "path": str(data_path),
                "duration_s": duration_s,
                "reference": reference_mode(meta),
                "probe_type": meta["imDatPrb_type"],
                "probe_sn": meta["imDatPrb_sn"],
                "headstage_sn": meta.get("imDatHs_sn", ""),
                "basestation": meta.get("imDatBsc_pn", ""),
                "err_flags": next(
                    (v for k, v in meta.items() if k.startswith("imErrFlags0")), ""
                ),
                "uv_per_bit": 1e6
                * float(meta["imAiRangeMax"])
                / float(meta["imMaxInt"])
                / float(meta["imChan0apGain"]),
                "fs": float(meta["imSampRate"]),
            }
        )
    return pd.DataFrame(rows)


def open_data(path: str):
    if path.endswith(".cbin"):
        from spks.spikeglx_utils import load_spikeglx_mtsdecomp

        return load_spikeglx_mtsdecomp(path)[0]
    from spks.spikeglx_utils import load_spikeglx_binary

    return load_spikeglx_binary(path)[0]


def robust_sd(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.median(np.abs(x - np.median(x, axis=axis)), axis=axis) / 0.6745


def line_amplitude(x: np.ndarray, fs: float) -> np.ndarray:
    """RMS (µV) at the line frequencies, per channel, above the nearby floor."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=int(fs), axis=0)
    df = freqs[1] - freqs[0]
    power = np.zeros(x.shape[1])
    for hz in LINE_HZ:
        peak = psd[np.abs(freqs - hz) <= 1].sum(axis=0)
        near = (np.abs(freqs - hz) > 2) & (np.abs(freqs - hz) <= 8)
        floor = np.median(psd[near], axis=0) * np.sum(np.abs(freqs - hz) <= 1)
        power += np.clip(peak - floor, 0, None)
    return np.sqrt(power * df)


def measure_recording(row: dict) -> tuple[dict, pd.DataFrame]:
    """Return the recording summary and the per-window values."""
    data = open_data(row["path"])
    fs = row["fs"]
    n = int(fs)
    pad = int(0.05 * fs)
    starts = np.linspace(
        EDGE_S * fs, (row["duration_s"] - EDGE_S) * fs - n, N_WINDOWS
    ).astype(int)
    sos = signal.butter(3, BAND_HZ, btype="band", fs=fs, output="sos")
    windows, floors = [], []
    for start in starts:
        raw = np.asarray(data[start - pad : start + n + pad, :384], dtype=np.float32)
        raw *= row["uv_per_bit"]
        raw -= np.median(raw, axis=0)
        ap = signal.sosfiltfilt(sos, raw, axis=0)[pad:-pad]
        common = np.median(ap, axis=1)
        floor = robust_sd(ap - common[:, None])
        wide = raw[pad:-pad]
        floors.append(floor)
        windows.append(
            {
                "time_s": start / fs,
                "floor": np.median(floor),
                "floor_raw": np.median(robust_sd(ap)),
                "common": robust_sd(common),
                "line_raw": np.median(line_amplitude(wide, fs)),
                "line_car": np.median(
                    line_amplitude(wide - np.median(wide, axis=1, keepdims=True), fs)
                ),
            }
        )
    windows = pd.DataFrame(windows)
    channel_floor = np.median(floors, axis=0)
    median_floor = np.median(channel_floor)
    summary = {
        **{k: row[k] for k in ("subject", "session", "run")},
        **windows.drop(columns="time_s").median().to_dict(),
        "floor_p10": np.percentile(channel_floor, 10),
        "saturated": np.mean(windows["common"] > SATURATED_UV),
        "bad_channels": np.mean(
            (channel_floor < 0.3 * median_floor) | (channel_floor > 2 * median_floor)
        ),
    }
    windows.insert(0, "run", row["run"])
    windows.insert(0, "session", row["session"])
    windows.insert(0, "subject", row["subject"])
    return summary, windows


def _measure_safe(row: dict):
    try:
        return measure_recording(row)
    except Exception as error:  # one bad file must not stop the survey
        print(f"failed {row['path']}: {error!r}", flush=True)
        return None


def measure(workers: int) -> None:
    from labdata.utils import prefs

    recordings = find_recordings(Path(prefs["local_paths"][0]))
    print(f"{len(recordings)} recordings", flush=True)
    with Pool(workers) as pool:
        results = [
            r
            for r in pool.imap_unordered(_measure_safe, recordings.to_dict("records"))
            if r
        ]
    summary = pd.DataFrame([r[0] for r in results])
    windows = pd.concat([r[1] for r in results], ignore_index=True)
    keys = ["subject", "session", "run"]
    summary = recordings.merge(summary, on=keys).sort_values(keys)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "recordings.csv", index=False)
    windows.to_csv(RESULTS / "windows.csv", index=False)
    print(summary.drop(columns=["path"]).to_string(), flush=True)


def plot() -> None:
    import matplotlib.pyplot as plt

    from thesis.plotting import FIGURE_STYLE, label_panels, save_figure

    df = pd.read_csv(RESULTS / "recordings.csv", dtype={"session": str})
    df["date"] = pd.to_datetime(df["session"].str[:8])
    df["day"] = (df["date"] - df.groupby("subject")["date"].transform("min")).dt.days
    order = (
        df.groupby("subject")["date"].min().sort_values().index.tolist()
    )  # mice in recording order
    x_of = {s: i for i, s in enumerate(order)}
    rng = np.random.default_rng(0)
    df["x"] = df["subject"].map(x_of) + rng.uniform(-0.2, 0.2, len(df))
    color = df["reference"].map(REFERENCE_COLORS)

    plt.rcParams.update(FIGURE_STYLE)
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), layout="constrained")
    strips = [
        ("floor", "AP noise after CAR (µV)"),
        ("common", "common-mode noise (µV)"),
        ("line_raw", "60 Hz + harmonics before CAR (µV)"),
        ("saturated", "share of time saturated"),
    ]
    for ax, (column, label) in zip(axes.flat[:4], strips):
        ax.scatter(df["x"], df[column], c=color, s=18, alpha=0.8, linewidths=0)
        ax.set_xticks(range(len(order)), order, rotation=60, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(1, 1000)

    ax = axes[2, 0]
    for subject, group in df.groupby("subject"):
        if len(group) < 3:
            continue
        group = group.sort_values("day")
        ref_color = REFERENCE_COLORS[group["reference"].mode()[0]]
        ax.plot(group["day"], group["floor"], "-o", ms=3, color=ref_color, lw=1)
        ax.text(
            group["day"].iloc[-1] + 2,
            group["floor"].iloc[-1] + LABEL_NUDGE.get(subject, 0),
            subject,
            color=ref_color,
            fontsize=7,
            va="center",
        )
    ax.set_xlabel("days since first recording")
    ax.set_ylabel("AP noise after CAR (µV)")

    axes[2, 1].remove()

    for i, (name, c) in enumerate(REFERENCE_COLORS.items()):
        if name in set(df["reference"]):
            axes[1, 1].text(
                0.3,
                0.97 - 0.07 * i,
                name,
                color=c,
                transform=axes[1, 1].transAxes,
                va="top",
                fontsize=8,
            )
    label_panels(axes.flat[:5])
    print(save_figure(fig, FIGURE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    )
    sub.add_parser("plot")
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.workers)
    else:
        plot()


if __name__ == "__main__":
    main()
