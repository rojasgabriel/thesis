"""Locate cortical layers on each shank from the flash-evoked current source density.

Question: at which probe depth are the flash-evoked laminar landmarks on each
shank, and so which layers do the recorded units come from?

Comparison: the depth profile of the iCSD after the first flash of a trial,
read against the landmarks of Senzai et al. (2019, Neuron 101:500, Fig. 1):
a flash gives an early sink at the layer 3/4 border (b'), the afferent input
from the LGN, with a source in the superficial layers (a'), and a sink at
the layer 5/6 border (d') with a source in lower layer 6 (e'). Mid layer 5
(c) is the depth of the largest multi-unit power (500-5000 Hz). Thus b' is
above c and d' is below c. Niell & Stryker (2008, J Neurosci 28:7520) put
layer 4 at the first sink and at the largest negativity of the first LFP dip,
and layer 5 at the depth where spontaneous rates become much higher. This
module gives the depths; which landmark a sink is stays a reading of the
figure with these rules, because a shank does not reach the pia.

Alignment: first flash of each trial (`first_stim_times_s`), mapped to probe
samples with the inverse of the imec0 StreamSync. The first flash follows
darkness, so its response is not mixed with the response to an earlier flash.
Baseline: mean LFP from -100 to 0 ms, per trial and channel.

LFP: the NP2.0 AP band is full band (no LF file). Low pass at 250 Hz
(Butterworth order 4, zero phase), then every 30th sample (1 kHz). The two
columns of a 15 um row are averaged, so each shank gives 48 rows.

iCSD: step iCSD (Pettersen et al. 2006, J Neurosci Methods 154:116), sources
are discs of radius `RADIUS_UM` with constant current density over each row,
conductivity `SIGMA` S/m. It gives a CSD at every row, also at the shank
ends. The CSD is then smoothed in depth with a Gaussian of `SMOOTH_UM`. The
choice of 45 um removes fine stripes that start with the response (probably
gain differences between channels); with 22 um the stripes stay.

Edge band: the step iCSD puts the current of sources outside the shank into
the end rows. In a test with a sink 30-250 um outside the shank, the iCSD
minimum is always at the end row and stays above 10 % of it for 75 um. So
every landmark must be a local extremum (lower or higher than both
neighbours) at least `EDGE_UM` = 75 um from both ends; else it is NaN.

Landmarks per shank:
- sink_um: the most negative local minimum of the CSD averaged over
  `SINK_WINDOW_MS`. The window starts at 20 ms so the LED light artifact
  (0-20 ms on exposed sites) is outside it. 95 % interval: bootstrap over
  trials (NaN draws, with no local minimum, are left out; `sink_found` is
  the fraction of draws with one). The figure marks the sink only if the
  interval is not wider than `SMOOTH_UM` and the sink peaks after the start
  of `SINK_WINDOW_MS`; else it is not a landmark.
- sink_onset_ms, sink_peak_ms: at the sink row, the first time the CSD is
  below 20 % of its minimum, and the time of the minimum (10-100 ms).
- lfp_dip_um: the most negative local minimum of the LFP in the same window.
- mua_peak_um: the largest local maximum of the MUA power after common median
  reference, median over `N_MUA_WINDOWS` windows of 1 s.

Figure: one CSD panel per shank (sink blue), the marked sinks, the MUA power
profile, and, for the mice of the atlas plan, the shifted atlas layer starts
(dotted lines; layer names between the CSD and the MUA panel, "<6" = below L6).

Exclusions: channels at the ADC limit in more than `MAX_RAIL_FRACTION` of
the trial windows (stuck or saturated channels) leave the row average; a row
with no channel left is interpolated in depth. Then trials with any sample at
the ADC limit on a remaining channel. Both counts are in the output.
Sampling unit: the trial (for the bootstrap), and the shank (for the
landmarks). A shank covers about 720 um and does not reach the pia, so the
CSD alone gives landmarks, not layer limits; the unit layers below add them.

Unit layers (table `UnitLayer` of `labdata_plugin`, `populate` step; its
`make` runs `measure` and `landmarks` from the raw data, on the stable units
of its `UnitStability` key): the planned probe track of GRB061, GRB064 and
GRB068 is one vvasp plan (`<mouse>_chronic_NP2.json`: entry AP -4188, ML
-2682 um, as in their `ProbeInsertion` rows; tip AP -1946, ML -1866, DV
-3440 um; elevation 131, azimuth -20, spin 0 deg; 3636.8 um along the axis;
allen_mouse_25um with the plan's bregma, rotation and scaling). The
probe is about 41 deg from vertical, so depth on the probe is not cortical
depth. `ATLAS_LAYER_STARTS_UM` is the Allen VISp layer at each shank centre
along that track, computed once with the vvasp transforms (5 um steps; the
computed entry point agrees with the plan). The real track differs from the
plan, mostly in depth: the offset is the median over shanks of (MUA peak -
atlas mid layer 5), and the atlas starts are shifted by it. A spread of the
per-shank values above `MAX_OFFSET_SPREAD_UM` makes the layers uncertain;
the `reliable` column is then 0. Check the offset against the sinks in the
figure (b' at the shifted L3/4 start, d' at the shifted L5 start) before using
the layers. `settings()` lists the constants that change the result; the
table stops when they differ from its `UnitLayerParam` row.
"below L6" is under VISp6b: white matter or hippocampus in this plan. vvasp
shank k is taken as SpikeGLX shank k: the measured MUA peaks rise from shank
0 to 3 as the atlas mid layer 5 does (GRB061, GRB064).

    uv run python -m thesis.ephys.analyses.csd measure GRB064 20261008_143359
    uv run python -m thesis.ephys.analyses.csd plot GRB064 20261008_143359
    uv run python -m thesis.ephys.analyses.csd populate GRB064 20261008_143359
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, signal
from scipy.ndimage import gaussian_filter1d

RESULTS = Path("results/csd")
FIGURES = Path("figures/csd")
PRE_MS, POST_MS = 200, 400
BASELINE_MS = (-100, 0)
LOWPASS_HZ = 250
STEP = 30  # 30 kHz -> 1 kHz
SIGMA = 0.3  # S/m
RADIUS_UM = 500
SMOOTH_UM = 45
SINK_WINDOW_MS = (20, 45)
EDGE_UM = 75
N_BOOT = 500
N_MUA_WINDOWS = 30
MAX_RAIL_FRACTION = 0.05
MUA_BAND_HZ = (500, 5000)
PLOT_MS = (-10, 100)
# Starts (um from tip) of VISp6b, 5, 4, 2/3, 1 and of the outside, per shank,
# on the vvasp plan of GRB061, GRB064 and GRB068 (see the docstring).
ATLAS_LAYER_STARTS_UM = {
    0: (2560, 2770, 3005, 3155, 3390, 3505),
    1: (2635, 2855, 3095, 3220, 3465, 3580),
    2: (2740, 2960, 3190, 3320, 3540, 3655),
    3: (2840, 3090, 3285, 3395, 3615, 3705),
}
PLAN_SUBJECTS = ("GRB061", "GRB064", "GRB068")
LAYER_NAMES = ("below L6", "L6", "L5", "L4", "L2/3", "L1", "outside")
LAYER_LABELS = ("<6", "6", "5", "4", "2/3", "1", "out")  # short names in the figure
MAX_OFFSET_SPREAD_UM = 150


def ap_path(subject: str, session: str) -> Path:
    from labdata.utils import prefs

    root = Path(prefs["local_paths"][0]) / subject / session
    metas = [
        p
        for p in sorted(root.glob("ephys_g*/ephys_g*_imec0/*.ap.meta"))
        if not p.name.startswith("._") and "backup" not in str(p)
    ]
    if len(metas) != 1:
        raise ValueError(
            f"{subject} {session}: expected one imec0 AP file, got {metas}"
        )
    return metas[0]


def geometry(meta: dict) -> np.ndarray:
    """Return shank, x, z, used for each saved channel (snsGeomMap)."""
    entries = re.findall(r"\(([\d:]+)\)", meta["snsGeomMap"])
    return np.array([[int(v) for v in e.split(":")] for e in entries])


def measure(subject: str, session: str) -> Path:
    """Save the per-trial LFP of each row and the MUA power of each row."""
    from labdata.schema import StreamSync

    import thesis.ephys  # noqa: F401  (import before labdata plugins on Hodgkin)
    from thesis.ephys.analyses.recording_noise import open_data, read_meta
    from thesis.ephys.trials import build_trial_table

    meta_path = ap_path(subject, session)
    meta = read_meta(meta_path)
    data_path = meta_path.with_suffix(".bin")
    if not data_path.exists():
        data_path = meta_path.with_suffix(".cbin")
    data = open_data(str(data_path))
    fs = float(meta["imSampRate"])
    uv = 1e6 * float(meta["imAiRangeMax"]) / float(meta["imMaxInt"])
    uv /= float(meta["imChan0apGain"])
    limit = int(meta["imMaxInt"]) - 1
    geom = geometry(meta)
    n_probe = len(geom)

    trials = build_trial_table(subject, session, include_frames=False)
    first = np.array([v[0] for v in trials["first_stim_times_s"] if len(v)], float)
    key = {"subject_name": subject, "session_name": session, "stream_name": "imec0"}
    source, clock = (StreamSync() & key).get_interp_data(warn=False)
    samples = np.interp(first, clock, source).round().astype(int)

    shanks = np.unique(geom[:, 0])
    rows_z = [np.unique(geom[geom[:, 0] == s, 2]) for s in shanks]
    n_rows = max(len(z) for z in rows_z)

    def to_rows(x: np.ndarray, good: np.ndarray) -> np.ndarray:
        """Average the good channels of each row: (..., channel) -> (..., shank, row).

        A row with no good channel is NaN.
        """
        out = np.full(x.shape[:-1] + (len(shanks), n_rows), np.nan, np.float32)
        for i, s in enumerate(shanks):
            for j, z in enumerate(rows_z[i]):
                use = good & (geom[:, 0] == s) & (geom[:, 2] == z)
                if use.any():
                    out[..., i, j] = x[..., use].mean(-1)
        return out

    pre, post = int(PRE_MS / 1e3 * fs), int(POST_MS / 1e3 * fs)
    pad = int(0.1 * fs)
    sos = signal.butter(4, LOWPASS_HZ, "low", fs=fs, output="sos")
    lfp, railed = [], []
    for s in samples:
        raw = np.asarray(data[s - pre - pad : s + post + pad, :n_probe])
        railed.append(np.abs(raw).max(axis=0) >= limit)
        x = signal.sosfiltfilt(sos, raw.astype(np.float32) * uv, axis=0)
        lfp.append(x[pad:-pad:STEP])
    railed = np.array(railed)  # trial, channel
    good = railed.mean(axis=0) <= MAX_RAIL_FRACTION
    rail = railed[:, good].any(axis=1)
    lfp = to_rows(np.stack(lfp), good)
    t_ms = np.arange(lfp.shape[1]) * STEP / fs * 1e3 - PRE_MS

    duration = float(meta["fileTimeSecs"])
    starts = np.linspace(60, duration - 61, N_MUA_WINDOWS) * fs
    sos = signal.butter(3, MUA_BAND_HZ, "band", fs=fs, output="sos")
    power = []
    for start in starts.astype(int):
        raw = np.asarray(data[start : start + int(fs), :n_probe], np.float32) * uv
        x = signal.sosfiltfilt(sos, raw, axis=0)
        x -= np.median(x, axis=1, keepdims=True)
        power.append(np.mean(x[pad:-pad] ** 2, axis=0))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{subject}_{session}.npz"
    np.savez_compressed(
        out,
        lfp=lfp,  # trial, time, shank, row
        t_ms=t_ms,
        rail=rail,
        bad_channels=np.flatnonzero(~good),
        rows_z=np.array(
            [np.pad(z, (0, n_rows - len(z)), constant_values=-1) for z in rows_z]
        ),
        mua=to_rows(np.median(power, axis=0), good),
    )
    print(
        f"{subject} {session}: {len(samples)} first flashes, {rail.sum()} at the ADC limit; "
        f"channels at the limit in > {MAX_RAIL_FRACTION:.0%} of trials: {np.flatnonzero(~good).tolist()}"
    )
    return out


def step_icsd_matrix(z_um: np.ndarray) -> np.ndarray:
    """Forward matrix F (uV per nA/mm^3) of the step iCSD: phi = F @ csd."""
    z = z_um / 1e3  # mm
    h, r = np.median(np.diff(z)), RADIUS_UM / 1e3
    u = np.linspace(-h / 2, h / 2, 51)  # points inside each step
    dz = z[:, None, None] - (z[None, :, None] + u)
    kernel = np.sqrt(dz**2 + r**2) - np.abs(dz)  # mm
    # 1/(2 sigma) * integral(kernel) over the step: mm^2 * nA/mm^3 / (S/m) -> uV
    return np.trapezoid(kernel, u, axis=-1) / (2 * SIGMA) * 1e-3 * 1e3


def icsd(lfp_rows: np.ndarray, z_um: np.ndarray) -> np.ndarray:
    """CSD (nA/mm^3, negative = sink) of LFP (..., row), smoothed in depth."""
    csd = linalg.solve(step_icsd_matrix(z_um), np.moveaxis(lfp_rows, -1, 0))
    csd = np.moveaxis(csd, 0, -1)
    spacing = np.median(np.diff(z_um))
    return gaussian_filter1d(csd, SMOOTH_UM / spacing, axis=-1, mode="nearest")


def fill_rows(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear interpolation in depth over NaN rows (last axis)."""
    bad = np.isnan(x).all(axis=tuple(range(x.ndim - 1)))
    if not bad.any():
        return x
    x = x.copy()
    for i in np.flatnonzero(bad):
        lo, hi = np.flatnonzero(~bad & (z < z[i])), np.flatnonzero(~bad & (z > z[i]))
        j = lo[-1] if lo.size else hi[0]
        k = hi[0] if hi.size else lo[-1]
        w = 0.5 if j == k else (z[i] - z[j]) / (z[k] - z[j])
        x[..., i] = (1 - w) * x[..., j] + w * x[..., k]
    return x


def local_extreme(profile: np.ndarray, z: np.ndarray, sign: int = -1) -> float:
    """Depth of the strongest local minimum (sign -1) or maximum (sign 1).

    Only rows at least `EDGE_UM` from both ends count; NaN if there is none.
    """
    x = sign * profile
    inner = (z >= z.min() + EDGE_UM) & (z <= z.max() - EDGE_UM)
    peak = np.zeros(len(x), bool)
    peak[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    candidates = np.flatnonzero(peak & inner)
    if candidates.size == 0:
        return np.nan
    return float(z[candidates[np.argmax(x[candidates])]])


def landmarks(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return the landmarks of each shank and the mean LFP/CSD for plotting."""
    d = np.load(path)
    t = d["t_ms"]
    keep = ~d["rail"]
    lfp = d["lfp"][keep]
    base = (t >= BASELINE_MS[0]) & (t < BASELINE_MS[1])
    lfp = lfp - lfp[:, base].mean(1, keepdims=True)
    win = (t >= SINK_WINDOW_MS[0]) & (t < SINK_WINDOW_MS[1])
    trace_t = (t >= 10) & (t <= 100)
    rng = np.random.default_rng(0)
    rows, maps = [], {}
    for sh in range(lfp.shape[2]):
        z = d["rows_z"][sh]
        valid = z >= 0
        z = z[valid]
        trials = fill_rows(lfp[:, :, sh, valid], z)  # trial, time, row
        mean = trials.mean(0)
        csd = icsd(mean, z)
        sink = local_extreme(csd[win].mean(0), z)
        boot = np.array(
            [
                local_extreme(
                    icsd(
                        trials[rng.integers(0, len(trials), len(trials))][:, win].mean(
                            (0, 1)
                        ),
                        z,
                    ),
                    z,
                )
                for _ in range(N_BOOT)
            ]
        )
        onset = peak = np.nan
        if np.isfinite(sink):
            trace = csd[trace_t][:, np.flatnonzero(z == sink)[0]]
            peak = float(t[trace_t][np.argmin(trace)])
            onset = float(t[trace_t][np.argmax(trace < 0.2 * trace.min())])
        mua = fill_rows(d["mua"][sh, valid], z)
        found = np.isfinite(boot)
        rows.append(
            {
                "shank": sh,
                "top_um": z.max(),
                "bottom_um": z.min(),
                "sink_um": sink,
                "sink_lo_um": np.percentile(boot[found], 2.5)
                if found.any()
                else np.nan,
                "sink_hi_um": np.percentile(boot[found], 97.5)
                if found.any()
                else np.nan,
                "sink_found": found.mean(),
                "sink_onset_ms": onset,
                "sink_peak_ms": peak,
                "lfp_dip_um": local_extreme(mean[win].mean(0), z),
                "mua_peak_um": local_extreme(mua, z, sign=1),
                "n_trials": int(keep.sum()),
                "n_rail": int((~keep).sum()),
            }
        )
        maps[sh] = {"z": z, "lfp": mean, "csd": csd, "mua": mua}
    return pd.DataFrame(rows), {"t_ms": t, "shanks": maps}


def plot(subject: str, session: str) -> None:
    import matplotlib.pyplot as plt

    path = RESULTS / f"{subject}_{session}.npz"
    table, maps = landmarks(path)
    table.insert(0, "session", session)
    table.insert(0, "subject", subject)
    table.to_csv(RESULTS / f"{subject}_{session}_landmarks.csv", index=False)
    print(table.to_string(index=False))
    offset = atlas_offset(table)[0] if subject in PLAN_SUBJECTS else None

    t = maps["t_ms"]
    w = (t >= PLOT_MS[0]) & (t <= PLOT_MS[1])
    shanks = maps["shanks"]
    vmax = np.percentile(
        np.abs(np.concatenate([m["csd"][w].ravel() for m in shanks.values()])), 99
    )
    lo = min(m["z"].min() for m in shanks.values())
    hi = max(m["z"].max() for m in shanks.values())
    fig, axes = plt.subplots(
        1,
        2 * len(shanks),
        figsize=(2.4 * len(shanks) + 1, 4.2),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1] * len(shanks), "wspace": 0.3},
    )
    for i, (sh, m) in enumerate(shanks.items()):
        ax, side = axes[2 * i], axes[2 * i + 1]
        row = table.loc[table["shank"] == sh].iloc[0]
        image = ax.pcolormesh(
            t[w],
            m["z"],
            m["csd"][w].T,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
            rasterized=True,
        )
        ax.axvline(0, color="k", ls="--", lw=0.8)
        for edge in (m["z"].min() - 10, m["z"].max() - EDGE_UM + 7.5):
            ax.axhspan(edge, edge + EDGE_UM + 2.5, color="0.85", alpha=0.8, lw=0)
        if (
            row["sink_hi_um"] - row["sink_lo_um"] <= SMOOTH_UM
            and row["sink_peak_ms"] >= SINK_WINDOW_MS[0]
        ):
            ax.plot(
                [SINK_WINDOW_MS[0], SINK_WINDOW_MS[1]],
                [row["sink_um"]] * 2,
                color="k",
                lw=2,
            )
        ax.set_title(f"shank {sh}", fontsize=9)
        ax.set_xlabel("time from first flash (ms)" if i == 0 else "")
        ax.set_ylim(lo - 10, hi + 10)
        side.plot(m["mua"] / m["mua"].max(), m["z"], color="0.3")
        if np.isfinite(row["mua_peak_um"]):
            side.axhline(row["mua_peak_um"], color="0.3", ls=":", lw=0.8)
        side.set_xticks([])
        side.tick_params(left=False)
        side.set_xlabel("mua", fontsize=8)
        if offset is not None:
            starts = np.array(ATLAS_LAYER_STARTS_UM[sh]) + offset
            inside = (starts >= m["z"].min()) & (starts <= m["z"].max())
            for start in starts[inside]:
                ax.axhline(start, color="k", ls=":", lw=0.7)
                side.axhline(start, color="k", ls=":", lw=0.7)
            bounds = np.clip(np.r_[-np.inf, starts, np.inf], m["z"].min(), m["z"].max())
            for low, high, name in zip(
                bounds[:-1], bounds[1:], LAYER_LABELS, strict=True
            ):
                if high - low >= 45:  # room for the label
                    side.text(
                        -0.17,
                        (low + high) / 2,
                        name,
                        transform=side.get_yaxis_transform(),
                        fontsize=7,
                        ha="center",
                        va="center",
                    )
        ax.text(-0.2, 1.04, "abcdefgh"[i], transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel("depth from probe tip (µm)")
    cbar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_ticks([-vmax, vmax], labels=["sink", "source"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{subject}_{session}.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def atlas_offset(table: pd.DataFrame) -> tuple[float, float]:
    """Return the median and the spread of (MUA peak - atlas mid layer 5)."""
    mid5 = np.array([np.mean(ATLAS_LAYER_STARTS_UM[k][1:3]) for k in table["shank"]])
    diff = table["mua_peak_um"].to_numpy() - mid5
    return float(np.nanmedian(diff)), float(np.nanmax(diff) - np.nanmin(diff))


def unit_layers(shank, depth, offset: float) -> list[str]:
    """Return the atlas layer of each unit from its shank and depth (um from tip)."""
    return [
        LAYER_NAMES[
            np.searchsorted(
                np.array(ATLAS_LAYER_STARTS_UM[int(k)]) + offset, z, "right"
            )
        ]
        for k, z in zip(shank, depth, strict=True)
    ]


def settings() -> dict:
    """Return the constants that change the result (stored in UnitLayerParam)."""
    return {
        "pre_post_ms": [PRE_MS, POST_MS],
        "baseline_ms": list(BASELINE_MS),
        "lowpass_hz": LOWPASS_HZ,
        "step": STEP,
        "sigma": SIGMA,
        "radius_um": RADIUS_UM,
        "smooth_um": SMOOTH_UM,
        "sink_window_ms": list(SINK_WINDOW_MS),
        "edge_um": EDGE_UM,
        "n_boot": N_BOOT,
        "n_mua_windows": N_MUA_WINDOWS,
        "max_rail_fraction": MAX_RAIL_FRACTION,
        "mua_band_hz": list(MUA_BAND_HZ),
        "atlas_layer_starts_um": {
            str(k): list(v) for k, v in ATLAS_LAYER_STARTS_UM.items()
        },
        "max_offset_spread_um": MAX_OFFSET_SPREAD_UM,
    }


def populate(subject: str, session: str) -> None:
    """Populate the UnitLayer table for one session and print the layer counts."""
    import thesis.ephys  # noqa: F401  (import before labdata plugins on Hodgkin)
    from labdata_plugin.schema import UnitLayer

    key = {"subject_name": subject, "session_name": session}
    UnitLayer.populate(key, display_progress=True)
    print((UnitLayer & key).fetch(format="frame").to_string())
    layers = pd.Series((UnitLayer.Unit & key).fetch("layer"), dtype=object)
    print(layers.value_counts().reindex(LAYER_NAMES, fill_value=0).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["measure", "plot", "populate"])
    parser.add_argument("subject")
    parser.add_argument("session")
    args = parser.parse_args()
    {"measure": measure, "plot": plot, "populate": populate}[args.step](
        args.subject, args.session
    )


if __name__ == "__main__":
    main()
