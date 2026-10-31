"""In the port, does the flash response change with how much the body moves?

Question: the flash × exit term of the GLM holds every change of the
after-exit flash response that is the same for both turns. That can be an
effect of movement on the flash response, or a visual change when the head
turns. In the center port the nose is in the poke, so the LED panel stays at
almost the same place in the visual field while the body still moves. If the
flash response changes with movement there, movement can change it directly,
with no visual shift.

Null hypothesis: in the port, a unit's flash response does not depend on the
motion energy at the flash (slope 0); over the units of a mouse, the median
slope is 0.

- Units: the flash-excited units of `raw_video_trust_classification.json`
  (30 / 15); each mouse is analyzed alone (different camera views).
- Flashes: completed GLM trials; flashes in the port (before the center exit)
  with a gap of at least `MIN_GAP_S` from the flash before; the first flash
  of a trial is left out (no flash before; it also starts the fixation drop).
  The window `WINDOW_MS` around the flash must lie inside the valid interval.
  Gaps come in 40-ms steps, so the shortest is 120 ms (46% / 32% of the
  flashes); then the -40 to 0 ms baseline is 80-120 ms after the flash
  before, on the tail of its response. Checked on 2026-10-30 with gaps of
  200 ms or more (134 / 66 flashes): the flash response still falls with
  movement (-18% / -26% per SD, p = 0.002 / 0.001), but the rise of the
  background is no longer significant (+5% / +20%, p = 0.54 / 0.06).
- Rates: the observed spike counts divided by exp(the part of every non-flash
  column of the design: fixation and pre-stimulus terms, previous choice,
  session drift and the motion-energy PCs), each trial with the fold model of
  the GLM cross-validation that held it out. This removes the drop over the
  wait (so the time in the port is not read as movement) and the gain on all
  activity that the motion-energy terms model. Spike history stays.
- Movement at a flash: the mean total motion energy of the video frames from
  `ME_WINDOW_MS[0]` to `ME_WINDOW_MS[1]` around the flash, z-scored over the
  analyzed flashes of the session. The window ends at the flash, so it holds
  no movement that the flash causes. Flashes with no frame are left out.
  Checked on 2026-10-30: windows of -50 to +100 and 0 to +100 ms give the
  same sign and size; the flash does not change the motion energy (0-50 ms
  after over 50-0 ms before: 0.95 / 0.97), so the light of the flash is not
  read as movement; leaving out the first 300 ms of the wait, or adding the
  time since the first flash and the gap as predictors, keeps the result
  (movement is larger early in the wait, rho -0.17 / -0.37).
- Flash response of one flash: the mean rate from `RESPONSE_MS[0]` to
  `RESPONSE_MS[1]` minus the mean rate from -40 to 0 ms (a mean, not the
  largest bin, which a single noisy flash would bias).
- Slope: per unit, a line fit of the flash response against the movement z;
  given as percent of the fitted response at average movement (100 ·
  slope / intercept). A unit whose fitted response is ≤ 0 has no percent and
  is counted. The background (the rate before the flash) gets the same fit,
  in percent of its mean.
- Statistics: per mouse, the median percent with a bootstrap 95% interval
  over units (`BOOTSTRAPS`) and a two-sided Wilcoxon signed-rank test against
  0. The sampling unit is the unit.
- Figure: per mouse, the flash-aligned rate of each motion-energy quintile
  (rate over the unit's mean observed rate over the valid bins, mean ± SEM
  over units), and the histogram of the unit percents (median dashed, 0
  solid). The axes are the same in both mice. Percents beyond
  ±`PERCENT_LIMIT` (units with a very small response) are drawn in the end
  bins and counted in the panel; they stay in the statistics.

The script writes `figures/glm/flash_movement_in_port.svg` and
`results/glm/flash_movement_in_port.json`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.flash_movement_in_port
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import to_rgb
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm.decoding_without_terms import FLASH_TERMS
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
MIN_GAP_S = 0.1
WINDOW_MS = (-40, 150)
RESPONSE_MS = (40, 100)
ME_WINDOW_MS = (-100, 0)
PERCENT_LIMIT = 150  # display limit of the histograms (%)
QUINTILES = 5
BIN_MS = 10
BOOTSTRAPS = 2000
SEED = 4
VISUAL = CONCEPT_COLORS["visual"]
_MOTION = np.array(to_rgb(CONCEPT_COLORS["motion energy"]))
SHADES = [tuple(1 - (1 - _MOTION) * f) for f in np.linspace(0.3, 1.0, QUINTILES)]


def in_port_flashes(trials, prepared) -> tuple[np.ndarray, np.ndarray]:
    """Return the window rows (flash, 1-ms bin) and the times (s) of the analyzed flashes."""
    n_bins = len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared).reshape(len(trials), -1)
    lags = np.arange(*WINDOW_MS)
    rows, times = [], []
    for row, (_, trial) in enumerate(trials.iterrows()):
        flashes = np.asarray(trial["stim_pulse_times_s"], dtype=float)
        gaps = np.diff(flashes, prepend=-np.inf)
        for time, gap in zip(flashes, gaps, strict=True):
            if (
                time >= trial["center_exit_s"]
                or not np.isfinite(gap)
                or gap < MIN_GAP_S
            ):
                continue
            start = time - prepared["alignments"][row] - prepared["relative_centers"][0]
            columns = int(np.rint(start / dmat.BINWIDTH_S)) + lags
            if columns[0] < 0 or columns[-1] >= n_bins or not valid[row, columns].all():
                continue
            rows.append(row * n_bins + columns)
            times.append(time)
    return np.asarray(rows), np.asarray(times)


def flash_motion(root: Path, times: np.ndarray) -> np.ndarray:
    """Return the mean total motion energy of the frames around each flash (NaN if none)."""
    with np.load(root / "video_me_features.npz", allow_pickle=False) as video:
        frame_times, total = video["frame_times_s"], video["total"]
    order = np.argsort(frame_times)
    frame_times, total = frame_times[order], total[order]
    lo = np.searchsorted(frame_times, times + ME_WINDOW_MS[0] / 1000)
    hi = np.searchsorted(frame_times, times + ME_WINDOW_MS[1] / 1000)
    cumulative = np.concatenate([[0.0], np.cumsum(total)])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cumulative[hi] - cumulative[lo]) / (hi - lo)


def session_values(session: str, excited: list[int]) -> dict:
    """Return per unit the flash responses, backgrounds and PSTHs of the analyzed flashes."""
    root = OUT / session
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    metadata = json.loads((root / "common_design_me.json").read_text())
    design = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    names = metadata["column_names"]
    other = [i for i, n in enumerate(names) if not n.startswith(FLASH_TERMS)]
    scaling = {
        "training_mean": np.asarray(metadata["training_mean"])[other],
        "training_scale": np.asarray(metadata["training_scale"])[other],
    }
    rows, times = in_port_flashes(trials, prepared)
    motion = flash_motion(root, times)
    keep = np.isfinite(motion)
    rows, motion = rows[keep], motion[keep]
    z = (motion - motion.mean()) / motion.std()
    n_bins = len(prepared["relative_centers"])
    flash_trial = rows[:, 0] // n_bins
    x = np.asarray(design[rows.ravel()][:, other]).reshape(*rows.shape, len(other))
    folds = fit.trial_folds(len(trials), fit.CV_FOLDS, fit.CV_SEED)
    records = {
        r["unit_id"]: r
        for r in fit.load_unit_records(
            root / "all_fit_me/checkpoints", "unit_*_folds.json"
        )
    }
    subject, stamp = session.split("_", 1)
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )
    valid = dmat._valid_bin_mask(prepared)
    lags = np.arange(*WINDOW_MS)
    before = lags < 0
    response = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    out = {"units": [], "response": [], "background": [], "psth": []}
    for unit in excited:
        if unit not in records:
            continue
        counts = fit.build_unit_counts(
            prepared["alignments"], table.loc[unit, "spike_times_s"]
        )
        mean_rate = counts[valid].sum() / (valid.sum() * dmat.BINWIDTH_S)
        coefficients = np.asarray([f["coefficients"] for f in records[unit]["folds"]])
        part = np.empty(rows.shape)
        for fold in range(fit.CV_FOLDS):
            held = folds[flash_trial] == fold
            if not held.any():
                continue
            part[held] = raw_contribution(
                x[held].reshape(-1, len(other)),
                coefficients[fold][other],
                scaling,
                np.arange(len(other)),
            ).reshape(-1, rows.shape[1])
        rate = (
            counts[rows] * np.exp(-part) / dmat.BINWIDTH_S
        )  # (flash, 1-ms bin), spikes/s
        background = rate[:, before].mean(1)
        out["units"].append(int(unit))
        out["response"].append(rate[:, response].mean(1) - background)
        out["background"].append(background)
        groups = np.digitize(z, np.quantile(z, np.linspace(0, 1, QUINTILES + 1)[1:-1]))
        binned = rate.reshape(len(rate), -1, BIN_MS).mean(-1) / mean_rate
        out["psth"].append([binned[groups == g].mean(0) for g in range(QUINTILES)])
    return {
        "subject": subject,
        "flashes": int(len(z)),
        "flashes_without_video": int((~keep).sum()),
        "z": z,
        **{k: np.asarray(v) for k, v in out.items()},
    }


def percent_slopes(z: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return 100 · slope / intercept of a line fit per unit (NaN if the intercept is ≤ 0)."""
    out = []
    for y in values:
        slope, intercept = np.polyfit(z, y, 1)
        out.append(100 * slope / intercept if intercept > 0 else np.nan)
    return np.asarray(out)


def summary(values: np.ndarray, rng) -> dict:
    finite = values[np.isfinite(values)]
    draws = [np.median(rng.choice(finite, len(finite))) for _ in range(BOOTSTRAPS)]
    return {
        "units": int(len(finite)),
        "units_without_a_percent": int(np.sum(~np.isfinite(values))),
        "median_percent_per_sd": float(np.median(finite)),
        "median_95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "wilcoxon_p": float(wilcoxon(finite).pvalue),
        "units_above_0": int(np.sum(finite > 0)),
    }


def plot(sessions: list[dict]) -> Path:
    centers = np.arange(*WINDOW_MS, BIN_MS) + BIN_MS / 2
    edges = np.linspace(-PERCENT_LIMIT, PERCENT_LIMIT, 31)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(sessions),
            2,
            figsize=(9.0, 2.7 * len(sessions)),
            gridspec_kw={"width_ratios": (1.3, 1)},
            squeeze=False,
        )
        for row, s in enumerate(sessions):
            ax = axes[row, 0]
            psth = s["psth"]  # (unit, quintile, bin)
            for g in range(QUINTILES):
                mean = psth[:, g].mean(0)
                sem = psth[:, g].std(0, ddof=1) / np.sqrt(len(psth))
                ax.fill_between(
                    centers, mean - sem, mean + sem, color=SHADES[g], alpha=0.25, lw=0
                )
                ax.plot(centers, mean, color=SHADES[g], lw=1.6)
            ax.axvline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            ax.text(
                0.98,
                0.95,
                "most movement",
                color=SHADES[-1],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            ax.text(
                0.98,
                0.86,
                "stillest",
                color=SHADES[0],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            ax.set_xlim(WINDOW_MS)
            ax.set_xlabel("time from flash (ms), in port")
            ax.set_ylabel("rate / unit's mean observed rate\n(mean ± SEM over units)")
            ax.set_title(
                f"{s['subject']}: {len(psth)} flash-excited units, {s['flashes']} flashes",
                fontsize=8,
                loc="left",
            )
            ax = axes[row, 1]
            finite = s["percent"][np.isfinite(s["percent"])]
            outside = int(np.sum(np.abs(finite) > PERCENT_LIMIT))
            ax.hist(
                np.clip(finite, -PERCENT_LIMIT, PERCENT_LIMIT),
                bins=edges,
                color=VISUAL,
                alpha=0.6,
            )
            ax.axvline(0, color=REFERENCE_COLOR, lw=0.8)
            st = s["stats"]["flash response"]
            ax.axvline(st["median_percent_per_sd"], color=VISUAL, ls="--", lw=1.6)
            ax.text(
                0.97,
                0.97,
                f"median {st['median_percent_per_sd']:+.0f}% per SD\n"
                f"95%: {st['median_95'][0]:+.0f} to {st['median_95'][1]:+.0f}\n"
                f"Wilcoxon p = {st['wilcoxon_p']:.2g}"
                + (
                    f"\n{outside} beyond ±{PERCENT_LIMIT}% (end bins)"
                    if outside
                    else ""
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=VISUAL,
            )
            ax.set_xlabel("change of the flash response\nper 1 SD of motion energy (%)")
            ax.set_ylabel("units")
        for col in range(2):
            lows, highs = zip(*(a.get_ylim() for a in axes[:, col]), strict=True)
            for a in axes[:, col]:
                a.set_ylim(0 if col else min(lows), max(highs) * (1.25 if col else 1))
        figure.tight_layout()
        label_panels(axes.ravel())
        return save_figure(figure, OUT / "flash_movement_in_port")


def main() -> None:
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
    rng = np.random.default_rng(SEED)
    sessions, report = [], {}
    for session in SESSIONS:
        s = session_values(session, sorted(excited[session]))
        s["percent"] = percent_slopes(s["z"], s["response"])
        background = percent_slopes(s["z"], s["background"])
        s["stats"] = {
            "flash response": summary(s["percent"], rng),
            "background": summary(background, rng),
        }
        report[s["subject"]] = {
            "flashes": s["flashes"],
            "flashes_without_video": s["flashes_without_video"],
            "unit_ids": s["units"].tolist(),
            "percent_per_sd": s["percent"].tolist(),
            "background_percent_per_sd": background.tolist(),
            **s["stats"],
        }
        sessions.append(s)
    (OUT / "flash_movement_in_port.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: {
                    kk: v
                    for kk, v in r.items()
                    if "percent_per_sd" not in kk and kk != "unit_ids"
                }
                for k, r in report.items()
            },
            indent=1,
        )
    )
    print(plot(sessions))


if __name__ == "__main__":
    main()
