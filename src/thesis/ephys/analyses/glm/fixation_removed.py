"""What do V1 neurons look like around the first flash without the fixation drop?

Question: after the first flash of a trial, most neurons fire less for the
whole wait in the port (`dataset_overview.py`, the population figure). The GLM
gives this drop its own term, the fixation kernel (from the first flash to the
center exit). With that term divided out of the data, what is left?

Comparison: the same neurons and trials, three versions of each neuron's rate
around the first flash: observed; observed divided by the fitted fixation
contribution, exp(fixation part of the log rate), of the model in use; the
same with the fit of `penalty_sensitivity.py` (100× weaker penalty on the
fixation kernel, which keeps 0.90 / 0.98 of a known level against 0.68 /
0.76). Nothing else of the model is used, so the flash responses are the
observed ones.

- Units: the GLM units (`fit.py`, unit criteria 1, stability 0, exclusions),
  both sessions of `flash_kernels.SESSIONS`.
- Trials and grid: the selected completed trials of the GLM on its 1-ms grid,
  aligned to the first flash; only valid bins enter. Bins of `BIN_S`.
- Contribution: the raw fixation columns times their fitted weights
  (`results.raw_contribution`), per 1-ms bin, so it ends at the exit.
- Heatmaps: log2 of the rate over the unit's own mean rate in the window of
  that version, from `WINDOW_S[0]` to `WINDOW_S[1]`; rows sorted by the
  observed value from `SORT_S[0]` to `SORT_S[1]`, the same order in each
  version; the color limit is `LIMIT` and values past it take the end color.
- PSTH: the mean over units of the rate over the unit's mean rate, ± SEM over
  units. The sampling unit is the unit.

- Example (`fixation_removed_example.svg`): the example neuron of each mouse
  of `dataset_overview.py` and its ten trials (spread over the session), the
  1-ms rate smoothed with a Gaussian of SD `SMOOTH_S`, for the three versions;
  flashes (blue ticks) and the center exit (orange); below, the mean over all
  trials of that version with the observed one in black.

The script writes `figures/glm/fixation_removed.svg`,
`figures/glm/fixation_removed_example.svg` and
`results/glm/fixation_removed.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.fixation_removed
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.fit import build_unit_counts, load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
FITS = {
    "observed": None,
    "fixation removed": OUT,
    "fixation removed, weaker penalty": OUT / "weaker_fixation_penalty",
}
WINDOW_S = (-0.3, 0.6)
BIN_S = 0.04  # one flash slot
SORT_S = (0.15, 0.5)
LIMIT = 1.5
TRIAL_WINDOW_S = (-0.3, 1.5)  # as the trial raster of `dataset_overview.py`
EXAMPLE_TRIALS = 10
SMOOTH_S = 0.015  # Gaussian SD of the trial rates
COLORS = {
    "observed": OBSERVED_COLOR,
    "fixation removed": CONCEPT_COLORS["task"],
    "fixation removed, weaker penalty": "#F0A070",
}


def session_inputs(root: Path, session: str) -> dict:
    """Load what the correction needs for one session of one fit."""
    prepared = dmat._load_windows(root / session / "stimulus_windows_me.npz")
    metadata = json.loads((root / session / "common_design_me.json").read_text())
    design = np.load(
        root / session / "common_design_me.npy", mmap_mode="r", allow_pickle=False
    )
    hold = [
        i for i, n in enumerate(metadata["column_names"]) if n.startswith("hold_basis")
    ]
    subject, stamp = session.split("_", 1)
    n_trials = len(prepared["alignments"])
    return {
        "prepared": prepared,
        "trials": dmat._load_selected_trials(prepared),
        "hold": hold,
        "hold_design": np.asarray(design[:, hold]),  # (row, hold column), all bins
        "hold_mean": np.asarray(metadata["training_mean"])[hold],
        "hold_scale": np.asarray(metadata["training_scale"])[hold],
        "final": {
            int(r["unit_id"]): r
            for r in load_unit_records(
                root / session / "all_fit_me/checkpoints", "unit_*_final.json"
            )
        },
        "table": fetch_unit_table(
            subject, stamp, 1, 0, include_metrics=False
        ).set_index("unit_id"),
        "valid": dmat._valid_bin_mask(prepared).reshape(n_trials, -1),
    }


def unit_trials(inputs: dict, unit_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1-ms spike counts and the same divided by exp(fixation part), (trial, bin)."""
    prepared, hold = inputs["prepared"], inputs["hold"]
    shape = inputs["valid"].shape
    counts = build_unit_counts(
        prepared["alignments"], inputs["table"].loc[unit_id, "spike_times_s"]
    ).reshape(shape)
    part = raw_contribution(
        inputs["hold_design"],
        np.asarray(inputs["final"][unit_id]["coefficients"], dtype=float)[hold],
        {"training_mean": inputs["hold_mean"], "training_scale": inputs["hold_scale"]},
        np.arange(len(hold)),
    ).reshape(shape)
    return counts, counts * np.exp(-part)


def unit_rates(root: Path, session: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return unit ids, observed and fixation-removed rates (unit, bin), spikes/s."""
    inputs = session_inputs(root, session)
    centers = inputs["prepared"]["relative_centers"]
    valid = inputs["valid"]
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + 1e-9, BIN_S)
    which = np.digitize(centers, edges) - 1  # 40-ms bin of each 1-ms bin
    inside = (which >= 0) & (which < len(edges) - 1)
    exposure = np.bincount(
        which[inside], weights=valid[:, inside].sum(0), minlength=len(edges) - 1
    )
    ids, observed, removed = [], [], []
    for unit_id in sorted(inputs["final"]):
        rates = []
        for values in unit_trials(inputs, unit_id):
            total = np.bincount(
                which[inside],
                weights=(values * valid)[:, inside].sum(0),
                minlength=len(edges) - 1,
            )
            rates.append(total / np.maximum(exposure, 1) / dmat.BINWIDTH_S)
        ids.append(unit_id)
        observed.append(rates[0])
        removed.append(rates[1])
    return ids, np.asarray(observed), np.asarray(removed)


def smoothed(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Gaussian-smoothed rate (spikes/s) of each trial row, NaN outside the valid bins."""
    width = int(round(SMOOTH_S / dmat.BINWIDTH_S))
    x = np.arange(-4 * width, 4 * width + 1)
    kernel = np.exp(-0.5 * (x / width) ** 2)
    rows = []
    for row, ok in zip(values * valid, valid, strict=True):
        num = np.convolve(row, kernel, mode="same")
        den = np.convolve(ok.astype(float), kernel, mode="same")
        rows.append(np.where(ok, num / np.maximum(den, 1e-9) / dmat.BINWIDTH_S, np.nan))
    return np.asarray(rows)


def example_figure() -> Path:
    """One neuron per mouse: ten trials and the PSTH, observed and fixation removed."""
    overview = json.loads(
        Path("results/dataset_overview/dataset_overview.json").read_text()
    )
    columns = list(FITS)
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(11.0, 6.4), layout="constrained")
        grid = figure.add_gridspec(2, len(columns), height_ratios=(1, 1))
        panels = []
        for row, session in enumerate(SESSIONS):
            subject = session.split("_")[0]
            unit_id = overview[subject]["example_unit_id"]
            inputs = {
                name: session_inputs(root, session)
                for name, root in FITS.items()
                if root is not None
            }
            first = inputs["fixation removed"]
            centers = first["prepared"]["relative_centers"]
            show = (centers >= TRIAL_WINDOW_S[0]) & (centers < TRIAL_WINDOW_S[1])
            picks = (
                np.linspace(0, len(first["trials"]) - 1, EXAMPLE_TRIALS)
                .round()
                .astype(int)
            )
            rates = {}
            counts, corrected = unit_trials(first, unit_id)
            rates["observed"] = smoothed(counts, first["valid"])
            rates["fixation removed"] = smoothed(corrected, first["valid"])
            weak = inputs["fixation removed, weaker penalty"]
            rates["fixation removed, weaker penalty"] = smoothed(
                unit_trials(weak, unit_id)[1], weak["valid"]
            )
            top = np.nanpercentile(rates["observed"][picks][:, show], 99)
            cell = grid[row, :].subgridspec(2, len(columns), height_ratios=(1.4, 1))
            psth_axes = []
            for col, name in enumerate(columns):
                axis = figure.add_subplot(cell[0, col])
                image = axis.imshow(
                    rates[name][picks][:, show],
                    aspect="auto",
                    cmap="Greys",
                    vmin=0,
                    vmax=top,
                    interpolation="nearest",
                    extent=(centers[show][0], centers[show][-1], EXAMPLE_TRIALS, 0),
                )
                for y, trial in enumerate(picks):
                    record = first["trials"].iloc[trial]
                    flashes = np.asarray(record["stim_pulse_times_s"], float) - float(
                        record["stim_pulse_times_s"][0]
                    )
                    exit_s = float(record["center_exit_s"]) - float(
                        record["stim_pulse_times_s"][0]
                    )
                    flashes = flashes[flashes < TRIAL_WINDOW_S[1]]
                    axis.vlines(
                        flashes, y, y + 0.25, color=CONCEPT_COLORS["visual"], lw=0.8
                    )
                    if exit_s < TRIAL_WINDOW_S[1]:
                        axis.plot(
                            exit_s, y + 0.5, ">", color=CONCEPT_COLORS["task"], ms=3
                        )
                axis.set_yticks([])
                axis.tick_params(labelbottom=False)
                if row == 0:
                    axis.set_title(name, fontsize=8)
                if col == 0:
                    axis.set_ylabel(
                        f"{subject}: unit {unit_id}\n{EXAMPLE_TRIALS} trials"
                    )
                panels.append(axis)
                psth = figure.add_subplot(
                    cell[1, col],
                    sharex=axis,
                    sharey=psth_axes[0] if psth_axes else None,
                )
                for other in ("observed", name):
                    mean = np.nanmean(rates[other][:, show], 0)
                    psth.plot(centers[show], mean, color=COLORS[other], lw=1.2)
                psth.axvline(0, color="0.6", lw=0.6)
                psth.set_xlabel("time from the first flash (s)")
                if col == 0:
                    psth.set_ylabel("rate (spikes/s)\nall trials")
                psth_axes.append(psth)
            figure.colorbar(
                image, ax=panels[-len(columns) :], shrink=0.8, label="rate (spikes/s)"
            )
        label_panels(panels)
        return save_figure(figure, OUT / "fixation_removed_example")


def main() -> None:
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + 1e-9, BIN_S)
    centers = (edges[:-1] + edges[1:]) / 2
    sort = (centers >= SORT_S[0]) & (centers < SORT_S[1])
    data, report = {}, {}
    for session in SESSIONS:
        subject = session.split("_")[0]
        versions = {}
        ids, observed, removed = unit_rates(FITS["fixation removed"], session)
        versions["observed"], versions["fixation removed"] = observed, removed
        weak_ids, _, weak = unit_rates(
            FITS["fixation removed, weaker penalty"], session
        )
        keep = np.isin(weak_ids, ids)
        if list(np.asarray(weak_ids)[keep]) != ids:
            raise ValueError(f"The two fits have different units in {session}.")
        versions["fixation removed, weaker penalty"] = weak[keep]
        # log2 of each version over the unit's own mean rate in the window.
        logs = {
            name: np.log2(
                np.maximum(r, 0.1) / np.maximum(r.mean(1, keepdims=True), 1e-9)
            )
            for name, r in versions.items()
        }
        order = np.argsort(logs["observed"][:, sort].mean(1))
        data[subject] = (logs, order)
        report[subject] = {
            name: {
                "units": len(ids),
                "fraction_below_mean_150_500_ms": float(
                    np.mean(v[:, sort].mean(1) < 0)
                ),
                "values_past_the_color_limit": int(np.sum(np.abs(v) > LIMIT)),
            }
            for name, v in logs.items()
        }
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(data),
            len(FITS) + 1,
            figsize=(11.5, 3.0 * len(data)),
            layout="constrained",
            squeeze=False,
            gridspec_kw={"width_ratios": (1, 1, 1, 1.1)},
        )
        image = None
        for row, (subject, (logs, order)) in enumerate(data.items()):
            for col, name in enumerate(FITS):
                axis = axes[row, col]
                image = axis.imshow(
                    logs[name][order],
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-LIMIT,
                    vmax=LIMIT,
                    interpolation="nearest",
                    extent=(edges[0], edges[-1], len(order), 0),
                )
                axis.axvline(0, color=OBSERVED_COLOR, lw=0.6)
                if row == 0:
                    axis.set_title(name, fontsize=8)
                if col == 0:
                    axis.set_ylabel(f"{subject}: {len(order)} neurons")
                axis.set_xlabel("time from the first flash (s)")
            axis = axes[row, -1]
            for index, name in enumerate(FITS):
                ratio = 2 ** logs[name]
                mean = ratio.mean(0)
                sem = ratio.std(0, ddof=1) / np.sqrt(len(ratio))
                axis.plot(centers, mean, color=COLORS[name], lw=1.4)
                axis.fill_between(
                    centers,
                    mean - sem,
                    mean + sem,
                    color=COLORS[name],
                    alpha=0.15,
                    lw=0,
                )
                if row == 0:
                    axis.text(
                        0.98,
                        0.97 - 0.08 * index,
                        name,
                        color=COLORS[name],
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7,
                    )
            axis.axhline(1, color="0.6", lw=0.6)
            axis.axvline(0, color=OBSERVED_COLOR, lw=0.6)
            axis.set_xlabel("time from the first flash (s)")
            axis.set_ylabel("rate ÷ own mean rate")
        figure.colorbar(
            image,
            ax=axes[:, : len(FITS)].ravel().tolist(),
            shrink=0.6,
            label="log2 rate ÷ own mean rate",
        )
        label_panels(axes.ravel())
        print(save_figure(figure, OUT / "fixation_removed"))
    print(example_figure())
    (OUT / "fixation_removed.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
