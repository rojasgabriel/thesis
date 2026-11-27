"""What do V1 neurons look like around the first flash without the fixation drop?

Question: after the first flash of a trial, most neurons fire less for the
whole wait in the port (`dataset_overview.py`, the population figure). The GLM
gives this drop its own term, the fixation kernel (from the first flash to the
center exit). With that term divided out of the data, what is left?

Comparison: the same neurons and trials, two versions of each neuron's rate
around the first flash: observed, and observed divided by the fitted fixation
contribution, exp(fixation part of the log rate), of the model in use. Nothing
else of the model is used, so the flash responses are the observed ones.

- Units: the GLM units (`fit.py`, unit criteria 1, stability 0, exclusions),
  both sessions of `flash_kernels.SESSIONS`.
- Trials and grid: the selected completed trials of the GLM on its 1-ms grid,
  aligned to the first flash; only valid bins enter. Bins of `BIN_S`.
- Contribution: the raw fixation columns times their fitted weights
  (`results.raw_contribution`), per 1-ms bin, so it ends at the exit.
- Heatmaps, shown from `WINDOW_S[0]` to `WINDOW_S[1]`: (rate − baseline) ÷
  mean rate. Baseline: the observed rate in the 40 ms before the first flash
  (the same in both versions: the fixation kernel starts at the first flash).
  Mean rate: the observed mean from `MEAN_S[0]` to `MEAN_S[1]`. Both come from
  the observed rate, so the two versions share one scale. Rows sorted by the
  observed value from `SORT_S[0]` to `SORT_S[1]`, the same order in each
  version; the color limit is ±`LIMIT` and values past it take the end color
  (count in the json).
- Fraction below the mean (json): the mean from `SORT_S[0]` to `SORT_S[1]` of
  log2 of the rate over that version's own mean in the shown window, below 0.
- Hidden: in the corrected version, `HIDDEN_S[0]` to `HIDDEN_S[1]` after
  the first flash is not shown (gray in the heatmaps, a gap in the curves):
  there the fixation kernel overlaps the first-flash response, and dividing
  it out inflates that response (in GRB006 about 2.5 times), so the result is
  not reliable. The mean rate of a corrected unit is over its shown bins.
- Units of all mice pooled (Gabriel, 2026-11-27): one heatmap per version and
  the mean ± SEM over units of the heatmap value (`plotting.unit_curves`).

- Example (`fixation_removed_example.svg`): the example neuron of each mouse
  of `dataset_overview.py` and ten of its trials (spread over the session),
  the 1-ms rate smoothed with a Gaussian of SD `SMOOTH_S`, observed and with
  the fixation part divided out; flashes (blue ticks) and the center exit
  (orange); below, the mean over all trials, the observed one in black. Run
  `dataset_overview.py` first.

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
    unit_curves,
)

OUT = Path("results/glm")
FITS = {
    "observed": None,
    "fixation removed": OUT,
}
WINDOW_S = (-0.28, 0.6)  # 0 is a bin edge
MEAN_S = (-0.48, 1.52)  # the 1-ms grid is -0.5 to 2.1 s
BIN_S = 0.04  # one flash slot
SORT_S = (0.15, 0.5)
LIMIT = 1.5
TRIAL_WINDOW_S = (-0.3, 1.5)  # as the trial raster of `dataset_overview.py`
EXAMPLE_TRIALS = 10
SMOOTH_S = 0.015  # Gaussian SD of the trial rates
# Not shown for the corrected versions: the fixation kernel starts at the
# first flash and overlaps its response (they share a part of the rate), so
# dividing it out there inflates the first-flash response.
HIDDEN_S = (0.0, 0.2)
COLORS = {
    "observed": OBSERVED_COLOR,
    "fixation removed": CONCEPT_COLORS["task"],
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
    edges = np.arange(MEAN_S[0], MEAN_S[1] + 1e-9, BIN_S)
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
            # A bin with no valid trial has no rate (NaN), not a rate of 0.
            with np.errstate(invalid="ignore", divide="ignore"):
                rates.append(
                    np.where(exposure > 0, total / exposure, np.nan) / dmat.BINWIDTH_S
                )
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
        # Landscape for a slide: two mice in each row.
        wide = -(-len(SESSIONS) // 2)
        figure = plt.figure(figsize=(7.6 * wide, 6.4), layout="constrained")
        outer = figure.add_gridspec(2, wide)
        grid = [outer[divmod(i, wide)] for i in range(len(SESSIONS))]
        panels = []
        for row, session in enumerate(SESSIONS):
            subject = session.split("_")[0]
            unit_id = overview[subject]["example_unit_id"]
            inputs = session_inputs(OUT, session)
            centers = inputs["prepared"]["relative_centers"]
            show = (centers >= TRIAL_WINDOW_S[0]) & (centers < TRIAL_WINDOW_S[1])
            picks = (
                np.linspace(0, len(inputs["trials"]) - 1, EXAMPLE_TRIALS)
                .round()
                .astype(int)
            )
            counts, corrected = unit_trials(inputs, unit_id)
            rates = {
                "observed": smoothed(counts, inputs["valid"]),
                "fixation removed": smoothed(corrected, inputs["valid"]),
            }
            hidden = (centers >= HIDDEN_S[0]) & (centers < HIDDEN_S[1])
            rates["fixation removed"][:, hidden] = np.nan  # the terms overlap there
            top = np.nanpercentile(rates["observed"][picks][:, show], 99)
            cell = grid[row].subgridspec(2, len(columns), height_ratios=(1.4, 1))
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
                    record = inputs["trials"].iloc[trial]
                    start = float(record["stim_pulse_times_s"][0])
                    flashes = np.asarray(record["stim_pulse_times_s"], float) - start
                    exit_s = float(record["center_exit_s"]) - start
                    flashes = flashes[flashes < TRIAL_WINDOW_S[1]]
                    axis.vlines(
                        flashes, y, y + 0.25, color=CONCEPT_COLORS["visual"], lw=0.8
                    )
                    if exit_s < TRIAL_WINDOW_S[1]:
                        axis.plot(
                            exit_s, y + 0.5, ">", color=CONCEPT_COLORS["task"], ms=3
                        )
                if name != "observed":  # the hidden window, not "no spikes"
                    axis.axvspan(
                        *HIDDEN_S,
                        facecolor="white",
                        edgecolor="0.6",
                        hatch="////",
                        lw=0,
                    )
                axis.set_yticks([])
                axis.tick_params(labelbottom=False)
                if row < wide:
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
    full = np.arange(MEAN_S[0], MEAN_S[1] + 1e-9, BIN_S)
    middle = (full[:-1] + full[1:]) / 2
    shown = (middle > WINDOW_S[0]) & (middle < WINDOW_S[1])
    base = int(np.argmin(np.abs(middle + BIN_S / 2)))  # the bin that ends at 0
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + 1e-9, BIN_S)
    centers = middle[shown]
    sort = (centers >= SORT_S[0]) & (centers < SORT_S[1])
    data, report = {}, {}
    for session in SESSIONS:
        subject = session.split("_")[0]
        versions = {}
        ids, observed, removed = unit_rates(FITS["fixation removed"], session)
        baseline = observed[:, base : base + 1]
        scale = np.maximum(np.nanmean(observed, 1, keepdims=True), 1e-9)
        versions["observed"], versions["fixation removed"] = (
            observed[:, shown],
            removed[:, shown],
        )
        hidden = (centers >= HIDDEN_S[0]) & (centers < HIDDEN_S[1])
        for name in list(FITS)[1:]:  # not reliable where the terms overlap
            versions[name] = versions[name].copy()
            versions[name][:, hidden] = np.nan
        values = {name: (r - baseline) / scale for name, r in versions.items()}
        order = np.argsort(values["observed"][:, sort].mean(1))
        data[subject] = (values, order)
        report[subject] = {
            name: {
                "units": len(ids),
                "fraction_below_mean_150_500_ms": float(
                    np.mean(
                        np.nanmean(
                            np.log2(
                                np.maximum(r[:, sort], 0.1)
                                / np.maximum(np.nanmean(r, 1, keepdims=True), 1e-9)
                            ),
                            1,
                        )
                        < 0
                    )
                ),
                "values_past_the_color_limit": int(
                    np.sum(np.abs(values[name]) > LIMIT)
                ),
            }
            for name, r in versions.items()
        }
    # The units of all mice in one map per version, sorted by the observed
    # rate in the wait (Gabriel, 2026-11-27: pool the units of all mice).
    pooled = {
        name: np.vstack([values[name] for values, _ in data.values()]) for name in FITS
    }
    order = np.argsort(pooled["observed"][:, sort].mean(1))
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(9.6, 5.2), layout="constrained")
        grid = figure.add_gridspec(len(FITS), 2, width_ratios=(1, 1.1))
        heat = []
        image = None
        for row, name in enumerate(FITS):
            axis = figure.add_subplot(grid[row, 0])
            heat.append(axis)
            image = axis.imshow(
                pooled[name][order],
                aspect="auto",
                cmap=matplotlib.colormaps["RdBu_r"].with_extremes(bad="0.8"),
                vmin=-LIMIT,
                vmax=LIMIT,
                interpolation="nearest",
                extent=(edges[0], edges[-1], len(order), 0),
            )
            axis.axvline(0, color=OBSERVED_COLOR, lw=0.6)
            axis.set_ylabel(f"{name}\nunits")
            if row == 0:
                axis.set_title(
                    f"{len(order)} units, {len(data)} mice", fontsize=8, loc="left"
                )
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("time from the first flash (s)")
        axis = figure.add_subplot(grid[:, 1])
        for index, name in enumerate(FITS):
            unit_curves(
                axis,
                centers,
                [values[name] for values, _ in data.values()],
                COLORS[name],
            )
            axis.text(
                0.98,
                0.97 - 0.05 * index,
                name,
                color=COLORS[name],
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
        axis.axhline(0, color="0.6", lw=0.6, ls="--")
        axis.axvline(0, color=OBSERVED_COLOR, lw=0.6)
        axis.set_xlabel("time from the first flash (s)")
        axis.set_ylabel("(rate − pre-flash rate) ÷ mean rate")
        figure.colorbar(
            image,
            ax=heat,
            location="bottom",
            shrink=0.8,
            aspect=30,
            label="(rate − pre-flash rate) ÷ mean rate",
        )
        label_panels([*heat, axis])
        print(save_figure(figure, OUT / "fixation_removed"))
    print(example_figure())
    (OUT / "fixation_removed.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
