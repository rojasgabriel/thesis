"""Show how the flash response depends on the gap from the flash before and on the period.

Question: how does the response of a V1 unit to a flash change with the gap
from the flash before, in the fixation and after the center exit, for units
that fire more and for units that fire less after a flash? Does the model
predict the same?

Comparison: the observed firing rate and the firing rate that the model
predicts, around the same flashes. Both rates have the response to the flash
before in them, so they are measured in the same way. The comparisons over
gaps and between periods are paired: each unit has a value in each condition.

- Flashes: every flash that is not the first of its trial. In the fixation:
  its response window (`LAGS_MS` after the flash) ends before the center
  exit. After the exit: the flash and the flash before it are both at or
  after the center exit, and the response window ends before the response
  entry. The gaps are multiples of 40 ms. Each gap up to `LONG_GAP_MS` is a
  group (40, 80, 120, and 160 ms), and all longer gaps are one group.
- Kind of unit: from the flash kernels of the model (`flash_kernels.npz`, the
  mean response over the three gap classes in the fixation). A unit fires more
  after a flash when that response is above `KIND_LIMIT`, and less when it is
  below -`KIND_LIMIT`. Units between the limits, and units with a mean rate
  below `MIN_RATE` spikes/s around the flashes, are not used.
- Observed rate: the spike count of the unit in the 1-ms bins of the model
  around each flash, as a rate.
- Predicted rate: the rate of the final refit of the unit in the same bins
  (all its terms, with the observed spike history). The final refit is fit on
  all trials, so this is not a held-out prediction.
- The figure: one block for each kind of unit. In a block, one row for each
  gap group, with the fixation in the left column and the time after the exit
  in the right column. A row shows the rate against time from the flash
  before, from `TRACE_BEFORE_MS` before it to the end of the response window
  of the flash: the response to the flash before, and then the flash of the
  row at its gap. The rate of a unit is divided by its mean observed rate
  around all these flashes; lines are the mean over the units of the kind,
  smoothed (Gaussian, `SMOOTH_MS`). The dashed vertical line is the flash
  before and the solid line is the flash. In the group with the longest gaps
  the flash comes at different times: the solid line is the shortest gap and
  the band goes to the longest one.
- Measures in the output: the rate before the flash (mean from `BASELINE_MS`)
  and the response (mean from `RESPONSE_MS`), over the unit mean, for each
  condition; and for each kind and gap, the two-sided Wilcoxon signed-rank
  test of the response after the exit against in the fixation. The rate
  before a flash is not a neutral level: it is a part of the response to the
  flash before.

Trials, alignment, and exclusions are those of `fit.py`. The sampling unit is
the unit. The script writes `flash_by_gap.svg` and `flash_by_gap.json` in the
session directory (`--root`). Run `flash_kernels` first.

    uv run python -m thesis.ephys.analyses.glm.flash_by_gap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import KIND_LIMIT
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    save_figure,
)

# The gaps are multiples of 40 ms. Each gap up to `LONG_GAP_MS` is a group, and
# all longer gaps are one group.
LONG_GAP_MS = 200
# The window of the response around a flash, and the time that a row shows
# before the flash before.
LAGS_MS = (-40, 150)
TRACE_BEFORE_MS = 40
BASELINE_MS = (-20, 20)
RESPONSE_MS = (40, 100)
SMOOTH_MS = 5
MIN_RATE = 2.0
VISUAL = CONCEPT_COLORS["visual"]
NAMES = ("observed", "model")
COLORS = (OBSERVED_COLOR, MODEL_COLOR)
PERIODS = ("still", "move")
KINDS = ("units that fire more after a flash", "units that fire less after a flash")


def flashes_by_period(trials) -> dict[str, np.ndarray]:
    """Return the trial row, time, gap (ms), and period of each flash that is used."""
    out = {key: [] for key in ("row", "time", "gap", "period")}
    window = LAGS_MS[1] / 1000
    for row, (flashes, entry, leave, response) in enumerate(
        zip(
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            trials["response_port_entry_s"],
            strict=True,
        )
    ):
        flashes = np.asarray(flashes, dtype=float)
        flashes = flashes[(flashes >= entry) & (flashes <= response)]
        before, times = flashes[:-1], flashes[1:]
        in_fixation = times + window <= leave
        after_exit = (before >= leave) & (times + window <= response)
        keep = in_fixation | after_exit
        out["row"].append(np.full(int(keep.sum()), row))
        out["time"].append(times[keep])
        out["gap"].append(1000 * (times - before)[keep])
        out["period"].append(after_exit[keep].astype(int))
    return {key: np.concatenate(values) for key, values in out.items()}


def _window_mean(values: np.ndarray, lags: np.ndarray, limits) -> np.ndarray:
    return values[..., (lags >= limits[0]) & (lags < limits[1])].mean(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    model = cli.PoissonGLM(root=args.root)
    kernel_path = args.root / "flash_kernels.npz"
    if not kernel_path.exists():
        raise FileNotFoundError(f"{kernel_path}: run `flash_kernels` first.")
    with np.load(kernel_path) as saved:
        kernel_response = dict(
            zip(saved["unit_ids"].tolist(), saved["fixation_response"], strict=True)
        )
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared)
    table = flashes_by_period(trials)
    rows = table["row"]
    slot = 1000 * dmat.FLASH_SLOT_S
    gaps = slot * np.rint(table["gap"] / slot)
    group = (np.minimum(gaps, LONG_GAP_MS) / slot).astype(int) - 1
    groups = int(group.max()) + 1
    condition = table["period"] * groups + group  # the period, then the gap group
    conditions = len(PERIODS) * groups
    # Two alignments of the bins of the model grid. For the measures: the flash.
    # For the rows of the figure: the flash before it.
    lags = np.arange(*LAGS_MS)
    trace_lags = np.arange(-TRACE_BEFORE_MS, LONG_GAP_MS + LAGS_MS[1])

    def grid_columns(event_times, offsets):
        first = np.rint(
            (event_times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S
        )
        return first.astype(int)[:, None] + offsets[None, :]

    columns = grid_columns(table["time"], lags)
    trace_columns = grid_columns(table["time"] - table["gap"] / 1000, trace_lags)
    if columns.min() < 0 or columns.max() >= grid[1]:
        raise ValueError("A response window is outside the model grid.")
    # A bin of a row is used up to the end of the response window of the flash.
    used = (
        (trace_columns >= 0)
        & (trace_columns < grid[1])
        & (trace_lags[None, :] < (gaps + LAGS_MS[1])[:, None])
    )
    trace_columns = np.clip(trace_columns, 0, grid[1] - 1)
    # weights[c]: the mean over the flashes of condition c, for each bin.
    members = np.stack([used & (condition == c)[:, None] for c in range(conditions)])
    flashes_in_bin = members.sum(1)
    weights = members / np.maximum(flashes_in_bin, 1)[:, None, :]
    in_condition = np.stack([condition == c for c in range(conditions)]).astype(float)
    in_condition /= in_condition.sum(1, keepdims=True)

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, args.units)]
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
        )
    }
    out = {key: [] for key in ("trace", "before", "response", "kind")}
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        size = float(kernel_response[int(unit_id)])
        if abs(size) <= KIND_LIMIT:
            continue
        spikes = np.asarray(spikes, dtype=float)
        rate_grid = (
            build_unit_counts(alignments, spikes).reshape(grid) / dmat.BINWIDTH_S
        )
        observed = rate_grid[rows[:, None], columns]
        scale = observed.mean()  # the mean observed rate around these flashes
        if scale < MIN_RATE:
            continue
        # The rate of the final refit in every bin, with the observed history.
        record = final[int(unit_id)]
        width = int(metadata["base_columns"]) + int(record["components"])
        coefficients = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        log_rate = (
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ coefficients[:width]
            + history @ coefficients[width:]
        )
        predicted_grid = np.exp(log_rate).reshape(grid) / dmat.BINWIDTH_S
        sources = (rate_grid, predicted_grid)
        # (condition, observed or model, bin) and (condition, observed or model)
        traces = np.stack([source[rows[:, None], trace_columns] for source in sources])
        out["trace"].append(np.einsum("cfl,mfl->cml", weights, traces) / scale)
        around = np.stack([source[rows[:, None], columns] for source in sources])
        mean = np.einsum("cf,mfl->cml", in_condition, around) / scale
        out["before"].append(_window_mean(mean, lags, BASELINE_MS))
        out["response"].append(_window_mean(mean, lags, RESPONSE_MS))
        out["kind"].append(int(size < 0))
    kind = np.asarray(out["kind"])
    trace, before, response = (
        np.asarray(out[key]) for key in ("trace", "before", "response")
    )  # (unit, condition, observed or model[, bin])
    group_names = [
        f"{gaps[group == g].min():.0f}" + ("" if np.ptp(gaps[group == g]) == 0 else "+")
        for g in range(groups)
    ]
    stats = {
        "units": int(len(units)),
        "units_of_each_kind": {
            name: int(np.sum(kind == k)) for k, name in enumerate(KINDS)
        },
        "gap_groups_ms": group_names,
        "flashes": {
            period: np.bincount(group[table["period"] == p], minlength=groups).tolist()
            for p, period in enumerate(PERIODS)
        },
    }
    for k, kind_name in enumerate(KINDS):
        of_kind = kind == k
        entry = {}
        for m, name in enumerate(NAMES):
            for p, period in enumerate(PERIODS):
                block = slice(p * groups, (p + 1) * groups)
                entry[f"{name}, {period}"] = {
                    "median_rate_before_the_flash": np.median(
                        before[of_kind, block, m], 0
                    ).tolist(),
                    "median_response": np.median(
                        response[of_kind, block, m], 0
                    ).tolist(),
                }
            change = response[of_kind, groups:, m] - response[of_kind, :groups, m]
            entry[f"{name}, response move minus still"] = {
                "median": np.median(change, 0).tolist(),
                "units_higher": (change > 0).sum(0).tolist(),
                "wilcoxon_p": [
                    float(wilcoxon(change[:, g]).pvalue) for g in range(groups)
                ],
            }
        stats[kind_name] = entry

    has_data = (flashes_in_bin > 0).astype(float)
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 5.6))
        outer = figure.add_gridspec(1, len(KINDS), wspace=0.22)
        letter_axes = []
        for k in range(len(KINDS)):
            # The bins after the end of a row have no data, so the smoothing uses
            # only the bins of the row. (condition, observed or model, bin)
            mean = trace[kind == k].mean(0) * has_data[:, None, :]
            smooth = gaussian_filter1d(mean, SMOOTH_MS, axis=-1) / np.maximum(
                gaussian_filter1d(has_data, SMOOTH_MS, axis=-1)[:, None, :], 1e-9
            )
            block = outer[k].subgridspec(groups, len(PERIODS), hspace=0.2, wspace=0.08)
            axes = np.empty((groups, len(PERIODS)), dtype=object)
            for g in range(groups):
                for p in range(len(PERIODS)):
                    axis = figure.add_subplot(
                        block[g, p], sharex=axes[0, 0], sharey=axes[0, 0]
                    )
                    axes[g, p] = axis
                    c = p * groups + g
                    shortest = gaps[group == g].min()
                    drawn = trace_lags < shortest + LAGS_MS[1]
                    for m, color in enumerate(COLORS):
                        axis.plot(
                            trace_lags[drawn], smooth[c, m, drawn], color=color, lw=1.0
                        )
                    axis.axvline(0, color=VISUAL, linestyle="--", linewidth=0.8)
                    axis.axvline(shortest, color=VISUAL, linewidth=1.0)
                    if np.ptp(gaps[group == g]) > 0:  # the later flashes of the group
                        axis.axvspan(
                            shortest,
                            gaps[group == g].max(),
                            color=VISUAL,
                            alpha=0.08,
                            lw=0,
                        )
                    axis.tick_params(labelbottom=g == groups - 1, labelleft=p == 0)
                    if p == 0:
                        axis.text(
                            0.98,
                            0.93,
                            f"{group_names[g]} ms gap",
                            transform=axis.transAxes,
                            va="top",
                            ha="right",
                            fontsize=6,
                        )
                    if g == 0:
                        axis.text(
                            0.5,
                            1.05,
                            PERIODS[p],
                            transform=axis.transAxes,
                            va="bottom",
                            ha="center",
                            fontsize=7,
                        )
            axes[0, 0].set_xlim(-TRACE_BEFORE_MS, trace_lags[-1] + 10)
            axes[groups // 2, 0].set_ylabel("normalized firing rate")
            figure.text(
                (axes[-1, 0].get_position().x0 + axes[-1, -1].get_position().x1) / 2,
                axes[-1, 0].get_position().y0 - 0.075,
                "time from the flash before (ms)",
                ha="center",
            )
            figure.text(
                axes[0, 0].get_position().x0,
                axes[0, 0].get_position().y1 + 0.055,
                KINDS[k],
                fontsize=7.5,
            )
            letter_axes.append(axes[0, 0])
        for m, (name, color) in enumerate(zip(NAMES, COLORS, strict=True)):
            figure.text(0.84 + 0.085 * m, 0.955, name, color=color, ha="right")
        figure.text(
            0.5,
            0.005,
            "the flash before (dashed), the flash (solid)",
            color=VISUAL,
            ha="center",
            fontsize=6.5,
        )
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("ab", letter_axes, strict=True):
            edge = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                edge - 0.02,
                axis.get_position().y1 + 0.05,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        print(save_figure(figure, args.root / "flash_by_gap"))
    output = args.root / "flash_by_gap.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
