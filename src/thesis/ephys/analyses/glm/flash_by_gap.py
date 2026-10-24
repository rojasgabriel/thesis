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
  mean response over the two gap classes in the fixation). A unit fires more
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
  smoothed (Gaussian, `SMOOTH_MS`). The dotted vertical line is the flash
  before and the solid line is the flash. In the group with the longest gaps
  the flash comes at different times: the solid line is the shortest gap and
  the band goes to the longest one.
- Measure in the output: the matched peak of `peaks.py` (spikes/s; baseline
  from the still rate before the flash), for each gap group, in the port and after the exit,
  on the observed rate and the model rate, for the flash-excited units of
  `raw_video_trust_classification.json` (StimulusResponse of 2026-10-24, after the gain fix); and the two-sided Wilcoxon signed-rank test of move minus
  still for each gap group. The traces are of all units of each kind, as
  above.
- The units of both mice are pooled in the traces (the title gives the units
  and mice); the tests are within each mouse.

Trials, alignment, and exclusions are those of `fit.py`. The sampling unit is
the unit, and each test is within one mouse (`SESSIONS`); the units of the two
mice are not pooled. Panel a: units that fire more; b: units that fire less. The
script writes `flash_by_gap.svg` and `flash_by_gap.json` (by session) in
`--out`. Run `flash_kernels` first.

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

from thesis.ephys.analyses.glm import cli, dmat, peaks
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import KIND_LIMIT, SESSIONS
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
SMOOTH_MS = 5
MIN_RATE = 2.0
VISUAL = CONCEPT_COLORS["visual"]
NAMES = ("observed", "model")
COLORS = (OBSERVED_COLOR, MODEL_COLOR)
PERIODS = ("in port", "after exit")
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


def session_traces(root: Path, unit_set: str) -> dict:
    """Return the mean traces of each unit of one session, and the statistics."""
    model = cli.PoissonGLM(root=root)
    kernel_path = root / "flash_kernels.npz"
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
    units = units.iloc[analysis_unit_indices(units, unit_set)]
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
        )
    }
    # The saved flash-excited units, the same as in every other figure.
    excited = set(
        json.loads((root.parent / "raw_video_trust_classification.json").read_text())[
            root.name
        ]
    )
    in_peak = lags < peaks.PEAK_MS[1]  # the bins of the matched peak
    out = {key: [] for key in ("trace", "kind", "peak")}
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        size = float(kernel_response[int(unit_id)])
        spikes = np.asarray(spikes, dtype=float)
        rate_grid = (
            build_unit_counts(alignments, spikes).reshape(grid) / dmat.BINWIDTH_S
        )
        observed = rate_grid[rows[:, None], columns]
        scale = observed.mean()  # the mean observed rate around these flashes
        traced = abs(size) > KIND_LIMIT and scale >= MIN_RATE
        if not traced and int(unit_id) not in excited:
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
        if int(unit_id) in excited:
            # The matched peak (`peaks.py`) of each gap group, in the port and after the exit,
            # for the observed rate and the model: (group, observed or model, period).
            around = np.stack([source[rows[:, None], columns] for source in sources])
            mean = np.einsum("cf,mfl->cml", in_condition, around)[:, :, in_peak]
            out["peak"].append(
                [
                    [peaks.peaks(mean[[g, groups + g], m]) for m in range(len(NAMES))]
                    for g in range(groups)
                ]
            )
        if traced:
            traces = np.stack(
                [source[rows[:, None], trace_columns] for source in sources]
            )
            out["trace"].append(np.einsum("cfl,mfl->cml", weights, traces) / scale)
            out["kind"].append(int(size < 0))
    kind = np.asarray(out["kind"])
    trace = np.asarray(out["trace"])  # (unit, condition, observed or model, bin)
    peak = np.asarray(out["peak"])  # (excited unit, group, observed or model, period)
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
    change = peak[..., 1] - peak[..., 0]  # (excited unit, group, observed or model)
    stats["matched peak (spikes/s), after exit minus in port, excited units"] = {
        "units": len(peak),
        **{
            name: {
                "median": np.median(change[:, :, m], 0).tolist(),
                "units_higher_in_move": (change[:, :, m] > 0).sum(0).tolist(),
                "wilcoxon_p": [
                    float(wilcoxon(change[:, g, m]).pvalue) for g in range(groups)
                ],
            }
            for m, name in enumerate(NAMES)
        },
    }

    return {
        "subject": root.name.split("_")[0],
        "trace": trace,
        "kind": kind,
        "gaps": gaps,
        "group": group,
        "groups": groups,
        "group_names": group_names,
        "trace_lags": trace_lags,
        "has_data": (flashes_in_bin > 0).astype(float),
        "stats": stats,
    }


def _smooth(sessions: list[dict], k: int) -> np.ndarray:
    """Return the smoothed mean trace of one kind over the pooled units of all mice.

    (condition, observed or model, bin). The bins after the end of a row have
    no data, so each mouse counts only where its rows have data, and the
    smoothing uses only those bins.
    """
    total = sum(
        data["trace"][data["kind"] == k].sum(0) * data["has_data"][:, None, :]
        for data in sessions
    )
    count = sum(
        np.sum(data["kind"] == k) * data["has_data"][:, None, :] for data in sessions
    )
    return gaussian_filter1d(total, SMOOTH_MS, axis=-1) / np.maximum(
        gaussian_filter1d(count, SMOOTH_MS, axis=-1), 1e-9
    )


def draw_block(figure, slot, sessions: list[dict], k: int):
    """Draw the rows of one kind of unit, both mice in each axis; return the top-left axis."""
    first = sessions[0]
    gaps, group, groups = first["gaps"], first["group"], first["groups"]
    if any(data["group_names"] != first["group_names"] for data in sessions):
        raise ValueError("The mice have different gap groups.")
    smooth = _smooth(sessions, k)
    block = slot.subgridspec(groups, len(PERIODS), hspace=0.2, wspace=0.08)
    axes = np.empty((groups, len(PERIODS)), dtype=object)
    for g in range(groups):
        for p in range(len(PERIODS)):
            axis = figure.add_subplot(block[g, p], sharex=axes[0, 0], sharey=axes[0, 0])
            axes[g, p] = axis
            c = p * groups + g
            shortest = gaps[group == g].min()
            lags = first["trace_lags"]
            drawn = lags < shortest + LAGS_MS[1]
            for m, color in enumerate(COLORS):
                axis.plot(lags[drawn], smooth[c, m, drawn], color=color, lw=1.0)
            axis.axvline(0, color=VISUAL, linestyle=":", linewidth=0.9)
            axis.axvline(shortest, color=VISUAL, linewidth=1.0)
            if np.ptp(gaps[group == g]) > 0:  # the later flashes of the group
                axis.axvspan(
                    shortest, gaps[group == g].max(), color=VISUAL, alpha=0.08, lw=0
                )
            axis.tick_params(labelbottom=g == groups - 1, labelleft=p == 0)
            if p == 0:
                axis.text(
                    0.98,
                    0.93,
                    f"{first['group_names'][g]} ms gap",
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
    axes[0, 0].set_xlim(-TRACE_BEFORE_MS, first["trace_lags"][-1] + 10)
    axes[groups // 2, 0].set_ylabel("normalized firing rate")
    figure.text(
        (axes[-1, 0].get_position().x0 + axes[-1, -1].get_position().x1) / 2,
        axes[-1, 0].get_position().y0 - 0.06,
        "time from the flash before (ms)",
        ha="center",
    )
    counts = (
        f"{sum(int(np.sum(data['kind'] == k)) for data in sessions)} units,"
        f" {len(sessions)} mice"
    )
    figure.text(
        axes[0, 0].get_position().x0,
        axes[0, 0].get_position().y1 + 0.04,
        f"{KINDS[k]} ({counts})",
        fontsize=7.5,
    )
    return axes[0, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    sessions = [session_traces(args.out / session, args.units) for session in SESSIONS]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 6.2))
        outer = figure.add_gridspec(
            1, len(KINDS), wspace=0.22, left=0.08, right=0.98, top=0.86, bottom=0.12
        )
        letter_axes = [
            draw_block(figure, outer[0, k], sessions, k) for k in range(len(KINDS))
        ]
        for m, (name, color) in enumerate(zip(NAMES, COLORS, strict=True)):
            figure.text(0.8 + 0.1 * m, 0.955, name, color=color, ha="left")
        figure.text(
            0.5,
            0.02,
            "the flash before (dotted), the flash (solid)",
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
                axis.get_position().y1 + 0.04,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        print(save_figure(figure, args.out / "flash_by_gap"))
    stats = {
        session: data["stats"] for session, data in zip(SESSIONS, sessions, strict=True)
    }
    output = args.out / "flash_by_gap.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
