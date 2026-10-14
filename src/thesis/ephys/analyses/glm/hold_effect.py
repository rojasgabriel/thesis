"""Show the suppression of V1 firing while the animal holds with the stimulus on.

Question: V1 units fire less from about 80 ms after the first flash until the
animal leaves the center port. Does that follow the stimulus (adaptation to the
flashes) or the fixation (the time in the center port)? The flashes go on from the first flash to the response
entry, so the time after the center exit has the stimulus without the fixation.

Comparison: the fixation (first flash to center exit) against the time from the
center exit to the response entry, in the same trials and the same units. The
comparison is paired: each unit has a value in both periods. Panel a is
GRB006. In panels b to h the units of both mice (`SESSIONS`) are pooled
(standard practice); in d, e, and h the large marker is the pooled median
with its hierarchical-bootstrap 95% interval (mice, then units), and the open
circles are the median of each mouse. The tests are within each mouse.

- Panel a (examples): spikes of two units against time from the first flash
  (left) and from the center exit (right), on the time axes of panels b and c,
  with the trials sorted by the length of the fixation. The vertical line is the
  event at time 0. Top: the unit with
  the lowest fixation level among the units with a rate in `SUPPRESSED_RATES`.
  Bottom: the unit with the highest fixation level among the units that fire at
  `EXAMPLE_MIN_RATE` or more.
- Panel b (observation): raw firing rate, no model, against time from the first
  flash and from the center exit. The rate of a unit is divided by its mean
  rate over the valid bins; the line is the mean over units and the band is its
  SEM. The
  top row is the fraction of trials with a flash in each 40 ms.
- Panel c (characterization): the kernels of a descriptive model, as a change
  in log rate. The model is the design of `dmat.py` with one change: the
  withdrawal kernel (0 to 100 ms) is replaced with a kernel from the center
  exit to the response entry (`AFTER_BASES` raised cosines, `AFTER_WINDOW_S`).
  Both kernels are relative to the time before the first flash. Thin lines are
  units; the thick line and band are the median and the interquartile range.
- Panel d (the paired comparison): each unit's mean kernel value over
  `LEVEL_MS` of the fixation against its mean over the same time after the center
  exit. Error bars are the SD over the 10 cross-validation folds, in both
  directions. Two-sided Wilcoxon signed-rank test of the two values. Points
  outside `LEVEL_LIMITS` are not drawn; they are in the statistics, and their
  number is in `hold_effect_stats.json`.
- Panel e (control for the order of fitting): does the fixation kernel take
  variance from the other terms? Each unit's fixation level in the main model (all
  terms fit together) against its level in a two-step fit: the design with no
  fixation kernel first, then the fixation kernel alone on top of that fixed
  prediction (`two_step_levels`). The two-step value is a lower bound. Both are
  in-sample. The axis is `ORDER_LIMITS`; a unit with a two-step value beyond
  `NO_ESTIMATE` has no estimate (its fit with no penalty did not converge). It
  is listed in the output and stays in the rank test.
- Panel f (control for stillness): raw rate against motion energy in the
  center port and at a side port (`port_vs_stillness.py`), with its slope test.
  The two animals have different camera views, so motion energy is not the
  same measure.
- Panel g (control for the movement after the exit): the rate around a flash
  in the fixation, and around a flash after the exit with less and with more
  motion energy of the video (lower and upper half), from
  `matched_flashes.py`. The first flash of a trial is not used. The rate of a
  unit is divided by its mean over the flashes in the fixation and after the
  exit; lines are the mean and bands the SEM over units. The two groups after
  the exit are matched for the time from the first flash, the gap from the
  flash before, and the time from the exit. The fixation is not matched to
  them for time: the fixation has almost the same length in all trials.
- Panel h (which units): each unit's fixation level (mean of its fixation
  kernel over `LEVEL_MS`, main model) against the unique ΔD² of its visual
  flash regressor. A value below `UNIQUE_FLOOR` is drawn there, and a level
  outside `LEVEL_LIMITS` is not drawn; all units are in the Spearman
  correlation of `matched_flashes.json`.

The descriptive model and the design with no fixation kernel are fit here (`glm
fit` stages, with cross-validation) in the directories with the suffixes
`ROOT_SUFFIX` and `NO_HOLD_SUFFIX` next to the session directory. It
is not the model of the main figure: the longer kernel after the exit does not
predict reliably better than the withdrawal kernel (`docs/glm.md`).

Units, trials, alignment (first flash), baseline, and exclusions are those of
`fit.py`. Bins after the response entry are excluded. The sampling unit is the
unit. The script writes `hold_effect.svg`, `hold_effect_stats.json` (by
session), and `hold_effect_caption.md` in `--out`.

    uv run python -m thesis.ephys.analyses.glm.hold_effect --fit --root <session>
    uv run python -m thesis.ephys.analyses.glm.hold_effect

The first command fits the two variants of one session (about 20 minutes each
for all units; a finished fit is not run again) and draws nothing. The second
command makes no fit: it draws the figure from the fits of both sessions and
stops with the command to run when a fit is missing. Run `port_vs_stillness`
and `matched_flashes` for each session first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem, wilcoxon
from sklearn.linear_model import PoissonRegressor

from thesis.ephys.analyses.glm import cli, dmat, port_vs_stillness
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    crossvalidate,
    fit_models,
    load_unit_records,
    refit_final,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.grid import INPUTS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    pooled_points,
    save_figure,
    separate_axes,
)

CAPTION = """\
**V1 firing changes during fixation and at the withdrawal, while the flashes
go on.** The fixation is the time from the first flash to the withdrawal from
the center port. The flashes continue until the animal enters a response port,
so the time after the withdrawal has the stimulus without the fixation. Two
mice: {units}. In b to h the units of both mice are pooled; in d, e, and h
the large marker is the pooled median with its hierarchical-bootstrap 95%
interval (the mice, then the units of each), and the open circles are the
median of each mouse ({subject} and {other}). The tests are within each mouse. **a**, Spikes of two example units of
{subject} against time from the first flash (left) and from the withdrawal
(right), with the trials sorted by the length of the fixation. The vertical
line is the event at time 0; in the left panels the orange line is the
withdrawal of each trial. Top: a unit with a lower rate in fixation. Bottom: a
unit with a higher rate in fixation. **b**, Observation, with no model. Top:
the fraction of trials with a flash in each 40 ms. Bottom: the firing rate,
with the rate of each unit divided by its mean rate (mean and SEM over units).
The rate is {before} before the first flash, {late} in the last 200 ms of the
fixation, and {after_rate} from 100 to 400 ms after the withdrawal. **c**, The
kernels of a descriptive model, as a change in log rate relative to the time
before the first flash. This model is the GLM of the main figure with one
change: a kernel from the withdrawal to the response entry in place of the
100-ms withdrawal kernel. Thin lines are units; the thick lines and the
bands are the median and the interquartile range. **d**, The paired comparison. For
each unit, the mean kernel value from {lo} to {hi} ms after the first flash
against the mean over the same time after the withdrawal (error bars: SD over
the 10 cross-validation folds). {paired} {outside_d} **e**, Control for the
order of fitting. The level in fixation when all terms are fit together
against the level when the fixation kernel is fit last, on what the other
terms do not explain (a lower bound). {two_step} {outside_e} {no_estimate}
**f**, Control
for stillness. Raw firing rate against motion energy of the video, in the
center port and at a side port, in each animal. {stillness} The two animals
have different camera views, so motion energy is not the same measure in
both. **g**, Control for the movement after the withdrawal. The firing rate
around a flash in fixation, and around a flash after the withdrawal with less
and with more motion energy of the video (lower and upper half; mean and SEM
over units; the first flash of a trial is not used), in each animal. {energy}
**h**, The level in fixation against the unique ΔD² of the visual flash.
{relation} {outside_h}
"""
ROOT_SUFFIX = "_after_exit_kernel"
NO_HOLD_SUFFIX = "_no_hold_term"
# The excited example unit must fire at this rate or more (spikes/s).
EXAMPLE_MIN_RATE = 2.0
# The suppressed example unit must have a rate in this range (spikes/s): enough
# spikes to see the hold, few enough to see single spikes.
SUPPRESSED_RATES = (5.0, 20.0)
AFTER = "after_exit"
AFTER_WINDOW_S = 0.6
AFTER_BASES = 8
LEVEL_MS = (150, 500)
# Axis limits of the kernel panels (Δ log rate).
LEVEL_LIMITS = (-1.6, 1.6)
# Panel e: a wider axis, so that only the units with no two-step estimate are
# outside it. A two-step value beyond `NO_ESTIMATE` is a fit with no penalty
# that did not converge (a unit with very few spikes in fixation).
ORDER_LIMITS = (-2.5, 1.6)
NO_ESTIMATE = 3.0
UNIQUE_FLOOR = 1e-4  # panel h: a smaller unique ΔD² is drawn here
RATE_BIN_MS = 20
FLASH_SLOT_S = 0.04
FLASH_WINDOW_MS = (-100, 500)
EXIT_WINDOW_MS = (-200, 500)
TASK, VISUAL = CONCEPT_COLORS["task"], CONCEPT_COLORS["visual"]
MOTION = CONCEPT_COLORS["motion energy"]


def after_exit_basis() -> dmat.BasisFunction:
    """Bases of the kernel after the center exit, spaced like the hold kernel."""
    post = AFTER_WINDOW_S + dmat.BINWIDTH_S
    if dmat.HOLD_LOG_STRETCH_S:
        return dmat.StretchedCosineBasis(
            AFTER_BASES, post, dmat.BINWIDTH_S, dmat.HOLD_LOG_STRETCH_S
        )
    return dmat.RaisedCosineBasis(AFTER_BASES, 0, post, dmat.BINWIDTH_S)


def _fit_variant(source: Path, unit_set: str, suffix: str, attributes: dict) -> Path:
    """Fit the design with some `dmat` attributes replaced, in a sibling directory."""
    root = source.with_name(source.name + suffix)
    model = cli.PoissonGLM(root=root)
    directory = model.fit_dir(unit_set)
    root.mkdir(exist_ok=True)
    for name in INPUTS:
        if not (root / name).exists():
            shutil.copy(source / name, root / name)
    # ponytail: the design reads module attributes; set them for this fit and
    # restore them. Make them arguments if the design gets more callers.
    saved = {name: getattr(dmat, name) for name in attributes}
    for name, value in attributes.items():
        setattr(dmat, name, value)
    try:
        if not model.design.exists():
            dmat.prepare_common_design(model.windows, model.video, model.design)
        # Each stage keeps its finished units, so a stopped fit goes on.
        checkpoints = directory / "checkpoints"
        if not list(checkpoints.glob("unit_*_final.json")) or len(
            list(checkpoints.glob("unit_*_final.json"))
        ) < len(list(checkpoints.glob("unit_*_validation.json"))):
            fit_models(model.windows, model.design, unit_set, directory)
            summary = json.loads((directory / "validation_summary.json").read_text())
            components = int(summary["selected_video_components"])
            crossvalidate(model.windows, model.design, unit_set, directory, components)
            refit_final(model.windows, model.design, unit_set, directory, components)
    finally:
        for name, value in saved.items():
            setattr(dmat, name, value)
    return root


def _time_in_trial() -> np.ndarray:
    centers, _, _ = dmat.construct_timebins(dmat.PRE_S, dmat.POST_S, dmat.BINWIDTH_S)
    return centers


def _kernel_block(basis: np.ndarray, since: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Place the basis rows at each bin's time since an event; zero where not kept."""
    index = np.rint(since / dmat.BINWIDTH_S).astype(int)
    inside = (index >= 0) & (index < len(basis)) & keep
    block = np.where(inside[..., None], basis[np.clip(index, 0, len(basis) - 1)], 0.0)
    return block.reshape(-1, basis.shape[1])


def fit_descriptive(source: Path, unit_set: str) -> Path:
    """Fit the design with a kernel from the center exit to the response entry."""
    basis = after_exit_basis().basis
    original = dmat.build_task_design

    def with_after_kernel(alignments, trials):
        values, manifest = original(alignments, trials)
        leave = trials["center_exit_s"].to_numpy(dtype=float)
        since = _time_in_trial()[None, :] - (leave - np.asarray(alignments))[:, None]
        limit = trials["response_port_entry_s"].to_numpy(dtype=float) - leave
        manifest.append(
            {
                "name": AFTER,
                "group": "task",
                "events": len(alignments),
                "columns": basis.shape[1],
                "coding": "kernel on the center exit, zero from the response entry on",
            }
        )
        block = _kernel_block(basis, since, since < limit[:, None])
        return np.column_stack((values, block)), manifest

    return _fit_variant(
        source,
        unit_set,
        ROOT_SUFFIX,
        {"build_task_design": with_after_kernel, "WITHDRAWAL_KERNEL": False},
    )


def two_step_levels(source: Path, unit_set: str) -> dict:
    """Return each unit's hold level from the joint fit and from a two-step fit.

    Step 1 is the design with no hold kernel. Step 2 fits the hold kernel alone
    on top of the fixed prediction of step 1 (Poisson, no intercept, no
    penalty), so the other terms keep all that they can explain. Both fits use
    all valid bins; the values are in-sample.
    """
    cache = source / f"hold_two_step_{unit_set}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    first = cli.PoissonGLM(
        root=_fit_variant(source, unit_set, NO_HOLD_SUFFIX, {"HOLD_KERNEL": False})
    )
    joint = cli.PoissonGLM(root=source)
    prepared = dmat._load_windows(joint.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments = prepared["alignments"]
    valid = dmat._valid_bin_mask(prepared)
    basis = dmat.task_temporal_bases()[dmat.HOLD]
    times = 1000 * basis.basis_time
    level = (times >= LEVEL_MS[0]) & (times < LEVEL_MS[1])
    leave = trials["center_exit_s"].to_numpy(dtype=float) - alignments
    since = np.broadcast_to(_time_in_trial(), (len(alignments), len(_time_in_trial())))
    hold_columns = _kernel_block(basis.basis, since, since < leave[:, None])[valid]
    metadata = json.loads(first.design.with_suffix(".json").read_text())
    common = np.load(first.design, mmap_mode="r", allow_pickle=False)
    joint_metadata = json.loads(joint.design.with_suffix(".json").read_text())
    columns = [
        index
        for index, name in enumerate(joint_metadata["column_names"])
        if name.startswith(f"{dmat.HOLD}_basis_")
    ]
    scale = np.asarray(joint_metadata["training_scale"])[columns]
    joint_records = {
        item["unit_id"]: item
        for item in load_unit_records(
            joint.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
        )
    }
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")
    out = {"unit_ids": [], "joint": [], "two_step": [], "rate": []}
    for record in load_unit_records(
        first.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
    ):
        unit_id = record["unit_id"]
        spikes = np.asarray(units.loc[unit_id, "spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).astype(float)[valid]
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        width = int(metadata["base_columns"]) + int(record["components"])
        weights = np.asarray(record["coefficients"], dtype=float)
        predicted = np.exp(
            record["intercept"]
            + np.asarray(common[:, :width])[valid] @ weights[:width]
            + history[valid] @ weights[width:]
        )
        second = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=1000).fit(
            hold_columns, counts / predicted, sample_weight=predicted
        )
        joint_weights = np.asarray(joint_records[unit_id]["coefficients"])[columns]
        out["unit_ids"].append(unit_id)
        out["rate"].append(joint_records[unit_id]["fit_mean_count"] / dmat.BINWIDTH_S)
        out["two_step"].append((second.coef_ @ basis.basis.T)[level].mean())
        out["joint"].append(((joint_weights / scale) @ basis.basis.T)[level].mean())
        print(f"Two-step fit of unit {unit_id}.", flush=True)
    out = {key: np.asarray(value) for key, value in out.items()}
    np.savez(cache, **out)
    return out


def kernel_values(root: Path, unit_set: str) -> dict:
    """Return the hold and after-exit kernels (Δ log rate): final refit and folds."""
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names = metadata["column_names"]
    scale = np.asarray(metadata["training_scale"])
    checkpoints = model.fit_dir(unit_set) / "checkpoints"
    final = load_unit_records(checkpoints, "unit_*_final.json")
    folds = {
        item["unit_id"]: item
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    out = {"unit_ids": np.asarray([item["unit_id"] for item in final])}
    for name, basis in (
        (dmat.HOLD, dmat.task_temporal_bases()[dmat.HOLD]),
        (AFTER, after_exit_basis()),
    ):
        columns = [i for i, column in enumerate(names) if column.startswith(name)]
        times = 1000 * basis.basis_time
        level = (times >= LEVEL_MS[0]) & (times < LEVEL_MS[1])
        values_of = lambda weights: (  # noqa: E731
            (np.asarray(weights)[columns] / scale[columns]) @ basis.basis.T
        )
        curves = np.asarray([values_of(item["coefficients"]) for item in final])
        fold_levels = [
            [
                values_of(fold["coefficients"])[level].mean()
                for fold in folds[item["unit_id"]]["folds"]
            ]
            for item in final
        ]
        out[name] = {
            "times": times,
            "values": curves,
            "level": curves[:, level].mean(axis=1),
            "level_sd": np.asarray([np.std(values) for values in fold_levels]),
        }
    return out


def raw_rates(source: Path, unit_set: str, examples: tuple[int, ...]) -> dict:
    """Return each unit's normalized rate from the first flash and from the center exit."""
    prepared = dmat._load_windows(source / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared).reshape(grid)
    leave = trials["center_exit_s"].to_numpy(dtype=float) - alignments
    step = RATE_BIN_MS  # 1-ms bins per rate bin

    def window(events, limits_ms, keep):
        """Index of each 1-ms bin around the events, and the bins that count."""
        fine = (np.arange(*limits_ms) + 0.5) / 1000
        columns = np.rint(
            (events[:, None] + fine[None, :] - relative[0]) / dmat.BINWIDTH_S
        ).astype(int)
        inside = (columns >= 0) & (columns < grid[1])
        columns = np.clip(columns, 0, grid[1] - 1)
        rows = np.arange(grid[0])[:, None]
        used = inside & valid[rows, columns] & keep(relative[columns])
        centers = np.arange(*limits_ms, step) + step / 2
        return rows, columns, used, centers

    in_hold = lambda times: times < leave[:, None]  # noqa: E731
    after_flash = lambda times: times >= 0  # noqa: E731
    windows = {
        "flash": window(np.zeros(grid[0]), FLASH_WINDOW_MS, in_hold),
        "exit": window(leave, EXIT_WINDOW_MS, after_flash),
    }
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, unit_set)]
    rates = {name: [] for name in windows}
    order = np.argsort(leave)
    rasters = {}
    for _, unit in units.iterrows():
        spikes = np.asarray(unit["spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).reshape(grid)
        mean_count = counts[valid].mean()
        if mean_count <= 0:
            continue
        if int(unit["unit_id"]) in examples:
            # Spike times of each trial from the first flash and from the exit.
            rows, bins = np.nonzero(counts[order] > 0)
            spike_ms = 1000 * relative[bins]
            rasters[int(unit["unit_id"])] = {
                "flash": (spike_ms, rows),
                "exit": (spike_ms - 1000 * leave[order][rows], rows),
            }
        for name, (rows, columns, used, _) in windows.items():
            shape = (grid[0], -1, step)
            spikes_in = (counts[rows, columns] * used).reshape(shape).sum((0, 2))
            exposure = used.reshape(shape).sum((0, 2))
            rates[name].append(spikes_in / np.maximum(exposure, 1) / mean_count)
    flashes = [
        np.asarray(times, dtype=float) - first
        for times, first in zip(trials["stim_pulse_times_s"], alignments, strict=True)
    ]

    def flash_fraction(events, limits_ms):
        edges = np.arange(limits_ms[0] / 1000, limits_ms[1] / 1000 + 1e-9, FLASH_SLOT_S)
        hits = [
            np.histogram(times - event, bins=edges)[0] > 0
            for times, event in zip(flashes, events, strict=True)
        ]
        return 1000 * edges[:-1], np.mean(hits, axis=0)

    return {
        "flash": (windows["flash"][3], np.asarray(rates["flash"])),
        "exit": (windows["exit"][3], np.asarray(rates["exit"])),
        "flash_fraction": flash_fraction(np.zeros(grid[0]), FLASH_WINDOW_MS),
        "exit_fraction": flash_fraction(leave, EXIT_WINDOW_MS),
        "median_exit_ms": float(1000 * np.median(leave)),
        "rasters": rasters,
        "sorted_exit_ms": 1000 * leave[order],
    }


def _fits_done(root: Path, unit_set: str) -> bool:
    """Return True when every stage of the fit in `root` finished for all units."""
    checkpoints = cli.PoissonGLM(root=root).fit_dir(unit_set) / "checkpoints"
    finals = len(list(checkpoints.glob("unit_*_final.json")))
    return finals > 0 and finals >= len(
        list(checkpoints.glob("unit_*_validation.json"))
    )


def session_effect(root: Path, unit_set: str, fit: bool) -> dict:
    """Return the measures of one session.

    With `fit`, the descriptive model and the design with no fixation kernel
    are fit when they are not there. Without it, nothing is fit: a missing or
    unfinished fit raises an error with the command to run.
    """
    loaded = {}
    for name, command in (
        ("matched_flashes", "matched_flashes"),
        ("port_vs_stillness", "port_vs_stillness"),
    ):
        path = root / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run `{command} --root {root}` first.")
        with np.load(path, allow_pickle=False) as data:
            loaded[name] = {key: data[key] for key in data.files}
        loaded[f"{name}_json"] = json.loads(path.with_suffix(".json").read_text())
    descriptive = root.with_name(root.name + ROOT_SUFFIX)
    if fit:
        descriptive = fit_descriptive(root, unit_set)
        steps = two_step_levels(root, unit_set)
    else:
        cache = root / f"hold_two_step_{unit_set}.npz"
        if not (_fits_done(descriptive, unit_set) and cache.exists()):
            raise FileNotFoundError(
                f"The fits of {root.name} are not finished. Run "
                f"`hold_effect --fit --root {root} --units {unit_set}`."
            )
        steps = two_step_levels(root, unit_set)  # reads the cache
    kernels = kernel_values(descriptive, unit_set)
    # Example units: the most suppressed and the most excited unit, each among
    # the units with a rate that makes its raster readable.
    ids, joint = steps["unit_ids"], steps["joint"]
    active = steps["rate"] >= EXAMPLE_MIN_RATE
    readable = (steps["rate"] >= SUPPRESSED_RATES[0]) & (
        steps["rate"] <= SUPPRESSED_RATES[1]
    )
    readable = readable if readable.any() else active
    suppressed = int(ids[readable][np.argmin(joint[readable])])
    excited = int(ids[active][np.argmax(joint[active])])
    rates = raw_rates(root, unit_set, (suppressed, excited))
    hold, after = kernels[dmat.HOLD], kernels[AFTER]
    limits = LEVEL_LIMITS
    centers, values = rates["exit"]
    matched = loaded["matched_flashes"]
    stats = {
        "units": len(hold["level"]),
        "rate_before_first_flash": float(
            rates["flash"][1][:, rates["flash"][0] < 0].mean()
        ),
        "rate_last_200_ms_of_hold": float(values[:, centers < 0].mean()),
        "rate_100_to_400_ms_after_exit": float(
            values[:, (centers > 100) & (centers < 400)].mean()
        ),
        "median_delta_log_rate_hold": float(np.median(hold["level"])),
        "median_delta_log_rate_after_exit": float(np.median(after["level"])),
        "fraction_units_hold_below_0": float(np.mean(hold["level"] < 0)),
        "fraction_units_after_exit_below_0": float(np.mean(after["level"] < 0)),
        "units_higher_after_exit": int(np.sum(after["level"] > hold["level"])),
        "wilcoxon_p": float(wilcoxon(after["level"], hold["level"]).pvalue),
        "units_outside_the_axis_limits_of_panel_d": int(
            np.sum(
                (np.minimum(hold["level"], after["level"]) < limits[0])
                | (np.maximum(hold["level"], after["level"]) > limits[1])
            )
        ),
        "example_units_suppressed_and_excited": [suppressed, excited],
        "two_step": {
            "median_delta_log_rate_joint": float(np.median(steps["joint"])),
            "median_delta_log_rate_two_step": float(np.median(steps["two_step"])),
            "fraction_units_two_step_below_0": float(np.mean(steps["two_step"] < 0)),
            "wilcoxon_p_two_step_against_0": float(wilcoxon(steps["two_step"]).pvalue),
            "units_outside_the_axis_limits_of_panel_e": int(
                np.sum(
                    (np.minimum(steps["joint"], steps["two_step"]) < ORDER_LIMITS[0])
                    | (np.maximum(steps["joint"], steps["two_step"]) > ORDER_LIMITS[1])
                )
            ),
            "units_with_no_two_step_estimate": [
                int(unit)
                for unit in steps["unit_ids"][np.abs(steps["two_step"]) > NO_ESTIMATE]
            ],
        },
        "units_outside_the_axis_limits_of_panel_h": int(
            np.sum(
                (matched["fixation_level"] < limits[0])
                | (matched["fixation_level"] > limits[1])
            )
        ),
        "median_center_exit_ms_from_first_flash": rates["median_exit_ms"],
    }
    flashes = loaded["matched_flashes_json"]
    return {
        "subject": root.name.split("_")[0],
        "kernels": kernels,
        "steps": steps,
        "rates": rates,
        "stillness": loaded["port_vs_stillness"],
        "matched": matched,
        "examples": (suppressed, excited),
        "stats": stats,
        "least": loaded["port_vs_stillness_json"]["least_motion_group"],
        "slope": loaded["port_vs_stillness_json"]["slope_against_motion_energy_group"],
        "by_energy": flashes["after exit, lower against upper half of motion energy"][
            "response, second / first group"
        ],
        "relation": flashes["fixation level against unique dD2, flash (Spearman)"],
    }


def _draw_examples(figure, examples, data: dict) -> list:
    """Draw panel a: the rasters of the two example units; return its first axis."""
    rates = data["rates"]
    suppressed, excited = data["examples"]
    exits = rates["sorted_exit_ms"]
    trial_rows = np.arange(len(exits))
    first = None
    for row, (unit_id, label) in enumerate(
        ((suppressed, "suppressed"), (excited, "excited"))
    ):
        for column, (name, limits) in enumerate(
            (("flash", FLASH_WINDOW_MS), ("exit", EXIT_WINDOW_MS))
        ):
            axis = figure.add_subplot(examples[row, column])
            spike_ms, rows = rates["rasters"][unit_id][name]
            keep = (spike_ms >= limits[0]) & (spike_ms < limits[1])
            axis.plot(
                spike_ms[keep], rows[keep], "|", color=OBSERVED_COLOR, markersize=1.5
            )
            # The event at time 0, and in the left column the exit of each trial.
            event = VISUAL if name == "flash" else TASK
            axis.axvline(0, color=event, linewidth=0.8)
            if name == "flash":
                axis.plot(exits, trial_rows, color=TASK, linewidth=1)
            axis.set_xlim(*limits)
            axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel("trial, sorted\nby fixation length")
                axis.text(
                    0.01,
                    1.02,
                    f"{data['subject']}, unit {unit_id}, {label}",
                    transform=axis.transAxes,
                    va="bottom",
                    fontsize=6.5,
                )
            else:
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.text(
                    0.99,
                    1.02,
                    "first flash" if name == "flash" else "center exit",
                    color=event,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                )
            first = first or axis
    return first


def _draw_matched(axis, data: dict, labels: bool, style: str = "-") -> None:
    """Draw panel g for one mouse: the rate around flashes in three conditions."""
    matched = data["matched"]
    lags = matched["lags_ms"]
    reference = matched["rate__fixation against after exit, matched gap"]
    by_energy = matched["rate__after exit, lower against upper half of motion energy"]
    scale = reference.mean((0, 2))
    used = scale > 0
    curves = (
        (reference[0], TASK, "still"),
        (by_energy[0], "0.35", "move, less motion energy"),
        (by_energy[1], MOTION, "move, more motion energy"),
    )
    # The labels are in the band between the curve of the fixation and the
    # curves after the exit.
    for (values, color, label), height in zip(curves, (0.14, 0.56, 0.48), strict=True):
        values = gaussian_filter1d(values[used] / scale[used, None], 3, axis=-1)
        mean, error = values.mean(0), sem(values, axis=0)
        axis.plot(lags, mean, color=color, ls=style)
        axis.fill_between(
            lags, mean - error, mean + error, color=color, alpha=0.2, lw=0
        )
        if labels:
            axis.text(
                0.03,
                height,
                label,
                color=color,
                transform=axis.transAxes,
                va="top",
                fontsize=6,
            )
    axis.axvline(0, color=VISUAL, linewidth=0.8)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("time from flash (ms)")


def _pair(sessions, text) -> str:
    """Return one value for each session, as 'x (A) and y (B)'."""
    return " and ".join(f"{text(s)} ({s['subject']})" for s in sessions)


def _direction(p_value: float, ratio: float) -> str:
    """Return the direction of a ratio as a word, or "not reliably different"."""
    if p_value >= 0.05:
        return "not reliably different"
    return "higher" if ratio > 1 else "lower"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument(
        "--fit",
        action="store_true",
        help="fit the variants of the session in --root; draw no figure",
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    if args.fit:
        for root in [args.root] if args.root else [args.out / s for s in SESSIONS]:
            session_effect(root, args.units, fit=True)
        return
    sessions = [session_effect(args.out / s, args.units, fit=False) for s in SESSIONS]
    limits = LEVEL_LIMITS

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 12.6))
        examples = figure.add_gridspec(
            2, 2, left=0.11, right=0.97, top=0.9794, bottom=0.8349, hspace=0.25
        )
        top = figure.add_gridspec(
            2, 2, left=0.11, right=0.97, top=0.7895, bottom=0.6492, height_ratios=(1, 4)
        )
        middle = figure.add_gridspec(
            1, 2, left=0.11, right=0.97, top=0.5832, bottom=0.4635
        )
        bottom = figure.add_gridspec(
            1, 3, left=0.11, right=0.86, top=0.385, bottom=0.25, wspace=0.8
        )
        last = figure.add_gridspec(
            1, 2, left=0.11, right=0.97, top=0.165, bottom=0.045, wspace=0.45
        )
        panel_axes = [_draw_examples(figure, examples, sessions[0])]
        shared = None
        for column, (name, xlabel) in enumerate(
            (("flash", "time from first flash (ms)"), ("exit", "time from center exit"))
        ):
            xlabel += " (ms)" if name == "exit" else ""
            flash_axis = figure.add_subplot(top[0, column])
            axis = figure.add_subplot(top[1, column], sharex=flash_axis, sharey=shared)
            shared = shared or axis
            kernel_axis = figure.add_subplot(
                middle[column], sharey=panel_axes[-1] if column else None
            )
            # The units of both mice pooled (standard practice).
            slots, _ = sessions[0]["rates"][f"{name}_fraction"]
            fraction = np.mean(
                [data["rates"][f"{name}_fraction"][1] for data in sessions], axis=0
            )
            flash_axis.step(slots, fraction, where="post", color=VISUAL)
            centers = sessions[0]["rates"][name][0]
            values = np.concatenate([data["rates"][name][1] for data in sessions])
            mean, error = values.mean(axis=0), sem(values, axis=0)
            axis.fill_between(
                centers,
                mean - error,
                mean + error,
                color=OBSERVED_COLOR,
                alpha=0.25,
                lw=0,
            )
            axis.plot(centers, mean, color=OBSERVED_COLOR)
            key = dmat.HOLD if name == "flash" else AFTER
            times_all = sessions[0]["kernels"][key]["times"]
            shown = times_all <= (FLASH_WINDOW_MS if column == 0 else EXIT_WINDOW_MS)[1]
            times = times_all[shown]
            curves = np.concatenate(
                [data["kernels"][key]["values"][:, shown] for data in sessions]
            )
            kernel_axis.plot(
                times, curves.T, color="0.6", linewidth=0.3, alpha=0.25, zorder=1
            )
            low, median, high = np.percentile(curves, [25, 50, 75], axis=0)
            kernel_axis.fill_between(
                times, low, high, color=TASK, alpha=0.4, linewidth=0, zorder=2
            )
            kernel_axis.plot(times, median, color=TASK, linewidth=1.6, zorder=3)
            flash_axis.set_ylim(0, 1.05)
            flash_axis.tick_params(labelbottom=False)
            axis.set_xlabel(xlabel)
            kernel_axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            kernel_axis.set_xlim(axis.get_xlim())
            kernel_axis.set_xlabel(xlabel)
            if column == 0:
                flash_axis.text(
                    0.3,
                    0.75,
                    "fraction of trials with a flash",
                    color=VISUAL,
                    transform=flash_axis.transAxes,
                    fontsize=6.5,
                )
                axis.set_ylabel("normalized firing rate")
                axis.text(
                    0.02,
                    0.04,
                    f"{len(values)} units, {len(sessions)} mice",
                    transform=axis.transAxes,
                    fontsize=6.5,
                )
                panel_axes.append(flash_axis)
                kernel_axis.set_ylim(*limits)
                kernel_axis.set_ylabel(r"$\Delta$ log rate")
                kernel_axis.text(
                    0.98,
                    0.96,
                    "units",
                    color="0.55",
                    transform=kernel_axis.transAxes,
                    ha="right",
                    va="top",
                )
                kernel_axis.text(
                    0.98,
                    0.86,
                    "median and IQR",
                    color=TASK,
                    transform=kernel_axis.transAxes,
                    ha="right",
                    va="top",
                )
                panel_axes.append(kernel_axis)
            else:
                flash_axis.tick_params(labelleft=False)
                axis.tick_params(labelleft=False)
                kernel_axis.tick_params(labelleft=False)

        window = f"{LEVEL_MS[0]} to {LEVEL_MS[1]} ms"
        paired = figure.add_subplot(bottom[0])
        order_axis = figure.add_subplot(bottom[1])
        relation = figure.add_subplot(last[1])
        subjects = [data["subject"] for data in sessions]
        holds = [data["kernels"][dmat.HOLD] for data in sessions]
        afters = [data["kernels"][AFTER] for data in sessions]
        # Panel d: each unit with its SD over the folds in both directions.
        paired.errorbar(
            np.concatenate([h["level"] for h in holds]),
            np.concatenate([a["level"] for a in afters]),
            xerr=np.concatenate([h["level_sd"] for h in holds]),
            yerr=np.concatenate([a["level_sd"] for a in afters]),
            fmt="none",
            ecolor=TASK,
            elinewidth=0.4,
            alpha=0.4,
        )
        for axis, shown in ((paired, limits), (order_axis, ORDER_LIMITS)):
            axis.set_xlim(shown)
            axis.set_ylim(shown)
        pooled_points(
            paired,
            [h["level"] for h in holds],
            [a["level"] for a in afters],
            subjects,
            TASK,
        )
        pooled_points(
            order_axis,
            [data["steps"]["joint"] for data in sessions],
            [data["steps"]["two_step"] for data in sessions],
            subjects,
            TASK,
        )
        relation.set_xscale("log")
        relation.set_ylim(limits)
        pooled_points(
            relation,
            [
                np.maximum(data["matched"]["flash_unique"], UNIQUE_FLOOR)
                for data in sessions
            ],
            [data["matched"]["fixation_level"] for data in sessions],
            subjects,
            TASK,
        )
        for axis, shown in ((paired, limits), (order_axis, ORDER_LIMITS)):
            axis.plot(
                shown, shown, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
            )
            for line in (axis.axhline, axis.axvline):
                line(0, color=REFERENCE_COLOR, linewidth=0.5)
            axis.set_xlim(shown)
            axis.set_ylim(shown)
        paired.set_xlabel(
            r"$\Delta$ log rate in fixation," + f"\n{window} from first flash"
        )
        paired.set_ylabel(r"$\Delta$ log rate after exit," + f"\n{window} from exit")
        order_axis.set_xlabel(
            r"$\Delta$ log rate in fixation," + "\nall terms fit together"
        )
        order_axis.set_ylabel(
            r"$\Delta$ log rate in fixation," + "\nfixation kernel fit last"
        )
        # Panels f and g: the units of both mice pooled.
        stillness_axis = figure.add_subplot(bottom[2])
        port_vs_stillness.plot_rates(
            stillness_axis,
            {
                "rate": np.concatenate([d["stillness"]["rate"] for d in sessions]),
                "frames": np.sum([d["stillness"]["frames"] for d in sessions], axis=0),
            },
            states=(0, 2),
        )
        stillness_axis.set_xlabel("motion energy\n(percentile in each mouse)")
        matched_axis = figure.add_subplot(last[0])
        keys = (
            "rate__fixation against after exit, matched gap",
            "rate__after exit, lower against upper half of motion energy",
        )
        _draw_matched(
            matched_axis,
            {
                "matched": {
                    "lags_ms": sessions[0]["matched"]["lags_ms"],
                    **{
                        key: np.concatenate(
                            [d["matched"][key] for d in sessions], axis=1
                        )
                        for key in keys
                    },
                }
            },
            labels=True,
        )
        matched_axis.set_ylabel("normalized firing rate")
        relation.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        relation.set_xlabel(r"unique $\Delta D^2$ of the visual flash")
        relation.set_ylabel(r"$\Delta$ log rate in fixation")
        panel_axes += [paired, order_axis, stillness_axis, matched_axis, relation]
        for axis in panel_axes[2:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdefgh", panel_axes, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                left,
                axis.get_position().y1 + 0.015,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        output = save_figure(figure, args.out / "hold_effect")

    stats = {s: data["stats"] for s, data in zip(SESSIONS, sessions, strict=True)}
    output.with_name("hold_effect_stats.json").write_text(json.dumps(stats, indent=1))
    first, other = sessions[0]["subject"], sessions[1]["subject"]
    st = lambda data: data["stats"]  # noqa: E731
    paired_text = " ".join(
        f"{d['subject']}: median Δ log rate {st(d)['median_delta_log_rate_hold']:.2f}"
        f" in fixation (below zero in {st(d)['fraction_units_hold_below_0']:.0%} of"
        f" the units) and {st(d)['median_delta_log_rate_after_exit']:+.2f} after the"
        f" withdrawal; higher after the withdrawal in"
        f" {st(d)['units_higher_after_exit']} of {st(d)['units']} units (two-sided"
        f" Wilcoxon signed-rank test, p = {st(d)['wilcoxon_p']:.1g})."
        for d in sessions
    )
    two_text = " ".join(
        f"{d['subject']}: median {st(d)['two_step']['median_delta_log_rate_joint']:.2f}"
        f" and {st(d)['two_step']['median_delta_log_rate_two_step']:.2f}; with the"
        f" kernel fit last, the level is below zero in"
        f" {st(d)['two_step']['fraction_units_two_step_below_0']:.0%} of the units"
        f" (p = {st(d)['two_step']['wilcoxon_p_two_step_against_0']:.1g})."
        for d in sessions
    )
    stillness_text = (
        "At the same low motion energy, the rate in the center port against the"
        " rate at a side port is "
        + "; ".join(
            f"{_direction(d['least']['wilcoxon_p'], d['least']['median_center_over_side'])}"
            f" in {d['subject']} (median ratio"
            f" {d['least']['median_center_over_side']:.2f}, lower in"
            f" {d['least']['fraction_units_lower_in_center_port']:.0%} of the units,"
            f" p = {d['least']['wilcoxon_p']:.2g})"
            for d in sessions
        )
        + ". Over all motion-energy groups, the rate at a side port goes up with"
        " motion energy and the rate in the center port does not: the slope of"
        " each unit over the groups is higher at the side port in "
        + "; ".join(
            f"{d['slope']['side minus center']['units_higher_at_side_port']} of"
            f" {d['slope']['side minus center']['units']} units of {d['subject']}"
            f" (p = {d['slope']['side minus center']['wilcoxon_p']:.1g})"
            for d in sessions
        )
        + ". So the low rate in the port is not only the low rate of a still"
        " animal."
    )
    energy_text = (
        "The response after the withdrawal, more over less motion energy: "
        + "; ".join(
            f"{d['by_energy']['median_ratio']:.2f} in {d['subject']} (p ="
            f" {d['by_energy']['wilcoxon_p']:.2g})"
            for d in sessions
        )
        + "."
    )
    relation_text = (
        "Spearman correlation: "
        + "; ".join(
            f"ρ = {d['relation']['rho']:.2f}, p = {d['relation']['p']:.2g} in"
            f" {d['subject']}"
            for d in sessions
        )
        + "."
    )

    def outside(key: str) -> str:
        return (
            _pair(sessions, lambda d: str(_find(st(d), key)))
            + " units are outside the axis limits and are in the tests."
        )

    output.with_name("hold_effect_caption.md").write_text(
        CAPTION.format(
            units=_pair(sessions, lambda d: f"{st(d)['units']} units"),
            subject=first,
            other=other,
            before=_pair(sessions, lambda d: f"{st(d)['rate_before_first_flash']:.2f}"),
            late=_pair(sessions, lambda d: f"{st(d)['rate_last_200_ms_of_hold']:.2f}"),
            after_rate=_pair(
                sessions, lambda d: f"{st(d)['rate_100_to_400_ms_after_exit']:.2f}"
            ),
            lo=LEVEL_MS[0],
            hi=LEVEL_MS[1],
            paired=paired_text,
            outside_d=outside("units_outside_the_axis_limits_of_panel_d"),
            two_step=two_text,
            outside_e=outside("units_outside_the_axis_limits_of_panel_e"),
            no_estimate=(
                "Units with no two-step estimate (the fit with no penalty did not"
                " converge, because the unit has very few spikes in fixation): "
                + _pair(
                    sessions,
                    lambda d: str(
                        len(st(d)["two_step"]["units_with_no_two_step_estimate"])
                    ),
                )
                + "; they are outside the axis and in the rank test."
            ),
            stillness=stillness_text,
            energy=energy_text,
            relation=relation_text,
            outside_h=outside("units_outside_the_axis_limits_of_panel_h"),
        )
    )
    print(json.dumps(stats, indent=1))
    print(output)


def _find(stats: dict, key: str):
    """Return a value of the session statistics, also from the two-step block."""
    return stats[key] if key in stats else stats["two_step"][key]


if __name__ == "__main__":
    main()
