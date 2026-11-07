"""How much does each kind of signal add to the V1 GLM: flash, task, movement, history, drift?

Question: the term-by-term gains (`main_figure.py`, panel f) split the flash
and the task signals into many terms, and correlated terms share what they
explain, so each term alone looks small. How much do the flash terms and the
task terms explain as groups, next to the spike history, the motion energy and
the session drift?

Comparison, as in `unique_dev.py` and with its shuffles, seed, folds and
per-fold penalties (no new penalty is chosen):

- unique gain of a group = full-model held-out D² minus D² with every term of
  the group shuffled at the same time (each term with its own rule: inside the
  trial; over trials for the session drift; the side of the trial for the
  previous side and the turn direction);
- maximal gain of a group = D² with only that group unshuffled minus D² with
  every term shuffled.

Groups (`GROUPS`):

- flash: `FLASH`, all five flash terms (the flash, the short gap, after the
  exit, the recovery, the side after the exit);
- task events: `TASK_EVENTS`, the pre-stimulus level, the fixation, the
  previous choice, the response period and the turn direction;
- spike history, motion energy (10 PCs) and session drift are one term each;
  their values are those of `unique_dev.py` (same shuffles and seed), read
  from its checkpoints, not fit again.

Units: all GLM units with a final fit (172 / 79), the two GLM sessions
(`flash_kernels.SESSIONS`). The sampling unit is the unit. Each fold's ΔD² is
turned into bits per spike with the factor of `unique_dev.add_bits_per_spike`.
The figure shows the pooled median with its hierarchical-bootstrap 95% interval
(mice, then units) on a log axis that starts at `main_figure.FLOOR`; a smaller
value is drawn there. The numbers are the fraction of units with a unique gain
above zero (`inclusion.regressor_tests`). Left: the groups; right: each term
(`main_figure._plot_unique`).

Fit, then the figure (about 50 refits per unit; the fits are saved per unit,
so a rerun goes on where it stopped):

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.term_groups --fit
    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.term_groups

Writes `figures/glm/term_groups.svg` and `results/glm/term_groups.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.dmat import (
    PREVIOUS_SIDE,
    TURN_DIRECTION,
    _design_with_history,
    _load_selected_trials,
    _load_windows,
    build_unit_history,
    training_zscore,
)
from thesis.ephys.analyses.glm.fit import (
    _SHARED,
    analysis_unit_indices,
    build_unit_counts,
    collect_records,
    fit_poisson_at_alpha,
    load_unit_records,
    poisson_metrics,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.inclusion import regressor_tests
from thesis.ephys.analyses.glm.main_figure import (
    FLOOR,
    REGRESSORS,
    _plot_unique,
    session_population,
)
from thesis.ephys.analyses.glm.unique_dev import (
    MAXIMAL_BITS,
    OVER_TRIALS,
    SHUFFLE_SEED,
    SIGN_OVER_TRIALS,
    UNIQUE_BITS,
    _scored_folds,
    add_bits_per_spike,
    block_slices,
    shuffled_side_columns,
)
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    hierarchical_interval,
    label_panels,
    save_figure,
)

FLASH = (
    "visual_flash",
    "flash_just_before",
    "flash_after_exit",
    "flash_recovery",
    "flash_side",
)
TASK_EVENTS = ("fixation", "hold", "previous_side", "response_period", "turn_direction")
# Fit here (several terms) or read from `unique_dev` (one term).
GROUPS = {"flash": FLASH, "task events": TASK_EVENTS}
SINGLE = {
    "spike history": "history",
    "motion energy": "video",
    "session drift": "session_drift",
}
# Top to bottom in the figure, under the full model.
ROWS = ("spike history", "task events", "flash", "motion energy", "session drift")
COLORS = {
    "spike history": CONCEPT_COLORS["history"],
    "task events": CONCEPT_COLORS["task"],
    "flash": CONCEPT_COLORS["visual"],
    "motion energy": CONCEPT_COLORS["motion energy"],
    "session drift": REGRESSORS["session_drift"][1],
}
OUT = Path("results/glm")
CHECKPOINTS = "term_groups"


def _group_task(job: dict) -> dict:
    """Score the unique and maximal D² of each group of `GROUPS` for one unit."""
    prepared, common = _SHARED["prepared"], _SHARED["common"]
    within, between = _SHARED["within_trial"], _SHARED["between_trials"]
    spikes = np.asarray(job["spike_times"], dtype=float)
    counts = build_unit_counts(prepared["alignments"], spikes)
    raw_history = build_unit_history(prepared["alignments"], spikes)
    slices = job["slices"]
    trials = len(prepared["alignments"])
    bins = len(between) // trials
    partner = between.reshape(trials, bins)[:, 0] // bins
    side_blocks = {
        name: shuffled_side_columns(
            np.asarray(common[:, slices[name]]).reshape(trials, bins, -1),
            job["common_mean"][slices[name]],
            job["common_scale"][slices[name]],
            job["side_codes"][name],
            partner,
            job["unknown_previous_support"] if name == PREVIOUS_SIDE else None,
        ).reshape(len(common), -1)
        for name in SIGN_OVER_TRIALS
    }
    # Every block of the full model, as in `unique_dev` (not the broad "task").
    blocks = [name for name in slices if name != "task"]
    per_fold = {group: {"unique": [], "maximal": []} for group in GROUPS}
    folds = _scored_folds(counts, _SHARED["partitions"], job["alphas"])
    for fold_index, (fold, alpha) in enumerate(folds):
        fit, test = fold["fit"], fold["test"]
        history, _, _ = training_zscore(raw_history, fit)
        rows = {"fit": np.flatnonzero(fit), "test": np.flatnonzero(test)}
        designs = {
            part: _design_with_history(
                common, history, job["common_columns"], fold[part]
            )
            for part in ("fit", "test")
        }
        fit_counts, test_counts = counts[fit], counts[test]

        def score() -> float:
            model = fit_poisson_at_alpha(designs["fit"], fit_counts, float(alpha))
            return poisson_metrics(
                test_counts, model.predict(designs["test"]), float(fit_counts.mean())
            )["deviance_explained"]

        def put(name: str, shuffled: bool) -> None:
            columns = slices[name]
            width = job["common_columns"]
            if columns.stop <= width:
                source = np.asarray(common[:, columns])
            else:
                source = history[:, columns.start - width : columns.stop - width]
            for part, matrix in designs.items():
                if not shuffled:
                    matrix[:, columns] = source[rows[part]]
                elif name in side_blocks:
                    matrix[:, columns] = side_blocks[name][rows[part]]
                else:
                    mapping = between if name in OVER_TRIALS else within
                    matrix[:, columns] = source[mapping[rows[part]]]

        complete = float(job["complete_folds"][fold_index])
        for name in blocks:
            put(name, True)
        reference = score()
        for group, members in GROUPS.items():
            for name in blocks:
                put(name, name not in members)
            alone = score()
            for name in blocks:
                put(name, name in members)
            removed = score()
            for name in members:
                put(name, False)
            per_fold[group]["unique"].append(complete - removed)
            per_fold[group]["maximal"].append(alone - reference)
    return {
        "unit_id": job["unit_id"],
        "shuffle_seed": SHUFFLE_SEED,
        "members": {group: list(members) for group, members in GROUPS.items()},
        "groups": {
            group: {
                "unique_test_deviance_explained": float(np.mean(values["unique"])),
                "maximal_test_deviance_explained": float(np.mean(values["maximal"])),
                "folds": values,
            }
            for group, values in per_fold.items()
        },
    }


def fit_groups(root: Path, unit_set: str = "all", workers: int | None = None) -> None:
    """Fit the group comparators of every unit of one session (resumes per unit)."""
    model = cli.PoissonGLM(root=root)
    fit_dir = model.fit_dir(unit_set)
    prepared = _load_windows(model.windows)
    with model.design.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    folds = load_unit_records(fit_dir / "checkpoints", "unit_*_folds.json")
    (components,) = {int(item["components"]) for item in folds}
    slices = block_slices(metadata, components)
    missing = {name for members in GROUPS.values() for name in members} - set(slices)
    if missing:
        raise ValueError(f"Terms not in the design: {sorted(missing)}")
    trials = _load_selected_trials(prepared)
    unknown = np.nan_to_num(trials["last_side"].to_numpy(dtype=float)) == 0
    before_exit = (
        prepared["bin_times"].reshape(len(trials), -1)[unknown]
        < trials["center_exit_s"].to_numpy(dtype=float)[unknown, None]
    )
    side_codes = {
        PREVIOUS_SIDE: np.nan_to_num(trials["last_side"].to_numpy(dtype=float)),
        TURN_DIRECTION: trials["response"].to_numpy(dtype=float),
    }
    by_unit = {int(item["unit_id"]): item for item in folds}
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    output = fit_dir / "checkpoints" / CHECKPOINTS
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in analysis_unit_indices(units, unit_set):
        unit_id = int(units.iloc[row]["unit_id"])
        if unit_id not in by_unit:
            continue
        jobs.append(
            {
                "unit_id": unit_id,
                "spike_times": np.asarray(
                    units.iloc[row]["spike_times_s"], dtype=float
                ),
                "common_columns": int(metadata["base_columns"]) + components,
                "slices": slices,
                "common_mean": np.asarray(metadata["training_mean"]),
                "common_scale": np.asarray(metadata["training_scale"]),
                "side_codes": side_codes,
                "unknown_previous_support": before_exit[..., None].astype(float),
                "alphas": [float(f["alpha"]) for f in by_unit[unit_id]["folds"]],
                "complete_folds": [
                    float(f["test"]["deviance_explained"])
                    for f in by_unit[unit_id]["folds"]
                ],
                "output": str(output / f"unit_{unit_id}.json"),
            }
        )
    collect_records(
        "term-group analysis",
        _group_task,
        jobs,
        model.windows,
        model.design,
        workers,
        shuffle_seed=SHUFFLE_SEED,
    )


def session_groups(root: Path, unit_set: str = "all") -> dict:
    """Return the bits-per-spike gains of each row of `ROWS`, per unit."""
    checkpoints = cli.PoissonGLM(root=root).fit_dir(unit_set) / "checkpoints"
    groups = add_bits_per_spike(
        load_unit_records(checkpoints / CHECKPOINTS, "unit_*.json"), checkpoints
    )
    single = {
        int(item["unit_id"]): item
        for item in add_bits_per_spike(
            load_unit_records(checkpoints / "unique", "unit_*.json"), checkpoints
        )
    }
    if set(single) != {int(item["unit_id"]) for item in groups}:
        raise ValueError(f"{root.name}: the group fits and unique_dev differ in units.")
    rows = {}
    for row in ROWS:
        values = [
            (
                item["groups"][row]
                if row in GROUPS
                else single[int(item["unit_id"])]["groups"][SINGLE[row]]
            )
            for item in groups
        ]
        rows[row] = {
            "unique": [v[UNIQUE_BITS] for v in values],
            "maximal": [v[MAXIMAL_BITS] for v in values],
            "tests": regressor_tests(
                [np.asarray(v["folds"]["unique_bits"]) for v in values]
            ),
        }
    return {
        "subject": root.name.split("_")[0],
        "units": len(groups),
        "full": [item["complete_bits_per_spike"] for item in groups],
        "rows": rows,
    }


def _plot_groups(axis, sessions: list[dict]) -> None:
    def draw(y, groups, color, face):
        stats = hierarchical_interval(groups)
        low, high = np.maximum(stats["interval_95"], FLOOR)
        axis.plot([low, high], [y, y], color=color, linewidth=1.0)
        axis.scatter(
            max(stats["median"], FLOOR),
            y,
            s=22,
            zorder=3,
            facecolor=face,
            edgecolor=color,
        )
        return stats["median"]

    positions = dict(zip(reversed(ROWS), range(len(ROWS)), strict=True))
    for row, y in positions.items():
        color = COLORS[row]
        for key, offset, face in (("unique", -0.17, color), ("maximal", 0.17, "white")):
            draw(y + offset, [s["rows"][row][key] for s in sessions], color, face)
        need = sum(s["rows"][row]["tests"]["units_that_need_it"] for s in sessions)
        total = sum(s["units"] for s in sessions)
        axis.text(
            1.03,
            y,
            f"{need / total:.0%}",
            color=color,
            va="center",
            fontsize=6.5,
            transform=axis.get_yaxis_transform(),
        )
    draw(len(ROWS), [s["full"] for s in sessions], "black", "black")
    axis.text(
        1.03,
        len(ROWS) + 0.5,
        "units with a unique\ngain > 0",
        fontsize=6,
        transform=axis.get_yaxis_transform(),
    )
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 3)
    axis.set_yticks(range(len(ROWS) + 1), [*reversed(ROWS), "full model"])
    for tick, row in zip(axis.get_yticklabels(), reversed(ROWS), strict=False):
        tick.set_color(COLORS[row])
    axis.set_xlabel("held-out bits per spike\n(filled: unique; open: maximal)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", action="store_true", help="fit the groups first")
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.fit:
        for name in SESSIONS:
            fit_groups(OUT / name, args.units, args.workers)
    sessions = [session_groups(OUT / name, args.units) for name in SESSIONS]
    terms = [session_population(OUT / name, args.units) for name in SESSIONS]
    order = sorted(
        terms[0]["tests"],
        key=lambda name: np.median(
            [item["groups"][name][UNIQUE_BITS] for item in terms[0]["records"]]
        ),
    )
    with plt.rc_context(FIGURE_STYLE):
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(7.4, 3.6), layout="constrained", width_ratios=(1, 1.25)
        )
        _plot_groups(left, sessions)
        _plot_unique(right, terms, order)
        right.set_xlim(left.get_xlim())
        label_panels((left, right), x=-0.3)
        print(save_figure(figure, OUT / "term_groups"))
    summary = {
        s["subject"]: {
            "units": s["units"],
            "full_model_median_bits": float(np.median(s["full"])),
            **{
                row: {
                    "unique_median_bits": float(np.median(v["unique"])),
                    "maximal_median_bits": float(np.median(v["maximal"])),
                    "fraction_units_that_need_it": v["tests"][
                        "fraction_units_that_need_it"
                    ],
                }
                for row, v in s["rows"].items()
            },
        }
        for s in sessions
    }
    (OUT / "term_groups.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
