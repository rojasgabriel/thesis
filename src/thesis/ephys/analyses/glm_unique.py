"""Measure each block's unique explained deviance in the fitted V1 GLM.

Shuffle one regressor block, refit, and take complete minus shuffled test D².
All bases of a block move together, within trial. One shuffle is shared across
units. Motion-energy PCs stay one block.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import (
    _SHARED,
    HISTORY_COLUMNS,
    VIDEO_BASIS_COLUMNS,
    _cached,
    _design_with_history,
    _load_windows,
    _write_json_atomic,
    build_unit_counts,
    build_unit_history,
    code_version,
    fit_poisson_at_alpha,
    poisson_metrics,
    run_over_units,
    sample_unit_indices,
    training_zscore,
)
from thesis.ephys.units import fetch_unit_table

SHUFFLE_SEED = 20260910
BROAD_GROUPS = ("task", "video", "history")
DETAILED_GROUPS = (
    "visual_flash",
    "center_poke",
    "center_exit",
    "response_entry",
    "response_side",
    "video",
    "history",
)
DISPLAY_LABELS = {
    "task": "Task",
    "video": "Motion energy",
    "history": "Spike history",
    "visual_flash": "Visual flash",
    "center_poke": "Center poke",
    "center_exit": "Center exit",
    "response_entry": "Response entry",
    "response_side": "Response side",
}
GROUP_COLORS = {
    "task": "C0",
    "video": "C1",
    "history": "C2",
}
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def block_slices(metadata: dict, video_components: int) -> dict[str, slice]:
    """Return the contiguous full-model columns for each reported block."""
    task_columns = int(metadata["task_columns"])
    base_columns = int(metadata["base_columns"])
    video_stop = base_columns + VIDEO_BASIS_COLUMNS * video_components
    groups = {
        "task": slice(0, task_columns),
        "video": slice(base_columns, video_stop),
        "history": slice(video_stop, video_stop + HISTORY_COLUMNS),
    }
    cursor = 0
    for item in metadata["task_manifest"]:
        stop = cursor + int(item["columns"])
        groups[item["name"]] = slice(cursor, stop)
        cursor = stop
    if cursor != task_columns:
        raise ValueError("Task manifest and task-column count differ.")
    return groups


def shuffle_permutations(
    n_trials: int,
    bins_per_trial: int,
    seed: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return within-trial permutations, only among valid bins when given."""
    if n_trials <= 0 or bins_per_trial <= 1:
        raise ValueError(
            "Trials and bins per trial must define a two-dimensional grid."
        )
    rows = np.arange(n_trials * bins_per_trial, dtype=np.int64).reshape(
        n_trials, bins_per_trial
    )
    if valid is None:
        valid_trials = np.ones(rows.shape, dtype=bool)
    else:
        valid_trials = np.asarray(valid, dtype=bool).reshape(rows.shape)
    rng = np.random.default_rng(seed)
    within_trial = np.empty_like(rows)
    for trial, trial_rows in enumerate(rows):
        keep = np.flatnonzero(valid_trials[trial])
        permuted = trial_rows.copy()
        if keep.size:
            permuted[keep] = trial_rows[keep][rng.permutation(keep.size)]
        within_trial[trial] = permuted
    return within_trial.ravel()


def _load_fold_records(fit_dir: Path) -> list[dict]:
    """Read the per-fold penalties that `fit` selected for every unit."""
    records = []
    for path in sorted(fit_dir.glob("unit_*_folds.json")):
        with path.open() as handle:
            records.append(json.load(handle))
    if not records:
        raise ValueError(f"No cross-validated fits in {fit_dir}; run `fit` first.")
    return records


def _shuffled_deviance(
    fit_design: np.ndarray,
    fit_counts: np.ndarray,
    test_design: np.ndarray,
    test_counts: np.ndarray,
    alpha: float,
) -> float:
    """Refit one shuffled design at a fixed penalty and score the held-out fold.

    The penalty comes from the full model rather than being reselected. The
    shuffled design has the same width, so the optimum barely moves, and holding
    it fixed keeps the difference in deviance attributable to the shuffle rather
    than to two models landing on different penalties.
    """
    fit_mean = float(fit_counts.mean())
    model = fit_poisson_at_alpha(fit_design, fit_counts, alpha)
    return poisson_metrics(test_counts, model.predict(test_design), fit_mean)[
        "deviance_explained"
    ]


def _scored_folds(counts, partitions, alphas) -> list[tuple[dict, float]]:
    """Pair saved alphas with the folds that contained test spikes."""
    folds = [fold for fold in partitions if counts[fold["test"]].sum() > 0]
    if len(folds) != len(alphas):
        raise ValueError(
            f"{len(folds)} folds contained spikes but {len(alphas)} alphas were saved."
        )
    return list(zip(folds, alphas, strict=True))


def _plot_groups(axis, records: list[dict], groups: tuple[str, ...]) -> None:
    """Draw maximal deviance behind unique, so the shared part is the gap."""
    values = [
        np.asarray(
            [
                record["groups"][group]["unique_test_deviance_explained"]
                for record in records
            ]
        )
        for group in groups
    ]
    maximal = [
        np.asarray(
            [
                record["groups"][group]["maximal_test_deviance_explained"]
                for record in records
            ]
        )
        for group in groups
    ]
    positions = np.arange(len(groups))
    axis.boxplot(
        maximal,
        positions=positions,
        widths=0.78,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "0.90", "edgecolor": "0.65", "linewidth": 0.6},
        medianprops={"color": "0.55", "linewidth": 0.8},
        whiskerprops={"color": "0.7", "linewidth": 0.6},
        capprops={"color": "0.7", "linewidth": 0.6},
        zorder=1,
    )
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "0.3", "linewidth": 0.7},
        capprops={"color": "0.3", "linewidth": 0.7},
    )
    for position, (group, group_values, patch) in enumerate(
        zip(groups, values, boxes["boxes"], strict=True)
    ):
        color = GROUP_COLORS.get(group, GROUP_COLORS["task"])
        patch.set(facecolor=color, edgecolor="black", alpha=0.45, linewidth=0.7)
        axis.scatter(
            position,
            np.mean(group_values),
            facecolor="white",
            edgecolor="0.25",
            linewidth=0.6,
            s=20,
            zorder=4,
        )
    axis.axhline(0, color="0.35", linestyle="--", linewidth=0.7, zorder=0)
    axis.set_xticks(positions, [DISPLAY_LABELS[group] for group in groups])
    axis.tick_params(axis="x", labelrotation=48)
    for label, group in zip(axis.get_xticklabels(), groups, strict=True):
        label.set_ha("right")
        label.set_color(GROUP_COLORS.get(group, GROUP_COLORS["task"]))
    axis.margins(x=0.04)


def plot_conditional_deviance(records: list[dict], output: Path) -> tuple[Path, Path]:
    """Plot Oesch-style conditional deviance for broad and detailed blocks."""
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.5, 3.5),
            gridspec_kw={"width_ratios": [0.85, 2.15], "wspace": 0.3},
        )
        _plot_groups(axes[0], records, BROAD_GROUPS)
        _plot_groups(axes[1], records, DETAILED_GROUPS)
        axes[0].set_ylabel("Test deviance explained ($\\Delta D^2$)")
        axes[1].set_ylabel("Test deviance explained ($\\Delta D^2$)")
        for y, label, color in (
            (0.97, "maximal (block alone)", "0.55"),
            (0.89, "unique (only this block)", "black"),
        ):
            axes[1].text(
                0.99,
                y,
                label,
                transform=axes[1].transAxes,
                ha="right",
                va="top",
                color=color,
                fontsize=7,
            )
        for letter, axis in zip("ab", axes, strict=True):
            axis.text(
                -0.12,
                1.04,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.subplots_adjust(bottom=0.35, left=0.09, right=0.99, top=0.94)
        output.parent.mkdir(parents=True, exist_ok=True)
        pdf = output.with_suffix(".pdf")
        png = output.with_suffix(".png")
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return pdf, png


def _write_summary(records: list[dict], output_dir: Path) -> dict:
    group_order = tuple(dict.fromkeys((*BROAD_GROUPS, *DETAILED_GROUPS)))
    summary = {
        "units": len(records),
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_repetitions": 1,
        "groups": {},
    }
    rows = []
    for group in group_order:
        values = np.asarray(
            [
                record["groups"][group]["unique_test_deviance_explained"]
                for record in records
            ]
        )
        summary["groups"][group] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
        rows.extend(
            {
                "unit_id": record["unit_id"],
                "depth": record["depth"],
                "group": group,
                "complete_test_deviance_explained": record[
                    "complete_test_deviance_explained"
                ],
                "unique_test_deviance_explained": record["groups"][group][
                    "unique_test_deviance_explained"
                ],
                "unique_test_deviance_explained_sem": record["groups"][group][
                    "unique_test_deviance_explained_sem"
                ],
                "maximal_test_deviance_explained": record["groups"][group][
                    "maximal_test_deviance_explained"
                ],
                "maximal_test_deviance_explained_sem": record["groups"][group][
                    "maximal_test_deviance_explained_sem"
                ],
            }
            for record in records
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    partial = summary_path.with_suffix(".json.partial")
    with partial.open("w") as handle:
        json.dump(summary, handle, indent=2)
    partial.replace(summary_path)
    csv_path = output_dir / "unit_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _block_task(job: dict) -> dict:
    """Score every block's unique and maximal deviance for one unit."""
    output = Path(job["output"])
    cached = _cached(output, "unique")
    if cached is not None:
        if cached["shuffle_seed"] != SHUFFLE_SEED:
            raise ValueError(f"Shuffle seed differs in {output}.")
        return cached
    prepared = _SHARED["prepared"]
    common = _SHARED["common"]
    spikes = np.asarray(job["spike_times"], dtype=float)
    counts = build_unit_counts(prepared["alignments"], spikes)
    raw_history = build_unit_history(prepared["alignments"], spikes)
    groups, order = job["groups"], job["group_order"]
    within = _SHARED["within_trial"]
    per_fold = {name: {"unique": [], "maximal": []} for name in order}
    complete_folds = []
    for fold, alpha in _scored_folds(counts, _SHARED["partitions"], job["alphas"]):
        fit, test = fold["fit"], fold["test"]
        alpha = float(alpha)
        history, _, _ = training_zscore(raw_history, fit)
        fit_rows = np.flatnonzero(fit)
        test_rows = np.flatnonzero(test)
        fit_design = _design_with_history(common, history, job["common_columns"], fit)
        test_design = _design_with_history(common, history, job["common_columns"], test)
        fit_counts = counts[fit]
        test_counts = counts[test]

        def score() -> float:
            return _shuffled_deviance(
                fit_design, fit_counts, test_design, test_counts, alpha
            )

        complete = score()
        complete_folds.append(complete)

        # Shuffle the two scored matrices in place and restore from the source.
        # Each fold copies its train and test rows only once.
        def put(columns, shuffled):
            width = job["common_columns"]
            if columns.stop <= width:
                source = np.asarray(common[:, columns])
            else:
                source = history[:, columns.start - width : columns.stop - width]
            for matrix, rows in ((fit_design, fit_rows), (test_design, test_rows)):
                matrix[:, columns] = source[within[rows]] if shuffled else source[rows]

        # These detailed blocks are non-overlapping and cover the full model.
        blocks = [groups[name] for name in DETAILED_GROUPS]
        for columns in blocks:
            put(columns, True)
        reference = score()
        for columns in blocks:
            put(columns, False)

        for group in order:
            columns = groups[group]
            put(columns, True)
            removed = score()
            put(columns, False)

            for other in blocks:
                put(other, True)
            put(columns, False)
            alone = score()
            for other in blocks:
                put(other, False)

            per_fold[group]["unique"].append(complete - removed)
            per_fold[group]["maximal"].append(alone - reference)
    if len(complete_folds) < 2:
        raise ValueError(
            f"only {len(complete_folds)} test folds contained spikes; "
            "at least 2 are required"
        )
    group_results = {}
    for group in order:
        unique = np.asarray(per_fold[group]["unique"])
        maximal = np.asarray(per_fold[group]["maximal"])
        group_results[group] = {
            "columns": groups[group].stop - groups[group].start,
            "shuffle": "rows jointly within trial",
            "unique_test_deviance_explained": float(unique.mean()),
            "unique_test_deviance_explained_sem": float(
                unique.std(ddof=1) / np.sqrt(len(unique))
            ),
            "maximal_test_deviance_explained": float(maximal.mean()),
            "maximal_test_deviance_explained_sem": float(
                maximal.std(ddof=1) / np.sqrt(len(maximal))
            ),
            "folds": {"unique": unique.tolist(), "maximal": maximal.tolist()},
        }
    record = {
        "record_version": code_version("unique"),
        "unit_id": job["unit_id"],
        "depth": job["depth"],
        "video_components": job["video_components"],
        "shuffle_seed": SHUFFLE_SEED,
        "folds": len(complete_folds),
        "complete_test_deviance_explained": float(np.mean(complete_folds)),
        "groups": group_results,
    }
    _write_json_atomic(output, record)
    return record


def run_unique(
    windows: Path,
    design_path: Path,
    fit_dir: Path,
    unit_set: str,
    workers: int | None = None,
) -> None:
    """Fit same-width shuffled comparators and write the summary figure."""
    prepared = _load_windows(windows)
    common = np.load(design_path, mmap_mode="r", allow_pickle=False)
    with design_path.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    checkpoints = fit_dir / "checkpoints"
    fold_records = _load_fold_records(checkpoints)
    selected_components = {int(item["components"]) for item in fold_records}
    if len(selected_components) != 1:
        raise ValueError("All cross-validated fits must use one video-PC count.")
    video_components = selected_components.pop()
    common_columns = int(metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * video_components
    )
    groups = block_slices(metadata, video_components)
    group_order = tuple(dict.fromkeys((*BROAD_GROUPS, *DETAILED_GROUPS)))

    with np.load(windows, allow_pickle=False) as saved:
        trial_split = saved["trial_split"].copy()
        relative_times = saved["relative_bin_centers_s"].copy()
    bins_per_trial = len(relative_times)
    if len(common) != len(trial_split) * bins_per_trial:
        raise ValueError("Common design rows do not match the saved trial grid.")
    fold_alphas = {
        int(item["unit_id"]): [float(entry["alpha"]) for entry in item["folds"]]
        for item in fold_records
    }

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    if unit_set == "sample":
        unit_rows = sample_unit_indices(len(units))
    else:
        unit_rows = np.arange(len(units))
    output_dir = fit_dir / "unique_deviance"
    checkpoint_dir = checkpoints / "unique"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "unit_id": int(units.iloc[row]["unit_id"]),
            "depth": float(units.iloc[row]["depth"]),
            "spike_times": np.asarray(units.iloc[row]["spike_times_s"], dtype=float),
            "video_components": video_components,
            "common_columns": common_columns,
            "groups": groups,
            "group_order": group_order,
            "alphas": fold_alphas[int(units.iloc[row]["unit_id"])],
            "output": str(
                checkpoint_dir / f"unit_{int(units.iloc[row]['unit_id'])}.json"
            ),
        }
        for row in unit_rows
        if int(units.iloc[row]["unit_id"]) in fold_alphas
    ]
    missing = len(unit_rows) - len(jobs)
    if missing:
        raise ValueError(
            f"{missing} units have no cross-validated record; run fit again."
        )
    records = []
    failures = 0
    for position, (record, reused) in enumerate(
        run_over_units(
            _block_task,
            jobs,
            windows,
            design_path,
            workers,
            "unique",
            SHUFFLE_SEED,
        ),
        start=1,
    ):
        if record is None:
            failures += 1
            continue
        records.append(record)
        action = "Reused" if reused else "Completed"
        print(
            f"{action} unique analysis {position}/{len(jobs)}: "
            f"unit {record['unit_id']}",
            flush=True,
        )
    if failures:
        raise RuntimeError(
            f"Unique analysis incomplete: {failures}/{len(jobs)} units failed. "
            "No summary or figure was written. Per-unit results are safe to reuse; "
            "rerun the same command."
        )

    summary = _write_summary(records, output_dir)
    pdf, png = plot_conditional_deviance(records, output_dir / "conditional_deviance")
    print(
        json.dumps(
            {
                **summary,
                "pdf": str(pdf),
                "png": str(png),
            },
            indent=2,
        )
    )
