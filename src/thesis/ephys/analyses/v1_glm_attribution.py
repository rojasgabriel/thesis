"""Measure conditional predictive contributions in the fitted V1 GLM.

Scientific comparison
---------------------
For GRB006 session 20240821_121447, compare the held-out Poisson deviance
explained by the complete V1 encoding model with the deviance explained by a
same-width model in which one regressor block is shuffled and the model is
refit. The difference, complete minus shuffled, estimates the predictive
information unique to that block conditional on all other regressors.

All basis columns for one regressor are shuffled together. The same row
permutation is used for every unit and stays within each trial, so it destroys
temporal alignment without moving covariates between chronological data
splits. Session-drift columns instead move together between whole trials within
each split because they change too little inside one trial. For every shuffled
model, the L2 penalty is selected on training/validation data, coefficients are
refit on training plus validation, and the locked test trials are scored. One
fixed shuffle is used; units, not shuffles, are the displayed observations.

The analysis reports broad task/event, video-derived movement, spike-history,
and drift blocks, then the actual event regressors. It does not split the 25
camera PCs because they form one correlated movement representation. Negative
differences remain visible. These values are predictive associations, not
causal effects, and detailed contributions need not add to a broad-block value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.v1_glm import (
    HISTORY_COLUMNS,
    VIDEO_BASIS_COLUMNS,
    _contiguous_slices,
    _design_with_history,
    _load_windows,
    _write_json_atomic,
    build_unit_design,
    fit_poisson_alpha_path,
    fit_poisson_at_alpha,
    pilot_indices,
    poisson_metrics,
)
from thesis.ephys.units import fetch_unit_table

SHUFFLE_SEED = 20260910
BROAD_GROUPS = ("task", "video", "history", "drift")
DETAILED_GROUPS = (
    "visual_flash",
    "go_cue_command",
    "wrong_punishment_command",
    "center_entry",
    "center_exit",
    "response_entry",
    "response_side",
    "outcome",
    "video",
    "history",
    "drift",
)
DISPLAY_LABELS = {
    "task": "Task",
    "video": "Video SVD",
    "history": "Spike history",
    "drift": "Session drift",
    "visual_flash": "Visual flash",
    "go_cue_command": "Go cue",
    "wrong_punishment_command": "Punishment cue",
    "center_entry": "Center entry",
    "center_exit": "Center exit",
    "response_entry": "Response entry",
    "response_side": "Response side",
    "outcome": "Outcome",
}
GROUP_COLORS = {
    "task": "C0",
    "video": "C1",
    "history": "C2",
    "drift": "black",
}
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def attribution_slices(metadata: dict, video_components: int) -> dict[str, slice]:
    """Return the contiguous full-model columns for each reported block."""
    task_columns = int(metadata["task_columns"])
    base_columns = int(metadata["base_columns"])
    video_stop = base_columns + VIDEO_BASIS_COLUMNS * video_components
    groups = {
        "task": slice(0, task_columns),
        "drift": slice(task_columns, base_columns),
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
    trial_split: np.ndarray, bins_per_trial: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return within-trial and split-restricted whole-trial permutations."""
    trial_split = np.asarray(trial_split)
    if trial_split.ndim != 1 or bins_per_trial <= 1:
        raise ValueError(
            "Trials and bins per trial must define a two-dimensional grid."
        )
    rows = np.arange(len(trial_split) * bins_per_trial, dtype=np.int64).reshape(
        len(trial_split), bins_per_trial
    )
    rng = np.random.default_rng(seed)
    within_trial = np.empty_like(rows)
    for trial, trial_rows in enumerate(rows):
        within_trial[trial] = trial_rows[rng.permutation(bins_per_trial)]

    whole_trial = np.empty_like(rows)
    for split in np.unique(trial_split):
        trials = np.flatnonzero(trial_split == split)
        if len(trials) < 2:
            raise ValueError(
                "Each split needs at least two trials for drift shuffling."
            )
        sources = rng.permutation(trials)
        while np.any(sources == trials):
            sources = rng.permutation(trials)
        whole_trial[trials] = rows[sources]
    return within_trial.ravel(), whole_trial.ravel()


def apply_row_permutation(
    destination: np.ndarray,
    source: np.ndarray,
    permutation: np.ndarray,
    chunk_rows: int,
) -> None:
    """Copy permuted rows in small chunks to limit peak memory."""
    if destination.shape != source.shape or len(permutation) != len(source):
        raise ValueError("Permutation inputs must have matching row and column counts.")
    for start in range(0, len(source), chunk_rows):
        stop = min(start + chunk_rows, len(source))
        destination[start:stop] = source[permutation[start:stop]]


def _load_fit_records(fit_dir: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    tests = {}
    for path in fit_dir.glob("unit_*_test.json"):
        with path.open() as handle:
            item = json.load(handle)
        tests[int(item["unit_id"])] = item
    selections = {}
    for path in fit_dir.glob("unit_*_validation.json"):
        with path.open() as handle:
            item = json.load(handle)
        selections[int(item["unit_id"])] = item
    if not tests or tests.keys() != selections.keys():
        raise ValueError("Complete test and validation records are required.")
    return tests, selections


def _fit_shuffled_model(
    design: np.ndarray,
    counts: np.ndarray,
    split_slices: tuple[slice, slice, slice],
) -> dict:
    train, validation, test = split_slices
    training_mean = float(counts[train].mean())
    selected_model, path = fit_poisson_alpha_path(
        design[train], counts[train], design[validation], counts[validation]
    )
    validation_metrics = poisson_metrics(
        counts[validation], selected_model.predict(design[validation]), training_mean
    )
    fit_stop = validation.stop
    fit_mean = float(counts[:fit_stop].mean())
    final_model = fit_poisson_at_alpha(
        design[:fit_stop], counts[:fit_stop], path["best_alpha"]
    )
    test_metrics = poisson_metrics(
        counts[test], final_model.predict(design[test]), fit_mean
    )
    return {
        "alpha_path": path,
        "validation": validation_metrics,
        "fit_mean_count": fit_mean,
        "test": test_metrics,
    }


def _group_color(group: str) -> str:
    return GROUP_COLORS.get(group, GROUP_COLORS["task"])


def _plot_groups(axis, records: list[dict], groups: tuple[str, ...]) -> None:
    values = [
        np.asarray(
            [
                record["groups"][group]["unique_test_deviance_explained"]
                for record in records
            ]
        )
        for group in groups
    ]
    positions = np.arange(len(groups))
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "0.3", "linewidth": 0.7},
        capprops={"color": "0.3", "linewidth": 0.7},
    )
    for position, (group, group_values, patch) in enumerate(
        zip(groups, values, boxes["boxes"], strict=True)
    ):
        color = _group_color(group)
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
        label.set_color(_group_color(group))
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
        axes[0].set_ylabel("Unique test deviance explained ($\\Delta D^2$)")
        axes[1].set_ylabel("Unique test deviance explained ($\\Delta D^2$)")
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
                "full_test_deviance_explained": record["full_test_deviance_explained"],
                "shuffled_test_deviance_explained": record["groups"][group]["test"][
                    "deviance_explained"
                ],
                "unique_test_deviance_explained": record["groups"][group][
                    "unique_test_deviance_explained"
                ],
                "alpha": record["groups"][group]["alpha_path"]["best_alpha"],
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


def run_attribution(args: argparse.Namespace) -> None:
    """Fit same-width shuffled comparators and write the summary figure."""
    prepared = _load_windows(args.windows)
    common = np.load(args.design, mmap_mode="r", allow_pickle=False)
    with args.design.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    tests, selections = _load_fit_records(args.fit_dir)
    selected_components = {
        int(item["plus_video"]["components"]) for item in tests.values()
    }
    if len(selected_components) != 1:
        raise ValueError("All complete models must use one video-PC count.")
    video_components = selected_components.pop()
    common_columns = int(metadata["base_columns"]) + (
        VIDEO_BASIS_COLUMNS * video_components
    )
    groups = attribution_slices(metadata, video_components)
    group_order = tuple(dict.fromkeys((*BROAD_GROUPS, *DETAILED_GROUPS)))

    with np.load(args.windows, allow_pickle=False) as windows:
        trial_split = windows["trial_split"].copy()
        relative_times = windows["relative_bin_centers_s"].copy()
    bins_per_trial = len(relative_times)
    if len(common) != len(trial_split) * bins_per_trial:
        raise ValueError("Common design rows do not match the saved trial grid.")
    within_trial, whole_trial = shuffle_permutations(
        trial_split, bins_per_trial, SHUFFLE_SEED
    )
    split_slices = _contiguous_slices(prepared["split"])

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    if args.units == "pilot":
        unit_rows = pilot_indices(len(units))
    else:
        unit_rows = np.arange(len(units))
    output_dir = args.output or args.fit_dir / "conditional_deviance_fit"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for position, row in enumerate(unit_rows, start=1):
        unit = units.iloc[row]
        unit_id = int(unit["unit_id"])
        output = output_dir / f"unit_{unit_id}.json"
        if output.exists():
            with output.open() as handle:
                record = json.load(handle)
            if record["shuffle_seed"] != SHUFFLE_SEED:
                raise ValueError(f"Shuffle seed differs in {output}.")
            records.append(record)
            print(f"Loaded unit {position} of {len(unit_rows)}: {unit_id}", flush=True)
            continue

        counts, history = build_unit_design(
            prepared["alignments"], np.asarray(unit["spike_times_s"], dtype=float)
        )
        selection = selections[unit_id]
        history_mean = np.asarray(selection["history_training_mean"], dtype=float)
        history_scale = np.asarray(selection["history_training_scale"], dtype=float)
        history_scaled = (history - history_mean) / history_scale
        design = _design_with_history(common, history_scaled, common_columns)
        if design.shape[1] != len(tests[unit_id]["plus_video"]["coefficients"]):
            raise ValueError("Saved full model and attribution design widths differ.")

        full_deviance = float(
            tests[unit_id]["plus_video"]["test"]["deviance_explained"]
        )
        group_results = {}
        for group_position, group in enumerate(group_order, start=1):
            columns = groups[group]
            source = design[:, columns].copy()
            permutation = whole_trial if group == "drift" else within_trial
            apply_row_permutation(
                design[:, columns], source, permutation, bins_per_trial
            )
            fitted = _fit_shuffled_model(design, counts, split_slices)
            design[:, columns] = source
            del source
            fitted.update(
                columns=columns.stop - columns.start,
                shuffle=(
                    "whole trials within split"
                    if group == "drift"
                    else "rows jointly within trial"
                ),
                unique_test_deviance_explained=(
                    full_deviance - fitted["test"]["deviance_explained"]
                ),
            )
            group_results[group] = fitted
            print(
                f"  Fitted {group_position} of {len(group_order)} blocks: {group}",
                flush=True,
            )

        record = {
            "unit_id": unit_id,
            "depth": float(unit["depth"]),
            "video_components": video_components,
            "shuffle_seed": SHUFFLE_SEED,
            "full_test_deviance_explained": full_deviance,
            "groups": group_results,
        }
        _write_json_atomic(output, record)
        records.append(record)
        print(f"Saved unit {position} of {len(unit_rows)}: {unit_id}", flush=True)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("figures/v1_glm/stimulus_windows.npz"),
    )
    parser.add_argument(
        "--design", type=Path, default=Path("figures/v1_glm/common_design.npy")
    )
    parser.add_argument("--fit-dir", type=Path, default=Path("figures/v1_glm/all_fit"))
    parser.add_argument("--units", choices=("pilot", "all"), default="pilot")
    parser.add_argument("--output", type=Path)
    run_attribution(parser.parse_args())


if __name__ == "__main__":
    main()
