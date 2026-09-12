"""Measure conditional predictive contributions in the fitted V1 GLM.

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

from thesis.ephys.analyses.v1_glm import (
    VIDEO_BASIS_COLUMNS,
    _load_windows,
    _split_masks,
    _valid_bin_mask,
    _write_json_atomic,
    build_unit_counts,
    fit_poisson_alpha_path,
    fit_poisson_at_alpha,
    poisson_metrics,
    test_unit_indices,
)
from thesis.ephys.units import fetch_unit_table

SHUFFLE_SEED = 20260910
BROAD_GROUPS = ("task", "video")
DETAILED_GROUPS = (
    "visual_flash",
    "center_poke",
    "center_exit",
    "response_side",
    "video",
)
DISPLAY_LABELS = {
    "task": "Task",
    "video": "Motion energy",
    "visual_flash": "Visual flash",
    "center_poke": "Center poke",
    "center_exit": "Center exit",
    "response_side": "Response side",
}
GROUP_COLORS = {
    "task": "C0",
    "video": "C1",
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
        "video": slice(base_columns, video_stop),
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


def _load_fit_records(fit_dir: Path) -> dict[int, dict]:
    tests = {}
    for path in fit_dir.glob("unit_*_test.json"):
        with path.open() as handle:
            item = json.load(handle)
        tests[int(item["unit_id"])] = item
    selected = {
        int(path.name.split("_")[1]) for path in fit_dir.glob("unit_*_validation.json")
    }
    if not tests or tests.keys() != selected:
        raise ValueError("Complete test and validation records are required.")
    return tests


def _fit_shuffled_model(
    design: np.ndarray,
    counts: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> dict:
    training_mean = float(counts[train].mean())
    selected_model, path = fit_poisson_alpha_path(
        design[train], counts[train], design[validation], counts[validation]
    )
    validation_metrics = poisson_metrics(
        counts[validation], selected_model.predict(design[validation]), training_mean
    )
    fit = train | validation
    fit_mean = float(counts[fit].mean())
    final_model = fit_poisson_at_alpha(design[fit], counts[fit], path["best_alpha"])
    test_metrics = poisson_metrics(
        counts[test], final_model.predict(design[test]), fit_mean
    )
    return {
        "alpha_path": path,
        "validation": validation_metrics,
        "fit_mean_count": fit_mean,
        "test": test_metrics,
    }


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


def run_attribution(
    windows: Path, design_path: Path, fit_dir: Path, unit_set: str
) -> None:
    """Fit same-width shuffled comparators and write the summary figure."""
    prepared = _load_windows(windows)
    common = np.load(design_path, mmap_mode="r", allow_pickle=False)
    with design_path.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    tests = _load_fit_records(fit_dir)
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

    with np.load(windows, allow_pickle=False) as saved:
        trial_split = saved["trial_split"].copy()
        relative_times = saved["relative_bin_centers_s"].copy()
    bins_per_trial = len(relative_times)
    if len(common) != len(trial_split) * bins_per_trial:
        raise ValueError("Common design rows do not match the saved trial grid.")
    valid = _valid_bin_mask(prepared)
    train, validation, test = _split_masks(prepared["split"], valid)
    within_trial = shuffle_permutations(
        len(trial_split), bins_per_trial, SHUFFLE_SEED, valid=valid
    )

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    if unit_set == "test":
        unit_rows = test_unit_indices(len(units))
    else:
        unit_rows = np.arange(len(units))
    output_dir = fit_dir / "conditional_deviance_fit"
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

        counts = build_unit_counts(
            prepared["alignments"], np.asarray(unit["spike_times_s"], dtype=float)
        )
        design = np.asarray(common[:, :common_columns])
        if design.shape[1] != len(tests[unit_id]["plus_video"]["coefficients"]):
            raise ValueError("Saved full model and attribution design widths differ.")

        full_deviance = float(
            tests[unit_id]["plus_video"]["test"]["deviance_explained"]
        )
        group_results = {}
        for group_position, group in enumerate(group_order, start=1):
            columns = groups[group]
            source = design[:, columns].copy()
            design[:, columns] = source[within_trial]
            fitted = _fit_shuffled_model(design, counts, train, validation, test)
            design[:, columns] = source
            del source
            fitted.update(
                columns=columns.stop - columns.start,
                shuffle="rows jointly within trial",
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
