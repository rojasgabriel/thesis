"""Measure the unique and maximal explained deviance of each regressor block in the V1 GLM.

Use the ten whole-trial cross-validation folds and the per-fold penalties from
`fit`. In each fold, refit on the fit trials at that fold's penalty and score
test D² on the held-out trials. Do not select the penalty again. Skip a fold if
the unit has no test spikes in it.

- Unique ΔD² = complete-model D² minus D² with that block shuffled.
- Maximal ΔD² = D² with only that block unshuffled minus D² with all blocks
  shuffled.

Control for motion energy: trial-specific ΔD² = complete-model D² minus D² with
the motion-energy block taken from another trial at the same time in the trial
(`trial_swap_permutation`). The swap keeps the average time course of motion
energy in a trial, so this value is the part of the unique ΔD² that needs the
movement of that specific trial.

A shuffle permutes design rows among the valid bins of each trial. All columns
of a block move together. One permutation (`SHUFFLE_SEED`) is shared by all
units, folds, and blocks. Motion-energy PCs stay one block. The blocks are the
three broad groups (task, video, history) and the eight groups in
`DETAILED_GROUPS`.

Units, event alignment, excluded bins and units, and the D² null (a constant
rate equal to the fit-trial mean) are the same as in `fit.py`. The sampling
unit is the unit. Each unit reports the mean and SEM over its scored folds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import sem, ttest_1samp, wilcoxon

from thesis.ephys.analyses.glm.dmat import (
    HISTORY_COLUMNS,
    _design_with_history,
    _load_windows,
    build_unit_history,
    training_zscore,
)
from thesis.ephys.analyses.glm.fit import (
    _SHARED,
    _write_json_atomic,
    analysis_unit_indices,
    build_unit_counts,
    collect_records,
    fit_poisson_at_alpha,
    load_unit_records,
    poisson_metrics,
)
from thesis.ephys.units import fetch_unit_table

SHUFFLE_SEED = 20260910
BROAD_GROUPS = ("task", "video", "history")
DETAILED_GROUPS = (
    "visual_flash",
    "flash_adaptation",
    "hold",
    "withdrawal",
    "video",
    "history",
)
GROUP_ORDER = tuple(dict.fromkeys((*BROAD_GROUPS, *DETAILED_GROUPS)))
RELIABLE_ALPHA = 0.05
UNIQUE = "unique_test_deviance_explained"
MAXIMAL = "maximal_test_deviance_explained"


def block_slices(metadata: dict, video_components: int) -> dict[str, slice]:
    """Return the contiguous full-model columns for each reported block."""
    task_columns = int(metadata["task_columns"])
    base_columns = int(metadata["base_columns"])
    video_stop = base_columns + video_components
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
    n_trials: int, bins_per_trial: int, seed: int, valid: np.ndarray
) -> np.ndarray:
    """Return within-trial permutations among each trial's valid bins."""
    if n_trials <= 0 or bins_per_trial <= 1:
        raise ValueError(
            "Trials and bins per trial must define a two-dimensional grid."
        )
    rows = np.arange(n_trials * bins_per_trial, dtype=np.int64).reshape(
        n_trials, bins_per_trial
    )
    valid = np.asarray(valid, dtype=bool).reshape(rows.shape)
    rng = np.random.default_rng(seed)
    for trial_rows, keep in zip(rows, valid, strict=True):
        trial_rows[keep] = rng.permutation(trial_rows[keep])
    return rows.ravel()


def trial_swap_permutation(n_trials: int, bins_per_trial: int, seed: int) -> np.ndarray:
    """Map each row to the same bin of another trial.

    The trials form one random cycle, so no trial keeps its own rows.
    """
    order = np.random.default_rng(seed).permutation(n_trials)
    partner = np.empty(n_trials, dtype=np.int64)
    partner[order] = np.roll(order, -1)
    return (partner[:, None] * bins_per_trial + np.arange(bins_per_trial)).ravel()


def _scored_folds(counts, partitions, alphas) -> list[tuple[dict, float]]:
    """Pair saved alphas with the folds that contained test spikes."""
    folds = [fold for fold in partitions if counts[fold["test"]].sum() > 0]
    if len(folds) != len(alphas):
        raise ValueError(
            f"{len(folds)} folds contained spikes but {len(alphas)} alphas were saved."
        )
    return list(zip(folds, alphas, strict=True))


def _write_summary(records: list[dict], output_dir: Path) -> dict:
    summary = {
        "units": len(records),
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_repetitions": 1,
        "groups": {},
    }
    group_keys = (
        "unique_test_deviance_explained",
        "unique_test_deviance_explained_sem",
        "maximal_test_deviance_explained",
        "maximal_test_deviance_explained_sem",
    )
    rows = []
    for group in GROUP_ORDER:
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
                **{key: record["groups"][group][key] for key in group_keys},
            }
            for record in records
        )

    swapped = np.asarray(
        [
            record["groups"]["video"]["trial_specific_test_deviance_explained"]
            for record in records
        ]
    )
    summary["video_trial_specific"] = {
        "mean": float(np.mean(swapped)),
        "median": float(np.median(swapped)),
        "q25": float(np.quantile(swapped, 0.25)),
        "q75": float(np.quantile(swapped, 0.75)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "summary.json", summary)
    csv_path = output_dir / "unit_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _block_task(job: dict) -> dict:
    """Score every block's unique and maximal deviance for one unit."""
    prepared = _SHARED["prepared"]
    common = _SHARED["common"]
    spikes = np.asarray(job["spike_times"], dtype=float)
    counts = build_unit_counts(prepared["alignments"], spikes)
    raw_history = build_unit_history(prepared["alignments"], spikes)
    groups, order = job["groups"], job["group_order"]
    within = _SHARED["within_trial"]
    between = _SHARED["between_trials"]
    per_fold = {name: {"unique": [], "maximal": []} for name in order}
    complete_folds, trial_specific = [], []
    for fold_index, (fold, alpha) in enumerate(
        _scored_folds(counts, _SHARED["partitions"], job["alphas"]), start=1
    ):
        fit, test = fold["fit"], fold["test"]
        alpha = float(alpha)
        history, _, _ = training_zscore(raw_history, fit)
        fit_rows = np.flatnonzero(fit)
        test_rows = np.flatnonzero(test)
        fit_design = _design_with_history(common, history, job["common_columns"], fit)
        test_design = _design_with_history(common, history, job["common_columns"], test)
        fit_counts = counts[fit]
        test_counts = counts[test]
        fit_mean = float(fit_counts.mean())

        def score(comparison: str) -> float:
            # Refit at this fold's full-model penalty, not a reselected one, so a
            # change in D² comes from the shuffle and not from a new penalty.
            try:
                model = fit_poisson_at_alpha(fit_design, fit_counts, alpha)
                return poisson_metrics(
                    test_counts, model.predict(test_design), fit_mean
                )["deviance_explained"]
            except Exception as error:
                raise RuntimeError(
                    f"fold {fold_index}, {comparison}: {error}"
                ) from error

        complete = score("full model")
        complete_folds.append(complete)

        # Shuffle the two scored matrices in place and restore from the source.
        # Each fold copies its train and test rows only once. `mapping` is a row
        # permutation, or None to restore the real rows.
        def put(columns, mapping):
            width = job["common_columns"]
            if columns.stop <= width:
                source = np.asarray(common[:, columns])
            else:
                source = history[:, columns.start - width : columns.stop - width]
            for matrix, rows in ((fit_design, fit_rows), (test_design, test_rows)):
                matrix[:, columns] = source[rows if mapping is None else mapping[rows]]

        # Control: motion energy of another trial at the same time in the trial.
        if "video" in order:
            put(groups["video"], between)
            trial_specific.append(complete - score("video swapped between trials"))
            put(groups["video"], None)

        # These detailed blocks are non-overlapping and cover the full model.
        blocks = [groups[name] for name in DETAILED_GROUPS]
        for columns in blocks:
            put(columns, within)
        reference = score("all regressor groups shuffled")
        for columns in blocks:
            put(columns, None)

        for group in order:
            columns = groups[group]
            put(columns, within)
            removed = score(f"{group} shuffled")
            put(columns, None)

            for other in blocks:
                put(other, within)
            put(columns, None)
            alone = score(f"only {group} unshuffled")
            for other in blocks:
                put(other, None)

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
            "columns": sum(
                part.stop - part.start
                for part in (
                    groups[group]
                    if isinstance(groups[group], tuple)
                    else (groups[group],)
                )
            ),
            "shuffle": "rows jointly within trial",
            "unique_test_deviance_explained": float(unique.mean()),
            "unique_test_deviance_explained_sem": float(sem(unique)),
            "maximal_test_deviance_explained": float(maximal.mean()),
            "maximal_test_deviance_explained_sem": float(sem(maximal)),
            "folds": {"unique": unique.tolist(), "maximal": maximal.tolist()},
        }
    if "video" in order:
        group_results["video"]["trial_specific_test_deviance_explained"] = float(
            np.mean(trial_specific)
        )
        group_results["video"]["trial_specific_test_deviance_explained_sem"] = float(
            sem(trial_specific)
        )
        group_results["video"]["folds"]["trial_specific"] = trial_specific
    record = {
        "unit_id": job["unit_id"],
        "depth": job["depth"],
        "video_components": job["video_components"],
        "shuffle_seed": SHUFFLE_SEED,
        "folds": len(complete_folds),
        "complete_test_deviance_explained": float(np.mean(complete_folds)),
        "groups": group_results,
    }
    return record


def block_table(samples: dict[str, list[dict]]) -> dict:
    """Summarize each block within each sample and over all units."""
    complete = np.concatenate(
        [
            [record["complete_test_deviance_explained"] for record in records]
            for records in samples.values()
        ]
    )
    table = {
        "samples": list(samples),
        "units": int(len(complete)),
        "complete_d2_median": float(np.median(complete)),
        "reliable_alpha": RELIABLE_ALPHA,
        "blocks": {},
    }
    for group in DETAILED_GROUPS:
        per_sample = []
        for records in samples.values():
            entries = [record["groups"][group] for record in records]
            unique = np.asarray([entry[UNIQUE] for entry in entries])
            # Does this unit gain or lose? Test its fold values against zero,
            # one-sided in each direction.
            gain_p, loss_p = (
                np.asarray(
                    [
                        ttest_1samp(
                            entry["folds"]["unique"], 0, alternative=side
                        ).pvalue
                        for entry in entries
                    ]
                )
                for side in ("greater", "less")
            )
            per_sample.append(
                {
                    "columns": int(entries[0]["columns"]),
                    "unique": unique,
                    "maximal": np.asarray([entry[MAXIMAL] for entry in entries]),
                    "reliable_fraction": float(np.mean(gain_p < RELIABLE_ALPHA)),
                    "gain": gain_p < RELIABLE_ALPHA,
                    "loss": loss_p < RELIABLE_ALPHA,
                    "wilcoxon_p": float(wilcoxon(unique, alternative="greater").pvalue),
                }
            )
        pooled_unique = np.concatenate([item["unique"] for item in per_sample])
        share = float(np.median(pooled_unique)) / table["complete_d2_median"]
        reliable = [item["reliable_fraction"] for item in per_sample]
        table["blocks"][group] = {
            "columns": [item["columns"] for item in per_sample],
            "unique_median_by_sample": [
                float(np.median(item["unique"])) for item in per_sample
            ],
            "maximal_median_by_sample": [
                float(np.median(item["maximal"])) for item in per_sample
            ],
            "reliable_fraction_by_sample": reliable,
            "gain_fraction": float(
                np.mean(np.concatenate([item["gain"] for item in per_sample]))
            ),
            "loss_fraction": float(
                np.mean(np.concatenate([item["loss"] for item in per_sample]))
            ),
            "wilcoxon_p_by_sample": [item["wilcoxon_p"] for item in per_sample],
            "unique_median": float(np.median(pooled_unique)),
            "unique_q25_q75": np.quantile(pooled_unique, [0.25, 0.75]).tolist(),
            "maximal_median": float(
                np.median(np.concatenate([item["maximal"] for item in per_sample]))
            ),
            "share_of_complete_d2": share,
        }
    return table


def run_unique(
    windows: Path,
    design_path: Path,
    fit_dir: Path,
    unit_set: str,
    workers: int | None = None,
) -> None:
    """Fit same-width shuffled comparators; plot the summary and the example unit."""
    prepared = _load_windows(windows)
    common = np.load(design_path, mmap_mode="r", allow_pickle=False)
    with design_path.with_suffix(".json").open() as handle:
        metadata = json.load(handle)
    checkpoints = fit_dir / "checkpoints"
    fold_records = load_unit_records(checkpoints, "unit_*_folds.json")
    if not fold_records:
        raise ValueError(f"No cross-validated fits in {checkpoints}; run `fit` first.")
    selected_components = {int(item["components"]) for item in fold_records}
    if len(selected_components) != 1:
        raise ValueError("All cross-validated fits must use one video-PC count.")
    video_components = selected_components.pop()
    common_columns = int(metadata["base_columns"]) + video_components
    groups = block_slices(metadata, video_components)

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
    unit_rows = analysis_unit_indices(units, unit_set)
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
            "group_order": GROUP_ORDER,
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
    print(
        f"Starting unique-deviance analysis: {len(jobs)} units, "
        "full and shuffled comparator fits within each cross-validation fold.",
        flush=True,
    )
    records = collect_records(
        "unique-deviance analysis",
        _block_task,
        jobs,
        windows,
        design_path,
        workers,
        shuffle_seed=SHUFFLE_SEED,
    )

    summary = _write_summary(records, output_dir)
    from thesis.ephys.analyses.glm.figures import (
        plot_deviance_explained,
        plot_example_unit,
    )
    from thesis.ephys.analyses.glm.results import example_unit_id

    figure_path = plot_deviance_explained(
        {unit_set: records}, fit_dir / "deviance_explained"
    )
    print(json.dumps({**summary, "figure": str(figure_path)}, indent=2))
    final_records = load_unit_records(checkpoints, "unit_*_final.json")
    plot_example_unit(
        windows, design_path, fit_dir, example_unit_id(records, final_records)
    )
