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

A shuffle permutes design rows among the valid bins of each trial. A block
that is a variable of the trial (`OVER_TRIALS`: drift, the previous outcome) is
instead taken from another trial at the same bin (`trial_swap_permutation`),
because a shuffle inside the trial keeps its value there. A block that is a
side (±1, or 0 for unknown) on a time course of the trial itself
(`SIGN_OVER_TRIALS`: previous side and turn direction) keeps its own time
course and takes the actual side code of the other trial. The swap is in raw
columns before applying the saved centering and scale. Taking the rows of the other trial would
also move a time course that is aligned to the events of that trial, and
test the timing and not the side. All columns of a block move
together. One permutation (`SHUFFLE_SEED`) is shared by all
units, folds, and blocks. Motion-energy PCs stay one block. The blocks are the
three broad groups (task, video, history) and each regressor of the design
(the task manifest of the saved design).

Units, event alignment, excluded bins and units, and the D² null (a constant
rate equal to the fit-trial mean) are the same as in `fit.py`. The sampling
unit is the unit. Each unit reports the mean and SEM over its scored folds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import sem

from thesis.ephys.analyses.glm.dmat import (
    DRIFT,
    HISTORY_COLUMNS,
    PREVIOUS_OUTCOME,
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
    _write_json_atomic,
    analysis_unit_indices,
    build_unit_counts,
    collect_records,
    fit_poisson_at_alpha,
    load_unit_records,
    poisson_metrics,
)
from thesis.ephys.units import fetch_unit_table

# The regressors that are a variable of the trial, shuffled over trials.
OVER_TRIALS = (DRIFT, PREVIOUS_OUTCOME)
# The regressors that are a side (±1) on a time course of the trial: the sign
# is shuffled over trials, the time course stays.
SIGN_OVER_TRIALS = (PREVIOUS_SIDE, TURN_DIRECTION)
# The name of the shuffle rule, saved in each record.
LEGACY_SHUFFLE_RULE = "trial variables over trials, sides by sign, 2026-10-16"
SHUFFLE_RULE = "trial variables over trials, raw sides with unknowns, 2026-10-18"

SHUFFLE_SEED = 20260910
BROAD_GROUPS = ("task", "video", "history")
RELIABLE_ALPHA = 0.05
UNIQUE = "unique_test_deviance_explained"
MAXIMAL = "maximal_test_deviance_explained"
UNIQUE_BITS = "unique_bits_per_spike"
MAXIMAL_BITS = "maximal_bits_per_spike"


def add_bits_per_spike(records: list[dict], checkpoints: Path) -> list[dict]:
    """Add the unique and maximal gains in bits per spike to each unique record.

    In one test fold, D² and bits per spike are the same log-likelihood gain over
    a constant rate, divided by the null deviance and by the spike count times
    ln 2. That factor does not depend on the model, so a fold's ΔD² times the
    bits/D² of the full model in that fold (`unit_*_folds.json`) is its Δ bits
    per spike, with the same sign. Adds `UNIQUE_BITS`, `MAXIMAL_BITS`, the fold
    values (`unique_bits`, `maximal_bits`), and `complete_bits_per_spike`.
    """
    folds = {
        int(item["unit_id"]): [fold["test"] for fold in item["folds"]]
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    for record in records:
        tests = folds[int(record["unit_id"])]
        d2 = np.asarray([t["deviance_explained"] for t in tests], dtype=float)
        bits = np.asarray([t["bits_per_spike"] for t in tests], dtype=float)
        # A fold with D² of about 0 gives 0/0; its factor is the median of the
        # other folds of the unit (the factor is about the same in each fold).
        good = np.abs(d2) > 1e-6
        if not good.any():
            raise ValueError(f"Unit {record['unit_id']}: D² is 0 in every fold.")
        factor = np.full(len(tests), np.median(bits[good] / d2[good]))
        factor[good] = bits[good] / d2[good]
        record["complete_bits_per_spike"] = float(np.mean(bits))
        for group in record["groups"].values():
            for kind, key in (("unique", UNIQUE_BITS), ("maximal", MAXIMAL_BITS)):
                values = np.asarray(group["folds"][kind], dtype=float)
                if len(values) != len(factor):
                    raise ValueError(f"Unit {record['unit_id']}: folds do not match.")
                group["folds"][f"{kind}_bits"] = (values * factor).tolist()
                group[key] = float(np.mean(values * factor))
    return records


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


def shuffled_side_columns(
    standardized: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    sides: np.ndarray,
    partner: np.ndarray,
    unknown_support: np.ndarray | None = None,
) -> np.ndarray:
    """Swap actual side codes while retaining each recipient trial's time course.

    `standardized` is trial × bin × column. A side of 0 means unknown. Its
    unsigned time course must be supplied in `unknown_support` (unknown trials
    × bin × column), because a raw zero cannot reveal that time course.
    """
    sides = np.asarray(sides)
    if not np.isin(sides, (-1, 0, 1)).all():
        raise ValueError("Side codes must be -1, 0, or 1.")
    raw = np.asarray(standardized, dtype=float) * scale + mean
    unsigned = raw * sides[:, None, None]
    unknown = sides == 0
    if unknown.any():
        if unknown_support is None or unknown_support.shape != unsigned[unknown].shape:
            raise ValueError("Unknown sides need their unsigned time course.")
        unsigned[unknown] = unknown_support
    shuffled = unsigned * sides[partner, None, None]
    return ((shuffled - mean) / scale).astype(standardized.dtype)


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
    for group in records[0]["groups"]:  # the groups of the fitted design
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
    previous = job.get("reuse_unique")
    # Shuffle before centering: flipping a centered zero invents an effect in
    # bins where the raw regressor is off.
    trials = len(prepared["alignments"])
    bins = len(between) // trials
    partner = between.reshape(trials, bins)[:, 0] // bins
    side_blocks = {}
    for name in SIGN_OVER_TRIALS:
        if name in groups:
            columns = groups[name]
            block = np.asarray(common[:, columns]).reshape(trials, bins, -1)
            side_blocks[name] = shuffled_side_columns(
                block,
                job["common_mean"][columns],
                job["common_scale"][columns],
                job["side_codes"][name],
                partner,
                job.get("unknown_previous_support") if name == PREVIOUS_SIDE else None,
            ).reshape(len(common), -1)
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

        complete = (
            float(job["complete_folds"][fold_index - 1])
            if previous is not None
            else score("full model")
        )
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
                if isinstance(mapping, tuple):  # ("side", raw-space side shuffle)
                    matrix[:, columns] = side_blocks[mapping[1]][rows]
                else:
                    matrix[:, columns] = source[
                        rows if mapping is None else mapping[rows]
                    ]

        # Control: motion energy of another trial at the same time in the trial.
        if "video" in order and previous is None:
            put(groups["video"], between)
            trial_specific.append(complete - score("video swapped between trials"))
            put(groups["video"], None)

        # These detailed blocks are non-overlapping and cover the full model.
        blocks = [name for name in order if name != "task"]

        def shuffle_of(name):
            # A variable of the trial (drift, the previous side, the side of the
            # response, the turn direction, the previous outcome) is shuffled over trials. A
            # shuffle inside a trial keeps its sign there, so it would test only
            # when the column turns on, not the variable.
            if name in side_blocks:
                return ("side", name)
            return between if name in OVER_TRIALS else within

        for name in blocks:
            put(groups[name], shuffle_of(name))
        reference = score("all regressor groups shuffled")
        for name in blocks:
            put(groups[name], None)

        for group in order:
            columns = groups[group]
            if previous is None or group in SIGN_OVER_TRIALS:
                put(columns, shuffle_of(group))
                removed = score(f"{group} shuffled")
                put(columns, None)
                unique = complete - removed
            else:
                unique = previous["groups"][group]["folds"]["unique"][fold_index - 1]

            for name in blocks:
                put(groups[name], shuffle_of(name))
            put(columns, None)
            alone = score(f"only {group} unshuffled")
            for name in blocks:
                put(groups[name], None)

            per_fold[group]["unique"].append(unique)
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
            "shuffle": (
                "raw side codes over trials, recipient timing retained"
                if group in SIGN_OVER_TRIALS
                else (
                    "rows jointly over trials"
                    if group in OVER_TRIALS
                    else "rows jointly within trial"
                )
            ),
            "unique_test_deviance_explained": float(unique.mean()),
            "unique_test_deviance_explained_sem": float(sem(unique)),
            "maximal_test_deviance_explained": float(maximal.mean()),
            "maximal_test_deviance_explained_sem": float(sem(maximal)),
            "folds": {"unique": unique.tolist(), "maximal": maximal.tolist()},
        }
    if "video" in order:
        if previous is not None:
            trial_specific = previous["groups"]["video"]["folds"]["trial_specific"]
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
        "shuffle_rule": SHUFFLE_RULE,
        "folds": len(complete_folds),
        "complete_test_deviance_explained": float(np.mean(complete_folds)),
        "groups": group_results,
        "reused_unique_groups": (
            [name for name in order if name not in SIGN_OVER_TRIALS]
            if previous is not None
            else []
        ),
    }
    return record


def run_unique(
    windows: Path,
    design_path: Path,
    fit_dir: Path,
    unit_set: str,
    workers: int | None = None,
    *,
    repair_side_shuffle: bool = False,
) -> None:
    """Fit shuffled comparators and write the summary and example-unit data.

    `repair_side_shuffle` preserves legacy records in `unique_centered_sides`.
    It reuses unaffected unique folds and trial-swapped video scores. It refits
    the two side-unique tests and every maximal test, whose all-shuffled
    reference also depends on the side rule. Restarting reuses repaired units.
    """
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
    trials = _load_selected_trials(prepared)
    side_codes = {
        PREVIOUS_SIDE: np.nan_to_num(trials["last_side"].to_numpy(dtype=float)),
        TURN_DIRECTION: trials["response"].to_numpy(dtype=float),
    }
    unknown = side_codes[PREVIOUS_SIDE] == 0
    before_exit = (
        prepared["bin_times"].reshape(len(trials), -1)[unknown]
        < trials["center_exit_s"].to_numpy(dtype=float)[unknown, None]
    )

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
    preserved_dir = checkpoints / "unique_centered_sides"
    stale = [
        path
        for path in checkpoint_dir.glob("unit_*.json")
        if json.loads(path.read_text()).get("shuffle_rule") != SHUFFLE_RULE
    ]
    if stale and not repair_side_shuffle:
        raise ValueError(
            f"{len(stale)} records in {checkpoint_dir} use an older shuffle rule. "
            "Call run_unique(..., repair_side_shuffle=True) to preserve and repair "
            "the centered-side records, or move that folder aside for a full rerun."
        )
    if repair_side_shuffle:
        for path in stale:
            if json.loads(path.read_text()).get("shuffle_rule") != LEGACY_SHUFFLE_RULE:
                raise ValueError(f"Cannot repair an unknown shuffle rule: {path}")
            if (preserved_dir / path.name).exists():
                raise FileExistsError(preserved_dir / path.name)
        if stale:
            preserved_dir.mkdir(parents=True, exist_ok=True)
            for path in stale:
                path.rename(preserved_dir / path.name)
    preserved = (
        {
            int(item["unit_id"]): item
            for item in load_unit_records(preserved_dir, "unit_*.json")
        }
        if repair_side_shuffle
        else {}
    )
    group_order = tuple(dict.fromkeys((*BROAD_GROUPS, *groups)))
    fold_deviance = {
        int(item["unit_id"]): [
            float(fold["test"]["deviance_explained"]) for fold in item["folds"]
        ]
        for item in fold_records
    }
    for unit_id, record in preserved.items():
        if unit_id not in fold_deviance:
            raise ValueError(
                f"Preserved unique unit {unit_id} has no current fold fit."
            )
        if (
            record.get("shuffle_rule") != LEGACY_SHUFFLE_RULE
            or record.get("shuffle_seed") != SHUFFLE_SEED
            or set(record["groups"]) != set(group_order)
            or any(
                len(group["folds"]["unique"]) != len(fold_deviance[unit_id])
                for group in record["groups"].values()
            )
            or not np.isclose(
                record["complete_test_deviance_explained"],
                np.mean(fold_deviance[unit_id]),
                atol=1e-6,
                rtol=1e-5,
            )
        ):
            raise ValueError(
                f"Preserved unique unit {unit_id} differs from current fits."
            )

    jobs = [
        {
            "unit_id": int(units.iloc[row]["unit_id"]),
            "depth": float(units.iloc[row]["depth"]),
            "spike_times": np.asarray(units.iloc[row]["spike_times_s"], dtype=float),
            "video_components": video_components,
            "common_columns": common_columns,
            "groups": groups,
            "common_mean": np.asarray(metadata["training_mean"]),
            "common_scale": np.asarray(metadata["training_scale"]),
            "side_codes": side_codes,
            "unknown_previous_support": before_exit[..., None].astype(float),
            "reuse_unique": preserved.get(int(units.iloc[row]["unit_id"])),
            "complete_folds": fold_deviance[int(units.iloc[row]["unit_id"])],
            # The broad groups, then every regressor of this design.
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
    from thesis.ephys.analyses.glm.results import example_unit_id, write_figure_data

    print(json.dumps(summary, indent=2))
    final_records = load_unit_records(checkpoints, "unit_*_final.json")
    write_figure_data(
        windows, design_path, fit_dir, example_unit_id(records, final_records)
    )
