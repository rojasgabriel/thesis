"""Test whether the flash response differs between the fixation and the time after it.

Question: V1 units fire less in the fixation (first flash to center exit). The
fixation kernel adds one value to the log rate, so it lowers the baseline and
the flash response by the same factor. Is that enough, or does the flash
response change by a different amount (or in shape) between the two periods?

Comparison: the design in use against the same design with one more term, a
second flash kernel (the same raised cosines) for a flash at or after the
center exit against a flash in the fixation (`dmat.flash_after_exit`). The
first flash of a trial is not in the contrast. Both designs are scored on the
same units and the same validation trials, so the comparison is paired over
units (two-sided Wilcoxon signed-rank test).

A term that does not help means "the same factor": a gain change of the unit.
It does not mean that the fixation has no effect on the visual response. The
sign and the shape of the new kernel (Δ log rate, after the exit minus in the
fixation) answer the question; the mean over `RESPONSE_MS` is one value for
each unit, with the SD over the cross-validation folds. For each unit, the
held-out D² of the 10 folds with the term is also compared with that of the
same folds without it (paired t-test, one-sided, p < `RELIABLE_ALPHA`), as for
the regressors of the main figure; a term with no effect gives about 5%.

Units (`--units`, the 20 selection units by default), trials, alignment (first
flash), baseline, and exclusions are those of `fit.py`. The sampling unit is
the unit. The fit with the term goes to a directory next to the session
directory (suffix `ROOT_SUFFIX`) and is not run again when it is there. The
script writes `flash_fixation_test.json` in the session directory.

    uv run python -m thesis.ephys.analyses.glm.flash_fixation_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.hold_effect import _fit_variant
from thesis.ephys.analyses.glm.unique_dev import RELIABLE_ALPHA

ROOT_SUFFIX = "_flash_after_exit_term"
RESPONSE_MS = (30, 120)


def _validation_scores(root: Path, unit_set: str) -> dict[int, float]:
    """Return each unit's validation D² with the largest motion-energy PC count."""
    checkpoints = cli.PoissonGLM(root=root).fit_dir(unit_set) / "checkpoints"
    count = str(max(dmat.VIDEO_COMPONENT_COUNTS))
    return {
        int(item["unit_id"]): item["plus_video"][count]["validation"][
            "deviance_explained"
        ]
        for item in load_unit_records(checkpoints, "unit_*_validation.json")
    }


def _fold_scores(root: Path, unit_set: str) -> dict[int, np.ndarray]:
    """Return each unit's held-out D² in each cross-validation fold."""
    checkpoints = cli.PoissonGLM(root=root).fit_dir(unit_set) / "checkpoints"
    return {
        int(item["unit_id"]): np.asarray(
            [fold["test"]["deviance_explained"] for fold in item["folds"]]
        )
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="sample")
    args = parser.parse_args()
    root = _fit_variant(
        args.root, args.units, ROOT_SUFFIX, {"FLASH_AFTER_EXIT_TERM": True}
    )
    in_use = _validation_scores(args.root, args.units)
    with_term = _validation_scores(root, args.units)
    unit_ids = sorted(in_use)
    if unit_ids != sorted(with_term):
        raise ValueError("The two designs were not fit on the same units.")
    gain = np.asarray([with_term[unit] - in_use[unit] for unit in unit_ids])

    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    columns = [
        index
        for index, name in enumerate(metadata["column_names"])
        if name.startswith(dmat.FLASH_AFTER_EXIT)
    ]
    scale = np.asarray(metadata["training_scale"])[columns]
    basis = dmat.task_temporal_bases()["visual_flash"]
    times = 1000 * basis.basis_time
    window = (times >= RESPONSE_MS[0]) & (times < RESPONSE_MS[1])
    kernel = lambda record: (  # noqa: E731
        (np.asarray(record["coefficients"])[columns] / scale) @ basis.basis.T
    )
    checkpoints = model.fit_dir(args.units) / "checkpoints"
    final = {
        int(item["unit_id"]): kernel(item)
        for item in load_unit_records(checkpoints, "unit_*_final.json")
    }
    folds = {
        int(item["unit_id"]): [kernel(fold)[window].mean() for fold in item["folds"]]
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    curves = np.asarray([final[unit] for unit in unit_ids])
    level = curves[:, window].mean(1)
    # For each unit, the same folds with and without the term (paired t-test).
    base_folds, term_folds = (_fold_scores(r, args.units) for r in (args.root, root))
    paired = [
        unit for unit in unit_ids if len(base_folds[unit]) == len(term_folds[unit])
    ]
    fold_gain = np.asarray([(term_folds[u] - base_folds[u]).mean() for u in paired])
    fold_p = np.asarray(
        [ttest_rel(term_folds[u], base_folds[u]).pvalue for u in paired]
    )
    reliable = fold_p / 2 < RELIABLE_ALPHA  # one-sided, as in the main figure
    needs_term = reliable & (fold_gain > 0)
    stats = {
        "units": len(unit_ids),
        "cross_validated_d2_gain": {
            "units_with_the_same_folds": len(paired),
            "median": float(np.median(fold_gain)),
            "wilcoxon_p": float(wilcoxon(fold_gain).pvalue),
            "fraction_units_better_with_term": float(needs_term.mean()),
            "fraction_units_worse_with_term": float(
                (reliable & (fold_gain < 0)).mean()
            ),
            "median_kernel_level_of_units_better_with_term": float(
                np.median(level[[unit_ids.index(u) for u in paired]][needs_term])
            )
            if needs_term.any()
            else None,
            "units_better_with_kernel_above_0": int(
                np.sum(level[[unit_ids.index(u) for u in paired]][needs_term] > 0)
            ),
            "unit_ids": paired,
            "by_unit": fold_gain.tolist(),
            "p_by_unit": fold_p.tolist(),
        },
        "unit_ids": unit_ids,
        "validation_d2_gain": {
            "median": float(np.median(gain)),
            "median_relative_to_in_use": float(
                np.median(gain) / np.median([in_use[unit] for unit in unit_ids])
            ),
            "units_better": int(np.sum(gain > 0)),
            "wilcoxon_p": float(wilcoxon(gain).pvalue),
            "by_unit": gain.tolist(),
        },
        "delta_log_rate_after_exit_minus_fixation": {
            "response_window_ms": RESPONSE_MS,
            "median": float(np.median(level)),
            "units_above_0": int(np.sum(level > 0)),
            "wilcoxon_p": float(wilcoxon(level).pvalue),
            "by_unit": level.tolist(),
            "fold_sd_by_unit": [float(np.std(folds[unit])) for unit in unit_ids],
        },
        "kernel_times_ms": times.tolist(),
        "kernel_median": np.median(curves, 0).tolist(),
        "kernel_quartiles": np.percentile(curves, [25, 75], 0).tolist(),
        "kernel_by_unit": curves.tolist(),
    }
    output = args.root / "flash_fixation_test.json"
    output.write_text(json.dumps(stats, indent=1))
    printed = {key: stats[key] for key in list(stats)[:5]}
    print(json.dumps(printed, indent=1))
    print(output)


if __name__ == "__main__":
    main()
