"""Test whether another parameterization of a term makes the other kernels harder to recover.

Question: a parameterization of a term can predict better and still make the
weights of the other terms less accurate. With the parameterization under
test in the design, are the known kernels of the model recovered as well as
with the parameterization in use?

This is step 2 of the design optimization (`design_optimization.py`),
together with the collinearity diagnostics: it runs for a parameterization
that predicts reliably better in step 1. The method is parameter recovery
(Wilson and Collins 2019): simulate from known parameters, fit, and compare.

Comparison: for each unit, spikes are simulated from its final refit, so the
kernels of that fit are the truth. The model is fit to the simulated spikes
two times: with the design in use, and with the same design where the term
has the parameterization under test (`--grid`, `--design`). The comparison is
paired: each unit has the error of each fit, on the same simulated spikes.

- Simulation: `leak_test.simulate` (the rate of the final refit in each 1-ms
  bin, with the history of the simulated spikes). `SIMULATIONS` simulations
  for each unit (`SEED`); the values of a unit are means over them. The
  penalty of each fit is the penalty of the unit; it is not selected again.
- Measures (`leak_test.measures`, Δ log rate): the fixation level (the mean
  of the fixation kernel from `leak_test.LEVEL_MS`), and the flash × move
  effect (the response of the flash kernel in move minus still) at each gap
  class. These are the kernels that the results of the model are read from.
- Error of a fit: the absolute difference between the recovered and the true
  value of a measure.
- Statistics: two-sided Wilcoxon signed-rank test over units of the error
  with the parameterization under test minus the error with the one in use.
  The parameterization fails when the error of a measure is reliably larger.
- Recovery of each regressor of the design in use: the agreement of each
  recovered kernel with the true one, 1 - |a - b|² / (|a|² + |b|²) (1 for
  equal kernels, 0 for unrelated kernels), and its size over the true size.
  For motion energy and the session drift, which have no kernel over time,
  the same two measures on the weights of all their columns.
  They are saved in `parameter_recovery.npz` in each session directory;
  `design_optimization.py` draws them.
- Figure: one row for each session, one panel for each measure: the error of
  each unit with the parameterization under test against its error with the
  one in use. A point above the diagonal is a unit that is recovered worse.
  A point outside `AXIS_LIMIT` is drawn at the limit; it is in the
  statistics, and the count is in the output.

Units (`--units`), trials, alignment, and exclusions are those of `fit.py`.
The sampling unit is the unit. The script writes `parameter_recovery.svg` and
`parameter_recovery.json` in `figures/glm/`. The design under test is built
in a work directory there and removed at the end.

    uv run python -m thesis.ephys.analyses.glm.parameter_recovery
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, leak_test
from thesis.ephys.analyses.glm.fit import fit_poisson_at_alpha, load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.kernel_reliability import agreement
from thesis.ephys.analyses.glm.results import fitted_kernels
from thesis.ephys.analyses.glm.unique_dev import block_slices
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

SIMULATIONS = 3
SEED = 0
AXIS_LIMIT = 0.6
MEASURES = (*(f"flash × move, {gap}" for gap in leak_test.GAPS_MS), "fixation level")
COLORS = (CONCEPT_COLORS["visual"], CONCEPT_COLORS["visual"], CONCEPT_COLORS["task"])
_OTHER: dict = {}


def _load(windows: Path, design: Path, other: Path, settings: dict) -> None:
    leak_test._load(windows, design)
    _OTHER.update(
        common=np.load(other, mmap_mode="r", allow_pickle=False),
        metadata=json.loads(other.with_suffix(".json").read_text()),
        settings=settings,
    )


def weight_vectors(result: dict, metadata: dict, history_scale: np.ndarray) -> dict:
    """Return the values that represent each regressor of one fit.

    For a kernel over time, the kernel (`results.fitted_kernels`). For the
    regressors with no kernel over time, the weights of all their columns, on
    the scale of the raw columns: the 10 motion-energy PCs and the columns of
    the session drift.
    """
    values = {
        name: kernel
        for name, (_, kernel) in fitted_kernels(result, metadata, history_scale).items()
    }
    weights = np.asarray(result["coefficients"], dtype=float)
    scale = np.asarray(metadata["training_scale"], dtype=float)
    for name, columns in block_slices(metadata, int(result["components"])).items():
        if name in ("video", dmat.DRIFT):
            values[name] = weights[columns] / scale[columns]
    return values


def _unit(job: dict) -> dict:
    """Return the measures of one unit, and the recovery of each of its kernels.

    "values": the true and the two recovered values of each measure, (3,
    measure). "agreement" and "size": for each regressor of the design in use,
    the agreement of the recovered kernel with the true one (1 for equal
    kernels; `kernel_reliability.agreement`) and the size of the recovered
    kernel over the size of the true one (the ratio of the weights for a
    regressor with one column), means over the simulations.
    """
    shared = leak_test._SHARED
    prepared, valid = shared["prepared"], shared["valid"]
    record = job["record"]
    components = int(record["components"])
    raw = dmat.build_unit_history(prepared["alignments"], job["spike_times"])
    _, mean, scale = dmat.training_zscore(raw, valid)
    truth = leak_test.measures(record, scale)
    true_kernels = weight_vectors(record, shared["metadata"], scale)
    kernel_agreement, kernel_size, recovered_kernels = [], [], []
    coefficients = np.asarray(record["coefficients"], dtype=float)
    rng = np.random.default_rng(SEED + job["unit_id"])
    recovered = []
    for _ in range(SIMULATIONS):
        counts, history, history_scale = leak_test.simulate(
            record, coefficients, mean, scale, rng
        )
        counts = counts.ravel()[valid]
        both = []
        for common, metadata, settings in (
            (shared["common"], shared["metadata"], {}),
            (_OTHER["common"], _OTHER["metadata"], _OTHER["settings"]),
        ):
            width = int(metadata["base_columns"]) + components
            design = dmat._design_with_history(common, history, width, valid)
            fitted = fit_poisson_at_alpha(design, counts, job["alpha"]).coef_
            if not settings:  # the design in use: the recovery of each kernel
                kernels = weight_vectors(
                    {"components": components, "coefficients": fitted},
                    metadata,
                    history_scale,
                )
                recovered_kernels.append(kernels)
                kernel_agreement.append(
                    [
                        agreement(true_kernels[name][None], kernels[name][None])[0]
                        for name in true_kernels
                    ]
                )
                kernel_size.append(
                    [
                        np.linalg.norm(kernels[name])
                        / np.linalg.norm(true_kernels[name])
                        for name in true_kernels
                    ]
                )
            # The kernels are read with the bases and the metadata of that design.
            saved = {name: getattr(dmat, name) for name in settings}
            in_use = shared["metadata"]
            for name, value in settings.items():
                setattr(dmat, name, value)
            shared["metadata"] = metadata
            try:
                both.append(
                    leak_test.measures(
                        {"components": components, "coefficients": fitted},
                        history_scale,
                    )
                )
            finally:
                shared["metadata"] = in_use
                for name, value in saved.items():
                    setattr(dmat, name, value)
        recovered.append(both)
    return {
        "values": np.vstack((truth, np.mean(recovered, axis=0))),
        "regressors": list(true_kernels),
        "agreement": np.mean(kernel_agreement, axis=0),
        "size": np.mean(kernel_size, axis=0),
        # The true and the recovered (mean over simulations) values of each
        # regressor, to see where a part of one goes.
        "true": true_kernels,
        "recovered": {
            name: np.mean([item[name] for item in recovered_kernels], axis=0)
            for name in true_kernels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--grid", default="hold_grid", help="Grid module.")
    parser.add_argument("--design", default="fine\nkernel", help="Name in its DESIGNS.")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    grid = importlib.import_module(f"thesis.ephys.analyses.glm.{args.grid}")
    settings = grid.DESIGNS[args.design]
    stats = {"grid": args.grid, "under_test": args.design, "in_use": grid.IN_USE}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(SESSIONS),
            len(MEASURES),
            figsize=(7.4, 2.6 * len(SESSIONS)),
            layout="constrained",
        )
        for row_axes, session in zip(axes, SESSIONS):
            model = cli.PoissonGLM(root=args.out / session)
            subject, _, name = session.partition("_")
            work = args.out / f"{session}_parameter_recovery_work"
            work.mkdir(exist_ok=True)
            other = work / "common_design_me.npy"
            saved = {key: getattr(dmat, key) for key in settings}
            for key, value in settings.items():
                setattr(dmat, key, value)
            try:
                if not other.exists():
                    dmat.prepare_common_design(model.windows, model.video, other)
            finally:
                for key, value in saved.items():
                    setattr(dmat, key, value)
            final = load_unit_records(
                model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
            )
            table = fetch_unit_table(
                subject,
                name,
                unit_criteria_id=1,
                stability_param_id=0,
                include_metrics=False,
            )
            spikes = dict(zip(table["unit_id"].astype(int), table["spike_times_s"]))
            jobs = [
                {
                    "unit_id": int(item["unit_id"]),
                    "record": item,
                    "alpha": float(item["alpha"]),
                    "spike_times": np.asarray(
                        spikes[int(item["unit_id"])], dtype=float
                    ),
                }
                for item in final
            ]
            with Pool(
                args.workers, _load, (model.windows, model.design, other, settings)
            ) as pool:
                units = pool.map(_unit, jobs)
            values = np.asarray(
                [unit["values"] for unit in units]
            )  # (unit, 3, measure)
            regressors = units[0]["regressors"]
            kernel_agreement = np.asarray([unit["agreement"] for unit in units])
            kernel_size = np.asarray([unit["size"] for unit in units])
            np.savez(
                model.root / "parameter_recovery.npz",
                regressors=np.asarray(regressors),
                agreement=kernel_agreement,
                size=kernel_size,
                unit_ids=np.asarray([job["unit_id"] for job in jobs]),
                **{
                    f"{kind}__{name}": np.asarray([unit[kind][name] for unit in units])
                    for kind in ("true", "recovered")
                    for name in regressors
                },
            )
            shutil.rmtree(work)
            truth, in_use, under_test = values[:, 0], values[:, 1], values[:, 2]
            errors = np.abs(in_use - truth), np.abs(under_test - truth)
            entry = {"units": len(values), "simulations": SIMULATIONS}
            entry["recovery of each regressor of the design in use"] = {
                name: {
                    "median_agreement": float(np.median(kernel_agreement[:, index])),
                    "median_size_recovered_over_true": float(
                        np.median(kernel_size[:, index])
                    ),
                }
                for index, name in enumerate(regressors)
            }
            for index, measure in enumerate(MEASURES):
                change = errors[1][:, index] - errors[0][:, index]
                entry[measure] = {
                    "median_error_in_use": float(np.median(errors[0][:, index])),
                    "median_error_under_test": float(np.median(errors[1][:, index])),
                    "median_change_of_error": float(np.median(change)),
                    "units_with_larger_error": int(np.sum(change > 0)),
                    "wilcoxon_p": float(wilcoxon(change).pvalue),
                    "units_outside_the_axis_limit": int(
                        np.sum(
                            (errors[0][:, index] > AXIS_LIMIT)
                            | (errors[1][:, index] > AXIS_LIMIT)
                        )
                    ),
                }
                ax = row_axes[index]
                ax.scatter(
                    np.minimum(errors[0][:, index], AXIS_LIMIT),
                    np.minimum(errors[1][:, index], AXIS_LIMIT),
                    s=5,
                    color=COLORS[index],
                    alpha=0.6,
                    lw=0,
                )
                ax.plot(
                    (0, AXIS_LIMIT),
                    (0, AXIS_LIMIT),
                    color=REFERENCE_COLOR,
                    ls="--",
                    lw=0.8,
                )
                ax.set(xlim=(0, AXIS_LIMIT), ylim=(0, AXIS_LIMIT), aspect="equal")
                ax.set_title(measure, color=COLORS[index], fontsize=8)
                ax.set_xlabel(f"error, {grid.IN_USE} (in use)")
                ax.text(
                    0.04,
                    0.96,
                    f"larger error in\n{entry[measure]['units_with_larger_error']} of "
                    f"{len(values)} units",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=6.5,
                )
            level = MEASURES.index("fixation level")
            large = np.abs(truth[:, level]) > leak_test.MIN_LEVEL
            entry["fixation level, recovered over true (median)"] = {
                "in_use": float(np.median(in_use[large, level] / truth[large, level])),
                "under_test": float(
                    np.median(under_test[large, level] / truth[large, level])
                ),
            }
            row_axes[0].set_ylabel(f"{subject}\nerror, {args.design}")
            stats[session] = entry
        label_panels(axes[0], x=-0.28, y=1.12)
        print(save_figure(figure, args.out / "parameter_recovery"))
    (args.out / "parameter_recovery.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
