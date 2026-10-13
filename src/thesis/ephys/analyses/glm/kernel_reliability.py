"""Test whether the kernels of the V1 GLM are the same in two halves of the trials.

Question: can the weights of each regressor be trusted? Regressors that are
correlated can give weights that change from one set of trials to another,
while the prediction stays the same. If the kernels are stable, a fit on one
half of the trials gives the same kernels as a fit on the other half.

Comparison: for each unit, the full model is fit two times, on the trials with
an even and with an odd position in the session. The two halves share no
trial, and each one covers the full session. The comparison is paired: each
unit has one kernel of each regressor from each half.

- Fit: the design, the scaling of the common columns, and the penalty of the
  unit are those of the final refit (`fit.py`); the penalty is not selected
  again. The spike-history columns are scaled on the rows of the half.
- Kernel of a regressor: `results.fitted_kernels` (Δ log rate). A regressor
  with one column (initiation, choice, previous choice) has one weight.
  Session drift is not compared.
- Reliability of a kernel in a unit: the Pearson correlation of the two
  kernels over the time points (`shape_r`), and the size of their difference
  against their size, `1 - |a - b|² / (|a|² + |b|²)` (`agreement`: 1 for equal
  kernels, 0 for unrelated kernels, below 0 for opposite kernels). A
  correlation alone is high for two kernels with the same shape and different
  sizes; the agreement is not.
- Reliability over units: the Spearman correlation over units of the response
  in the two halves. The response is the mean of the kernel over its time
  points for the kernels (`RESPONSE_MS` for the flash kernels), and the weight
  for a regressor with one column.
- Each measure is given for all units and for the units that need the
  regressor (unique ΔD² above 0 over the folds, the test of `inclusion.py`).
- Figure: a, the kernels of the example unit from the two halves. b, the
  agreement of each kernel regressor in each unit (points; the line is the
  median). A point below `DISPLAY_FLOOR` is drawn at the floor; it is in the
  statistics, and the count is in the output. c, the response of each unit in
  one half against the other half.

Units (`--units`), trials, alignment (first flash), baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit. The script writes
`kernel_reliability.svg`, `.json`, and `.npz` in the session directory.

    uv run python -m thesis.ephys.analyses.glm.kernel_reliability
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, ttest_1samp

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    fit_poisson_at_alpha,
    load_unit_records,
)
from thesis.ephys.analyses.glm.inclusion import ALPHA
from thesis.ephys.analyses.glm.main_figure import REGRESSORS
from thesis.ephys.analyses.glm.results import EXAMPLE_UNIT_ID, fitted_kernels
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, label_panels, save_figure

RESPONSE_MS = (40, 100)
DISPLAY_FLOOR = -0.5
_SHARED: dict = {}


def _load(windows: Path, design: Path) -> None:
    _SHARED["prepared"] = dmat._load_windows(windows)
    _SHARED["common"] = np.load(design, mmap_mode="r", allow_pickle=False)
    _SHARED["metadata"] = json.loads(design.with_suffix(".json").read_text())


def _fit_halves(job: dict) -> dict:
    """Return the kernels of one unit from the even and the odd trials."""
    prepared, common, metadata = (
        _SHARED[key] for key in ("prepared", "common", "metadata")
    )
    alignments = prepared["alignments"]
    counts = build_unit_counts(alignments, job["spike_times"])
    raw_history = dmat.build_unit_history(alignments, job["spike_times"])
    columns = int(metadata["base_columns"]) + job["components"]
    bins = len(prepared["relative_centers"])
    odd = np.repeat(np.arange(len(alignments)) % 2 == 1, bins)
    out = {}
    for half, members in enumerate((~odd, odd)):
        rows = dmat._valid_bin_mask(prepared) & members
        history, _, scale = dmat.training_zscore(raw_history, rows)
        design = dmat._design_with_history(common, history, columns, rows)
        model = fit_poisson_at_alpha(design, counts[rows], job["alpha"])
        result = {"components": job["components"], "coefficients": model.coef_}
        for name, (_, values) in fitted_kernels(result, metadata, scale).items():
            out[f"{name}__{half}"] = values
    return out


def agreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return 1 - |a - b|² / (|a|² + |b|²) for each row."""
    return 1 - ((first - second) ** 2).sum(1) / (first**2 + second**2).sum(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    model = cli.PoissonGLM(
        args.subject, args.session, Path(f"figures/glm/{args.subject}_{args.session}")
    )
    checkpoints = model.fit_dir(args.units) / "checkpoints"
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_final.json")
    }
    unique = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints / "unique", "unit_*.json")
    }
    table = fetch_unit_table(
        args.subject,
        args.session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    spikes = dict(zip(table["unit_id"].astype(int), table["spike_times_s"]))
    unit_ids = sorted(final)
    jobs = [
        {
            "spike_times": np.asarray(spikes[unit], dtype=float),
            "components": int(final[unit]["components"]),
            "alpha": float(final[unit]["alpha"]),
        }
        for unit in unit_ids
    ]
    with Pool(args.workers, _load, (model.windows, model.design)) as pool:
        fits = pool.map(_fit_halves, jobs)
    bases = dmat.task_temporal_bases()
    history_ms = 1 + 1000 * dmat.spike_history_basis().basis_time
    times = {name: 1000 * basis.basis_time for name, basis in bases.items()}
    times["history"] = history_ms
    # Session drift is not compared: its columns are set by the trials of the fit.
    names = [
        key[:-3] for key in fits[0] if key.endswith("__0") and key[:-3] != dmat.DRIFT
    ]
    halves = {
        name: [np.asarray([fit[f"{name}__{half}"] for fit in fits]) for half in (0, 1)]
        for name in names
    }
    stats = {"subject": args.subject, "session": args.session, "units": len(unit_ids)}
    arrays = {"unit_ids": np.asarray(unit_ids)}
    for name in names:
        first, second = halves[name]
        needs = np.asarray(
            [
                ttest_1samp(
                    unique[unit]["groups"][name]["folds"]["unique"],
                    0,
                    alternative="greater",
                ).pvalue
                < ALPHA
                for unit in unit_ids
            ]
        )
        kernel = name in times
        if kernel and name.startswith("flash"):
            window = (times[name] >= RESPONSE_MS[0]) & (times[name] <= RESPONSE_MS[1])
        else:
            window = slice(None)
        # A regressor with one column is a step of two equal values.
        response = [values[:, window].mean(1) for values in (first, second)]
        entry = {"units_that_need_it": int(needs.sum())}
        for label, members in (
            ("all units", slice(None)),
            ("units that need it", needs),
        ):
            group = {
                "response_spearman_over_units": float(
                    spearmanr(response[0][members], response[1][members])[0]
                )
            }
            if kernel:
                agree = agreement(first[members], second[members])
                shape = [
                    np.corrcoef(a, b)[0, 1]
                    for a, b in zip(first[members], second[members])
                ]
                group |= {
                    "median_agreement": float(np.median(agree)),
                    "fraction_agreement_above_0.5": float(np.mean(agree > 0.5)),
                    "median_shape_r": float(np.median(shape)),
                }
            entry[label] = group
        if kernel:
            arrays[f"agreement__{name}"] = agreement(first, second)
            arrays[f"kernel__{name}"] = np.asarray((first, second))
            entry["units_below_the_display_floor"] = int(
                np.sum(arrays[f"agreement__{name}"] < DISPLAY_FLOOR)
            )
        arrays[f"response__{name}"] = np.asarray(response)
        arrays[f"needs__{name}"] = needs
        stats[name] = entry
    kernels = [name for name in names if name in times]
    example = unit_ids.index(EXAMPLE_UNIT_ID) if EXAMPLE_UNIT_ID in unit_ids else 0
    stats["example_unit"] = unit_ids[example]
    rng = np.random.default_rng(0)
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(11, 6.6), layout="constrained")
        rows = figure.subfigures(3, 1, height_ratios=(1, 1.05, 0.8))
        top = rows[0].subplots(1, len(kernels))
        for ax, name in zip(top, kernels):
            label, color = REGRESSORS[name]
            for half, style in enumerate(("-", "--")):
                ax.plot(times[name], halves[name][half][example], color=color, ls=style)
            ax.axhline(0, color=REFERENCE_COLOR, lw=0.8, ls="--")
            ax.set_title(label, color=color, fontsize=8)
            ax.set_xlabel("time (ms)")
        top[0].set_ylabel("Δ log rate")
        top[-1].text(
            0.98,
            0.3,
            "even trials\nodd trials (dashed)",
            transform=top[-1].transAxes,
            ha="right",
            fontsize=6.5,
        )
        middle = rows[1].subplots()
        for position, name in enumerate(kernels):
            label, color = REGRESSORS[name]
            values = np.maximum(arrays[f"agreement__{name}"], DISPLAY_FLOOR)
            needs = arrays[f"needs__{name}"]
            jitter = rng.uniform(-0.25, 0.25, len(values))
            middle.scatter(
                position + jitter[~needs], values[~needs], s=6, color="0.75", lw=0
            )
            middle.scatter(
                position + jitter[needs], values[needs], s=6, color=color, lw=0
            )
            middle.hlines(
                np.median(arrays[f"agreement__{name}"][needs]),
                position - 0.35,
                position + 0.35,
                color="black",
                lw=1.5,
            )
        middle.axhline(0, color=REFERENCE_COLOR, lw=0.8, ls="--")
        middle.set_xticks(range(len(kernels)), [REGRESSORS[n][0] for n in kernels])
        for tick, name in zip(middle.get_xticklabels(), kernels):
            tick.set_color(REGRESSORS[name][1])
        middle.set_ylabel("agreement of the two halves\n(1 = equal kernels)")
        middle.text(
            1.0,
            0.02,
            "color: units that need the regressor; gray: other units\n"
            "line: median of the units that need it",
            transform=middle.transAxes,
            ha="right",
            fontsize=6.5,
        )
        bottom = rows[2].subplots(1, len(names))
        for ax, name in zip(bottom, names):
            label, color = REGRESSORS[name]
            first, second = arrays[f"response__{name}"]
            needs = arrays[f"needs__{name}"]
            ax.scatter(first[~needs], second[~needs], s=5, color="0.75", lw=0)
            ax.scatter(first[needs], second[needs], s=5, color=color, lw=0)
            limits = np.percentile(np.concatenate((first, second)), [1, 99])
            ax.plot(limits, limits, color=REFERENCE_COLOR, lw=0.8, ls="--")
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal")
            ax.set_title(label, color=color, fontsize=7)
            ax.tick_params(labelsize=6)
            stats[name]["units_outside_the_axis_limits"] = int(
                np.sum(
                    (np.minimum(first, second) < limits[0])
                    | (np.maximum(first, second) > limits[1])
                )
            )
        bottom[0].set_ylabel("response, odd trials\n(Δ log rate)")
        rows[2].supxlabel("response, even trials (Δ log rate)", fontsize=9)
        label_panels([top[0], middle], x=-0.02, y=1.08)
        label_panels([None, None, bottom[0]], x=-0.75, y=1.2)
        print(save_figure(figure, model.root / "kernel_reliability"))
    (model.root / "kernel_reliability.json").write_text(json.dumps(stats, indent=1))
    np.savez(model.root / "kernel_reliability.npz", **arrays)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
