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
- Figure: a, the kernels of the example unit of GRB006 from the two halves.
  b, the agreement of each kernel regressor in each unit, the units of both
  mice pooled; the black marker is the median of the units that need the
  regressor with its hierarchical-bootstrap 95% interval, and the open
  circles the median of each mouse. A point below `DISPLAY_FLOOR` is drawn at the floor; it is in the
  statistics, and the count is in the output. c, the response of each unit in
  one half against the other half, the units of both mice pooled.

Units (`--units`), trials, alignment (first flash), baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit, and each measure is
within one animal (`SESSIONS`); the units of the two animals are not pooled.
With `--refit`, the script fits the two halves of each session and writes
`kernel_reliability.npz` and `.json` in the session directory. It then draws
`kernel_reliability.svg` from those files and writes `kernel_reliability.json`
(by session) in `--out`.

    uv run python -m thesis.ephys.analyses.glm.kernel_reliability [--refit]
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
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.inclusion import ALPHA
from thesis.ephys.analyses.glm.main_figure import REGRESSORS
from thesis.ephys.analyses.glm.results import EXAMPLE_UNIT_ID, fitted_kernels
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    save_figure,
)

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


def fit_session(subject: str, session: str, unit_set: str, workers: int) -> None:
    """Fit each unit of one session on the two halves; write the `.npz` and `.json`."""
    model = cli.PoissonGLM(subject, session, Path(f"figures/glm/{subject}_{session}"))
    checkpoints = model.fit_dir(unit_set) / "checkpoints"
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_final.json")
    }
    unique = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints / "unique", "unit_*.json")
    }
    table = fetch_unit_table(
        subject,
        session,
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
    with Pool(workers, _load, (model.windows, model.design)) as pool:
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
    stats = {"subject": subject, "session": session, "units": len(unit_ids)}
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
    (model.root / "kernel_reliability.json").write_text(json.dumps(stats, indent=1))
    np.savez(model.root / "kernel_reliability.npz", **arrays)


def kernel_times() -> dict[str, np.ndarray]:
    """Return the time points (ms) of each kernel regressor."""
    times = {
        name: 1000 * basis.basis_time
        for name, basis in dmat.task_temporal_bases().items()
    }
    times["history"] = 1 + 1000 * dmat.spike_history_basis().basis_time
    return times


def draw_responses(axes, saved: list[dict], names: list, subjects: list) -> list:
    """Draw the response of each unit in the two halves, both mice in each axis.

    Returns, for each mouse, the number of its units outside the axis limits.
    """
    outside = [{} for _ in saved]
    for ax, name in zip(axes, names):
        label, color = REGRESSORS[name]
        values = [arrays[f"response__{name}"] for arrays in saved]
        limits = np.percentile(np.concatenate([v.ravel() for v in values]), [1, 99])
        for k, (arrays, (first, second)) in enumerate(zip(saved, values, strict=True)):
            needs = arrays[f"needs__{name}"]
            for members, shade in ((~needs, "0.75"), (needs, color)):
                ax.scatter(
                    first[members],
                    second[members],
                    s=6,
                    color=shade,
                    lw=0,
                )
            outside[k][name] = int(
                np.sum(
                    (np.minimum(first, second) < limits[0])
                    | (np.maximum(first, second) > limits[1])
                )
            )
        ax.plot(limits, limits, color=REFERENCE_COLOR, lw=0.8, ls="--")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal")
        ax.set_title(label, color=color, fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("response, odd trials\n(Δ log rate)")
    return outside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="fit the two halves of each session again before the figure",
    )
    args = parser.parse_args()
    if args.refit:
        for name in SESSIONS:
            subject, session = name.split("_", 1)
            fit_session(subject, session, args.units, args.workers)
    saved = []
    for name in SESSIONS:
        path = args.out / name / "kernel_reliability.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run with --refit first.")
        with np.load(path) as values:
            saved.append({key: values[key] for key in values.files})
    stats = {
        name: json.loads((args.out / name / "kernel_reliability.json").read_text())
        for name in SESSIONS
    }
    subjects = [name.split("_")[0] for name in SESSIONS]
    times = kernel_times()
    # Regressors in the order of the main figure.
    names = [n for n in REGRESSORS if f"response__{n}" in saved[0]]
    kernels = [n for n in names if f"kernel__{n}" in saved[0]]
    arrays = saved[0]
    unit_ids = arrays["unit_ids"].tolist()
    example = unit_ids.index(EXAMPLE_UNIT_ID) if EXAMPLE_UNIT_ID in unit_ids else 0
    rng = np.random.default_rng(0)
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(11, 7.0), layout="constrained")
        rows = figure.subfigures(3, 1, height_ratios=(1, 1.05, 0.85))
        top = rows[0].subplots(1, len(kernels))
        for ax, name in zip(top, kernels):
            label, color = REGRESSORS[name]
            for half, style in enumerate(("-", "--")):
                ax.plot(
                    times[name],
                    arrays[f"kernel__{name}"][half, example],
                    color=color,
                    ls=style,
                )
            ax.axhline(0, color=REFERENCE_COLOR, lw=0.8, ls="--")
            ax.set_title(label, color=color, fontsize=8)
            ax.set_xlabel("time (ms)")
        top[0].set_ylabel(f"{subjects[0]} example unit\nΔ log rate")
        top[0].text(
            0.98,
            0.08,
            "even trials\nodd trials (dashed)",
            transform=top[0].transAxes,
            ha="right",
            fontsize=6.5,
        )
        middle = rows[1].subplots()
        stats["pooled agreement, units that need the regressor"] = {}
        for position, name in enumerate(kernels):
            label, color = REGRESSORS[name]
            agree = [values_of[f"agreement__{name}"] for values_of in saved]
            needs = [values_of[f"needs__{name}"] for values_of in saved]
            pooled, pooled_needs = np.concatenate(agree), np.concatenate(needs)
            x = position + rng.uniform(-0.15, 0.15, len(pooled))
            shown = np.maximum(pooled, DISPLAY_FLOOR)
            middle.scatter(
                x[~pooled_needs], shown[~pooled_needs], s=5, color="0.75", lw=0
            )
            middle.scatter(
                x[pooled_needs], shown[pooled_needs], s=5, color=color, alpha=0.6, lw=0
            )
            # The median of the units that need it: pooled, with the
            # hierarchical-bootstrap 95% interval, and of each mouse (bars).
            summary = hierarchical_interval(
                [a[n] for a, n in zip(agree, needs, strict=True)]
            )
            low, high = summary["interval_95"]
            middle.errorbar(
                position + 0.28,
                summary["median"],
                yerr=[[summary["median"] - low], [high - summary["median"]]],
                fmt="o",
                ms=4,
                color="black",
                capsize=2,
                zorder=4,
            )
            for median in summary["median_of_each_mouse"]:
                middle.scatter(
                    position + 0.4,
                    median,
                    s=14,
                    facecolor="white",
                    edgecolor="black",
                    lw=0.8,
                    zorder=4,
                )
            stats["pooled agreement, units that need the regressor"][name] = summary
        middle.axhline(0, color=REFERENCE_COLOR, lw=0.8, ls="--")
        middle.set_xticks(range(len(kernels)), [REGRESSORS[n][0] for n in kernels])
        for tick, name in zip(middle.get_xticklabels(), kernels):
            tick.set_color(REGRESSORS[name][1])
        middle.set_ylabel("agreement of the two halves\n(1 = equal kernels)")
        middle.set_title(
            f"{sum(len(v['unit_ids']) for v in saved)} units, {len(saved)} mice."
            " Color: units that need the regressor; gray: other units. Black: the"
            " median of the units that need it, with its 95% interval; open circles:"
            " the median of each mouse",
            fontsize=7,
            loc="left",
        )
        axes = rows[2].subplots(1, len(names))
        for k, (counts, name) in enumerate(
            zip(draw_responses(axes, saved, names, subjects), SESSIONS, strict=True)
        ):
            for regressor, count in counts.items():
                stats[name][regressor]["units_outside_the_axis_limits"] = count
        rows[2].supxlabel(
            "response, even trials (Δ log rate); units of both mice", fontsize=8
        )
        label_panels([top[0], middle], x=-0.02, y=1.08)
        label_panels([None, None, axes[0]], x=-1.05, y=1.2)
        print(save_figure(figure, args.out / "kernel_reliability"))
    (args.out / "kernel_reliability.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
