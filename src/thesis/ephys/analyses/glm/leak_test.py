"""Test how much of the flash × move effect the GLM makes with no such effect in the data.

Question: the model finds a flash response that differs between the fixation
(still) and the time after the center exit (move). The rate level also
changes at the exit (the fixation kernel). A sum of flash kernels over a
flash train is close to a level, so the fit can give a part of a pure level
change to the flash × move kernels. How large is that part in the real design
of each unit?

Comparison: a parametric null, one for each unit. Spikes are simulated from
the final refit of the unit with the weights of the flash × move and the
flash × move × gap kernels set to 0, so the simulated unit has the fixation
effect and every other term of the real unit, and no change of the flash
response with the period. The full model is then fit to the simulated spikes
with the real design and the penalty of the unit. The flash × move effect of
that fit is what the model makes from nothing (the leak). The comparison is
paired: each unit has its real effect and its leak.

- Simulation: the rate of the final refit in each 1-ms bin of each trial,
  with the spike history of the simulated spikes (in the window of the trial
  only; the real history also has the spikes before the window). The rate is
  limited to `MAX_COUNT` per bin. `SIMULATIONS` simulations for each unit
  (`SEED`); the leak of a unit is the mean over them.
- Effect: the response of the flash kernel (its mean from `RESPONSE_MS`) for a
  flash in move minus a flash when still, at each gap class
  (`results.flash_kernel_at`), as in `flash_kernels.py`.
- Also given: the fixation level of the refit (the mean of the fixation
  kernel from `LEVEL_MS`) against the level that was simulated, to show how
  much of the level the fit keeps in the fixation kernel.
- Statistics: two-sided Wilcoxon signed-rank test over units of the real
  effect minus the leak, and the Spearman correlation of the leak with the
  fixation level of the unit.
- Figure: one row for each session. a, the real effect against the leak at
  the 80+ ms gap, one point for each unit; the error bar of the leak is the SD
  over the simulations. b, the same at the 40 ms gap. c, the leak against the
  fixation level of the unit. A point outside the axis limits is not drawn;
  it is in the statistics, and the count is in the output.

Units (`--units`), trials, alignment, and exclusions are those of `fit.py`.
The sampling unit is the unit. The script writes `leak_test.svg` and
`leak_test.json` in `figures/glm/`, and `leak_test.npz` in each session
directory.

    uv run python -m thesis.ephys.analyses.glm.leak_test
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
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    fit_poisson_at_alpha,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_by_period import SESSIONS
from thesis.ephys.analyses.glm.results import (
    fitted_kernels,
    flash_kernel_at,
    flash_kernel_levels,
)
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
MAX_COUNT = 0.5
RESPONSE_MS = (40, 100)
LEVEL_MS = (200, 450)
MIN_LEVEL = 0.1
GAPS_MS = {"80+ ms gap": 100, "40 ms gap": 40}
REMOVED = (dmat.FLASH_AFTER_EXIT, dmat.FLASH_MOVE_GAP)
AXIS_LIMITS = (-1.0, 1.5)
_SHARED: dict = {}


def _load(windows: Path, design: Path) -> None:
    prepared = dmat._load_windows(windows)
    _SHARED.update(
        prepared=prepared,
        valid=dmat._valid_bin_mask(prepared),
        common=np.load(design, mmap_mode="r", allow_pickle=False),
        metadata=json.loads(design.with_suffix(".json").read_text()),
        levels=flash_kernel_levels(dmat._load_selected_trials(prepared)),
    )


def measures(result: dict, history_scale: np.ndarray) -> np.ndarray:
    """Return move minus still at each gap, and the fixation level, of one fit."""
    kernels = fitted_kernels(result, _SHARED["metadata"], history_scale)
    lags = kernels[dmat.FLASH_AFTER_EXIT][0]
    window = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    out = [
        (
            flash_kernel_at(kernels, _SHARED["levels"], gap, True)
            - flash_kernel_at(kernels, _SHARED["levels"], gap, False)
        )[window].mean()
        for gap in GAPS_MS.values()
    ]
    time, hold = kernels[dmat.HOLD]
    out.append(hold[(time >= LEVEL_MS[0]) & (time < LEVEL_MS[1])].mean())
    return np.asarray(out)


def simulate(
    record: dict,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the spikes of one unit from a set of weights of its model.

    `coefficients` are weights for the design of the fit (`_SHARED["common"]`
    and the spike history); `mean` and `scale` are those of the history
    columns of the real spikes. The rate of a bin uses the history of the
    simulated spikes in the window of the trial, and is limited to
    `MAX_COUNT`. Returns the counts (trial × bin), the scaled history columns
    of the simulated spikes, and their scale.
    """
    prepared, valid, common, metadata = (
        _SHARED[key] for key in ("prepared", "valid", "common", "metadata")
    )
    alignments, centers = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(centers))
    width = int(metadata["base_columns"]) + int(record["components"])
    drive = (
        float(record["intercept"])
        + np.asarray(common[:, :width]) @ coefficients[:width]
    ).reshape(grid)
    # The history of a bin: the basis at the lags of 1 to 100 bins before it.
    basis = dmat.spike_history_basis().basis
    weights = coefficients[width:] / scale
    offset = -(mean / scale) @ coefficients[width:]
    counts = np.zeros(grid, dtype=int)
    for t in range(grid[1]):
        past = counts[:, max(0, t - len(basis)) : t][:, ::-1]
        history = (past @ basis[: past.shape[1]]) @ weights + offset
        rate = np.minimum(np.exp(drive[:, t] + history), MAX_COUNT)
        counts[:, t] = rng.poisson(rate)
    bin_times = alignments[:, None] + centers[None, :]
    spikes = np.sort(np.repeat(bin_times.ravel(), counts.ravel()))
    scaled, _, simulated_scale = dmat.training_zscore(
        dmat.build_unit_history(alignments, spikes), valid
    )
    return counts, scaled, simulated_scale


def _unit(job: dict) -> dict:
    """Simulate one unit with no flash × move effect and fit the full model."""
    prepared, valid, common, metadata = (
        _SHARED[key] for key in ("prepared", "valid", "common", "metadata")
    )
    alignments = prepared["alignments"]
    record = job["record"]
    width = int(metadata["base_columns"]) + int(record["components"])
    coefficients = np.asarray(record["coefficients"], dtype=float)
    raw = dmat.build_unit_history(alignments, job["spike_times"])
    _, mean, scale = dmat.training_zscore(raw, valid)
    real = measures(record, scale)
    null = coefficients.copy()
    groups = block_slices(metadata, int(record["components"]))
    for name in REMOVED:
        null[groups[name]] = 0
    rng = np.random.default_rng(SEED + job["unit_id"])
    fits = []
    for _ in range(SIMULATIONS):
        counts, scaled, simulated_scale = simulate(record, null, mean, scale, rng)
        design = dmat._design_with_history(common, scaled, width, valid)
        model = fit_poisson_at_alpha(design, counts.ravel()[valid], job["alpha"])
        result = {"components": record["components"], "coefficients": model.coef_}
        fits.append(measures(result, simulated_scale))
    null_level = measures({**record, "coefficients": null}, scale)[2]
    return {
        "unit_id": job["unit_id"],
        "real": real,
        "leak": np.asarray(fits),
        "simulated_level": null_level,
        "spikes_simulated_over_real": float(
            counts.ravel()[valid].sum()
            / max(build_unit_counts(alignments, job["spike_times"])[valid].sum(), 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stats = {}
    visual, task = CONCEPT_COLORS["visual"], CONCEPT_COLORS["task"]
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(SESSIONS), 3, figsize=(8.6, 2.9 * len(SESSIONS)), layout="constrained"
        )
        for row_axes, session in zip(axes, SESSIONS):
            model = cli.PoissonGLM(root=args.out / session)
            subject, _, name = session.partition("_")
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
            with Pool(args.workers, _load, (model.windows, model.design)) as pool:
                units = pool.map(_unit, jobs)
            real = np.asarray([unit["real"] for unit in units])  # (unit, measure)
            leaks = np.asarray([unit["leak"] for unit in units])
            leak, spread = leaks.mean(1), leaks.std(1)
            simulated_level = np.asarray([unit["simulated_level"] for unit in units])
            entry = {"units": len(units), "simulations": SIMULATIONS}
            for index, gap in enumerate(GAPS_MS):
                rest = real[:, index] - leak[:, index]
                both = np.column_stack((real[:, index], leak[:, index]))
                entry[gap] = {
                    "median_real_move_minus_still": float(np.median(real[:, index])),
                    "median_leak": float(np.median(leak[:, index])),
                    "units_with_leak_above_0": int(np.sum(leak[:, index] > 0)),
                    "leak_wilcoxon_p": float(wilcoxon(leak[:, index]).pvalue),
                    "median_real_minus_leak": float(np.median(rest)),
                    "units_with_real_above_leak": int(np.sum(rest > 0)),
                    "real_minus_leak_wilcoxon_p": float(wilcoxon(rest).pvalue),
                    "spearman_leak_and_fixation_level": float(
                        spearmanr(leak[:, index], simulated_level)[0]
                    ),
                    "units_outside_the_axis_limits": int(
                        np.sum(
                            (both < AXIS_LIMITS[0]).any(1)
                            | (both > AXIS_LIMITS[1]).any(1)
                        )
                    ),
                }
            entry["fixation level"] = {
                "median_simulated": float(np.median(simulated_level)),
                "median_refit": float(np.median(leak[:, 2])),
                # For the units with a level: the ratio has no meaning near 0.
                "median_refit_over_simulated": float(
                    np.median(
                        (leak[:, 2] / simulated_level)[
                            np.abs(simulated_level) > MIN_LEVEL
                        ]
                    )
                ),
            }
            entry["median_spikes_simulated_over_real"] = float(
                np.median([unit["spikes_simulated_over_real"] for unit in units])
            )
            stats[session] = entry
            np.savez(
                args.out / session / "leak_test.npz",
                unit_ids=np.asarray([unit["unit_id"] for unit in units]),
                measures=np.asarray([*GAPS_MS, "fixation level"]),
                real=real,
                leak=leaks,
                simulated_level=simulated_level,
            )
            style = {"fmt": "o", "ms": 2.5, "elinewidth": 0.4, "alpha": 0.8}
            for ax, (index, gap) in zip(row_axes, enumerate(GAPS_MS)):
                ax.errorbar(
                    leak[:, index],
                    real[:, index],
                    xerr=spread[:, index],
                    color=visual,
                    **style,
                )
                ax.plot(
                    AXIS_LIMITS, AXIS_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--"
                )
                ax.axhline(0, color=REFERENCE_COLOR, lw=0.6)
                ax.axvline(0, color=REFERENCE_COLOR, lw=0.6)
                ax.set(xlim=AXIS_LIMITS, ylim=AXIS_LIMITS, aspect="equal")
                ax.set_xlabel(f"leak, {gap}\n(no effect simulated)")
                ax.set_ylabel(
                    (f"{subject}\n" if index == 0 else "")
                    + "real move minus still\n(Δ log rate)"
                )
            level_axis = row_axes[2]
            level_axis.errorbar(
                simulated_level, leak[:, 0], yerr=spread[:, 0], color=task, **style
            )
            level_axis.axhline(0, color=REFERENCE_COLOR, lw=0.6)
            level_axis.axvline(0, color=REFERENCE_COLOR, lw=0.6)
            level_axis.set(xlim=(-1.5, 0.75), ylim=AXIS_LIMITS)
            level_axis.set_xlabel("fixation level of the unit\n(Δ log rate)")
            level_axis.set_ylabel("leak, 80+ ms gap")
        label_panels(axes[0], x=-0.3, y=1.05)
        print(save_figure(figure, args.out / "leak_test"))
    (args.out / "leak_test.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
