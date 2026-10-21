"""Test how much of the flash × after exit effect the GLM makes with no such effect in the data.

Question: the model finds a flash response that differs between the fixation
(in port) and the time after the center exit (after exit). The rate level also
changes at the exit (the fixation kernel). A sum of flash kernels over a
flash train is close to a level, so the fit can give a part of a pure level
change to the flash × after exit kernels. How large is that part in the real design
of each unit?

Comparison: a parametric null, one for each unit. Spikes are simulated from
the final refit of the unit with the raw flash × after exit regressors set
to 0. Their weights are zeroed and the intercept is adjusted for column
centering, so the in-port rate stays fixed. The simulated unit has the fixation
effect and every other term of the real unit, and no change of the flash
response with the period. The full model is then fit to the simulated spikes
with the real design and the penalty of the unit. The flash × after exit effect of
that fit is what the model makes from nothing (the leak). The comparison is
paired: each unit has its real effect and its leak.

- Simulation: the rate of the final refit in each 1-ms bin of each trial,
  with the spike history of the simulated spikes (in the window of the trial
  only; the real history also has the spikes before the window). The rate is
  limited to `MAX_COUNT` per bin. `SIMULATIONS` simulations for each unit
  (`SEED`); the leak of a unit is the mean over them.
- Effect, in the matched measure of every flash comparison (`peaks.py`): the
  part of the after-exit-minus-in-port peak that the flash × after exit terms carry, on the
  paired flashes of `peaks.paired_flashes`: the peak change of the model rate
  minus the same with the raw flash × after exit regressors set to 0
  (spikes/s), keeping the in-port baseline fixed. The real
  part uses the real fit; the leak uses each fit to simulated spikes. `model_trust.py`
  reads it for the excited units of StimulusResponse. The old
  kernel measure (the mean of the kernel difference from `RESPONSE_MS`, Δ log
  rate) stays in the output.
- Also given: the fixation level of the refit (the mean of the fixation
  kernel from `LEVEL_MS`) against the level that was simulated, to show how
  much of the level the fit keeps in the fixation kernel.
- Statistics: two-sided Wilcoxon signed-rank test over units of the leak
  against 0, and the Spearman correlation of the leak with the fixation level
  of the unit. The test of the real effect minus the leak, on the excited
  units, and the figure (panels g to i) are in `model_trust.py`.

Units (`--units`), trials, alignment, and exclusions are those of `fit.py`.
The sampling unit is the unit. The script simulates and refits each unit and
writes `leak_test.json` in `--out` and `leak_test.npz` in each session
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
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, peaks
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    fit_poisson_at_alpha,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import LEVEL_MS as HOLD_LEVEL_MS
from thesis.ephys.analyses.glm.peaks import paired_flashes
from thesis.ephys.analyses.glm.results import (
    fitted_kernels,
    flash_kernel_at,
    raw_contribution,
    zero_raw_columns,
)
from thesis.ephys.units import unit_spikes

SIMULATIONS = 3
SEED = 0
MAX_COUNT = 0.5
RESPONSE_MS = (40, 100)
# The window of the fixation level: the one of the fixation figure.
LEVEL_MS = HOLD_LEVEL_MS
MIN_LEVEL = 0.1
GAPS_MS = {"80+ ms gap": 100, "40 ms gap": 40}
REMOVED = (dmat.FLASH_AFTER_EXIT,)
_SHARED: dict = {}


def _load(windows: Path, design: Path) -> None:
    prepared = dmat._load_windows(windows)
    _SHARED.update(
        prepared=prepared,
        valid=dmat._valid_bin_mask(prepared),
        common=np.load(design, mmap_mode="r", allow_pickle=False),
        metadata=json.loads(design.with_suffix(".json").read_text()),
    )
    # The bins of the still and the move windows of `peaks.paired_flashes`, for the
    # matched peak (`peaks.py`).
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    rows, still, move = paired_flashes(dmat._load_selected_trials(prepared))
    flat = np.stack(
        [
            (
                rows[:, None] * len(relative)
                + np.rint(
                    (times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S
                ).astype(int)[:, None]
                + peaks.LAGS_MS[None, :]
            )
            for times in (still, move)
        ]
    )  # (period, trial, bin)
    used = np.zeros(len(alignments) * len(relative), dtype=bool)
    used[flat.ravel()] = True
    names = _SHARED["metadata"]["column_names"]
    _SHARED.update(
        window_rows=used,
        window_position=np.searchsorted(np.flatnonzero(used), flat),
        removed_columns=[
            index
            for index, name in enumerate(names)
            if name.startswith(tuple(f"{term}_basis" for term in REMOVED))
        ],
    )


def matched_part(intercept: float, coefficients, history, width: int) -> float:
    """Return the part of the after-exit-minus-in-port peak that flash × after exit carries.

    The peak of the model rate (`peaks.py`, spikes/s) on the paired flashes of
    `peaks.paired_flashes`, minus the same with the flash × after exit columns set to 0.
    """
    design = dmat._design_with_history(
        _SHARED["common"], history, width, _SHARED["window_rows"]
    )
    coefficients = np.asarray(coefficients, dtype=float)
    removed = _SHARED["removed_columns"]
    log_rate = float(intercept) + design @ coefficients
    changes = []
    contribution = raw_contribution(design, coefficients, _SHARED["metadata"], removed)
    for values in (log_rate, log_rate - contribution):
        rate = np.exp(values)[_SHARED["window_position"]] / dmat.BINWIDTH_S
        peak = peaks.peaks(rate.mean(1))  # (period, bin) -> (period,)
        changes.append(peak[1] - peak[0])
    return float(changes[0] - changes[1])


def measures(result: dict, history_scale: np.ndarray) -> np.ndarray:
    """Return after exit minus in port at each gap, and the fixation level, of one fit."""
    kernels = fitted_kernels(result, _SHARED["metadata"], history_scale)
    lags = kernels[dmat.FLASH_AFTER_EXIT][0]
    window = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    out = [
        (flash_kernel_at(kernels, gap, True) - flash_kernel_at(kernels, gap, False))[
            window
        ].mean()
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
    """Simulate one unit with no flash × after exit effect and fit the full model."""
    prepared, valid, common, metadata = (
        _SHARED[key] for key in ("prepared", "valid", "common", "metadata")
    )
    alignments = prepared["alignments"]
    record = job["record"]
    width = int(metadata["base_columns"]) + int(record["components"])
    coefficients = np.asarray(record["coefficients"], dtype=float)
    raw = dmat.build_unit_history(alignments, job["spike_times"])
    real_history, mean, scale = dmat.training_zscore(raw, valid)
    real = measures(record, scale)
    real_part = matched_part(record["intercept"], coefficients, real_history, width)
    removed = _SHARED["removed_columns"]
    # Keep the raw baseline fixed when the interaction is removed. Zeroing
    # its standardized coefficients alone would also change in-port rates.
    null_intercept, null = zero_raw_columns(
        record["intercept"], coefficients, metadata, removed
    )
    null_record = {**record, "intercept": null_intercept}
    rng = np.random.default_rng(SEED + job["unit_id"])
    fits, leak_parts = [], []
    for _ in range(SIMULATIONS):
        counts, scaled, simulated_scale = simulate(null_record, null, mean, scale, rng)
        design = dmat._design_with_history(common, scaled, width, valid)
        model = fit_poisson_at_alpha(design, counts.ravel()[valid], job["alpha"])
        result = {"components": record["components"], "coefficients": model.coef_}
        fits.append(measures(result, simulated_scale))
        leak_parts.append(matched_part(model.intercept_, model.coef_, scaled, width))
    null_level = measures({**record, "coefficients": null}, scale)[2]
    return {
        "unit_id": job["unit_id"],
        "real": real,
        "leak": np.asarray(fits),
        "real_part": real_part,
        "leak_part": np.asarray(leak_parts),
        "simulated_level": null_level,
        "spikes_simulated_over_real": float(
            counts.ravel()[valid].sum()
            / max(build_unit_counts(alignments, job["spike_times"])[valid].sum(), 1)
        ),
    }


def simulate_session(out: Path, session: str, unit_set: str, workers: int) -> dict:
    """Simulate and refit each unit of one session; write `leak_test.npz`."""
    model = cli.PoissonGLM(root=out / session)
    subject, _, name = session.partition("_")
    final = load_unit_records(
        model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
    )
    spikes = unit_spikes(subject, name)
    jobs = [
        {
            "unit_id": int(item["unit_id"]),
            "record": item,
            "alpha": float(item["alpha"]),
            "spike_times": np.asarray(spikes[int(item["unit_id"])], dtype=float),
        }
        for item in final
    ]
    with Pool(workers, _load, (model.windows, model.design)) as pool:
        units = pool.map(_unit, jobs)
    real = np.asarray([unit["real"] for unit in units])  # (unit, measure)
    leaks = np.asarray([unit["leak"] for unit in units])
    leak = leaks.mean(1)
    simulated_level = np.asarray([unit["simulated_level"] for unit in units])
    entry = {"units": len(units), "simulations": SIMULATIONS}
    for index, gap in enumerate(GAPS_MS):
        rest = real[:, index] - leak[:, index]
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
        }
    entry["fixation level"] = {
        "median_simulated": float(np.median(simulated_level)),
        "median_refit": float(np.median(leak[:, 2])),
        # For the units with a level: the ratio has no meaning near 0.
        "median_refit_over_simulated": float(
            np.median(
                (leak[:, 2] / simulated_level)[np.abs(simulated_level) > MIN_LEVEL]
            )
        ),
    }
    entry["median_spikes_simulated_over_real"] = float(
        np.median([unit["spikes_simulated_over_real"] for unit in units])
    )
    np.savez(
        out / session / "leak_test.npz",
        unit_ids=np.asarray([unit["unit_id"] for unit in units]),
        measures=np.asarray([*GAPS_MS, "fixation level"]),
        real=real,
        leak=leaks,
        simulated_level=simulated_level,
        real_part=np.asarray([unit["real_part"] for unit in units]),
        leak_part=np.asarray([unit["leak_part"] for unit in units]),
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stats = {
        session: simulate_session(args.out, session, args.units, args.workers)
        for session in SESSIONS
    }
    (args.out / "leak_test.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
