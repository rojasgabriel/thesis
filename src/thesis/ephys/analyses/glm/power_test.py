"""Measure the power of the flash × after exit test of result 3.

Question: result 3 of `docs/glm.md` finds no consistent flash × after exit
effect in the flash-excited units. Could the test find an effect of a given
size with these units, trials and fits, and how often does it find one when
there is none?

Comparison: simulated spikes with a known effect, analyzed as the real data.

- Planted effect: for each flash-excited unit, its final refit with the raw
  flash × after exit weights set to `α` times its raw flash weights (the two
  terms have the same basis), so after the exit the flash kernel is (1 + α)
  times the in-port flash kernel, in log rate. The intercept is changed with
  the centering of `results.zero_raw_columns`, so the in-port rate stays the
  same. α = 0 has no effect and gives the false-positive rate. Planted
  sizes: `ALPHAS`, the same for both estimators.
- Simulation: as `leak_test.py` (the same simulator, spike history of the
  simulated spikes, rate limit, real design).
- Two estimators (`--mode`), each minus the leak of the unit
  (`leak_test.npz`, the mean over its final refits to simulations with no
  effect), as in the real test:
  - final: one refit to all trials at the penalty of the unit's final
    refit. This is the final-refit version of the test, less noisy than the
    main measure.
  - held_out: the main measure of result 3. Ten refits, one for each
    whole-trial fold of `fit.cross_validation_partitions`, each at the
    penalty that the real fit chose in that fold (the final penalty for a
    fold with no real test spike); each paired trial takes the rate of the
    fold model that did not see it. The history columns are z-scored over
    all valid bins, not over the fit rows of each fold (a change of offset
    and scale that the fit absorbs). The penalty is not selected again for
    each simulation.
- Measure: the flash × after exit part in the log-ratio measure of the main
  comparison (`leak_test.log_part`: log(after exit / in port) of the model
  peak minus the same without the raw flash × after exit contribution). The
  normalized measure (`leak_test.comparison_part` over the observed mean
  rate) is also given.
- True value: the same measure on the rate of the simulating model (no
  noise, with the real spike history).
- Replicate: one simulation of every excited unit of a mouse. In each, the
  two-sided Wilcoxon signed-rank test over units of the estimate, as in the
  real test; detected = p < 0.05 with a median above 0. Power: the fraction
  of replicates detected; at α = 0, the fraction with p < 0.05 (either
  sign). Both with a Clopper-Pearson 95% interval: the count of replicates
  (`REPLICATES`, no effect then each planted size) sets their
  precision. A unit with a peak ≤ 0 in a fit has no log ratio and is left
  out of that replicate, as in the real test. The effect at 80% power is the
  linear interpolation of power against the median true effect.
- Units: the flash-excited units of `flash_excited_units.json` (`peaks.excited_units`)
  (the units of result 3). Trials, alignment, pairs, baseline (in port, -40
  to 0 ms), peak window and exclusions: those of `flash_movement.py` and
  `leak_test.py`. The long-gap class (preceding gap of 80 ms or more) is the
  main measure; the 40 ms class is also given.
- The sampling unit is the unit. Each mouse is analyzed alone (the real
  effects have opposite signs, and n differs: the power of GRB064 is the
  limit).

The script writes `power_test.npz` (final) and `power_test_held_out.npz` in
each session directory, `results/glm/power_test.json`, and
`figures/glm/power_test.svg`. It needs `leak_test.npz` and
`flash_movement.json`. The run of 2026-11-04 (20 replicates, 4 sizes, final
only) is in `results/glm/_power_test_2026_11_04/`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.power_test --mode final --workers 4
    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.power_test --mode held_out --workers 4
"""

from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, flash_movement, leak_test, peaks
from thesis.ephys.analyses.glm.fit import (
    CV_FOLDS,
    CV_SEED,
    build_unit_counts,
    cross_validation_partitions,
    fit_poisson_at_alpha,
    load_unit_records,
    trial_folds,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import raw_contribution, zero_raw_columns
from thesis.ephys.units import unit_spikes
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MOUSE_MARKERS,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
MODES = ("final", "held_out")
ALPHAS = (0.0, 0.05, 0.1, 0.175, 0.25, 0.375, 0.5)
# Replicates with no effect, then with each planted size. No effect gets more:
# a false-positive rate of a few percent needs many replicates to be precise.
REPLICATES = (200, 50)
FILES = {"final": "power_test.npz", "held_out": "power_test_held_out.npz"}
SEED = 100
POWER = 0.8
COLOR = CONCEPT_COLORS["visual"]


def _replicates(alpha: float) -> int:
    return REPLICATES[0] if alpha == 0 else REPLICATES[1]


def _columns(names: list[str], term: str) -> list[int]:
    """Return the basis columns of a term, in the order of the basis index."""
    found = [(n, i) for i, n in enumerate(names) if n.startswith(f"{term}_basis_")]
    return [
        i for n, i in sorted(found, key=lambda item: int(item[0].rsplit("_", 1)[1]))
    ]


def planted(record: dict, metadata: dict, alpha: float) -> tuple[float, np.ndarray]:
    """Return the intercept and weights with flash × after exit = α × the flash weights (raw)."""
    names = metadata["column_names"]
    after = _columns(names, dmat.FLASH_AFTER_EXIT)
    flash = _columns(names, dmat.FLASH_COMMON)
    if len(after) != len(flash):
        raise ValueError("The flash and flash × after exit bases differ.")
    mean = np.asarray(metadata["training_mean"])
    scale = np.asarray(metadata["training_scale"])
    intercept, weights = zero_raw_columns(
        record["intercept"], record["coefficients"], metadata, after
    )
    raw_flash = np.asarray(record["coefficients"], dtype=float)[flash] / scale[flash]
    weights[after] = alpha * raw_flash * scale[after]
    # The standardized columns are centered; add back the centering so that
    # only the raw contribution x · w_raw is added.
    return intercept + float((mean[after] / scale[after]) @ weights[after]), weights


def held_out_peaks(counts, history, width: int, alphas: list[float]) -> np.ndarray:
    """Return the comparison peaks of the fold models, each on the trials it did not see."""
    common, metadata = leak_test._SHARED["common"], leak_test._SHARED["metadata"]
    valid, pair = leak_test._SHARED["valid"], leak_test._SHARED["pair_rows"]
    n_trials = len(leak_test._SHARED["prepared"]["alignments"])
    fold_rows = np.repeat(
        trial_folds(n_trials, CV_FOLDS, CV_SEED), len(valid) // n_trials
    )
    pair_fold = fold_rows[pair]
    removed = leak_test._SHARED["removed_columns"]
    log_rate = np.full(int(pair.sum()), np.nan)
    contribution = np.full(int(pair.sum()), np.nan)
    partitions = cross_validation_partitions(n_trials, len(valid) // n_trials, valid)
    model = None
    for fold, part in enumerate(partitions):
        here = pair_fold == fold
        if not here.any():
            continue
        design = dmat._design_with_history(common, history, width, part["fit"])
        # Warm start from the fold before: the same optimum, fewer iterations.
        model = fit_poisson_at_alpha(
            design, counts.ravel()[part["fit"]], alphas[fold], model
        )
        test = dmat._design_with_history(
            common, history, width, pair & (fold_rows == fold)
        )
        log_rate[here] = model.intercept_ + test @ model.coef_
        contribution[here] = raw_contribution(test, model.coef_, metadata, removed)
    return leak_test.peaks_from(log_rate, contribution)


def _unit(job: dict) -> dict:
    """Simulate one unit with each planted effect and refit; return the model peaks."""
    shared = leak_test._SHARED
    prepared, valid, common, metadata = (
        shared[key] for key in ("prepared", "valid", "common", "metadata")
    )
    alignments = prepared["alignments"]
    record, mode = job["record"], job["mode"]
    width = int(metadata["base_columns"]) + int(record["components"])
    raw = dmat.build_unit_history(alignments, job["spike_times"])
    real_history, mean, scale = dmat.training_zscore(raw, valid)
    observed = build_unit_counts(alignments, job["spike_times"])
    grid = observed.reshape(len(alignments), -1)
    valid_bins = valid.reshape(len(alignments), -1)
    paired = shared["paired_trials"]
    scale_rate = (grid * valid_bins)[paired].sum() / (
        valid_bins[paired].sum() * dmat.BINWIDTH_S
    )
    # The penalty of each fold: the saved one of each fold with real test spikes.
    saved = iter(job["fold_alphas"])
    fold_alphas = [
        next(saved) if observed[part["test"]].sum() > 0 else job["alpha"]
        for part in cross_validation_partitions(
            len(alignments), len(valid) // len(alignments), valid
        )
    ]
    rng = np.random.default_rng([SEED, job["unit_id"], MODES.index(mode)])
    most = max(REPLICATES)
    true = []
    fitted = np.full((len(ALPHAS), most, 2, 2, 2), np.nan)
    for a, alpha in enumerate(ALPHAS):
        intercept, weights = planted(record, metadata, alpha)
        true.append(leak_test.comparison_peaks(intercept, weights, real_history, width))
        for r in range(_replicates(alpha)):
            counts, scaled, _ = leak_test.simulate(
                {**record, "intercept": intercept}, weights, mean, scale, rng
            )
            if mode == "final":
                design = dmat._design_with_history(common, scaled, width, valid)
                model = fit_poisson_at_alpha(
                    design, counts.ravel()[valid], job["alpha"]
                )
                fitted[a, r] = leak_test.comparison_peaks(
                    model.intercept_, model.coef_, scaled, width
                )
            else:
                fitted[a, r] = held_out_peaks(counts, scaled, width, fold_alphas)
        print(f"unit {job['unit_id']} ({mode}): α = {alpha:g} done", flush=True)
    return {
        "unit_id": job["unit_id"],
        "true_peaks": np.asarray(true) / scale_rate,  # (alpha, gap, source, period)
        "fit_peaks": fitted / scale_rate,  # (alpha, replicate, ...), NaN padded
    }


def simulate_session(session: str, units: list[int], workers: int, mode: str) -> Path:
    """Simulate and refit the given units of one session; write the npz of `mode`."""
    model = cli.PoissonGLM(root=OUT / session)
    subject, _, name = session.partition("_")
    checkpoints = model.fit_dir("all") / "checkpoints"
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_final.json")
    }
    folds = {
        int(item["unit_id"]): [float(f["alpha"]) for f in item["folds"]]
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    spikes = unit_spikes(subject, name)
    jobs = [
        {
            "unit_id": unit,
            "mode": mode,
            "record": final[unit],
            "alpha": float(final[unit]["alpha"]),
            "fold_alphas": folds[unit],
            "spike_times": np.asarray(spikes[unit], dtype=float),
        }
        for unit in units
        if unit in final
    ]
    with Pool(workers, leak_test._load, (model.windows, model.design)) as pool:
        results = pool.map(_unit, jobs, chunksize=1)
    path = OUT / session / FILES[mode]
    np.savez(
        path,
        unit_ids=np.asarray([r["unit_id"] for r in results]),
        alphas=np.asarray(ALPHAS),
        gaps=np.asarray(flash_movement.GAP_LABELS),
        true_peaks=np.asarray([r["true_peaks"] for r in results]),
        fit_peaks=np.asarray([r["fit_peaks"] for r in results]),
    )
    return path


def _interval(hits: int, n: int) -> list[float]:
    ci = binomtest(hits, n).proportion_ci(confidence_level=0.95)
    return [float(ci.low), float(ci.high)]


def at_power(levels: list[dict], power: float = POWER) -> float | None:
    """Return the median true effect where power crosses `power` (linear), if it does."""
    true = [level["median_true"] for level in levels]
    values = [level["power"] for level in levels]
    for (x0, y0), (x1, y1) in zip(
        zip(true, values, strict=True),
        zip(true[1:], values[1:], strict=True),
        strict=False,
    ):
        if y0 < power <= y1:
            return float(x0 + (power - y0) * (x1 - x0) / (y1 - y0))
    return None


def session_power(session: str, leak: dict, mode: str) -> dict | None:
    """Return the true effects, estimates, and power of one session, by α and gap."""
    path = OUT / session / FILES[mode]
    if not path.exists():
        return None
    with np.load(path) as saved:
        ids = saved["unit_ids"].tolist()
        alphas = saved["alphas"].tolist()
        true_peaks, fit_peaks = saved["true_peaks"], saved["fit_peaks"]
    order = [leak["unit_ids"].index(unit) for unit in ids]
    # Each version of the test takes its own leak, as the real data does. Held
    # out, the leak is the mean of the α = 0 replicates of this file, so their
    # estimates are centered on 0 by construction.
    leak_key = "leak (held out)" if mode == "held_out" else "leak"
    out = {"units": len(ids), "by_gap": {}}
    for g, gap in enumerate(flash_movement.GAP_LABELS):
        entry = {}
        for scale in ("log", "normalized"):
            if scale == "log":
                true = leak_test.log_part(true_peaks)[..., g]  # (unit, alpha)
                fits = leak_test.log_part(fit_peaks)[..., g]  # (unit, alpha, rep)
                base = leak["log " + leak_key][order, g]
            else:
                true = leak_test.comparison_part(true_peaks)[..., g]
                fits = leak_test.comparison_part(fit_peaks)[..., g]
                base = leak[leak_key][order, g]
            estimates = fits - base[:, None, None]
            levels = []
            for a, alpha in enumerate(alphas):
                medians, ps, kept = [], [], []
                for r in range(_replicates(alpha)):
                    values = estimates[:, a, r]
                    values = values[np.isfinite(values)]
                    medians.append(float(np.median(values)))
                    ps.append(float(wilcoxon(values).pvalue))
                    kept.append(len(values))
                medians, ps = np.asarray(medians), np.asarray(ps)
                detected = int(((ps < 0.05) & (medians > 0)).sum())
                significant = int((ps < 0.05).sum())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    true_median = float(np.nanmedian(true[:, a]))
                levels.append(
                    {
                        "alpha": alpha,
                        "replicates": len(ps),
                        "median_true": true_median,
                        "median_estimate": float(np.median(medians)),
                        "estimate_95": np.percentile(medians, [2.5, 97.5]).tolist(),
                        "power": detected / len(ps),
                        "power_95": _interval(detected, len(ps)),
                        "fraction_p_below_0.05": significant / len(ps),
                        "fraction_p_below_0.05_95": _interval(significant, len(ps)),
                        "median_units_per_replicate": float(np.median(kept)),
                        "replicate_medians": medians.tolist(),
                    }
                )
            entry[scale] = {
                "levels": levels,
                "effect_at_80_percent_power": at_power(levels),
            }
        out["by_gap"][gap] = entry
    return out


def plot(stats: dict) -> Path:
    """Power and estimate against the true effect, one column per mouse."""
    gap = flash_movement.GAP_LABELS[0]
    subjects = list(stats)
    faces = {"final": COLOR, "held_out": "white"}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            2, len(subjects), figsize=(3.3 * len(subjects), 5.6), squeeze=False
        )
        limits = [np.inf, -np.inf]
        for column, subject in enumerate(subjects):
            marker = MOUSE_MARKERS.get(subject, "o")
            top, bottom = axes[0, column], axes[1, column]
            for row, mode in enumerate(MODES):
                if stats[subject].get(mode) is None:
                    continue
                levels = stats[subject][mode]["by_gap"][gap]["log"]["levels"]
                true = np.asarray([level["median_true"] for level in levels])
                power = np.asarray([level["power"] for level in levels])
                low, high = np.asarray([level["power_95"] for level in levels]).T
                top.errorbar(
                    true,
                    power,
                    yerr=[power - low, high - power],
                    color=COLOR,
                    mfc=faces[mode],
                    marker=marker,
                    lw=1.2,
                    capsize=2,
                )
                null = levels[0]
                top.text(
                    0.98,
                    0.26 - 0.16 * row,
                    f"{mode.replace('_', '-')}, no effect:\np < 0.05 in "
                    f"{null['fraction_p_below_0.05']:.0%} of {null['replicates']}",
                    color=COLOR,
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    transform=top.transAxes,
                )
                if mode != "final":
                    continue
                for level in levels:
                    bottom.scatter(
                        np.full(level["replicates"], level["median_true"]),
                        level["replicate_medians"],
                        color=COLOR,
                        marker=marker,
                        s=6,
                        alpha=0.3,
                        lw=0,
                    )
                    limits = [
                        min(
                            limits[0], level["median_true"], *level["replicate_medians"]
                        ),
                        max(
                            limits[1], level["median_true"], *level["replicate_medians"]
                        ),
                    ]
            top.axhline(POWER, color=REFERENCE_COLOR, ls="--", lw=0.8)
            top.axhline(0.05, color=REFERENCE_COLOR, ls=":", lw=0.8)
            top.set_ylim(-0.02, 1.02)
            top.set_title(
                f"{subject}: {stats[subject]['final']['units']} excited units",
                fontsize=8,
            )
            top.set_xlabel("planted effect, median of units\n(log ratio)")
            top.set_ylabel(
                "fraction of replicates detected\n(filled: final; open: held out)"
            )
            bottom.set_xlabel("planted effect, median of units\n(log ratio)")
            bottom.set_ylabel("final-refit estimate (minus leak),\nmedian of units")
        pad = 0.05 * (limits[1] - limits[0])
        span = (limits[0] - pad, limits[1] + pad)
        for column in range(len(subjects)):
            axes[1, column].plot(span, span, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axes[1, column].set_xlim(span)
            axes[1, column].set_ylim(span)
            axes[0, column].set_xlim(span)
        figure.tight_layout()
        label_panels(axes.ravel())
        return save_figure(figure, OUT / "power_test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=MODES, default="final")
    parser.add_argument(
        "--reuse", action="store_true", help="read the saved npz; simulate nothing"
    )
    args = parser.parse_args()
    excited = peaks.load_excited_units()
    if not args.reuse:
        for session in SESSIONS:
            print(
                simulate_session(
                    session, sorted(excited[session]), args.workers, args.mode
                ),
                flush=True,
            )
    values = leak_test.comparison_values(OUT)
    stats = {
        session.split("_")[0]: {
            mode: session_power(session, values[session.split("_")[0]], mode)
            for mode in MODES
        }
        for session in SESSIONS
    }
    (OUT / "power_test.json").write_text(json.dumps(stats, indent=1))
    print(
        json.dumps(
            {
                subject: {
                    mode: {
                        gap: {
                            "effect_at_80_percent_power": e["log"][
                                "effect_at_80_percent_power"
                            ],
                            "levels": [
                                {
                                    k: v
                                    for k, v in level.items()
                                    if k != "replicate_medians"
                                }
                                for level in e["log"]["levels"]
                            ],
                        }
                        for gap, e in result["by_gap"].items()
                    }
                    for mode, result in modes.items()
                    if result is not None
                }
                for subject, modes in stats.items()
            },
            indent=1,
        )
    )
    print(plot(stats))


if __name__ == "__main__":
    main()
