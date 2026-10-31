"""Measure the power of the flash × after exit test of result 3.

Question: result 3 of `docs/glm.md` finds no consistent flash × after exit
effect in the flash-excited units. Could the test find an effect of a given
size with these units, trials and fits?

Comparison: simulated spikes with a known effect, analyzed as the real data.

- Planted effect: for each flash-excited unit, its final refit with the raw
  flash × after exit weights set to `α` times its raw flash weights (the two
  terms have the same basis), so after the exit the flash kernel is (1 + α)
  times the in-port flash kernel, in log rate. The intercept is changed with
  the centering of `results.zero_raw_columns`, so the in-port rate stays the
  same. α = 0 has no effect and gives the false-positive rate.
- Simulation and refit: as `leak_test.py` (the same simulator, spike history
  of the simulated spikes, rate limit, real design and penalty of the unit).
- Estimate: the flash × after exit part of the refit in the log-ratio measure
  of the main comparison (`leak_test.log_part`: log(after exit / in port) of
  the model peak minus the same without the raw flash × after exit
  contribution), minus the leak of the unit (`leak_test.npz`, the mean over
  its simulations with no effect), as in the real test. The normalized
  measure (`leak_test.comparison_part` over the observed mean rate) is also
  given.
- True value: the same measure on the rate of the simulating model (no
  noise, with the real spike history).
- Replicate: one simulation of every excited unit of a mouse. In each, the
  two-sided Wilcoxon signed-rank test over units of the estimate, as in the
  real test; detected = p < 0.05 with a median above 0. Power: the fraction
  of `REPLICATES` replicates detected; at α = 0, the fraction with p < 0.05
  (either sign). A unit with a peak ≤ 0 in a fit has no log ratio and is
  left out of that replicate, as in the real test.
- Units: the flash-excited units of `raw_video_trust_classification.json`
  (the units of result 3). Trials, alignment, pairs, baseline (in port, -40
  to 0 ms), peak window and exclusions: those of `flash_movement.py` and
  `leak_test.py`. The long-gap class (preceding gap of 80 ms or more) is the
  main measure; the 40 ms class is also given.
- The sampling unit is the unit. Each mouse is analyzed alone (the real
  effects have opposite signs, and n differs: the power of GRB064 is the
  limit).

The script writes `power_test.npz` in each session directory,
`results/glm/power_test.json`, and `figures/glm/power_test.svg`. It needs
`leak_test.npz` and `flash_movement.json`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.power_test
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
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, flash_movement, leak_test
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    fit_poisson_at_alpha,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import zero_raw_columns
from thesis.ephys.units import unit_spikes
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
ALPHAS = (0.0, 0.1, 0.25, 0.5)
REPLICATES = 20
SEED = 100
POWER = 0.8
COLOR = CONCEPT_COLORS["visual"]
MARKERS = {"GRB006": "o", "GRB064": "^"}


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


def _unit(job: dict) -> dict:
    """Simulate one unit with each planted effect and refit; return the model peaks."""
    shared = leak_test._SHARED
    prepared, valid, common, metadata = (
        shared[key] for key in ("prepared", "valid", "common", "metadata")
    )
    alignments = prepared["alignments"]
    record = job["record"]
    width = int(metadata["base_columns"]) + int(record["components"])
    raw = dmat.build_unit_history(alignments, job["spike_times"])
    real_history, mean, scale = dmat.training_zscore(raw, valid)
    observed = build_unit_counts(alignments, job["spike_times"]).reshape(
        len(alignments), -1
    )
    valid_bins = valid.reshape(len(alignments), -1)
    paired = shared["paired_trials"]
    scale_rate = (observed * valid_bins)[paired].sum() / (
        valid_bins[paired].sum() * dmat.BINWIDTH_S
    )
    rng = np.random.default_rng(SEED + job["unit_id"])
    true, fitted = [], []
    for alpha in ALPHAS:
        intercept, weights = planted(record, metadata, alpha)
        true.append(leak_test.comparison_peaks(intercept, weights, real_history, width))
        level = []
        for _ in range(REPLICATES):
            counts, scaled, _ = leak_test.simulate(
                {**record, "intercept": intercept}, weights, mean, scale, rng
            )
            design = dmat._design_with_history(common, scaled, width, valid)
            model = fit_poisson_at_alpha(design, counts.ravel()[valid], job["alpha"])
            level.append(
                leak_test.comparison_peaks(model.intercept_, model.coef_, scaled, width)
            )
        fitted.append(level)
        print(f"unit {job['unit_id']}: α = {alpha:g} done", flush=True)
    return {
        "unit_id": job["unit_id"],
        "true_peaks": np.asarray(true) / scale_rate,  # (alpha, gap, source, period)
        "fit_peaks": np.asarray(fitted) / scale_rate,  # (alpha, replicate, ...)
    }


def simulate_session(session: str, units: list[int], workers: int) -> Path:
    """Simulate and refit the given units of one session; write `power_test.npz`."""
    model = cli.PoissonGLM(root=OUT / session)
    subject, _, name = session.partition("_")
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir("all") / "checkpoints", "unit_*_final.json"
        )
    }
    spikes = unit_spikes(subject, name)
    jobs = [
        {
            "unit_id": unit,
            "record": final[unit],
            "alpha": float(final[unit]["alpha"]),
            "spike_times": np.asarray(spikes[unit], dtype=float),
        }
        for unit in units
        if unit in final
    ]
    with Pool(workers, leak_test._load, (model.windows, model.design)) as pool:
        results = pool.map(_unit, jobs)
    path = OUT / session / "power_test.npz"
    np.savez(
        path,
        unit_ids=np.asarray([r["unit_id"] for r in results]),
        alphas=np.asarray(ALPHAS),
        gaps=np.asarray(flash_movement.GAP_LABELS),
        true_peaks=np.asarray([r["true_peaks"] for r in results]),
        fit_peaks=np.asarray([r["fit_peaks"] for r in results]),
    )
    return path


def session_power(session: str, leak: dict) -> dict:
    """Return the true effects, estimates, and power of one session, by α and gap."""
    with np.load(OUT / session / "power_test.npz") as saved:
        ids = saved["unit_ids"].tolist()
        true_peaks, fit_peaks = saved["true_peaks"], saved["fit_peaks"]
    order = [leak["unit_ids"].index(unit) for unit in ids]
    out = {"units": len(ids), "replicates": REPLICATES, "by_gap": {}}
    for g, gap in enumerate(flash_movement.GAP_LABELS):
        entry = {}
        for scale in ("log", "normalized"):
            if scale == "log":
                true = leak_test.log_part(true_peaks)[..., g]  # (unit, alpha)
                fits = leak_test.log_part(fit_peaks)[..., g]  # (unit, alpha, rep)
                base = leak["log leak"][order, g]
            else:
                true = leak_test.comparison_part(true_peaks)[..., g]
                fits = leak_test.comparison_part(fit_peaks)[..., g]
                base = leak["leak"][order, g]
            estimates = fits - base[:, None, None]
            levels = []
            for a, alpha in enumerate(ALPHAS):
                medians, ps, kept = [], [], []
                for r in range(REPLICATES):
                    values = estimates[:, a, r]
                    values = values[np.isfinite(values)]
                    medians.append(float(np.median(values)))
                    ps.append(float(wilcoxon(values).pvalue))
                    kept.append(len(values))
                medians, ps = np.asarray(medians), np.asarray(ps)
                detected = (ps < 0.05) & (medians > 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    true_median = float(np.nanmedian(true[:, a]))
                levels.append(
                    {
                        "alpha": alpha,
                        "median_true": true_median,
                        "median_estimate": float(np.median(medians)),
                        "estimate_95": np.percentile(medians, [2.5, 97.5]).tolist(),
                        "power": float(detected.mean()),
                        "fraction_p_below_0.05": float((ps < 0.05).mean()),
                        "median_units_per_replicate": float(np.median(kept)),
                        "replicate_medians": medians.tolist(),
                    }
                )
            entry[scale] = levels
        out["by_gap"][gap] = entry
    return out


def plot(stats: dict) -> Path:
    """Power and estimate against the true effect, one column per mouse."""
    gap = flash_movement.GAP_LABELS[0]
    subjects = list(stats)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            2, len(subjects), figsize=(3.3 * len(subjects), 5.6), squeeze=False
        )
        limits = [np.inf, -np.inf]
        for column, subject in enumerate(subjects):
            levels = stats[subject]["by_gap"][gap]["log"]
            true = np.asarray([level["median_true"] for level in levels])
            power = np.asarray([level["power"] for level in levels])
            marker = MARKERS.get(subject, "o")
            top, bottom = axes[0, column], axes[1, column]
            top.plot(true, power, color=COLOR, marker=marker, lw=1.2)
            top.axhline(POWER, color=REFERENCE_COLOR, ls="--", lw=0.8)
            top.axhline(0.05, color=REFERENCE_COLOR, ls=":", lw=0.8)
            top.text(
                true[0] + 0.15,
                levels[0]["fraction_p_below_0.05"] + 0.04,
                f"no effect: p < 0.05 in {levels[0]['fraction_p_below_0.05']:.0%}",
                color=COLOR,
                fontsize=6,
            )
            top.set_ylim(-0.02, 1.02)
            top.set_title(
                f"{subject}: {stats[subject]['units']} excited units", fontsize=8
            )
            top.set_xlabel("planted effect, median of units\n(log ratio)")
            top.set_ylabel("fraction of replicates detected")
            for level in levels:
                bottom.scatter(
                    np.full(REPLICATES, level["median_true"]),
                    level["replicate_medians"],
                    color=COLOR,
                    marker=marker,
                    s=8,
                    alpha=0.4,
                    lw=0,
                )
                limits = [
                    min(limits[0], level["median_true"], *level["replicate_medians"]),
                    max(limits[1], level["median_true"], *level["replicate_medians"]),
                ]
            bottom.set_xlabel("planted effect, median of units\n(log ratio)")
            bottom.set_ylabel("estimate (real minus leak),\nmedian of units")
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
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--reuse", action="store_true", help="read power_test.npz; simulate nothing"
    )
    args = parser.parse_args()
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
    if not args.reuse:
        for session in SESSIONS:
            print(simulate_session(session, sorted(excited[session]), args.workers))
    values = leak_test.comparison_values(OUT)
    stats = {
        session.split("_")[0]: session_power(session, values[session.split("_")[0]])
        for session in SESSIONS
    }
    (OUT / "power_test.json").write_text(json.dumps(stats, indent=1))
    print(
        json.dumps(
            {
                s: {
                    g: [
                        {k: v for k, v in level.items() if k != "replicate_medians"}
                        for level in e["log"]
                    ]
                    for g, e in st["by_gap"].items()
                }
                for s, st in stats.items()
            },
            indent=1,
        )
    )
    print(plot(stats))


if __name__ == "__main__":
    main()
