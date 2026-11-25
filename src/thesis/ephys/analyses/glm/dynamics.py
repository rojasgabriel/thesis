"""Check the held-out GLM's dynamics with generated spike history.

Comparison: observed counts, conditional predictions with observed history,
and recursively generated Poisson spikes under the same saved fold models.
No fitting or rate clipping. Units, exclusions, trials, first-flash alignment,
1-ms bins, and response-entry censoring follow fit.py. All eligible fitted
units enter; the sampling unit is the neuron, with trials paired across sources.
There is no baseline subtraction: rates and first-flash PSTHs are absolute.
The final grid bin is excluded from all sources because of the native event
builder's open final boundary; saved likelihoods are checked on the original mask.

Each trial starts with its observed 100-ms pre-window history, then uses only
generated spikes. Its direct history contribution ends after 100 ms; indirect
effects of the initial state can persist through subsequent generated spikes.
Task/video inputs remain observed; this is not a generative model of behavior.
Trajectories exceeding 10 expected spikes/bin (10,000 Hz) are stopped and
reported, never clipped or counted as successful simulations. This operational
threshold is not a mathematical stability test. Count variance pools trials
with different task inputs; it is not a repeated-identical-stimulus Fano factor.
ISIs are binned at 1 ms and never cross trial boundaries. Simulation intervals
describe Monte Carlo variability with fixed fits, not parameter uncertainty.

    uv run python -m thesis.ephys.analyses.glm.dynamics --root <session-dir>
    uv run python -m thesis.ephys.analyses.glm.dynamics --self-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from numba import njit
from scipy.signal import lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, MODEL_COLOR, save_figure

MAX_COUNT = 10.0
SEED = 20261116


def simulate(drive, kernel, initial, valid, repeats, rng):
    """Return uncapped counts, first failed bin per trajectory, and max intensity.

    drive: trial × bin, kernel: lag 1 onward, initial: trial × lag (oldest first).
    A failed trajectory is NaN over its entire window, so it cannot silently
    enter complete-trajectory statistics as a low-count or truncated response.
    """
    drive, kernel, initial = (
        np.asarray(x, dtype=float) for x in (drive, kernel, initial)
    )
    valid = np.asarray(valid, dtype=bool)
    trials, bins = drive.shape
    if (
        valid.shape != drive.shape
        or initial.shape != (trials, len(kernel))
        or not np.isfinite(drive).all()
        or not np.isfinite(kernel).all()
        or not np.isfinite(initial).all()
        or np.any(initial < 0)
        or repeats < 1
        or np.any(np.diff(valid.astype(int), axis=1) > 0)
    ):
        raise ValueError(
            "Invalid simulation grid, parameters, or non-prefix validity mask."
        )
    return _simulate(drive, kernel, initial, valid, repeats, rng)


@njit(cache=True)
def _simulate(drive, kernel, initial, valid, repeats, rng):
    """Sample independent trajectories without allocating a lag matrix per bin."""
    trials, bins = drive.shape
    lag = len(kernel)
    counts = np.full((repeats, trials, bins), np.nan)
    failed = np.full((repeats, trials), -1, dtype=int)
    maximum = 0.0
    for r in range(repeats):
        for i in range(trials):
            past = np.zeros(lag + bins)
            past[:lag] = initial[i]
            for t in range(bins):
                if not valid[i, t]:
                    break
                eta = drive[i, t]
                for k in range(lag):
                    eta += past[lag + t - k - 1] * kernel[k]
                rate = np.exp(min(eta, 700.0))
                maximum = max(maximum, rate)
                if not np.isfinite(eta) or eta > np.log(MAX_COUNT):
                    failed[r, i] = t
                    counts[r, i, :] = np.nan
                    break
                count = rng.poisson(rate)
                past[lag + t] = count
                counts[r, i, t] = count
    return counts, failed, maximum


def history_counts(alignments, spikes, centers, lag):
    """Bin pre-window spikes using the event regressor's nearest-bin convention."""
    initial = np.zeros((len(alignments), lag))
    for i, alignment in enumerate(alignments):
        shifted = spikes + dmat.BINWIDTH_S
        overlaps = (
            shifted + dmat.spike_history_basis().basis_time[-1] > alignment + centers[0]
        )
        index = (
            np.rint((shifted[overlaps] - alignment) / dmat.BINWIDTH_S).astype(int) - 1
        )
        index -= int(round(centers[0] / dmat.BINWIDTH_S))
        before = index[(index >= -lag) & (index < 0)] + lag
        initial[i] = np.bincount(before, minlength=lag)
    return initial


def isi_histogram(counts):
    """Pool within-trial integer-bin ISIs (including same-bin multiple spikes)."""
    histogram = np.zeros(102, dtype=int)
    for row in np.asarray(counts).reshape(-1, counts.shape[-1]):
        if np.isinf(row).any() or np.any(row < 0):
            raise ValueError("Invalid spike count.")
        indices = np.repeat(np.arange(len(row)), np.nan_to_num(row).astype(int))
        intervals = np.diff(indices)
        histogram += np.bincount(np.minimum(intervals, 101), minlength=102)
    return histogram


def run(root: Path, out: Path, repeats: int, unit_set: str):
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    centers = prepared["relative_centers"]
    alignments = prepared["alignments"]
    valid = dmat._valid_bin_mask(prepared)
    grid = (len(alignments), len(centers))
    mask = valid.reshape(grid).copy()
    # DAMN uses an open final event boundary; omit this one bin for every source.
    mask[:, -1] = False
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    if common.shape[0] != valid.size:
        raise ValueError("Saved design does not match trial windows.")
    partitions = fit.cross_validation_partitions(*grid, valid)
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    selected = set(units.iloc[fit.analysis_unit_indices(units, unit_set)]["unit_id"])
    records = fit.load_unit_records(
        model.fit_dir("all") / "checkpoints", "unit_*_folds.json"
    )
    records = [r for r in records if r["unit_id"] in selected]
    if {r["unit_id"] for r in records} != selected:
        raise ValueError("Some selected units lack saved held-out fits.")
    units = units.set_index("unit_id")
    basis = dmat.spike_history_basis().basis
    output, curves, failure_bins = [], [], []
    out.mkdir(parents=True, exist_ok=True)
    for record in records:
        unit_id = record["unit_id"]
        spikes = np.asarray(units.loc[unit_id, "spike_times_s"], dtype=float)
        observed = fit.build_unit_counts(alignments, spikes).reshape(grid)
        raw = dmat.build_unit_history(alignments, spikes)
        initial = history_counts(alignments, spikes, centers, len(basis))
        # Verify every basis column, including pre-window initialization and lag 1.
        extended = np.concatenate((initial, observed), axis=1)
        replay = np.stack(
            [
                lfilter(np.r_[0, b], [1], extended, axis=1)[:, len(basis) :]
                for b in basis.T
            ],
            axis=-1,
        )
        np.testing.assert_allclose(
            replay[mask], raw.reshape(*grid, -1)[mask], atol=1e-8
        )
        conditional = np.full(grid, np.nan)
        generated = np.full((repeats, *grid), np.nan)
        failures = np.full((repeats, grid[0]), -1, dtype=int)
        maximum = 0.0
        width = int(metadata["base_columns"]) + int(record["components"])
        scored = [
            (i, p)
            for i, p in enumerate(partitions)
            if observed.ravel()[p["test"]].sum() > 0
        ]
        for (fold_id, part), fold in zip(scored, record["folds"], strict=True):
            weights = np.asarray(fold["coefficients"])
            if weights.shape != (width + len(basis.T),):
                raise ValueError(
                    "Saved coefficients do not match the current history basis."
                )
            scaled, mean, scale = dmat.training_zscore(raw, part["fit"])
            trial = part["all"].reshape(grid)[:, 0]
            rows = np.repeat(trial, grid[1])
            drive = (
                float(fold["intercept"])
                + common[rows, :width] @ weights[:width]
                - (mean / scale) @ weights[width:]
            ).reshape(-1, grid[1])
            kernel = basis @ (weights[width:] / scale)
            expected = np.exp(
                float(fold["intercept"])
                + common[part["test"], :width] @ weights[:width]
                + scaled[part["test"]] @ weights[width:]
            )
            metrics = fit.poisson_metrics(
                observed.ravel()[part["test"]], expected, fold["fit_mean_count"]
            )
            np.testing.assert_allclose(
                metrics["bits_per_spike"], fold["test"]["bits_per_spike"], atol=1e-6
            )
            conditional.ravel()[part["test"]] = expected
            counts, failed, peak = simulate(
                drive,
                kernel,
                initial[trial],
                mask[trial],
                repeats,
                np.random.default_rng([SEED, unit_id, fold_id]),
            )
            generated[:, trial] = counts
            failures[:, trial] = failed
            maximum = max(maximum, peak)
        conditional[~mask] = np.nan
        scored_trials = np.isfinite(conditional).any(axis=1)
        full = (failures < 0) & scored_trials[None, :]
        # A unit with any failed trajectory keeps its diagnostics, but no aggregate
        # generated-vs-real estimate is reported as though sampling were complete.
        failed_count = int(np.count_nonzero(failures >= 0))
        m = mask & scored_trials[:, None]
        obs_total = observed[m].sum()
        conditional_ratio = float(np.nansum(conditional) / obs_total)
        simulated_totals = np.nansum(generated, axis=2)
        real_totals = np.sum(np.where(m, observed, 0), axis=1)[scored_trials]
        intervals = np.quantile(
            simulated_totals[:, scored_trials], [0.025, 0.975], axis=0
        )
        obs_isi = isi_histogram(np.where(m, observed, np.nan))
        sim_isi = isi_histogram(generated[:, full.all(axis=0)])
        entry = {
            "unit_id": unit_id,
            "trials": int(scored_trials.sum()),
            "failed_trajectories": failed_count,
            "trajectories": int(full.size - repeats * (~scored_trials).sum()),
            "max_expected_spikes_per_bin": maximum,
            "conditional_over_observed_count": conditional_ratio,
            "generated_over_observed_count": None
            if failed_count
            else float(np.nansum(generated) / repeats / obs_total),
            "generated_count_ratio_interval": None
            if failed_count
            else (
                np.quantile(
                    np.nansum(generated, axis=(1, 2)) / obs_total, [0.025, 0.975]
                )
            ).tolist(),
            "trial_count_95pct_coverage": None
            if failed_count
            else float(
                np.mean((real_totals >= intervals[0]) & (real_totals <= intervals[1]))
            ),
            "observed_count_variance": float(real_totals.var()),
            "generated_count_variance": None
            if failed_count
            else float(simulated_totals[:, scored_trials].var(axis=1).mean()),
            "observed_isi_histogram": obs_isi.tolist(),
            "generated_isi_histogram": sim_isi.tolist(),
            "failed_trial_numbers": prepared["selected_trial_numbers"][
                np.any(failures >= 0, axis=0)
            ].tolist(),
        }
        # Common complete bins: no changing trial denominator around first flash.
        window = (centers >= -0.1) & (centers < 0.3)
        complete = scored_trials & mask[:, window].all(axis=1)
        if not complete.any():
            raise ValueError("No complete first-flash windows.")
        psth = (
            np.stack(
                (
                    observed[complete][:, window].mean(axis=0),
                    conditional[complete][:, window].mean(axis=0),
                    generated[:, complete][:, :, window].mean(axis=(0, 1)),
                )
            )
            / dmat.BINWIDTH_S
        )
        if failed_count:
            psth[2] = np.nan
        entry["conditional_first_flash_rmse_hz"] = float(
            np.sqrt(np.mean((psth[1] - psth[0]) ** 2))
        )
        entry["generated_first_flash_rmse_hz"] = (
            None if failed_count else float(np.sqrt(np.mean((psth[2] - psth[0]) ** 2)))
        )
        curves.append(psth)
        failure_bins.append(failures)
        output.append(entry)
        print(
            f"{root.name} unit {unit_id}: failures {failed_count}, generated/observed {entry['generated_over_observed_count']}",
            flush=True,
        )
    provenance = {
        "session": root.name,
        "repeats": repeats,
        "seed": SEED,
        "unit_set": unit_set,
        "failure_threshold_expected_count": MAX_COUNT,
        "binwidth_s": dmat.BINWIDTH_S,
        "initialization": "observed 100 ms before window; generated thereafter",
        "excluded_grid_bins": "final 1-ms bin: native event design has an open final boundary",
        "windows_sha256": hashlib.sha256(model.windows.read_bytes()).hexdigest(),
        "design_metadata_sha256": hashlib.sha256(
            model.design.with_suffix(".json").read_bytes()
        ).hexdigest(),
        "fold_files_sha256": {
            r["unit_id"]: hashlib.sha256(
                (
                    model.fit_dir("all")
                    / "checkpoints"
                    / f"unit_{r['unit_id']}_folds.json"
                ).read_bytes()
            ).hexdigest()
            for r in records
        },
        "units": output,
    }
    (out / "dynamics.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        out / "dynamics.npz",
        first_flash_time_s=centers[window],
        psth=np.array(curves),
        failure_bin=np.array(failure_bins),
        trial_numbers=prepared["selected_trial_numbers"],
        bin_time_s=centers,
    )
    plot(provenance, centers[window], np.array(curves), out)


def plot(report, times, curves, out):
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
        colors = ["black", "#777777", MODEL_COLOR]
        labels = ["observed", "observed history", "generated history"]
        # Incomplete final blocks are dropped only from this 10-ms display average.
        n = len(times) // 10 * 10
        good = np.isfinite(curves).all(axis=(1, 2))
        for source, (color, label) in enumerate(zip(colors, labels, strict=True)):
            if good.any():
                y = (
                    curves[good, source, :n]
                    .reshape(good.sum(), -1, 10)
                    .mean(axis=(0, 2))
                )
                axes[0].plot(times[:n].reshape(-1, 10).mean(axis=1), y, color=color)
            axes[0].text(
                0.98,
                0.98 - source * 0.11,
                label,
                color=color,
                ha="right",
                va="top",
                transform=axes[0].transAxes,
            )
        axes[0].set(
            xlabel="Time from first flash (s)", ylabel="Population rate (spikes/s)"
        )
        axes[0].text(
            0.02,
            0.02,
            f"{good.sum()} complete units; {(~good).sum()} flagged",
            transform=axes[0].transAxes,
            fontsize=8,
        )
        units = report["units"]
        for i, key in enumerate(
            ("conditional_over_observed_count", "generated_over_observed_count")
        ):
            values = [u[key] for u in units if u[key] is not None]
            axes[1].scatter(
                i + np.linspace(-0.1, 0.1, len(values)),
                values,
                s=10,
                color=colors[i + 1],
                alpha=0.5,
            )
            if values:
                axes[1].plot(
                    [i - 0.2, i + 0.2],
                    [np.median(values)] * 2,
                    color=colors[i + 1],
                    lw=2,
                )
        axes[1].axhline(1, color="black", lw=0.6)
        axes[1].set(
            xticks=[0, 1],
            xticklabels=[
                f"Observed history\n{len(units)} units",
                f"Generated history\n{good.sum()} units",
            ],
            ylabel="Predicted / observed spike count",
        )
        for name, key, color in (
            ("observed", "observed_isi_histogram", "black"),
            ("generated", "generated_isi_histogram", MODEL_COLOR),
        ):
            hist = np.sum(
                [u[key] for u in units if u["failed_trajectories"] == 0], axis=0
            )
            if np.ndim(hist) and hist.sum():
                axes[2].plot(np.arange(101), hist[:101] / hist.sum(), color=color)
            axes[2].text(
                0.98,
                0.98 if name == "observed" else 0.86,
                name,
                color=color,
                ha="right",
                va="top",
                transform=axes[2].transAxes,
            )
        axes[2].set(xlabel="Within-trial ISI (ms)", ylabel="Fraction of intervals")
        for letter, ax in zip("abc", axes, strict=True):
            ax.text(-0.15, 1.05, letter, weight="bold", transform=ax.transAxes)
        fig.savefig(out / "dynamics.png", dpi=160, bbox_inches="tight")
        save_figure(fig, out / "dynamics.svg")


def self_check():
    class Replay:
        def __init__(self):
            self.i = 0

        def poisson(self, rate):
            # Initial spikes affect bin 0; subsequent rates use the generated 2,0,1.
            np.testing.assert_allclose(rate, np.exp([0.7, 1.1, 0.6][self.i]))
            count = [2, 0, 1][self.i]
            self.i += 1
            return count

    result, _, _ = _simulate.py_func(
        np.zeros((1, 3)),
        np.array([0.4, 0.3]),
        np.array([[1, 1]]),
        np.ones((1, 3), bool),
        1,
        Replay(),
    )
    assert result.ravel().tolist() == [2, 0, 1]
    rng = np.random.default_rng(1)
    drive = np.full((5, 300), np.log(0.1))
    valid = np.ones(drive.shape, dtype=bool)
    counts, failed, maximum = simulate(
        drive, np.zeros(3), np.zeros((5, 3)), valid, 200, rng
    )
    assert not (failed >= 0).any() and abs(counts.mean() - 0.1) < 0.003
    assert np.isclose(maximum, 0.1)
    counts, failed, _ = simulate(
        np.zeros((1, 4)),
        np.array([20.0]),
        np.ones((1, 1)),
        np.ones((1, 4), bool),
        1,
        rng,
    )
    assert failed[0, 0] == 0 and np.isnan(counts).all()
    valid[:, 100:] = False
    counts, _, _ = simulate(drive, np.zeros(3), np.zeros((5, 3)), valid, 2, rng)
    assert np.isnan(counts[:, :, 100:]).all()
    assert isi_histogram(np.array([[1, 2, 0, 1]]))[:3].tolist() == [1, 1, 1]
    print(
        "Dynamics self-check passed (Poisson mean, failure detection, censoring, ISIs)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/glm/dynamics"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--units", choices=("all", "sample"), default="all")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.root is None or args.repeats < 2:
            parser.error("--root and at least two repeats are required")
        run(args.root, args.out, args.repeats, args.units)


if __name__ == "__main__":
    main()
