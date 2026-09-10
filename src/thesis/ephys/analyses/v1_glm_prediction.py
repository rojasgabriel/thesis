"""Compare observed and predicted V1 spike trains on held-out trials.

Scientific comparison
---------------------
For the previously held-out chronological test trials from GRB006 session
20240821_121447, show observed spikes beside one recursive simulation from the
complete sensory, task, audio, history, drift, and video model. Each simulated
row keeps that trial's external covariates fixed. Self-history is updated from
simulated spikes after seeding each trial with its observed pre-window history.

The final panel shows the one-step conditional mean used for held-out deviance
and bits/spike. Unlike the recursive rasters, that prediction conditions on the
observed spike history. Test trials are distinct trials aligned to their first
measured flash, not repeats of one identical stimulus. The displayed mean rates
use 20 ms Gaussian smoothing; all fitting and scoring remain at 1 ms.

The example is selected by a fixed rule as the unit whose full-model test
deviance explained is nearest the population median. This post-fit display
choice does not change any fitted model or reported score. No sensory-response
or visual-quality filter is used. The unit is an observation from one session.
The simulations are offline conditional draws, not causal online forecasts,
because the design includes acausal video and peri-event regressors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.v1_glm import (
    HISTORY_COLUMNS,
    VIDEO_COMPONENT_COUNTS,
    _contiguous_slices,
    _load_windows,
    build_unit_design,
    spike_history_basis,
)
from thesis.ephys.preprocessing.prepare_v1_glm import BINWIDTH_S
from thesis.ephys.units import fetch_unit_table

OBSERVED_COLOR = "0.1"
MODEL_COLOR = "#008695"
SMOOTHING_MS = 20
SIMULATION_SEED = 2008
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def select_representative_result(results: list[dict]) -> tuple[dict, float]:
    """Return the full-model result nearest the population median test score."""
    if not results:
        raise ValueError("No fitted unit results were found.")
    scores = np.asarray(
        [item["plus_video"]["test"]["deviance_explained"] for item in results],
        dtype=float,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Full-model test scores must be finite.")
    median = float(np.median(scores))
    result = min(
        results,
        key=lambda item: (
            abs(item["plus_video"]["test"]["deviance_explained"] - median),
            int(item["unit_id"]),
        ),
    )
    return result, median


def conditional_prediction(
    common: np.ndarray,
    history: np.ndarray,
    history_mean: np.ndarray,
    history_scale: np.ndarray,
    fitted_model: dict,
) -> np.ndarray:
    """Rebuild the saved one-step prediction that uses observed history."""
    coefficients = np.asarray(fitted_model["coefficients"], dtype=float)
    common_columns = len(coefficients) - HISTORY_COLUMNS
    if common_columns <= 0 or common.shape[1] < common_columns:
        raise ValueError("Saved coefficient count does not match the common design.")
    if history.shape != (common.shape[0], HISTORY_COLUMNS):
        raise ValueError("Common and history rows must match.")
    scaled_history = (history - history_mean) / history_scale
    linear_prediction = (
        float(fitted_model["intercept"])
        + common[:, :common_columns] @ coefficients[:common_columns]
        + scaled_history @ coefficients[common_columns:]
    )
    prediction = np.exp(linear_prediction)
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError(
            "Reconstructed conditional predictions must be finite and positive."
        )
    return prediction


def simulation_history_kernel(
    fitted_model: dict, history_scale: np.ndarray
) -> np.ndarray:
    """Collapse scaled history coefficients into one 100-lag spike filter."""
    coefficients = np.asarray(fitted_model["coefficients"], dtype=float)
    history_coefficients = coefficients[-HISTORY_COLUMNS:] / history_scale
    return spike_history_basis().basis @ history_coefficients


def simulate_spike_counts(
    observed_counts: np.ndarray,
    conditional_mean: np.ndarray,
    history_kernel: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw spike counts while replacing observed with simulated self-history."""
    observed = np.asarray(observed_counts, dtype=float)
    conditional = np.asarray(conditional_mean, dtype=float)
    kernel = np.asarray(history_kernel, dtype=float)
    if observed.ndim != 2 or observed.shape != conditional.shape:
        raise ValueError(
            "Observed counts and conditional means must be trial-by-bin arrays."
        )
    if np.any(observed < 0) or not np.isfinite(observed).all():
        raise ValueError("Observed counts must be finite and nonnegative.")
    if np.any(conditional <= 0) or not np.isfinite(conditional).all():
        raise ValueError("Conditional means must be finite and positive.")
    if kernel.ndim != 1 or not np.isfinite(kernel).all():
        raise ValueError("The history kernel must be one finite vector.")

    simulated = np.zeros(observed.shape, dtype=np.int64)
    difference = np.zeros(observed.shape, dtype=float)
    log_conditional = np.log(conditional)
    for bin_index in range(observed.shape[1]):
        lag_count = min(bin_index, len(kernel))
        if lag_count:
            recent_difference = difference[:, bin_index - lag_count : bin_index]
            correction = recent_difference @ kernel[:lag_count][::-1]
        else:
            correction = 0.0
        with np.errstate(over="raise", invalid="raise"):
            try:
                mean = np.exp(log_conditional[:, bin_index] + correction)
            except FloatingPointError as error:
                raise RuntimeError(
                    "Recursive simulation became unstable; no rate clipping was applied."
                ) from error
        if not np.isfinite(mean).all():
            raise RuntimeError(
                "Recursive simulation became unstable; no rate clipping was applied."
            )
        try:
            simulated[:, bin_index] = rng.poisson(mean)
        except ValueError as error:
            raise RuntimeError(
                "Recursive simulation became unstable; no rate clipping was applied."
            ) from error
        difference[:, bin_index] = simulated[:, bin_index] - observed[:, bin_index]
    return simulated


def _raster_events(counts: np.ndarray, times: np.ndarray) -> list[np.ndarray]:
    return [np.repeat(times, row.astype(int, copy=False)) for row in np.asarray(counts)]


def _save_figure(figure, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output.with_suffix(".pdf")
    png_path = output.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def plot_population_summary(
    results: list[dict], selections: list[dict], output: Path
) -> tuple[Path, Path]:
    """Show camera-PC selection and complete-model held-out performance."""
    selected_components = {item["plus_video"]["components"] for item in results}
    if len(selected_components) != 1:
        raise ValueError("All final fits must use one camera-PC count.")
    selected = selected_components.pop()
    component_counts = (0, *VIDEO_COMPONENT_COUNTS)
    validation_deviance = [
        np.mean(
            [
                item["baseline"]["validation"]["deviance_explained"]
                for item in selections
            ]
        ),
        *[
            np.mean(
                [
                    item["plus_video"][str(count)]["validation"]["deviance_explained"]
                    for item in selections
                ]
            )
            for count in VIDEO_COMPONENT_COUNTS
        ],
    ]
    full_results = [item["plus_video"]["test"] for item in results]
    deviance = np.asarray([item["deviance_explained"] for item in full_results])
    bits = np.asarray([item["bits_per_spike"] for item in full_results])
    observed = np.asarray([item["observed_spikes"] for item in full_results])
    predicted = np.asarray([item["predicted_spikes"] for item in full_results])

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
        axes = axes.ravel()

        positions = np.arange(len(component_counts))
        axes[0].plot(positions, validation_deviance, color=MODEL_COLOR, marker="o")
        selected_position = component_counts.index(selected)
        axes[0].scatter(
            selected_position,
            validation_deviance[selected_position],
            color=MODEL_COLOR,
            s=75,
            zorder=3,
        )
        axes[0].text(
            selected_position,
            validation_deviance[selected_position],
            "selected",
            color=MODEL_COLOR,
            ha="center",
            va="bottom",
        )
        axes[0].set_xticks(positions, component_counts)
        axes[0].set_xlabel("Candidate camera PCs")
        axes[0].set_ylabel("Mean validation deviance explained")

        for axis, values, xlabel, digits in (
            (axes[1], deviance, "Test deviance explained", 3),
            (axes[2], bits, "Test prediction (bits/spike)", 3),
        ):
            median = float(np.median(values))
            axis.hist(values, bins=24, color=MODEL_COLOR, alpha=0.8)
            axis.axvline(0, color="0.65", linestyle="--", linewidth=0.8)
            axis.axvline(median, color=MODEL_COLOR, linewidth=1.5)
            axis.text(
                0.98,
                0.93,
                f"median = {median:.{digits}f}",
                color=MODEL_COLOR,
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Units")

        lower = float(min(observed.min(), predicted.min()) * 0.8)
        upper = float(max(observed.max(), predicted.max()) * 1.2)
        axes[3].scatter(observed, predicted, color=MODEL_COLOR, alpha=0.65, s=16)
        axes[3].plot([lower, upper], [lower, upper], color="0.65", linestyle="--")
        axes[3].set(
            xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper)
        )
        axes[3].set_xlabel("Observed test spikes")
        axes[3].set_ylabel("Predicted test spikes")
        axes[3].text(
            0.04,
            0.94,
            (
                f"complete model\n{predicted.sum():,.0f} predicted / "
                f"{observed.sum():,.0f} observed"
            ),
            color=MODEL_COLOR,
            transform=axes[3].transAxes,
            ha="left",
            va="top",
        )

        for letter, axis in zip("abcd", axes, strict=True):
            axis.text(
                -0.14,
                1.05,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.tight_layout(h_pad=2.0, w_pad=2.0)
        return _save_figure(figure, output)


def plot_prediction_figure(
    relative_times: np.ndarray,
    observed: np.ndarray,
    simulation: np.ndarray,
    prediction: np.ndarray,
    result: dict,
    population_median: float,
    output: Path,
) -> tuple[Path, Path]:
    """Write the Pillow-style raster and conditional-rate comparison."""
    raster_data = (observed, simulation)
    raster_labels = (
        "observed held-out spikes",
        "model-predicted spike trains",
    )
    raster_colors = (OBSERVED_COLOR, MODEL_COLOR)
    trial_count = observed.shape[0]

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(7.1, 6.0),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1.35], "hspace": 0.28},
        )
        for axis, counts, label, color in zip(
            axes[:2], raster_data, raster_labels, raster_colors, strict=True
        ):
            axis.eventplot(
                _raster_events(counts, relative_times),
                colors=color,
                lineoffsets=np.arange(1, trial_count + 1),
                linelengths=0.8,
                linewidths=0.35,
            )
            axis.set_ylim(0.5, trial_count + 0.5)
            axis.set_yticks([1, (trial_count + 1) // 2, trial_count])
            axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
            axis.text(
                0.01,
                1.02,
                label,
                color=color,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
        axes[1].set_ylabel("Test trials")
        axes[0].text(
            0.99,
            1.02,
            (
                f"median-performance unit {int(result['unit_id'])}: "
                f"test $D^2$={result['plus_video']['test']['deviance_explained']:.3f}; "
                f"population median={population_median:.3f}"
            ),
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            color="0.3",
            fontsize=7,
        )

        sigma_bins = SMOOTHING_MS / (BINWIDTH_S * 1000)
        rates = [
            gaussian_filter1d(values.mean(axis=0) / BINWIDTH_S, sigma_bins)
            for values in (observed, prediction)
        ]
        rate_axis = axes[2]
        for rate, color in zip(rates, raster_colors, strict=True):
            rate_axis.plot(relative_times, rate, color=color, linewidth=1.25)
        rate_axis.axvline(0, color="0.75", linewidth=0.8, zorder=0)
        rate_axis.set_ylabel("Mean rate (spikes/s)")
        rate_axis.set_xlabel("Time from first measured flash (s)")
        rate_axis.text(
            0.01,
            0.94,
            "one-step prediction (uses observed recent spikes)",
            transform=rate_axis.transAxes,
            ha="left",
            va="top",
            color="0.3",
            fontsize=7,
        )
        for y, label, color in (
            (0.86, "observed", OBSERVED_COLOR),
            (0.77, "complete model", MODEL_COLOR),
        ):
            rate_axis.text(
                0.99,
                y,
                label,
                color=color,
                transform=rate_axis.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
            )
        rate_axis.text(
            0.99,
            0.03,
            f"{SMOOTHING_MS} ms Gaussian smoothing for display only",
            transform=rate_axis.transAxes,
            ha="right",
            va="bottom",
            color="0.4",
            fontsize=7,
        )

        edges = (
            relative_times[0] - BINWIDTH_S / 2,
            relative_times[-1] + BINWIDTH_S / 2,
        )
        axes[-1].set_xlim(*edges)
        for letter, axis in zip("abc", axes, strict=True):
            axis.text(
                -0.075,
                1.02,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        figure.align_ylabels(axes)
        return _save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("figures/v1_glm/stimulus_windows.npz"),
    )
    parser.add_argument(
        "--design", type=Path, default=Path("figures/v1_glm/common_design.npy")
    )
    parser.add_argument("--fit-dir", type=Path, default=Path("figures/v1_glm/all_fit"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/v1_glm/all_fit/predicted_spike_trains"),
    )
    args = parser.parse_args()

    results = []
    for path in sorted(args.fit_dir.glob("unit_*_test.json")):
        with path.open() as handle:
            results.append(json.load(handle))
    selections = []
    for path in sorted(args.fit_dir.glob("unit_*_validation.json")):
        with path.open() as handle:
            selections.append(json.load(handle))
    if len(results) != len(selections):
        raise ValueError("Final and validation result counts differ.")
    summary_pdf, summary_png = plot_population_summary(
        results, selections, args.fit_dir / "summary"
    )
    result, population_median = select_representative_result(results)
    unit_id = int(result["unit_id"])
    selection = next(item for item in selections if int(item["unit_id"]) == unit_id)

    prepared = _load_windows(args.windows)
    _, _, test_rows = _contiguous_slices(prepared["split"])
    with np.load(args.windows, allow_pickle=False) as windows:
        relative_times = windows["relative_bin_centers_s"].copy()
        trial_split = windows["trial_split"].copy()
    test_alignments = prepared["alignments"][trial_split == 2]
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    matching_units = units.loc[units["unit_id"] == unit_id]
    if len(matching_units) != 1:
        raise ValueError(f"Expected one eligible row for unit {unit_id}.")
    spike_times = np.asarray(matching_units.iloc[0]["spike_times_s"], dtype=float)
    counts, history = build_unit_design(test_alignments, spike_times)

    common = np.load(args.design, mmap_mode="r", allow_pickle=False)[test_rows]
    if len(common) != len(counts):
        raise ValueError("Test design and response rows differ.")
    history_mean = np.asarray(selection["history_training_mean"], dtype=float)
    history_scale = np.asarray(selection["history_training_scale"], dtype=float)
    prediction = conditional_prediction(
        common, history, history_mean, history_scale, result["plus_video"]
    )
    saved = result["plus_video"]["test"]
    if not np.isclose(prediction.sum(), saved["predicted_spikes"], rtol=1e-6):
        raise ValueError("Reconstructed predictions differ from the saved fit.")

    trial_count = len(test_alignments)
    bin_count = len(relative_times)
    observed = counts.reshape(trial_count, bin_count)
    conditional = prediction.reshape(trial_count, bin_count)
    simulation = simulate_spike_counts(
        observed,
        conditional,
        simulation_history_kernel(result["plus_video"], history_scale),
        np.random.default_rng(SIMULATION_SEED),
    )
    pdf_path, png_path = plot_prediction_figure(
        relative_times,
        observed,
        simulation,
        conditional,
        result,
        population_median,
        args.output,
    )
    print(
        json.dumps(
            {
                "unit_id": unit_id,
                "selection": "full-model test deviance explained nearest population median",
                "population_median_test_deviance_explained": population_median,
                "unit_test_deviance_explained": result["plus_video"]["test"][
                    "deviance_explained"
                ],
                "test_trials": trial_count,
                "observed_spikes": int(observed.sum()),
                "simulated_spikes": int(simulation.sum()),
                "simulation_seed": SIMULATION_SEED,
                "summary_pdf": str(summary_pdf),
                "summary_png": str(summary_png),
                "pdf": str(pdf_path),
                "png": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
