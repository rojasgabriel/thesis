"""Check that the V1 GLM pipeline recovers known effects from synthetic spikes.

Question: does `glm prepare`, `fit`, and `unique` recover kernels and regressor
contributions that were planted in simulated data? This needs no database. The
check replaces the trial table and the unit table with a synthetic session,
then runs the real pipeline code end to end in a temporary directory.

Synthetic session: 80 trials, each aligned to its first flash, with
initiation, a 30–80 ms flash train, withdrawal, and response entry. Video has
5 motion-energy PCs; only PC 1 drives spikes. The pipeline compares 2 and 5
PCs instead of the fixed 10, so the check also covers the PC-count selection.

The ground truth is written in continuous time from the event times, not from
the design matrix, so a wrong event alignment or a non-causal basis fails the
check. Units (all 20 spikes/s baseline, 1 ms bins):

- 101: flash only, with a 2 ms refractory history.
- 102: motion-energy PC 1 only.
- 103: withdrawal only.
- 104: constant rate, no history (null unit).
- 105: stimulus during hold only: a drop of the rate from the first flash to
  the withdrawal.
- 197: every effect. Its figures are drawn if it is selected as the example unit.

Bins after response entry have 3x the rate. The pipeline masks them, so they
must not change the fitted kernels. Sampling unit is the unit. Each unit is
checked on its cross-validated D², its final-refit kernels, and its unique
deviance. Structural checks cover bugs that the planted effects cannot show:
video PC 1 must land on bin centers, the common design must be z-scored on
training rows, a saved fold must replay with history scaled on its fit rows,
and the unique-deviance shuffle must stay within trial.

Mutation tests on 2026-10-06 caught a +30 ms flash shift, a non-causal history,
a disabled response-entry mask, withdrawal placed at initiation, an identity
or across-trial shuffle, an uncentered adaptation term, a 50 ms video lag, and
z-scoring on all rows.

    uv run python -m thesis.ephys.analyses.glm.synthetic_check [--out DIR]

The command raises AssertionError when a check fails. Use --out to keep the
artifacts. It writes the planted and the recovered kernels of each unit to
`synthetic_recovery.json` there; `model_checks.py` draws them.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.ephys.analyses.glm import cli, dmat, fit, results, unique_dev

SEED = 7
TRIALS = 80
FRAME_RATE_HZ = 60.0
BASE_RATE_HZ = 20.0
PC_COUNTS = (2, 5)
FLASH_PEAK_S = 0.04
FLASH_GAIN = 1.5
ADAPT_GAIN = -0.6  # flash kernel scale after a flash in the last 50 ms (centered)
WITHDRAWAL_PEAK_S = 0.05
WITHDRAWAL_GAIN = 1.5
VIDEO_GAIN = 0.4
HOLD_GAIN = -0.6
HOLD_TAU_S = 0.08
POST_RESPONSE_GAIN = np.log(3)
UNITS = {
    101: {"flash", "history"},
    102: {"video"},
    103: {"withdrawal"},
    104: set(),
    105: {"hold"},
    197: {"flash", "adapt", "hold", "withdrawal", "video", "history"},
}


def flash_kernel(lag: np.ndarray) -> np.ndarray:
    """Alpha-function log-rate kernel, zero before the flash and after 150 ms."""
    lag = np.asarray(lag, dtype=float)
    shape = (lag / FLASH_PEAK_S) * np.exp(1 - lag / FLASH_PEAK_S)
    return np.where((lag >= 0) & (lag <= 0.15), FLASH_GAIN * shape, 0.0)


def withdrawal_kernel(lag: np.ndarray) -> np.ndarray:
    """Gaussian log-rate bump after withdrawal, inside the causal support."""
    lag = np.asarray(lag, dtype=float)
    bump = WITHDRAWAL_GAIN * np.exp(-0.5 * ((lag - WITHDRAWAL_PEAK_S) / 0.03) ** 2)
    return np.where((lag >= 0) & (lag <= dmat.WITHDRAWAL_WINDOW_S), bump, 0.0)


def hold_kernel(lag: np.ndarray) -> np.ndarray:
    """Log-rate drop that starts at the first flash and stays for the hold."""
    lag = np.asarray(lag, dtype=float)
    return np.where(lag >= 0, HOLD_GAIN * (1 - np.exp(-lag / HOLD_TAU_S)), 0.0)


def history_kernel(lag_bins: np.ndarray) -> np.ndarray:
    """Refractory log-rate kernel at strictly past lags of 1..100 bins."""
    lag_bins = np.asarray(lag_bins, dtype=float)
    return np.where(lag_bins <= 2, -4.0, -1.5 * np.exp(-(lag_bins - 2) / 5.0))


def synthetic_trials(rng: np.random.Generator) -> pd.DataFrame:
    """Return a trial table with the columns the GLM reads."""
    rows = []
    for index in range(TRIALS):
        first = 10.0 + 4.0 * index
        gaps = rng.uniform(0.03, 0.08, size=40)
        flashes = first + np.concatenate(([0.0], np.cumsum(gaps)))
        flashes = flashes[flashes <= first + 1.0]
        # Some holds must be longer than 845 ms, or the last hold basis is
        # zero in every training row.
        exit_s = first + rng.uniform(0.35, 0.93)
        rows.append(
            {
                "trial_num": index + 1,
                "response": 1 if index % 2 else -1,
                "early_withdrawal": 0,
                "stim_pulse_times_s": flashes.tolist(),
                "center_entry_s": first - rng.uniform(0.03, 0.1),
                "center_exit_s": exit_s,
                "response_port_entry_s": exit_s + rng.uniform(0.3, 0.8),
            }
        )
    return pd.DataFrame(rows)


def write_windows(trials: pd.DataFrame, frames: np.ndarray, output: Path) -> None:
    """Write the window file the way `dmat.write_stimulus_windows` does."""
    first = trials["stim_pulse_times_s"].str[0].to_numpy(dtype=float)
    bins = dmat.trial_bins(first)
    rows = np.full(len(frames), -1, dtype=int)
    for i, (start, stop) in enumerate(
        zip(bins["window_start_s"], bins["window_stop_s"], strict=True)
    ):
        lo = np.searchsorted(frames, start, side="right") - 1
        hi = np.searchsorted(frames, stop)
        rows[lo : hi + 1] = i
    split = np.full(len(frames), -1, dtype=np.int8)
    split[rows >= 0] = bins["trial_split"][rows[rows >= 0]]
    metadata = {
        "subject_name": "SYN001",
        "session_name": "synthetic",
        "video_aligned": True,
    }
    np.savez_compressed(
        output,
        metadata_json=json.dumps(metadata),
        trial_num=trials["trial_num"].to_numpy(),
        eligible_trials=np.ones(len(trials), dtype=bool),
        selected_trial_rows=np.arange(len(trials)),
        first_stim_s=first,
        center_entry_s=trials["center_entry_s"].to_numpy(dtype=float),
        center_exit_s=trials["center_exit_s"].to_numpy(dtype=float),
        response_entry_s=trials["response_port_entry_s"].to_numpy(dtype=float),
        frame_times_s=frames,
        frame_trial_row=rows,
        frame_split=split,
        **bins,
    )


def write_video(windows: Path, output: Path, rng: np.random.Generator) -> np.ndarray:
    """Write 5 z-scored motion PCs for trial frames; return the PC 1 trace.

    PC 1 is also saved as the total motion energy of each frame.
    """
    with np.load(windows) as data:
        keep = np.flatnonzero(data["frame_split"] >= 0)
        times = data["frame_times_s"][keep]
        trial_rows = data["frame_trial_row"][keep]
        split = data["frame_split"][keep]
    smooth = np.convolve(
        rng.standard_normal(len(times) + 30), np.ones(30) / 30, "valid"
    )
    scores = rng.standard_normal((len(times), max(PC_COUNTS)))
    scores[:, 0] = smooth[: len(times)]
    scores, _, _ = dmat.training_zscore(scores, split == 0)
    np.savez_compressed(
        output,
        frame_times_s=times,
        frame_trial_row=trial_rows,
        frame_split=split,
        scores=scores,
        total=scores[:, 0],
    )
    return np.column_stack((times, scores[:, 0]))


def simulate_unit(
    effects: set[str],
    trials: pd.DataFrame,
    centers: np.ndarray,
    video: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return spike times from the continuous-time ground truth."""
    # Select flashes here rather than through dmat.flash_events, so a timing or
    # selection bug in the pipeline cannot also shift the ground truth.
    modeled = [
        np.asarray(row.stim_pulse_times_s)[
            (np.asarray(row.stim_pulse_times_s) >= row.center_entry_s)
            & (np.asarray(row.stim_pulse_times_s) <= row.response_port_entry_s)
        ]
        for row in trials.itertuples()
    ]
    # 1 for a flash that follows another within 50 ms, computed here and not
    # with dmat.flash_recent.
    resources = [np.r_[0.0, np.diff(times) <= 0.05] for times in modeled]
    center = np.concatenate(resources).mean()
    history = history_kernel(np.arange(1, 101))
    spikes = []
    for index, row in enumerate(trials.itertuples()):
        alignment = row.stim_pulse_times_s[0]
        times = alignment + centers
        drive = np.full(len(centers), np.log(BASE_RATE_HZ * dmat.BINWIDTH_S))
        for flash, r in zip(modeled[index], resources[index] - center, strict=True):
            scale = ("flash" in effects) + ADAPT_GAIN * r * ("adapt" in effects)
            drive += scale * flash_kernel(times - flash)
        if "withdrawal" in effects:
            drive += withdrawal_kernel(times - row.center_exit_s)
        if "hold" in effects:
            drive += hold_kernel(centers) * (times < row.center_exit_s)
        if "video" in effects:
            drive += VIDEO_GAIN * np.interp(times, video[:, 0], video[:, 1])
        drive += POST_RESPONSE_GAIN * (centers > row.response_port_entry_s - alignment)
        if "history" not in effects:
            counts = rng.poisson(np.exp(drive))
        else:
            counts = np.zeros(len(centers), dtype=int)
            for t in range(len(centers)):
                past = counts[max(0, t - 100) : t][::-1]
                counts[t] = rng.poisson(np.exp(drive[t] + history[: len(past)] @ past))
        offsets = rng.uniform(-0.4, 0.4, size=counts.sum()) * dmat.BINWIDTH_S
        spikes.append(np.repeat(times, counts) + offsets)
    return np.sort(np.concatenate(spikes))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="Keep artifacts in this directory.")
    args = parser.parse_args()
    root = args.out or Path(tempfile.mkdtemp(prefix="glm_synthetic_"))
    # Fits reuse checkpoints by file presence, so old artifacts would be scored.
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; pass a new --out directory.")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    trials = synthetic_trials(rng)
    end = trials["response_port_entry_s"].max() + 5
    frames = np.arange(0, end, 1 / FRAME_RATE_HZ) + rng.uniform(0, 1 / FRAME_RATE_HZ)
    model = cli.PoissonGLM(subject="SYN001", session="synthetic", root=root)
    write_windows(trials, frames, model.windows)
    video = write_video(model.windows, model.video, rng)
    centers, _, _ = dmat.construct_timebins(dmat.PRE_S, dmat.POST_S, dmat.BINWIDTH_S)
    units = pd.DataFrame(
        [
            {
                "unit_id": unit_id,
                "depth": float(unit_id),
                "spike_times_s": simulate_unit(effects, trials, centers, video, rng),
            }
            for unit_id, effects in UNITS.items()
        ]
    )

    # Swap the database reads for the synthetic session and shrink the PC grid.
    dmat.build_trial_table = lambda *_, **__: trials.copy()
    for module in (fit, unique_dev, results):
        module.fetch_unit_table = lambda *_, **__: units.copy()
    for module in (dmat, fit):
        module.VIDEO_COMPONENT_COUNTS = PC_COUNTS

    model.prepare()
    model.fit("all", workers=1)
    model.unique("all", workers=1)

    checkpoints = model.fit_dir("all") / "checkpoints"
    alignments = trials["stim_pulse_times_s"].str[0].to_numpy(dtype=float)
    valid = dmat._valid_bin_mask(dmat._load_windows(model.windows))
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    report, failures, recovery = {}, [], {}

    def check(label: str, passed: bool) -> None:
        report[label] = bool(passed)
        if not passed:
            failures.append(label)

    # Structural checks for bugs that the planted effects cannot reveal.
    common = np.load(model.design, mmap_mode="r")
    pc1 = metadata["column_names"].index("video_pc_001")
    scale, mean = metadata["training_scale"][pc1], metadata["training_mean"][pc1]
    expected = np.interp(dmat._load_windows(model.windows)["bin_times"], *video.T)
    # PC 1 is smooth over 500 ms, so a lagged column would still fit well.
    check(
        "video PC 1 sampled at bin centers",
        np.abs(common[:, pc1] * scale + mean - expected).max() < 1e-3,
    )
    train, _, _ = dmat._split_masks(dmat._load_windows(model.windows)["split"], valid)
    scaled = np.asarray(common[train], dtype=float)
    check(
        "common design z-scored on training rows",
        np.allclose(scaled.mean(0), 0, atol=1e-3)
        and np.allclose(scaled.std(0), 1, atol=1e-3),
    )
    # workers=1 keeps the permutation that `unique` used in this process.
    within = fit._SHARED["within_trial"]
    rows = np.arange(len(within))
    check(
        "shuffle stays within trial",
        np.array_equal(within // len(centers), rows // len(centers)),
    )
    check("shuffle moves valid bins", np.mean(within[valid] != rows[valid]) > 0.9)
    between = fit._SHARED["between_trials"]
    check(
        "trial swap keeps the bin and changes the trial",
        np.array_equal(between % len(centers), rows % len(centers))
        and np.all(between // len(centers) != rows // len(centers))
        and np.array_equal(np.sort(between), rows),
    )
    partitions = fit.cross_validation_partitions(TRIALS, len(centers), valid)

    for unit_id, effects in UNITS.items():
        final = json.loads((checkpoints / f"unit_{unit_id}_final.json").read_text())
        block = json.loads(
            (checkpoints / "unique" / f"unit_{unit_id}.json").read_text()
        )
        spikes = units.loc[units["unit_id"] == unit_id, "spike_times_s"].iloc[0]
        raw_history = dmat.build_unit_history(alignments, spikes)
        _, _, history_scale = dmat.training_zscore(raw_history, valid)
        kernels = results.fitted_kernels(final, metadata, history_scale)
        # Replay the first scored fold with history scaled on its fit rows only.
        folds = json.loads((checkpoints / f"unit_{unit_id}_folds.json").read_text())
        counts = fit.build_unit_counts(alignments, spikes)
        part = next(p for p in partitions if counts[p["test"]].sum() > 0)
        first = folds["folds"][0]
        history, _, _ = dmat.training_zscore(raw_history, part["fit"])
        design = dmat._design_with_history(
            common,
            history,
            metadata["base_columns"] + folds["components"],
            part["test"],
        )
        replay = fit.poisson_metrics(
            counts[part["test"]],
            np.exp(first["intercept"] + design @ np.asarray(first["coefficients"])),
            first["fit_mean_count"],
        )["deviance_explained"]
        check(
            f"{unit_id} fold 1 replays with fit-row scaling",
            abs(replay - first["test"]["deviance_explained"]) < 1e-4,
        )
        unique = {
            name: block["groups"][name]["unique_test_deviance_explained"]
            for name in unique_dev.DETAILED_GROUPS
        }
        d2 = final["cross_validated_deviance_explained"]
        swapped = block["groups"]["video"]["trial_specific_test_deviance_explained"]
        if effects == {"video"}:
            # PC 1 is independent noise in each trial, so all of its effect is
            # trial-specific.
            check(
                f"{unit_id} video effect is trial-specific",
                swapped > 0.7 * unique["video"],
            )
        elif "video" not in effects:
            check(f"{unit_id} trial-specific video near 0", abs(swapped) < 0.005)
        report[unit_id] = {
            "cv_d2": round(d2, 4),
            "unique": {name: round(value, 4) for name, value in unique.items()},
        }
        recovery[unit_id] = {}
        flash_t, flash_v = kernels["visual_flash"]
        if "flash" in effects:
            # Flashes repeat every 30-80 ms, so a timing error smears the kernel
            # rather than shifting it: check shape and peak amplitude together.
            truth = flash_kernel(flash_t / 1000)
            check(f"{unit_id} flash kernel r>0.95", _correlation(flash_v, truth) > 0.95)
            check(
                f"{unit_id} flash peak within 20%",
                abs(flash_v.max() / truth.max() - 1) < 0.2,
            )
            recovery[unit_id]["visual flash"] = (flash_t, flash_v, truth)
        if "adapt" in effects:
            _, adapt_v = kernels["flash_adaptation"]
            truth = ADAPT_GAIN * flash_kernel(flash_t / 1000)
            check(
                f"{unit_id} adaptation kernel r>0.7", _correlation(adapt_v, truth) > 0.7
            )
        if "withdrawal" in effects:
            exit_t, exit_v = kernels["withdrawal"]
            truth = withdrawal_kernel(exit_t / 1000)
            check(
                f"{unit_id} withdrawal kernel r>0.9", _correlation(exit_v, truth) > 0.9
            )
            recovery[unit_id]["withdrawal"] = (exit_t, exit_v, truth)
        if "hold" in effects:
            hold_t, hold_v = kernels[dmat.HOLD]
            truth = hold_kernel(hold_t / 1000)
            seen = hold_t <= 350  # every hold is at least this long
            check(
                f"{unit_id} hold kernel r>0.9",
                _correlation(hold_v[seen], truth[seen]) > 0.9,
            )
            level = (hold_t >= 200) & seen
            check(
                f"{unit_id} hold level within 25%",
                abs(hold_v[level].mean() / truth[level].mean() - 1) < 0.25,
            )
            recovery[unit_id]["stimulus during hold"] = (hold_t, hold_v, truth)
        if "history" in effects:
            lag, values = kernels["history"]
            check(f"{unit_id} refractory history", values[lag <= 3].mean() < -1)
        if effects:
            check(f"{unit_id} cv D2 > 0.002", d2 > 0.002)
        else:
            check(f"{unit_id} null cv D2 near 0", abs(d2) < 0.005)
            check(
                f"{unit_id} null unique near 0",
                max(abs(value) for value in unique.values()) < 0.005,
            )
        drivers = {
            "flash": "visual_flash",
            "video": "video",
            "withdrawal": "withdrawal",
            "hold": dmat.HOLD,
        }
        if len(effects - {"history"}) == 1:
            (effect,) = effects - {"history"}
            top = max((name for name in unique if name != "history"), key=unique.get)
            # Require a positive value too: if every block were 0, max() would
            # return the first key.
            check(
                f"{unit_id} largest unique block is {drivers[effect]}",
                top == drivers[effect] and unique[top] > 0.001,
            )

    figure_dir = model.fit_dir("all")
    check("figure data written", (figure_dir / "figure_data.npz").exists())
    (root / "synthetic_recovery.json").write_text(
        json.dumps(
            {
                str(unit_id): {
                    name: {
                        "time_ms": np.asarray(times).tolist(),
                        "recovered": np.asarray(fitted).tolist(),
                        "planted": np.asarray(truth).tolist(),
                    }
                    for name, (times, fitted, truth) in kernels_of_unit.items()
                }
                for unit_id, kernels_of_unit in recovery.items()
            }
        )
    )

    print(
        json.dumps(
            {"artifacts": str(root), **{str(k): v for k, v in report.items()}}, indent=2
        )
    )
    if failures:
        raise AssertionError(f"{len(failures)} synthetic checks failed: {failures}")
    print("All synthetic checks passed.")


if __name__ == "__main__":
    main()
