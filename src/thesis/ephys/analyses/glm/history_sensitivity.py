"""Does stabilizing spike history change the flash × exit conclusion?

Paired sensitivity analysis of the same saved flash-excited units, main-session
trials, measured flash pairs, first-flash alignment, baseline and peak windows
as flash_movement.py. The sampling unit is the neuron. Keep task/video columns,
trial folds, and each fold's original overall alpha fixed. Select a history
penalty multiplier (10,100,1000) by inner-validation Poisson loss among candidates
with no runaway in five generated-history repeats of those validation trials.
If none passes, retain the strongest factor as an explicitly failed attempt.
Refit on the outer fitting trials. Compare with the saved model and a refitted
no-history model at the same overall alpha. This isolates history sensitivity;
it is not a new optimum over all penalties or independent confirmation.

Use the native raw-column centering correction when removing flash × exit.
The primary measure is the existing log peak-ratio contribution minus a new
variant-specific held-out null bias. A nonpositive peak excludes that unit's
log measure; the normalized peak-change measure retains such units. Preserve
paired units when comparing each variant with the current model and report all omissions.
The last grid bin is excluded uniformly as in dynamics.py. Task periods are
not measured locomotion. No existing fits or correction files are overwritten.

Null calibration: final refit at median fold alpha/history multiplier, with
raw flash × exit removed; generate Poisson spikes recursively without clipping,
initialized from real pre-window history. Stop above 10 expected spikes/bin.
Refit each null replicate in the same folds at the real selected penalties;
do not retune penalties in null fits, matching power_test.py's convention.
Use each replicate's observed peaks to select measurement windows. Null bias
is the median over replicates, with Monte Carlo uncertainty saved separately.
Any failed null trajectory makes that calibration unavailable, never truncated.

    uv run python -m thesis.ephys.analyses.glm.history_sensitivity \\
        --root <main-session-parent> --out results/glm/history_sensitivity --workers 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import lfilter
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, dynamics, fit, leak_test
from thesis.ephys.analyses.glm.results import raw_contribution, zero_raw_columns
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, MODEL_COLOR, save_figure

FACTORS = (10.0, 100.0, 1000.0)
VARIANTS = ("current", "stronger history", "no history")
SEED = 20261117
SESSIONS = ("GRB006_20240821_121447", "GRB064_20261008_143359")


def fitted(X, y, width, alpha, factor):
    """Fit scaled history columns, returning coefficients in original coordinates."""
    if factor is None:
        model = fit.fit_poisson_at_alpha(X[:, :width], y, alpha)
        weights = np.r_[model.coef_, np.zeros(X.shape[1] - width)]
    else:
        scaled = X.copy()
        scaled[:, width:] /= np.sqrt(factor)
        model = fit.fit_poisson_at_alpha(scaled, y, alpha)
        weights = model.coef_.astype(float)
        weights[width:] /= np.sqrt(factor)
    return {
        "intercept": float(model.intercept_),
        "coefficients": weights.tolist(),
        "alpha": float(alpha),
        "history_penalty": factor,
    }


def generated(record, raw, rows, trial, counts, repeats, seed):
    """Simulate one unit/fold using its fitting-row normalization and raw weights."""
    shared = leak_test._SHARED
    mean, scale = raw[rows].mean(0), raw[rows].std(0)
    if np.any(scale <= 0):
        raise ValueError("History columns must vary on the fitting rows.")
    width = len(record["coefficients"]) - dmat.HISTORY_COLUMNS
    weights = np.asarray(record["coefficients"])
    grid = counts.shape
    common = shared["common"][:, :width].reshape(*grid, width)[trial]
    drive = (
        record["intercept"]
        + common @ weights[:width]
        - (mean / scale) @ weights[width:]
    )
    kernel = dmat.spike_history_basis().basis @ (weights[width:] / scale)
    return dynamics.simulate(
        drive,
        kernel,
        shared["initial"][trial],
        shared["mask"][trial],
        repeats,
        np.random.default_rng(seed),
    )


def fold_peaks(records, raw, counts, width, native=False):
    """Measure the unchanged fixed-history flash × exit attribution on held-out trials."""
    s = leak_test._SHARED
    logs = np.full(s["pair_rows"].sum(), np.nan)
    contribution = logs.copy()
    pair_indices = np.flatnonzero(s["pair_rows"])
    parts = s["original_parts"] if native else s["parts"]
    for part, record in zip(parts, records, strict=True):
        scaled, _, _ = dmat.training_zscore(raw, part["fit"])
        here = part["test"][pair_indices]
        rows = part["test"] & s["pair_rows"]
        X = dmat._design_with_history(s["common"], scaled, width, rows)
        weights = np.asarray(record["coefficients"])
        logs[here] = record["intercept"] + X @ weights
        contribution[here] = raw_contribution(
            X, weights, s["metadata"], s["removed_columns"]
        )
    if not np.isfinite(logs).all():
        raise ValueError("A paired window is missing its held-out model.")
    return leak_test.peaks_from(logs, contribution, counts)


def raw_history(initial, counts):
    """Use the same native history basis with lag 1 and observed pre-window state."""
    basis = dmat.spike_history_basis().basis
    extended = np.concatenate((initial, counts), axis=1)
    return np.stack(
        [lfilter(np.r_[0, b], [1], extended, axis=1)[:, len(basis) :] for b in basis.T],
        axis=-1,
    ).reshape(-1, dmat.HISTORY_COLUMNS)


def setup(root, unit):
    leak_test._load(root / "stimulus_windows_me.npz", root / "common_design_me.npy")
    s = leak_test._SHARED
    p = s["prepared"]
    grid = (len(p["alignments"]), len(p["relative_centers"]))
    valid = s["valid"].copy().reshape(grid)
    valid[:, -1] = False
    s["valid"], s["mask"] = valid.ravel(), valid
    if np.any(s["pair_rows"] & ~s["valid"]):
        raise ValueError("A flash pair includes an excluded final grid bin.")
    spikes = np.asarray(
        fetch_unit_table(
            p["metadata"]["subject_name"],
            p["metadata"]["session_name"],
            1,
            0,
            include_metrics=False,
        )
        .set_index("unit_id")
        .loc[unit, "spike_times_s"],
        dtype=float,
    )
    counts = fit.build_unit_counts(p["alignments"], spikes).reshape(grid)
    raw = dmat.build_unit_history(p["alignments"], spikes)
    initial = dynamics.history_counts(
        p["alignments"],
        spikes,
        p["relative_centers"],
        len(dmat.spike_history_basis().basis),
    )
    np.testing.assert_allclose(
        raw_history(initial, counts)[s["valid"]], raw[s["valid"]], atol=1e-8
    )
    s["initial"] = initial
    original_parts = fit.cross_validation_partitions(*grid, dmat._valid_bin_mask(p))
    s["parts"] = [
        part
        for part in fit.cross_validation_partitions(*grid, s["valid"])
        if counts.ravel()[part["test"]].sum() > 0
    ]
    original = json.loads(
        (root / "all_fit_me/checkpoints" / f"unit_{unit}_folds.json").read_text()
    )
    original_scored = [
        part for part in original_parts if counts.ravel()[part["test"]].sum() > 0
    ]
    s["original_parts"] = original_scored
    if len(s["parts"]) != len(original_scored):
        raise ValueError("Boundary exclusion changed the scored fold set.")
    width = int(s["metadata"]["base_columns"]) + int(original["components"])
    for part, record in zip(original_scored, original["folds"], strict=True):
        h, _, _ = dmat.training_zscore(raw, part["fit"])
        X = dmat._design_with_history(s["common"], h, width, part["test"])
        score = fit.poisson_metrics(
            counts.ravel()[part["test"]],
            np.exp(record["intercept"] + X @ record["coefficients"]),
            record["fit_mean_count"],
        )
        np.testing.assert_allclose(
            score["bits_per_spike"], record["test"]["bits_per_spike"], atol=1e-6
        )
    return counts, raw, original, width


def real_fits(raw, counts, original, width, unit):
    s = leak_test._SHARED
    records = {"current": original["folds"], "stronger history": [], "no history": []}
    details = []
    failures = {name: 0 for name in VARIANTS}
    trajectories = 0
    scores = {name: [] for name in VARIANTS}
    for index, (part, old) in enumerate(
        zip(s["parts"], original["folds"], strict=True)
    ):
        h, _, _ = dmat.training_zscore(raw, part["fit"])
        X = dmat._design_with_history(s["common"], h, width)
        alpha = old["alpha"]
        candidates = []
        validation_trials = part["inner"].reshape(counts.shape).any(1)
        for factor in FACTORS:
            candidate = fitted(
                X[part["train"]], counts.ravel()[part["train"]], width, alpha, factor
            )
            predicted = np.exp(
                candidate["intercept"] + X[part["inner"]] @ candidate["coefficients"]
            )
            _, failed, _ = generated(
                candidate,
                raw,
                part["fit"],
                validation_trials,
                counts,
                5,
                [SEED, unit, index, int(factor)],
            )
            candidates.append(
                {
                    "factor": factor,
                    "nll": fit.poisson_nll(counts.ravel()[part["inner"]], predicted),
                    "failed": int((failed >= 0).sum()),
                }
            )
        allowed = [c for c in candidates if c["failed"] == 0]
        best = (
            min(allowed, key=lambda c: (c["nll"], -c["factor"]))["factor"]
            if allowed
            else max(FACTORS)
        )
        for name, factor in (("stronger history", best), ("no history", None)):
            records[name].append(
                fitted(
                    X[part["fit"]], counts.ravel()[part["fit"]], width, alpha, factor
                )
            )
        test_trials = part["test"].reshape(counts.shape).any(1)
        for name in VARIANTS:
            r = records[name][index]
            normalization = (
                s["original_parts"][index]["fit"] if name == "current" else part["fit"]
            )
            tested = X[part["test"]]
            if name == "current":
                native, _, _ = dmat.training_zscore(raw, normalization)
                tested = dmat._design_with_history(
                    s["common"], native, width, part["test"]
                )
            prediction = np.exp(r["intercept"] + tested @ r["coefficients"])
            scores[name].append(
                fit.poisson_metrics(
                    counts.ravel()[part["test"]],
                    prediction,
                    counts.ravel()[part["fit"]].mean(),
                )
            )
            _, failed, _ = generated(
                r,
                raw,
                normalization,
                test_trials,
                counts,
                20,
                [SEED, unit, index, VARIANTS.index(name), 1],
            )
            failures[name] += int((failed >= 0).sum())
        trajectories += int(test_trials.sum()) * 20
        details.append(
            {"candidates": candidates, "passed": bool(allowed), "chosen": best}
        )
        print(f"unit {unit} fold {index}: history penalty {best:g}", flush=True)
    return records, {
        "selection": details,
        "failures": failures,
        "trajectories": trajectories,
        "scores": scores,
    }


def unit_job(job):
    root, out = Path(job["root"]), Path(job["out"])
    unit, repeats = job["unit"], job["repeats"]
    out.mkdir(parents=True, exist_ok=True)
    counts, raw, original, width = setup(root, unit)
    s = leak_test._SHARED
    real_path = out / "real.json"
    if real_path.exists():
        real = json.loads(real_path.read_text())
        records = real["records"]
    else:
        records, diagnostics = real_fits(raw, counts, original, width, unit)
        real = {
            "unit_id": unit,
            "records": records,
            "diagnostics": diagnostics,
            "peaks": {
                name: fold_peaks(
                    records[name], raw, counts, width, native=name == "current"
                ).tolist()
                for name in VARIANTS
            },
        }
        saved = json.loads((root.parent / "flash_movement.json").read_text())[
            "sessions"
        ][root.name.split("_")[0]]
        expected = np.asarray(saved["rate_peaks"])[saved["unit_ids"].index(unit), :, 1:]
        np.testing.assert_allclose(
            real["peaks"]["current"], expected, rtol=2e-5, atol=2e-4
        )
        fit._write_json_atomic(real_path, real)
    nulls = {}
    for name in VARIANTS[1:]:
        null_dir = out / name.replace(" ", "_")
        null_dir.mkdir(exist_ok=True)
        h, _, _ = dmat.training_zscore(raw, s["valid"])
        factor = (
            float(np.median([r["history_penalty"] for r in records[name]]))
            if name == "stronger history"
            else None
        )
        final = fitted(
            dmat._design_with_history(s["common"], h, width, s["valid"]),
            counts.ravel()[s["valid"]],
            width,
            float(np.median([r["alpha"] for r in records[name]])),
            factor,
        )
        b, w = zero_raw_columns(
            final["intercept"],
            final["coefficients"],
            s["metadata"],
            s["removed_columns"],
        )
        final.update(intercept=b, coefficients=w.tolist())
        null_peaks = []
        status_path = null_dir / "unstable.json"
        if status_path.exists():
            nulls[name] = None
            continue
        for rep in range(repeats):
            path = null_dir / f"null_{rep:03d}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                simulated, failed, maximum = generated(
                    final,
                    raw,
                    s["valid"],
                    np.ones(counts.shape[0], dtype=bool),
                    counts,
                    1,
                    [SEED, unit, VARIANTS.index(name), rep, 2],
                )
                if (failed >= 0).any():
                    fit._write_json_atomic(
                        status_path,
                        {
                            "replicate": rep,
                            "failed_trajectories": int((failed >= 0).sum()),
                            "max_expected_count": maximum,
                        },
                    )
                    null_peaks = None
                    print(
                        f"UNSTABLE NULL unit {unit}, {name}, replicate {rep}",
                        flush=True,
                    )
                    break
                y = np.nan_to_num(simulated[0])
                history = raw_history(s["initial"], y)
                null_models = []
                for part, record in zip(s["parts"], records[name], strict=True):
                    scaled, _, _ = dmat.training_zscore(history, part["fit"])
                    X = dmat._design_with_history(
                        s["common"], scaled, width, part["fit"]
                    )
                    null_models.append(
                        fitted(
                            X,
                            y.ravel()[part["fit"]],
                            width,
                            record["alpha"],
                            record["history_penalty"],
                        )
                    )
                result = {
                    "peaks": fold_peaks(null_models, history, y, width).tolist(),
                    "max_expected_count": maximum,
                }
                fit._write_json_atomic(path, result)
            null_peaks.append(result["peaks"])
            print(f"unit {unit} {name}: null {rep + 1}/{repeats}", flush=True)
        nulls[name] = null_peaks
    result = {**real, "null_peaks": nulls}
    fit._write_json_atomic(out / "complete.json", result)
    return {"unit_id": unit, "session": root.name}


def summarize(root, out):
    report = {}
    repeats = json.loads((out / "manifest.json").read_text())["null_repeats"]
    source = json.loads((root / "flash_movement.json").read_text())["sessions"]
    rng = np.random.default_rng(SEED)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for ax, session in zip(axes, SESSIONS, strict=True):
            records = [
                json.loads(p.read_text())
                for p in sorted((out / session).glob("unit_*/complete.json"))
            ]
            subject = session.split("_")[0]
            with np.load(root / session / "power_test_held_out.npz") as saved:
                old_peaks = saved["fit_peaks"][:, saved["alphas"].tolist().index(0.0)]
                old_null = leak_test.log_part(old_peaks)[:, :, 0]
                old_bias = {
                    unit: float(np.nanmedian(values))
                    if np.isfinite(values).mean() >= 0.5
                    else np.nan
                    for unit, values in zip(
                        saved["unit_ids"].tolist(), old_null, strict=True
                    )
                }
                old_draws = dict(zip(saved["unit_ids"].tolist(), old_null, strict=True))
                old_normalized = dict(
                    zip(
                        saved["unit_ids"].tolist(),
                        leak_test.comparison_part(old_peaks)[:, :, 0],
                        strict=True,
                    )
                )
            values = []
            mc_values = []
            entry = {}
            for name in VARIANTS:
                actual = np.array(
                    [
                        leak_test.log_part(np.asarray(r["peaks"][name]))[0]
                        for r in records
                    ]
                )
                if name == "current":
                    bias = np.array([old_bias[r["unit_id"]] for r in records])
                    null = np.array([old_draws[r["unit_id"]] for r in records])
                    normalized_bias = np.array(
                        [np.nanmedian(old_normalized[r["unit_id"]]) for r in records]
                    )
                else:
                    null = np.full((len(records), repeats), np.nan)
                    for j, r in enumerate(records):
                        if r["null_peaks"][name] is not None:
                            null[j] = leak_test.log_part(
                                np.asarray(r["null_peaks"][name])
                            )[:, 0]
                    enough = np.isfinite(null).mean(1) >= 0.5
                    bias = np.array(
                        [
                            np.median(a[np.isfinite(a)]) if ok else np.nan
                            for a, ok in zip(null, enough, strict=True)
                        ]
                    )
                    normalized_bias = np.array(
                        [
                            np.nan
                            if r["null_peaks"][name] is None
                            else np.median(
                                leak_test.comparison_part(
                                    np.asarray(r["null_peaks"][name])
                                )[:, 0]
                            )
                            / source[subject]["normalization_rate_spikes_s"][
                                source[subject]["unit_ids"].index(r["unit_id"])
                            ]
                            for r in records
                        ]
                    )
                effects = actual - bias
                mc = np.full((2000, len(records)), np.nan)
                for j, draws in enumerate(null):
                    draws = draws[np.isfinite(draws)]
                    if len(draws) and np.isfinite(effects[j]):
                        mc[:, j] = actual[j] - np.median(
                            rng.choice(draws, (2000, len(draws))), axis=1
                        )
                values.append(effects)
                mc_values.append(mc)
                finite = np.isfinite(effects)
                normalized = (
                    np.array(
                        [
                            leak_test.comparison_part(np.asarray(r["peaks"][name]))[0]
                            / source[subject]["normalization_rate_spikes_s"][
                                source[subject]["unit_ids"].index(r["unit_id"])
                            ]
                            for r in records
                        ]
                    )
                    - normalized_bias
                )
                entry[name] = {
                    "unit_ids": [r["unit_id"] for r in records],
                    "log_effect": np.where(finite, effects, 0).tolist(),
                    "finite": finite.tolist(),
                    "units": int(finite.sum()),
                    "median": float(np.median(effects[finite]))
                    if finite.any()
                    else None,
                    "wilcoxon_p": float(wilcoxon(effects[finite]).pvalue)
                    if finite.any()
                    else None,
                    "uncorrected_log_effect": np.where(
                        np.isfinite(actual), actual, None
                    ).tolist(),
                    "log_bias": np.where(np.isfinite(bias), bias, None).tolist(),
                    "corrected_normalized_effect": np.where(
                        np.isfinite(normalized), normalized, None
                    ).tolist(),
                    "median_normalized_effect": float(np.nanmedian(normalized))
                    if np.isfinite(normalized).any()
                    else None,
                    "normalized_wilcoxon_p": float(
                        wilcoxon(normalized[np.isfinite(normalized)]).pvalue
                    )
                    if np.isfinite(normalized).any()
                    else None,
                    "unavailable_null_calibration_units": [
                        r["unit_id"]
                        for r in records
                        if name != "current" and r["null_peaks"][name] is None
                    ],
                    "mean_bits_per_spike": [
                        float(
                            np.mean(
                                [
                                    f["bits_per_spike"]
                                    for f in r["diagnostics"]["scores"][name]
                                ]
                            )
                        )
                        for r in records
                    ],
                    "failed_trajectories": sum(
                        r["diagnostics"]["failures"][name] for r in records
                    ),
                    "trajectories": sum(
                        r["diagnostics"]["trajectories"] for r in records
                    ),
                }
            main_pair = np.isfinite(values[0]) & np.isfinite(values[2])
            jitter = np.linspace(-0.1, 0.1, len(records))
            for j in np.flatnonzero(main_pair):
                ax.plot(
                    np.arange(3) + jitter[j],
                    [v[j] for v in values],
                    color="0.8",
                    lw=0.5,
                    alpha=0.6,
                    zorder=0,
                )
            for i, (name, effects) in enumerate(zip(VARIANTS, values, strict=True)):
                paired = main_pair & np.isfinite(effects)
                entry[name]["paired_units"] = int(paired.sum())
                entry[name]["paired_unit_ids"] = [
                    r["unit_id"]
                    for r, keep in zip(records, paired, strict=True)
                    if keep
                ]
                if not paired.any():
                    continue
                sampled_units = rng.choice(np.flatnonzero(paired), (2000, paired.sum()))
                y = effects[paired]
                ax.scatter(
                    i + jitter[paired],
                    y,
                    s=12,
                    color=MODEL_COLOR,
                    alpha=0.5,
                )
                ax.plot([i - 0.2, i + 0.2], [np.median(y)] * 2, color=MODEL_COLOR, lw=2)
                entry[name]["paired_median"] = float(np.median(y))
                entry[name]["paired_change_from_current"] = float(
                    np.median(y - values[0][paired])
                )
                entry[name]["paired_wilcoxon_p"] = float(wilcoxon(y).pvalue)
                entry[name]["paired_median_mc95"] = np.quantile(
                    np.median(mc_values[i][:, paired], axis=1), [0.025, 0.975]
                ).tolist()
                draws = mc_values[i][np.arange(2000)[:, None], sampled_units]
                entry[name]["paired_median_bootstrap95"] = np.quantile(
                    np.median(draws, axis=1), [0.025, 0.975]
                ).tolist()
                change = draws - mc_values[0][np.arange(2000)[:, None], sampled_units]
                entry[name]["paired_change_bootstrap95"] = np.quantile(
                    np.median(change, axis=1), [0.025, 0.975]
                ).tolist()
                interval = entry[name]["paired_median_bootstrap95"]
                ax.errorbar(
                    i,
                    np.median(y),
                    yerr=[
                        [np.median(y) - min(interval[0], np.median(y))],
                        [max(interval[1], np.median(y)) - np.median(y)],
                    ],
                    fmt="o",
                    color=MODEL_COLOR,
                    capsize=3,
                )
            entry["paired_units"] = int(main_pair.sum())
            report[subject] = entry
            ax.axhline(0, color="black", lw=0.7)
            ax.set(
                xticks=range(3),
                xticklabels=[
                    f"Current\nn={entry['current']['paired_units']}",
                    f"Stronger history\nn={entry['stronger history']['paired_units']}",
                    f"No history\nn={entry['no history']['paired_units']}",
                ],
                ylabel="Flash × exit log-ratio contribution, minus bias",
            )
            ax.text(
                0.03,
                1.02,
                f"{subject}: {main_pair.sum()} current/no-history pairs",
                va="bottom",
                transform=ax.transAxes,
            )
        for letter, ax in zip("ab", axes, strict=True):
            ax.text(-0.15, 1.05, letter, weight="bold", transform=ax.transAxes)
        fig.savefig(out / "history_sensitivity.png", dpi=160, bbox_inches="tight")
        save_figure(fig, out / "history_sensitivity.svg")
    fit._write_json_atomic(out / "summary.json", report)
    print(json.dumps(report, indent=2))


def self_check():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = rng.poisson(np.exp(-1 + X @ np.array([0.2, -0.1, 0.3, -0.2])))
    r = fitted(X, y, 2, 0.01, 100)
    scaled = X.copy()
    scaled[:, 2:] /= 10
    reference = fit.fit_poisson_at_alpha(scaled, y, 0.01)
    np.testing.assert_allclose(
        np.exp(r["intercept"] + X @ r["coefficients"]),
        reference.predict(scaled),
        rtol=1e-6,
    )
    r = fitted(X, y, 2, 0.01, None)
    assert r["coefficients"][2:] == [0, 0]
    initial = np.zeros((1, len(dmat.spike_history_basis().basis)))
    counts = np.zeros((1, 200))
    counts[0, 5] = 1
    np.testing.assert_allclose(
        raw_history(initial, counts)[6:106], dmat.spike_history_basis().basis
    )
    print("History sensitivity self-check passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("results/glm/history_sensitivity")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--null-repeats", type=int, default=20)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None or args.workers < 1 or args.null_repeats < 2:
        parser.error("--root, positive workers, and at least two null repeats required")
    args.out.mkdir(parents=True, exist_ok=True)
    excited = json.loads((args.root / "flash_excited_units.json").read_text())
    manifest = {
        "code": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "history_sensitivity.py",
                "dynamics.py",
                "fit.py",
                "dmat.py",
                "leak_test.py",
                "flash_movement.py",
            )
        },
        "factors": FACTORS,
        "null_repeats": args.null_repeats,
        "inputs": {},
    }
    manifest["inputs"]["flash_excited_units.json"] = hashlib.sha256(
        (args.root / "flash_excited_units.json").read_bytes()
    ).hexdigest()
    manifest["inputs"]["flash_movement.json"] = hashlib.sha256(
        (args.root / "flash_movement.json").read_bytes()
    ).hexdigest()
    jobs = []
    for session in SESSIONS:
        root = args.root / session
        for name in (
            "stimulus_windows_me.npz",
            "common_design_me.npy",
            "common_design_me.json",
            "power_test_held_out.npz",
        ):
            manifest["inputs"][f"{session}/{name}"] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
        for unit in excited[session]:
            name = f"all_fit_me/checkpoints/unit_{unit}_folds.json"
            manifest["inputs"][f"{session}/{name}"] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
            jobs.append(
                {
                    "root": str(root),
                    "out": str(args.out / session / f"unit_{unit}"),
                    "unit": unit,
                    "repeats": args.null_repeats,
                }
            )
    path = args.out / "manifest.json"
    canonical = json.loads(json.dumps(manifest))
    if path.exists() and json.loads(path.read_text()) != canonical:
        raise ValueError(
            "Existing sensitivity results have different inputs/settings; use another output directory."
        )
    fit._write_json_atomic(path, canonical)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work = {
            pool.submit(unit_job, job): job
            for job in jobs
            if not (Path(job["out"]) / "complete.json").exists()
        }
        for future in as_completed(work):
            job = work[future]
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                failed.append(
                    {
                        "unit": job["unit"],
                        "session": Path(job["root"]).name,
                        "error": repr(error),
                    }
                )
                print(f"FAILED {failed[-1]}", flush=True)
    fit._write_json_atomic(args.out / "failures.json", failed)
    if failed:
        raise RuntimeError(
            f"{len(failed)} units failed; preserved completed checkpoints."
        )
    summarize(args.root, args.out)


if __name__ == "__main__":
    main()
