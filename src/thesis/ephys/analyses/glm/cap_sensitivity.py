"""How much does the simulation cap affect the flash × exit bias correction?

Change only MAX_COUNT in the original leak_test.simulate. Keep the original
final-refit null weights, zero initial history, complete simulation windows,
history standardization, fold penalties, flash pairs, baseline and peak measure.
Use independent replicate streams, shared across cap conditions within each
replicate. Replicate zero uses the original power_test.SEED and held_out stream
to check the first saved 500-Hz simulation; later replicates use distinct seeds.
Record that first refit against the saved file per unit. Refit the 500-Hz
baseline using the same current fitter as all alternatives. Reuse a refit when
another cap gives identical count arrays; every distinct array is fit with
power_test.held_out_peaks unchanged.

Caps are 250, 500 and 1,000 Hz, plus uncapped generation stopped above 10,000 Hz
or nonfinite intensity. The uncapped correction is unavailable for a unit if
any replicate fails; never estimate it from survivors. Record whether the first
failure is inside a fitted bin or only in the censored tail. An uncapped failure
in that tail does not establish that the fitted interval itself is unstable.

Units are the same 30/22 flash-excited units of the main two sessions; sampling
unit is the neuron. Preserve paired units across caps, report unavailable log
ratios from nonpositive peaks, and retain normalized peak changes separately.
Intervals resample neurons and null replicates, conditional on real-data fits.
This assesses a numerical assumption in an existing correction; no new model
terms, real-data fits, unit selection, or primary results are changed.

    uv run python -m thesis.ephys.analyses.glm.cap_sensitivity \\
        --root <main-session-parent> --out results/glm/cap_sensitivity --workers 32
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, fit, leak_test, power_test
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, MODEL_COLOR, save_figure

SESSIONS = ("GRB006_20240821_121447", "GRB064_20261008_143359")
CAPS = {"250": 0.25, "500": 0.5, "1000": 1.0, "uncapped": np.inf}


class UnstableNull(RuntimeError):
    def __init__(self, audit):
        super().__init__("Uncapped intensity exceeded the stopping threshold.")
        self.audit = audit


class RateAudit:
    """Observe Poisson calls without changing the original random-number stream."""

    def __init__(self, rng, cap, valid, pair):
        self.rng, self.cap = rng, cap
        self.valid, self.pair = valid, pair
        self.t = 0
        self.all_hits = self.valid_hits = self.pair_hits = 0
        self.trials = np.zeros(valid.shape[0], bool)
        self.failure = None

    def poisson(self, rate):
        t = self.t
        self.t += 1
        if not np.isfinite(self.cap):
            bad = ~np.isfinite(rate) | (rate > 10.0)
            if bad.any():
                self.failure = {
                    "bin": t,
                    "trial_rows": np.flatnonzero(bad).tolist(),
                    "valid_trial_rows": np.flatnonzero(bad & self.valid[:, t]).tolist(),
                }
                raise UnstableNull(self.report())
        hits = rate >= self.cap
        self.all_hits += int(hits.sum())
        self.valid_hits += int((hits & self.valid[:, t]).sum())
        self.pair_hits += int((hits & self.pair[:, t]).sum())
        self.trials |= hits & self.valid[:, t]
        return self.rng.poisson(rate)

    def report(self):
        return {
            "all_bins_at_cap": self.all_hits,
            "valid_bins_at_cap": self.valid_hits,
            "pair_bins_at_cap": self.pair_hits,
            "valid_trials_at_cap": int(self.trials.sum()),
            "failure": self.failure,
        }


def simulate(record, weights, mean, scale, rng, cap):
    s = leak_test._SHARED
    shape = (len(s["prepared"]["alignments"]), len(s["prepared"]["relative_centers"]))
    audit = RateAudit(
        rng, cap, s["valid"].reshape(shape), s["pair_rows"].reshape(shape)
    )
    with patch.object(leak_test, "MAX_COUNT", cap), np.errstate(over="ignore"):
        counts, history, _ = leak_test.simulate(record, weights, mean, scale, audit)
    return counts, history, audit.report()


def unit_job(job):
    root, out = Path(job["root"]), Path(job["out"])
    unit, repeats = job["unit"], job["repeats"]
    out.mkdir(parents=True, exist_ok=True)
    leak_test._load(root / "stimulus_windows_me.npz", root / "common_design_me.npy")
    s = leak_test._SHARED
    p = s["prepared"]
    checkpoint = root / "all_fit_me/checkpoints"
    record = json.loads((checkpoint / f"unit_{unit}_final.json").read_text())
    folds = json.loads((checkpoint / f"unit_{unit}_folds.json").read_text())
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
    counts = fit.build_unit_counts(p["alignments"], spikes)
    raw = dmat.build_unit_history(p["alignments"], spikes)
    _, mean, scale = dmat.training_zscore(raw, s["valid"])
    width = int(s["metadata"]["base_columns"]) + int(record["components"])
    grid = counts.reshape(len(p["alignments"]), -1)
    paired = s["paired_trials"]
    valid = s["valid"].reshape(grid.shape)
    observed_scale = (grid * valid)[paired].sum() / (
        valid[paired].sum() * dmat.BINWIDTH_S
    )
    saved = iter(folds["folds"])
    alphas = [
        next(saved)["alpha"] if counts[part["test"]].sum() > 0 else record["alpha"]
        for part in fit.cross_validation_partitions(*grid.shape, s["valid"])
    ]
    if next(saved, None) is not None:
        raise ValueError("Saved folds do not match the original trial partitions.")
    intercept, weights = power_test.planted(record, s["metadata"], 0.0)
    generator = {**record, "intercept": intercept}
    with np.load(root / "power_test_held_out.npz") as reference:
        original = reference["fit_peaks"][
            reference["unit_ids"].tolist().index(unit),
            reference["alphas"].tolist().index(0.0),
        ].copy()
    if repeats > len(original):
        raise ValueError(
            "Requested more replicates than the original saved calibration."
        )
    results = []
    for rep in range(repeats):
        path = out / f"replicate_{rep:03d}.json"
        if path.exists():
            result = json.loads(path.read_text())
            results.append(result)
            continue
        seed = [power_test.SEED, unit, power_test.MODES.index("held_out")]
        master = np.random.default_rng(seed + ([rep] if rep else []))
        state = copy.deepcopy(master.bit_generator.state)
        baseline, history, stats = simulate(
            generator, weights, mean, scale, master, 0.5
        )
        replay = (
            power_test.held_out_peaks(baseline, history, width, alphas) / observed_scale
        )
        digest = hashlib.sha256(baseline.tobytes()).hexdigest()
        cache = {digest: replay.tolist()}
        caps = {
            "500": {
                "status": "ok",
                "peaks": replay.tolist(),
                "audit": stats,
                "counts_sha256": digest,
                "reused": False,
            }
        }
        for label, cap in CAPS.items():
            if label == "500":
                continue
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(state)
            try:
                y, h, audit = simulate(generator, weights, mean, scale, rng, cap)
            except UnstableNull as error:
                caps[label] = {
                    "status": "unstable",
                    "peaks": None,
                    "audit": error.audit,
                }
                continue
            key = hashlib.sha256(y.tobytes()).hexdigest()
            reused = key in cache
            if not reused:
                cache[key] = (
                    power_test.held_out_peaks(y, h, width, alphas) / observed_scale
                ).tolist()
            caps[label] = {
                "status": "ok",
                "peaks": cache[key],
                "audit": audit,
                "counts_sha256": key,
                "reused": reused,
            }
        result = {
            "unit_id": unit,
            "replicate": rep,
            "caps": caps,
            "original_500_peak_max_abs_difference": float(
                np.max(np.abs(replay - original[0]))
            )
            if rep == 0
            else None,
            "seed": seed + ([rep] if rep else []),
        }
        fit._write_json_atomic(path, result)
        results.append(result)
        print(f"{root.name} unit {unit}: cap replicate {rep + 1}/{repeats}", flush=True)
    complete = {
        "unit_id": unit,
        "valid_bins": int(s["valid"].sum()),
        "pair_bins": int(s["pair_rows"].sum()),
        "trials": len(grid),
        "replicates": results,
    }
    fit._write_json_atomic(out / "complete.json", complete)
    return {"session": root.name, "unit_id": unit}


def summarize(root, out):
    source = json.loads((root / "flash_movement.json").read_text())["sessions"]
    report = {}
    rng = np.random.default_rng(20261118)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for ax, session in zip(axes, SESSIONS, strict=True):
            rows = [
                json.loads(p.read_text())
                for p in sorted((out / session).glob("unit_*/complete.json"))
            ]
            subject = session.split("_")[0]
            real = source[subject]
            peaks = np.asarray(real["normalized_rate_peaks"])
            actual = np.array(
                [
                    leak_test.log_part(
                        peaks[real["unit_ids"].index(r["unit_id"]), :, 1:]
                    )[0]
                    for r in rows
                ]
            )
            normalized = np.array(
                [
                    leak_test.comparison_part(
                        peaks[real["unit_ids"].index(r["unit_id"]), :, 1:]
                    )[0]
                    for r in rows
                ]
            )
            entry = {"unit_ids": [r["unit_id"] for r in rows], "caps": {}}
            effects, biases, draws_by_cap = {}, {}, {}
            for label in CAPS:
                null = np.full((len(rows), len(rows[0]["replicates"])), np.nan)
                normal = null.copy()
                failed = []
                audits = []
                for j, row in enumerate(rows):
                    replicates = [r["caps"][label] for r in row["replicates"]]
                    audits.extend(r["audit"] for r in replicates)
                    if any(r["status"] != "ok" for r in replicates):
                        failed.append(row["unit_id"])
                        continue
                    a = np.asarray([r["peaks"] for r in replicates])
                    null[j] = leak_test.log_part(a)[:, 0]
                    normal[j] = leak_test.comparison_part(a)[:, 0]
                enough = np.isfinite(null).mean(1) >= 0.5
                bias = np.array(
                    [
                        np.median(a[np.isfinite(a)]) if ok else np.nan
                        for a, ok in zip(null, enough, strict=True)
                    ]
                )
                effect = actual - bias
                effects[label], biases[label], draws_by_cap[label] = effect, bias, null
                normal_bias = np.array(
                    [np.median(a) if np.isfinite(a).all() else np.nan for a in normal]
                )
                entry["caps"][label] = {
                    "bias": np.where(np.isfinite(bias), bias, None).tolist(),
                    "corrected_log_effect": np.where(
                        np.isfinite(effect), effect, None
                    ).tolist(),
                    "corrected_normalized_effect": np.where(
                        np.isfinite(normalized - normal_bias),
                        normalized - normal_bias,
                        None,
                    ).tolist(),
                    "unavailable_simulation_units": failed,
                    "failed_replicates": sum(a["failure"] is not None for a in audits),
                    "first_failure_in_fitted_interval": sum(
                        bool(a["failure"] and a["failure"]["valid_trial_rows"])
                        for a in audits
                    ),
                    "valid_bins_at_cap": sum(a["valid_bins_at_cap"] for a in audits),
                    "valid_trials_at_cap": sum(
                        a["valid_trials_at_cap"] for a in audits
                    ),
                    "pair_bins_at_cap": sum(a["pair_bins_at_cap"] for a in audits),
                    "units_at_cap": [
                        r["unit_id"]
                        for r in rows
                        if any(
                            a["caps"][label]["audit"]["valid_bins_at_cap"]
                            for a in r["replicates"]
                        )
                    ],
                    "valid_exposure_bins": sum(
                        r["valid_bins"] * len(r["replicates"]) for r in rows
                    ),
                    "trial_trajectories": sum(
                        r["trials"] * len(r["replicates"]) for r in rows
                    ),
                    "finite_log_units": int(np.isfinite(effect).sum()),
                    "normalized_median": float(np.nanmedian(normalized - normal_bias))
                    if np.isfinite(normalized - normal_bias).any()
                    else None,
                    "normalized_wilcoxon_p": float(
                        wilcoxon(
                            (normalized - normal_bias)[
                                np.isfinite(normalized - normal_bias)
                            ]
                        ).pvalue
                    )
                    if np.isfinite(normalized - normal_bias).any()
                    else None,
                }
            mc = {label: np.full((2000, len(rows)), np.nan) for label in CAPS}
            for j in range(len(rows)):
                indices = rng.integers(
                    len(rows[j]["replicates"]), size=(2000, len(rows[j]["replicates"]))
                )
                for label in CAPS:
                    if np.isfinite(biases[label][j]):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            mc[label][:, j] = actual[j] - np.nanmedian(
                                draws_by_cap[label][j, indices], axis=1
                            )
            common = np.isfinite(
                np.array([effects[k] for k in ("250", "500", "1000")])
            ).all(0)
            entry["finite_cap_paired_unit_ids"] = [
                r["unit_id"] for r, ok in zip(rows, common, strict=True) if ok
            ]
            for i, label in enumerate(CAPS):
                keep = common & np.isfinite(effects[label])
                n = int(keep.sum())
                e = effects[label][keep]
                d = entry["caps"][label]
                d["paired_units"] = n
                if not n:
                    continue
                d["paired_median"] = float(np.median(e))
                d["paired_median_bias"] = float(np.median(biases[label][keep]))
                d["paired_median_bias_change_from_500"] = float(
                    np.median(biases[label][keep] - biases["500"][keep])
                )
                d["paired_wilcoxon_p"] = float(wilcoxon(e).pvalue)
                d["paired_median_change_from_500"] = float(
                    np.median(e - effects["500"][keep])
                )
                sample = rng.choice(n, (2000, n))
                boot = np.nanmedian(
                    mc[label][:, keep][np.arange(2000)[:, None], sample], axis=1
                )
                d["paired_median_bootstrap95"] = np.quantile(
                    boot, [0.025, 0.975]
                ).tolist()
                change = (mc[label][:, keep] - mc["500"][:, keep])[
                    np.arange(2000)[:, None], sample
                ]
                d["change_bootstrap95"] = np.quantile(
                    np.nanmedian(change, axis=1), [0.025, 0.975]
                ).tolist()
                d["bias_change_bootstrap95"] = [
                    -d["change_bootstrap95"][1],
                    -d["change_bootstrap95"][0],
                ]
                ax.scatter(
                    i + np.linspace(-0.1, 0.1, n), e, s=12, color=MODEL_COLOR, alpha=0.5
                )
                ax.errorbar(
                    i,
                    np.median(e),
                    yerr=[
                        [np.median(e) - min(np.median(e), np.quantile(boot, 0.025))],
                        [max(np.median(e), np.quantile(boot, 0.975)) - np.median(e)],
                    ],
                    fmt="o",
                    color=MODEL_COLOR,
                    capsize=3,
                )
            report[subject] = entry
            ax.axhline(0, color="black", lw=0.7)
            ax.set(
                xticks=range(4),
                xticklabels=[
                    f"{k} Hz\nn={entry['caps'][k]['paired_units']}"
                    if k != "uncapped"
                    else f"Uncapped\nn={entry['caps'][k]['paired_units']}"
                    for k in CAPS
                ],
                ylabel="Corrected flash × exit log-ratio contribution",
            )
            ax.text(0.02, 1.03, subject, transform=ax.transAxes, va="bottom")
        for letter, ax in zip("ab", axes, strict=True):
            ax.text(-0.15, 1.05, letter, transform=ax.transAxes, weight="bold")
        fig.savefig(out / "cap_sensitivity.png", dpi=160, bbox_inches="tight")
        save_figure(fig, out / "cap_sensitivity.svg")
    fit._write_json_atomic(out / "summary.json", report)
    print(json.dumps(report, indent=2))


def self_check():
    rate = np.array([0.5, 0.1, 0.5])
    expected = np.random.default_rng(1).poisson(rate)
    audit = RateAudit(
        np.random.default_rng(1),
        0.5,
        np.array([[True], [False], [True]]),
        np.array([[True], [False], [False]]),
    )
    np.testing.assert_array_equal(audit.poisson(rate), expected)
    assert audit.valid_hits == 2 and audit.pair_hits == 1
    unsafe = RateAudit(
        np.random.default_rng(1), np.inf, np.ones((1, 1), bool), np.ones((1, 1), bool)
    )
    try:
        unsafe.poisson(np.array([11.0]))
    except UnstableNull:
        assert unsafe.failure["valid_trial_rows"] == [0]
    else:
        raise AssertionError("Unsafe uncapped simulation was not stopped.")
    print("Cap audit preserves RNG draws and detects capped bins/uncapped failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/glm/cap_sensitivity"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None or args.workers < 1 or not 2 <= args.repeats <= 200:
        parser.error("--root, positive workers, and 2..200 repeats are required")
    args.out.mkdir(parents=True, exist_ok=True)
    excited = json.loads((args.root / "flash_excited_units.json").read_text())
    manifest = {
        "repeats": args.repeats,
        "code": {
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in (
                "cap_sensitivity.py",
                "leak_test.py",
                "power_test.py",
                "fit.py",
                "dmat.py",
                "flash_movement.py",
            )
        },
        "inputs": {},
    }
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
            for suffix in ("final", "folds"):
                name = f"all_fit_me/checkpoints/unit_{unit}_{suffix}.json"
                manifest["inputs"][f"{session}/{name}"] = hashlib.sha256(
                    (root / name).read_bytes()
                ).hexdigest()
            jobs.append(
                {
                    "root": str(root),
                    "out": str(args.out / session / f"unit_{unit}"),
                    "unit": unit,
                    "repeats": args.repeats,
                }
            )
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Inputs or code changed; choose a fresh output directory.")
    fit._write_json_atomic(path, manifest)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        submitted = {
            pool.submit(unit_job, j): j
            for j in jobs
            if not (Path(j["out"]) / "complete.json").exists()
        }
        for future in as_completed(submitted):
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                job = submitted[future]
                failures.append(
                    {"unit": job["unit"], "root": job["root"], "error": repr(error)}
                )
                print(f"FAILED {failures[-1]}", flush=True)
    fit._write_json_atomic(args.out / "failures.json", failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} units failed; completed checkpoints retained."
        )
    summarize(args.root, args.out)


if __name__ == "__main__":
    main()
