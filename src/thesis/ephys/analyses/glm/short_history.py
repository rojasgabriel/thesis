"""Does a 20-ms history improve calibration and dynamics without changing flash × exit?

Paired refits of all 172/115 fitted neurons in the two main sessions, retaining
their quality/stability rules, completed trials, first-flash alignment, task
and video design, baseline, and whole-trial folds. Change only the duration of
the log-spaced history bases from 100 to 20 ms. One of the ten shortened
bases is identically zero at 1-ms resolution and is omitted (nine active bases). Keep each saved fold's
overall penalty fixed; this is a sensitivity test, not penalty optimization.
Exclude the final grid bin from both new fits and diagnostic comparisons, as
in dynamics.py; preserve native original history normalization for saved fits.

Compare observed-history held-out calibration in 20 predicted-rate quantiles,
mean bits/spike against a constant fitting rate, and 20 generated-history
repeats per held-out trial. Initialize with observed spikes before the window;
sample recursively without clipping, stopping above 10 expected spikes/ms.

For the same 30/22 flash-excited units, measure the original matched flash pairs,
peak windows and raw-column removal. Refit 20 null replicates for the 20-ms
model at its fixed fold penalties. The primary null uses the original 500-Hz
cap, zero initial history and full simulation windows, isolating the history
change from the cap convention. Attempt uncapped nulls too; any failed replicate
makes that unit's uncapped correction unavailable, never average survivors.
The current reference retains its saved 200-replicate 500-Hz null correction.

Sampling unit: neuron, paired within mouse; never pool simulated trajectories
as biological replicates. Report log-ratio omissions from nonpositive peaks,
normalized effects separately, and both unit and null Monte Carlo uncertainty.
No original fits are overwritten. The scope switch permits separate machines
to process excited/other units; merge checkpoints before --summarize.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from thesis.ephys.analyses.glm import cap_sensitivity as cap
from thesis.ephys.analyses.glm import dmat, dynamics, fit, leak_test
from thesis.ephys.analyses.glm import history_sensitivity as hs
from thesis.ephys.analyses.glm.results import zero_raw_columns
from thesis.ephys.units import fetch_unit_table

SEED = 20261118
REPEATS = 20
NAMES = ("100 ms", "20 ms")


def short_basis():
    basis = dmat.RaisedCosineBasis(10, 0, 0.020, dmat.BINWIDTH_S, log_scale=True)
    basis.basis = basis.basis[:, np.any(basis.basis != 0, axis=0)]
    basis.n_funcs = basis.basis.shape[1]
    return basis


@contextmanager
def short_context():
    with (
        patch.object(dmat, "spike_history_basis", short_basis),
        patch.object(dmat, "HISTORY_COLUMNS", short_basis().n_funcs),
    ):
        yield


def shortened_history(counts, unit):
    """Rebuild native short-window initial history, including its overlap boundary."""
    s = leak_test._SHARED
    p = s["prepared"]
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
    initial = dynamics.history_counts(
        p["alignments"], spikes, p["relative_centers"], 20
    )
    raw = hs.raw_history(initial, counts)
    native = dmat.build_unit_history(p["alignments"], spikes)
    np.testing.assert_allclose(raw[s["valid"]], native[s["valid"]], atol=1e-8)
    return initial, raw


def calibration(observed, predicted):
    return [
        float(observed[g].sum() / predicted[g].sum())
        for g in np.array_split(np.argsort(predicted), 20)
    ]


def real_job(job):
    root, out, unit = Path(job["root"]), Path(job["out"]), job["unit"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        **job["hashes"],
        "fold_sha256": hashlib.sha256(
            (root / f"all_fit_me/checkpoints/unit_{unit}_folds.json").read_bytes()
        ).hexdigest(),
    }
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Checkpoint inputs changed; choose a fresh output directory.")
    fit._write_json_atomic(path, manifest)
    if (out / "real.json").exists():
        return job
    counts, raw100, original, width = hs.setup(root, unit)
    s = leak_test._SHARED
    initial100 = s["initial"].copy()
    raw20 = None
    predictions = {name: np.full(counts.size, np.nan) for name in NAMES}
    models, diagnostics, peaks = {}, {}, {}
    baseline = Path(job["baseline"]) / root.name / "dynamics.json"
    cached = None
    if baseline.exists():
        previous = json.loads(baseline.read_text())
        assert previous["fold_files_sha256"][str(unit)] == manifest["fold_sha256"]
        cached = next(r for r in previous["units"] if r["unit_id"] == unit)
    for name in NAMES:
        context = short_context() if name == "20 ms" else nullcontext()
        with context:
            if name == "100 ms":
                s["initial"], raw = initial100, raw100
            else:
                s["initial"], raw = shortened_history(counts, unit)
            if name == "20 ms":
                raw20 = raw
            rows = []
            scores, failures, trajectories = [], 0, 0
            fold_ids = fit.trial_folds(len(counts), fit.CV_FOLDS, fit.CV_SEED)
            for index, (part, old) in enumerate(
                zip(s["parts"], original["folds"], strict=True)
            ):
                normalization = (
                    s["original_parts"][index]["fit"]
                    if name == "100 ms"
                    else part["fit"]
                )
                h, _, _ = dmat.training_zscore(raw, normalization)
                record = (
                    old
                    if name == "100 ms"
                    else hs.fitted(
                        dmat._design_with_history(s["common"], h, width, part["fit"]),
                        counts.ravel()[part["fit"]],
                        width,
                        old["alpha"],
                        1.0,
                    )
                )
                X = dmat._design_with_history(s["common"], h, width, part["test"])
                predicted = np.exp(record["intercept"] + X @ record["coefficients"])
                predictions[name][part["test"]] = predicted
                scores.append(
                    fit.poisson_metrics(
                        counts.ravel()[part["test"]],
                        predicted,
                        counts.ravel()[part["fit"]].mean(),
                    )
                )
                trial = part["test"].reshape(counts.shape).any(1)
                if name != "100 ms" or cached is None:
                    fold_id = int(np.unique(fold_ids[trial]).item())
                    _, failed, _ = hs.generated(
                        record,
                        raw,
                        normalization,
                        trial,
                        counts,
                        REPEATS,
                        [dynamics.SEED, unit, fold_id],
                    )
                    failures += int((failed >= 0).sum())
                trajectories += REPEATS * int(trial.sum())
                rows.append(record)
                print(f"{root.name} unit {unit}: {name} fold {index + 1}", flush=True)
            if name == "100 ms" and cached is not None:
                assert trajectories == cached["trajectories"]
                failures = cached["failed_trajectories"]
            mask = np.isfinite(predictions[name])
            models[name] = rows
            diagnostics[name] = {
                "scores": scores,
                "failures": failures,
                "trajectories": trajectories,
                "calibration": calibration(
                    counts.ravel()[mask], predictions[name][mask]
                ),
            }
            if job["excited"]:
                peaks[name] = hs.fold_peaks(
                    rows, raw, counts, width, native=name == "100 ms"
                ).tolist()
    if job["excited"]:
        saved = json.loads((root.parent / "flash_movement.json").read_text())[
            "sessions"
        ][root.name.split("_")[0]]
        expected = np.asarray(saved["rate_peaks"])[saved["unit_ids"].index(unit), :, 1:]
        np.testing.assert_allclose(peaks["100 ms"], expected, rtol=2e-5, atol=2e-4)
    np.testing.assert_array_equal(
        np.isfinite(predictions["100 ms"]), np.isfinite(predictions["20 ms"])
    )
    assert raw20 is not None
    fit._write_json_atomic(
        out / "real.json",
        {
            "unit_id": unit,
            "excited": job["excited"],
            "models": models,
            "diagnostics": diagnostics,
            "peaks": peaks,
        },
    )
    return job


def null_job(job):
    root, out, unit = Path(job["root"]), Path(job["out"]), job["unit"]
    if (out / "complete.json").exists():
        return job
    real = json.loads((out / "real.json").read_text())
    counts, _, _, width = hs.setup(root, unit)
    s = leak_test._SHARED
    with short_context():
        initial, raw = shortened_history(counts, unit)
        h, mean, scale = dmat.training_zscore(raw, s["valid"])
        original_final = json.loads(
            (root / f"all_fit_me/checkpoints/unit_{unit}_final.json").read_text()
        )
        final = hs.fitted(
            dmat._design_with_history(s["common"], h, width, s["valid"]),
            counts.ravel()[s["valid"]],
            width,
            original_final["alpha"],
            1.0,
        )
        b, w = zero_raw_columns(
            final["intercept"],
            final["coefficients"],
            s["metadata"],
            s["removed_columns"],
        )
        final.update(
            intercept=b,
            coefficients=w.tolist(),
            components=original_final["components"],
        )
        results = []
        unavailable = (out / "uncapped_unavailable.json").exists()
        for rep in range(REPEATS):
            path = out / f"null_{rep:03d}.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            record, cache = {}, {}
            for label, ceiling in (("500", 0.5), ("uncapped", np.inf)):
                if label == "uncapped" and unavailable:
                    record[label] = None
                    continue
                try:
                    y, _, audit = cap.simulate(
                        final,
                        w,
                        mean,
                        scale,
                        np.random.default_rng([SEED, unit, rep]),
                        ceiling,
                    )
                except cap.UnstableNull as error:
                    unavailable = True
                    fit._write_json_atomic(
                        out / "uncapped_unavailable.json",
                        {"replicate": rep, "audit": error.audit},
                    )
                    record[label] = None
                    continue
                digest = hashlib.sha256(y.tobytes()).hexdigest()
                if digest not in cache:
                    history = hs.raw_history(np.zeros_like(initial), y)
                    fitted = []
                    for part, original in zip(
                        s["parts"], real["models"]["20 ms"], strict=True
                    ):
                        scaled, _, _ = dmat.training_zscore(history, part["fit"])
                        X = dmat._design_with_history(
                            s["common"], scaled, width, part["fit"]
                        )
                        fitted.append(
                            hs.fitted(
                                X, y.ravel()[part["fit"]], width, original["alpha"], 1.0
                            )
                        )
                    cache[digest] = hs.fold_peaks(fitted, history, y, width).tolist()
                record[label] = {
                    "peaks": cache[digest],
                    "audit": audit,
                    "counts_sha256": digest,
                }
            fit._write_json_atomic(path, record)
            results.append(record)
            print(f"{root.name} unit {unit}: null {rep + 1}/{REPEATS}", flush=True)
    fit._write_json_atomic(
        out / "complete.json",
        {**real, "nulls": results, "uncapped_unavailable": unavailable},
    )
    return job


def self_check():
    with short_context():
        basis = short_basis().basis
        assert basis.shape == (20, 9)
        counts = np.zeros((1, 100))
        counts[0, 30] = 1
        raw = hs.raw_history(np.zeros((1, 20)), counts)
        np.testing.assert_array_equal(raw[30], np.zeros(9))
        np.testing.assert_allclose(raw[31:51], basis)
        np.testing.assert_array_equal(raw[51:], np.zeros((49, 9)))
        # Native construction must give the identical strictly causal 20-ms history.
        centers, _, _ = dmat.construct_timebins(
            dmat.PRE_S, dmat.POST_S, dmat.BINWIDTH_S
        )
        grid = np.zeros((1, len(centers)))
        grid[0, 700] = 1
        spikes = np.array([centers[0] - 0.020, centers[0] - 0.010, centers[700]])
        initial = dynamics.history_counts(np.array([0.0]), spikes, centers, 20)
        native = dmat.build_unit_history(np.array([0.0]), spikes)
        np.testing.assert_allclose(
            hs.raw_history(initial, grid)[:-1], native[:-1], atol=1e-8
        )
    np.testing.assert_allclose(calibration(np.ones(100), np.ones(100)), np.ones(20))
    print(
        "20-ms causal history, finite support, native design and calibration checks passed."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/glm/short_history"))
    parser.add_argument("--baseline", type=Path, default=Path("results/glm/dynamics"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scope", choices=("all", "excited", "other"), default="all")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None or args.workers < 1:
        parser.error("--root and positive workers are required")
    if args.summarize:
        from thesis.ephys.analyses.glm.short_history_summary import summarize

        summarize(args.root, args.out)
        return
    hashes = {
        "code": {
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in (
                "short_history.py",
                "history_sensitivity.py",
                "dynamics.py",
                "cap_sensitivity.py",
                "dmat.py",
                "fit.py",
                "leak_test.py",
            )
        }
    }
    jobs = []
    for session in cap.SESSIONS:
        root = args.root / session
        with np.load(root / "power_test_held_out.npz") as saved:
            excited = set(saved["unit_ids"].tolist())
        inputs = {
            n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in (
                "common_design_me.npy",
                "common_design_me.json",
                "stimulus_windows_me.npz",
                "power_test_held_out.npz",
            )
        }
        for record in fit.load_unit_records(
            root / "all_fit_me/checkpoints", "unit_*_folds.json"
        ):
            unit = record["unit_id"]
            selected = unit in excited
            if args.scope != "all" and selected != (args.scope == "excited"):
                continue
            jobs.append(
                {
                    "root": str(root),
                    "out": str(args.out / session / f"unit_{unit}"),
                    "unit": unit,
                    "excited": selected,
                    "baseline": str(args.baseline),
                    "hashes": {**hashes, "inputs": inputs},
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for function, selected in (
            (real_job, jobs),
            (null_job, [j for j in jobs if j["excited"]]),
        ):
            futures = {pool.submit(function, job): job for job in selected}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append(
                        {
                            "unit": job["unit"],
                            "root": job["root"],
                            "phase": function.__name__,
                            "error": repr(error),
                        }
                    )
                    print(f"FAILED {failures[-1]}", flush=True)
            if failures:
                break
    fit._write_json_atomic(args.out / f"failures_{args.scope}.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} unit jobs failed; checkpoints preserved.")


if __name__ == "__main__":
    main()
