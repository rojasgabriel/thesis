"""Switch off self-history in the saved GLM while retaining all other weights.

Compare the original flash-excited neurons, held-out whole trials, flash pairs,
response-entry censoring, and matched peak windows. Set raw history to zero
with a fold-specific intercept adjustment. Never refit the real-data weights.
Report the original long-gap log ratio and the normalized peak difference.
The sampling unit is the neuron, paired within each mouse.

For a matched null, take the original final fit, remove raw history and flash ×
exit, and sample independent Poisson counts. Fit the full model to each null
dataset at the original penalties, then ablate its history before predicting.
This calibrates the fit-then-ablate statistic under a history-free generator.
The original reference keeps its stored capped null. Save each null replicate.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import dmat, fit, leak_test
from thesis.ephys.analyses.glm import history_sensitivity as hs
from thesis.ephys.analyses.glm.results import raw_contribution, zero_raw_columns


def ablate(intercept, weights, width, mean, scale):
    return float(intercept) - (mean / scale) @ np.asarray(weights)[width:]


def predict(counts, parts, params, width):
    s = leak_test._SHARED
    pair = s["pair_rows"]
    log = np.full(pair.sum(), np.nan)
    term = log.copy()
    for part, (intercept, weights) in zip(parts, params, strict=True):
        use = part["test"][pair]
        X = s["common"][part["test"] & pair, :width]
        log[use] = intercept + X @ weights[:width]
        term[use] = raw_contribution(X, weights, s["metadata"], s["removed_columns"])
    if not np.isfinite(log).all():
        raise ValueError("Missing paired prediction")
    return leak_test.peaks_from(log, term, counts)


def run(job):
    root, out, unit, repeats = job
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    counts, raw, original, width = hs.setup(root, unit)
    s = leak_test._SHARED
    parts = s["original_parts"]
    folds = original["folds"]
    params = []
    for part, record in zip(parts, folds, strict=True):
        _, mean, scale = dmat.training_zscore(raw, part["fit"])
        w = np.asarray(record["coefficients"])
        intercept = ablate(record["intercept"], w, width, mean, scale)
        np.testing.assert_allclose(
            intercept, record["intercept"] + (-mean / scale) @ w[width:]
        )
        params.append((intercept, w))
    ablated = predict(counts, parts, params, width)
    current = hs.fold_peaks(folds, raw, counts, width, native=True)
    fit._write_json_atomic(
        out / "real.json",
        {"unit": unit, "current": current.tolist(), "ablated": ablated.tolist()},
    )
    if not repeats:
        return

    valid = dmat._valid_bin_mask(s["prepared"])
    final = json.loads(
        (root / f"all_fit_me/checkpoints/unit_{unit}_final.json").read_text()
    )
    _, mean, scale = dmat.training_zscore(raw, valid)
    intercept = ablate(final["intercept"], final["coefficients"], width, mean, scale)
    intercept, w = zero_raw_columns(
        intercept, final["coefficients"], s["metadata"], s["removed_columns"]
    )
    rate = np.zeros(counts.size)
    rate[valid] = np.exp(intercept + s["common"][valid, :width] @ w[:width])
    if not np.isfinite(rate).all():
        raise ValueError("Nonfinite open-loop rate")
    parts = fit.cross_validation_partitions(*counts.shape, valid)
    observed = iter(folds)
    alphas = [
        next(observed)["alpha"] if counts.ravel()[p["test"]].sum() else final["alpha"]
        for p in parts
    ]
    draws = []
    for rep in range(repeats):
        path = out / f"null_{rep:03}.json"
        if path.exists():
            draws.append(json.loads(path.read_text()))
            continue
        y = (
            np.random.default_rng([20261120, unit, rep])
            .poisson(rate)
            .reshape(counts.shape)
        )
        raw_null = hs.raw_history(np.zeros_like(s["initial"]), y)
        params = []
        model = None
        for part, alpha in zip(parts, alphas, strict=True):
            h, mean, scale = dmat.training_zscore(raw_null, part["fit"])
            X = dmat._design_with_history(s["common"], h, width, part["fit"])
            model = fit.fit_poisson_at_alpha(X, y.ravel()[part["fit"]], alpha, model)
            params.append(
                (
                    ablate(model.intercept_, model.coef_, width, mean, scale),
                    model.coef_.copy(),
                )
            )
        peak = predict(y, parts, params, width).tolist()
        fit._write_json_atomic(path, peak)
        draws.append(peak)
    fit._write_json_atomic(
        out / "complete.json",
        {"null_peaks": draws, "max_expected_hz": float(rate.max() / dmat.BINWIDTH_S)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    excited = json.loads((args.root / "flash_excited_units.json").read_text())
    jobs = [
        (args.root / s, args.out / s / f"unit_{u}", u, args.repeats)
        for s in hs.SESSIONS
        for u in excited[s]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            future.result()
            print(f"Completed {futures[future][1]}", flush=True)


if __name__ == "__main__":
    main()
