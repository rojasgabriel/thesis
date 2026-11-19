"""Locate the time-dependent rate error of the held-out GLM predictions.

Question: the pooled residual is positive before the first flash and negative
from 0 to about 0.7 s (poisson_gof.py). Does it follow center entry, the first
flash or center exit; does it depend on stimulus rate; do a few units cause it;
and is it the same in every cross-validation fold (a fold-intercept artifact
would differ between folds)?

Comparison: recorded 1-ms counts against out-of-fold expected counts, which
include observed spike history. No new fits. Units, trials, folds and validity
(bins up to response entry) are those of conditional_dispersion.unit_predictions;
folds with zero test spikes are not scored. No baseline subtraction.
Alignment: center entry, first flash, center exit; 10-ms blocks. Trial events
come from the same trial table as the design (dmat._load_selected_trials).
Stimulus rate: low < 12 Hz, high > 12 Hz; 12 Hz trials are excluded from that
split only.

Statistic per block: pooled z = sum(y - mu) / sqrt(sum mu) over units and
trials (~N(0, 1) if rates are calibrated). To ask whether a few units cause it:
per unit, relative error sum(y - mu) / sum(mu) in the fixation window (first
flash to center exit) and in the 0.5 s before the first flash, with a z score
per unit; and the pooled z without the 10 % of units with most spikes.
Sampling unit: unit for the per-unit counts; sessions are reported separately.

Run: uv run python -m thesis.ephys.analyses.glm.residual_timing \\
    --data-root <checkout containing saved GLM fits and spike cache>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from thesis.behavior.trials import SESSIONS
from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm.conditional_dispersion import unit_predictions

BLOCK = 10  # bins of 1 ms
SPAN = {
    "center entry": (-0.3, 1.5),
    "first flash": (-0.5, 1.2),
    "center exit": (-0.8, 0.4),
}


def realign(values, offsets, start, stop):
    """Sum trials x bins `values` over trials on a grid shifted by `offsets` bins."""
    width = stop - start
    out = np.zeros(width)
    bins = values.shape[1]
    for row, offset in zip(values, offsets):
        lo, hi = max(start + offset, 0), min(stop + offset, bins)
        if hi > lo:
            out[lo - offset - start : hi - offset - start] += row[lo:hi]
    return out


def self_check():
    values = np.arange(12.0).reshape(2, 6)
    out = realign(values, np.array([2, 3]), -1, 2)
    assert np.array_equal(out, [1 + 8, 2 + 9, 3 + 10]), out
    assert np.array_equal(realign(values, np.array([0, 0]), -2, 1), [0, 0, 6])


def blocks(a):
    stop = len(a) // BLOCK * BLOCK
    return a[:stop].reshape(-1, BLOCK).sum(axis=1)


def session(data_root, name):
    path = data_root / "results/glm" / name
    predictions = unit_predictions(data_root, name)
    predictions = itertools.chain(
        [next(predictions)], predictions
    )  # missing caches fail first
    prepared = dmat._load_windows(path / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    first = prepared["alignments"]
    centers = prepared["relative_centers"]
    step = centers[1] - centers[0]

    def to_bin(s):
        return np.round((s - centers[0]) / step).astype(int)

    offsets = {
        "center entry": to_bin(trials["center_entry_s"].to_numpy(float) - first),
        "first flash": np.full(len(first), to_bin(0.0)),
        "center exit": to_bin(trials["center_exit_s"].to_numpy(float) - first),
    }
    grids = {
        k: (int(round(a / step)), int(round(b / step))) for k, (a, b) in SPAN.items()
    }
    rate = trials["visual_stim_rate_hz"].to_numpy(float)
    groups = {"low": rate < 12, "high": rate > 12}
    folds = fit.trial_folds(len(first), fit.CV_FOLDS, fit.CV_SEED)
    exit_rel = trials["center_exit_s"].to_numpy(float) - first

    pooled = {k: [np.zeros(b - a) for _ in range(2)] for k, (a, b) in grids.items()}
    by_rate = {g: [np.zeros(len(centers)) for _ in range(2)] for g in groups}
    by_fold = np.zeros((fit.CV_FOLDS, 2, len(centers)))
    units, per_unit_first = [], []
    for unit in predictions:
        r = np.where(unit["scored"], unit["y"] - unit["mu"], 0)
        e = np.where(unit["scored"], unit["mu"], 0)
        for key, (a, b) in grids.items():
            pooled[key][0] += realign(r, offsets[key], a, b)
            pooled[key][1] += realign(e, offsets[key], a, b)
        for g, mask in groups.items():
            by_rate[g][0] += r[mask].sum(axis=0)
            by_rate[g][1] += e[mask].sum(axis=0)
        for f in range(fit.CV_FOLDS):
            by_fold[f, 0] += r[folds == f].sum(axis=0)
            by_fold[f, 1] += e[folds == f].sum(axis=0)
        fixation = (centers[None, :] >= 0) & (centers[None, :] < exit_rel[:, None])
        before = np.broadcast_to(centers[None, :] < 0, fixation.shape)
        summary = dict(
            unit_id=unit["unit_id"], spikes=int(unit["y"][unit["scored"]].sum())
        )
        for label, window in (("fixation", fixation), ("pre_flash", before)):
            rr, ee = r[window].sum(), e[window].sum()
            summary[f"{label}_relative"] = float(rr / ee)
            summary[f"{label}_z"] = float(rr / np.sqrt(ee))
        units.append(summary)
        per_unit_first.append((blocks(r.sum(axis=0)), blocks(e.sum(axis=0))))

    def z(pair):
        return (blocks(pair[0]) / np.sqrt(blocks(pair[1]))).tolist()

    def time_of(a, b):
        return (blocks(np.arange(a, b) * step) / BLOCK).tolist()

    spikes = np.array([u["spikes"] for u in units])
    keep = spikes < np.percentile(spikes, 90)
    rr = np.array([p[0] for p in per_unit_first])
    ee = np.array([p[1] for p in per_unit_first])
    flash_time = blocks(centers) / BLOCK
    fold_total = (by_fold[:, 0].sum(axis=1) / by_fold[:, 1].sum(axis=1)).tolist()
    result = dict(
        session=name,
        units=len(units),
        trials=len(first),
        aligned={k: dict(time_s=time_of(*grids[k]), z=z(pooled[k])) for k in grids},
        rate=dict(
            time_s=flash_time.tolist(),
            **{g: z(by_rate[g]) for g in groups},
            trials={g: int(m.sum()) for g, m in groups.items()},
        ),
        folds=dict(
            time_s=flash_time.tolist(),
            z=[z(by_fold[f]) for f in range(fit.CV_FOLDS)],
            relative_error=fold_total,
        ),
        without_top_decile=dict(
            time_s=flash_time.tolist(),
            z=(rr[keep].sum(0) / np.sqrt(ee[keep].sum(0))).tolist(),
        ),
        unit_results=units,
    )
    fz = np.array([u["fixation_z"] for u in units])
    pz = np.array([u["pre_flash_z"] for u in units])
    print(
        json.dumps(
            dict(
                session=name,
                units=len(units),
                fixation_below=int((fz < -1.96).sum()),
                fixation_above=int((fz > 1.96).sum()),
                fixation_relative_median=float(
                    np.median([u["fixation_relative"] for u in units])
                ),
                pre_flash_below=int((pz < -1.96).sum()),
                pre_flash_above=int((pz > 1.96).sum()),
                pre_flash_relative_median=float(
                    np.median([u["pre_flash_relative"] for u in units])
                ),
                fold_relative_error=np.round(fold_total, 3).tolist(),
            )
        ),
        flush=True,
    )
    return result


def plot_report(results, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = dict(GRB006="o", GRB064="s", GRB061="^", GRB068="D")
    shades = ("black", "gray", "silver")
    rate_colors = dict(low="#0072B2", high="#D55E00")  # Okabe–Ito
    lines = ("-", "--", ":", "-.")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), layout="constrained")
    for axis in axes[1, len(results) :]:
        axis.remove()
    for index, r in enumerate(results):
        mouse = r["session"].split("_")[0]
        style = dict(
            color=shades[index],
            lw=0.9,
            marker=markers.get(mouse, "o"),
            markevery=15,
            ms=4,
            mfc="white",
        )
        for axis, key in zip(axes[0], SPAN):
            axis.plot(r["aligned"][key]["time_s"], r["aligned"][key]["z"], **style)
            axis.set_xlabel(f"time from {key} (s)")
        axis = axes[0, 3]
        for g, color in rate_colors.items():
            axis.plot(
                r["rate"]["time_s"], r["rate"][g], color=color, lw=0.8, ls=lines[index]
            )
        fold_axis = axes[1, index]
        for f, curve in enumerate(r["folds"]["z"]):
            fold_axis.plot(r["folds"]["time_s"], curve, color="gray", lw=0.5, alpha=0.7)
        fold_axis.plot(
            r["without_top_decile"]["time_s"],
            np.array(r["without_top_decile"]["z"]),
            color="black",
            lw=1,
        )
        fold_axis.set_xlabel(f"time from first flash (s), {mouse}")
    axes[0, 3].set_xlabel("time from first flash (s)")
    low, high = axes[0, 3].get_ylim()
    axes[0, 3].set_ylim(low, high + 0.6 * (high - low))  # room for direct labels
    for g, color in rate_colors.items():
        axes[0, 3].text(
            0.98,
            0.95 if g == "low" else 0.87,
            f"{g} rate",
            color=color,
            transform=axes[0, 3].transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    axes[0, 3].text(
        0.98,
        0.79,
        ", ".join(
            f"{n.split('_')[0]} {style}"
            for n, style in zip(
                [r["session"] for r in results], ("solid", "dashed", "dotted")
            )
        ),
        color="gray",
        transform=axes[0, 3].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    for axis in axes[1, : len(results)]:
        axis.text(
            0.98,
            0.95,
            "each fold",
            color="gray",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        axis.text(
            0.98,
            0.87,
            "all folds, top 10 % units removed",
            color="black",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    for axis in fig.axes:
        axis.axvline(0, color="gray", ls="--", lw=0.8)
        axis.axhline(0, color="gray", lw=0.5)
        for level in (-1.96, 1.96):
            axis.axhline(level, color="gray", ls=":", lw=0.7)
    for axis in (axes[0, 0], axes[1, 0]):
        axis.set_ylabel("summed residual / √expected")
    for letter, axis in zip("abcdefgh", fig.axes):
        axis.text(-0.14, 1.04, letter, transform=axis.transAxes, weight="bold")
    fig.savefig(out / "residual_timing.svg")
    fig.savefig(out / "residual_timing.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fano"))
    args = parser.parse_args()
    self_check()
    args.out.mkdir(parents=True, exist_ok=True)
    results, missing = [], []
    for name in SESSIONS:
        try:
            results.append(session(args.data_root, name))
        except FileNotFoundError as error:
            missing.append(dict(session=name, reason=str(error)))
            print(f"Skipped {name}: {error}", flush=True)
        (args.out / "residual_timing.json").write_text(
            json.dumps(dict(sessions=results, missing=missing), indent=2)
        )
    plot_report(results, args.out)


if __name__ == "__main__":
    main()
