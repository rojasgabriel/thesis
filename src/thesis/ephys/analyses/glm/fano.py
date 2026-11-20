"""Measure spike-count dispersion in the existing GLM sessions, without fitting.

Comparison: variance/mean against the Poisson reference of 1. Units must be in
both the cached quality-1/stability-0 unit table and saved final GLM fits;
fit.py exclusions apply. Trial selection and first-flash alignment come from
saved GLM windows. Raw Fano factors pool nonoverlapping equal-width bins from
-0.5 to 2.1 s, excluding incomplete bins and bins extending past response entry.
No baseline subtraction. The primary repeated-trial comparison uses 0–100 ms
after the first flash, restricted to trials in the center port throughout that
window. Group by visual rate and presence of clicks; retain groups of >=10
trials. Sum within-group squared deviations and divide by the corresponding
sum of (n-1)*mean counts. This controls these conditions, but not exact flash
sequences, movement, drift, or spike history. It is not a conditional GLM test.
Sampling unit: neuron, reported separately by session. Bootstrap whole trials
within condition (500 draws) for descriptive per-neuron 95% intervals. Silent
units have undefined dispersion and are reported as missing, never zero.

Run: uv run python -m thesis.ephys.analyses.glm.fano --data-root <existing checkout>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTHS = (1, 10, 50, 100, 200)


def dispersion(groups: list[np.ndarray]) -> np.ndarray:
    """Return condition-adjusted variance/mean, with units on the last axis."""
    numerator = sum(((x - x.mean(axis=0)) ** 2).sum(axis=0) for x in groups)
    denominator = sum((len(x) - 1) * x.mean(axis=0) for x in groups)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )


def self_check() -> None:
    assert np.allclose(dispersion([np.array([[0.0], [2.0]])]), 2)
    assert np.isnan(dispersion([np.zeros((10, 1))])).all()
    rng = np.random.default_rng(7)
    assert abs(float(dispersion([rng.poisson(2, (100000, 1))])[0]) - 1) < 0.02


def session(root: Path, name: str) -> dict:
    subject, date = name.split("_", 1)
    cache = root / "figures/cache"
    units = pd.read_pickle(
        cache
        / f"fetch_unit_table_subject-{subject}_session-{date}_unit_criteria_id-1_stability_param_id-0_include_metrics-False.pkl"
    )
    trials = pd.read_pickle(
        cache
        / f"build_trial_table_subject-{subject}_session-{date}_include_frames-False.pkl"
    )
    folder = root / "results/glm" / name
    records = [
        json.loads(p.read_text())
        for p in (folder / "all_fit_me/checkpoints").glob("unit_*_final.json")
    ]
    from thesis.ephys.analyses.glm.fit import excluded_units

    ids = sorted(
        set(int(r["unit_id"]) for r in records) - set(excluded_units(subject, date))
    )
    if not ids:
        raise ValueError(f"{name}: no saved final fits")
    units = units.set_index("unit_id").loc[ids]
    with np.load(folder / "stimulus_windows_me.npz", allow_pickle=False) as w:
        selected = w["selected_trial_rows"]
        align = w["first_stim_s"][w["eligible_trials"]]
        response = w["response_entry_s"][w["eligible_trials"]]
        numbers = w["trial_num"][selected]
    t = trials.iloc[selected].reset_index(drop=True)
    assert np.array_equal(t.trial_num, numbers) and len(t) == len(align)
    spikes = [np.asarray(s, dtype=float) for s in units.spike_times_s]
    assert all(np.isfinite(s).all() and (np.diff(s) >= 0).all() for s in spikes)
    raw = []
    for width in WIDTHS:
        dt = width / 1000
        edges = -0.4995 + np.arange(int(2.599 / dt) + 1) * dt
        starts, stops = align[:, None] + edges[:-1], align[:, None] + edges[1:]
        keep = stops <= response[:, None]
        f = []
        for s in spikes:
            y = (np.searchsorted(s, stops) - np.searchsorted(s, starts))[keep]
            f.append(float(y.var(ddof=1) / y.mean()) if y.mean() > 0 else np.nan)
        raw.append(f)
    in_port = (t.center_entry_s.to_numpy() <= align) & (
        t.center_exit_s.to_numpy() >= align + 0.1
    )
    labels = list(
        zip(t.visual_stim_rate_hz, [len(a) > 0 for a in t.audio_stim_times_s])
    )
    groups, sizes, group_labels = [], [], []
    counts = np.column_stack(
        [np.searchsorted(s, align + 0.1) - np.searchsorted(s, align) for s in spikes]
    )
    for label in sorted(set(labels)):
        rows = np.array([i for i, x in enumerate(labels) if x == label and in_port[i]])
        if len(rows) >= 10:
            groups.append(counts[rows])
            sizes.append(len(rows))
            group_labels.append(list(label))
    if not groups:
        raise ValueError(f"{name}: no condition has >=10 trials")
    fano = dispersion(groups)
    rng = np.random.default_rng(20261118)
    boot = np.array(
        [
            dispersion([x[rng.integers(len(x), size=len(x))] for x in groups])
            for _ in range(500)
        ]
    )
    ci = np.full((len(ids), 2), np.nan)
    finite = np.isfinite(fano)
    ci[finite] = np.nanpercentile(boot[:, finite], [2.5, 97.5], axis=0).T
    raw = np.array(raw).T
    return dict(
        session=name,
        units=len(ids),
        glm_trials=len(t),
        matched_trials=sum(sizes),
        condition_sizes=sizes,
        conditions=group_labels,
        unit_ids=ids,
        raw_fano=raw.tolist(),
        matched_fano=fano.tolist(),
        interval_95=ci.tolist(),
        raw_median=np.nanmedian(raw, axis=0).tolist(),
        matched_median=float(np.nanmedian(fano)),
        matched_iqr=np.nanpercentile(fano, [25, 75]).tolist(),
        above_1=int(np.sum(ci[:, 0] > 1)),
        below_1=int(np.sum(ci[:, 1] < 1)),
        undefined=int(np.sum(~np.isfinite(fano))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fano"))
    args = parser.parse_args()
    self_check()
    results, missing = [], []
    from thesis.behavior.trials import SESSIONS

    for name in SESSIONS:
        try:
            r = session(args.data_root, name)
        except FileNotFoundError as error:
            missing.append(dict(session=name, reason=str(error)))
            print(f"Skipped {name}: {error}", flush=True)
            continue
        results.append(r)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("unit_ids", "raw_fano", "matched_fano", "interval_95")
                }
            ),
            flush=True,
        )
    if not results:
        raise ValueError("No sessions could be analyzed")
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(widths_ms=WIDTHS, sessions=results, missing=missing)
    # Represent undefined silent-unit estimates as JSON null.
    for r in results:
        for key in ("raw_fano", "matched_fano", "interval_95"):
            values = np.asarray(r[key], dtype=object)
            values[~np.isfinite(np.asarray(r[key], dtype=float))] = None
            r[key] = values.tolist()
    (args.out / "fano.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    colors = ["#3679ad", "#bd6836", "#66904c"]
    for index, (r, color) in enumerate(zip(results, colors)):
        axes[0].plot(WIDTHS, r["raw_median"], "o-", color=color)
        axes[0].annotate(
            r["session"].split("_")[0],
            (200, r["raw_median"][-1]),
            xytext=(5, -7 if index == 0 else 7),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )
    axes[0].set(
        xscale="log",
        xlabel="counting window (ms)",
        ylabel="pooled Fano factor (median neuron)",
        xlim=(0.8, 600),
    )
    rng = np.random.default_rng(1)
    for i, (r, color) in enumerate(zip(results, colors)):
        y = np.array(r["matched_fano"], dtype=float)
        ci = np.array(r["interval_95"], dtype=float)
        x = i + rng.uniform(-0.18, 0.18, len(y))
        axes[1].vlines(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0.6)
        axes[1].scatter(x, y, s=7, color=color, alpha=0.5)
        axes[1].plot([i - 0.2, i + 0.2], [r["matched_median"]] * 2, color="black", lw=2)
    axes[1].set(
        xticks=range(len(results)),
        xticklabels=[r["session"].split("_")[0] for r in results],
        ylabel="first 100 ms: condition-matched Fano factor",
        yscale="log",
    )
    from matplotlib.ticker import FixedLocator, FuncFormatter

    axes[1].yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 4]))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].yaxis.set_minor_locator(FixedLocator([]))
    for letter, ax in zip("ab", axes):
        ax.axhline(1, color="gray", linestyle="--", lw=0.8)
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, weight="bold")
    fig.savefig(args.out / "fano.svg")
    fig.savefig(args.out / "fano.png", dpi=180)


if __name__ == "__main__":
    main()
