"""Does raw video change prediction, task weights, or split-half stability?

Use all eligible units (167 GRB006, 76 GRB064), paired by unit ID with the
unchanged main model. Trial selection, first-flash alignment, pre-entry
baseline, response-entry mask, and exclusions follow fit.py. The comparison
is 10 motion-energy PCs versus these PCs plus 10 raw-video PCs on the same
10 trial folds and inner penalty selection. Preprocessing is fixed to the
original training split, not nested within folds. No sensory filter is used.

Refit final candidate models at each unit's median fold penalty. Split-half
fits reuse kernel_reliability: alternate completed trials, fixed final-model
penalty per model, shared common scaling, half-specific history scaling.
Compare each raw split-half kernel agreement with the saved main-model
agreement in the same unit. All eligible units enter statistics; do not
reselect units using regressor inclusion from either model. The sampling
unit is the unit; tests are within mouse, Holm across six task terms.
Pooled median intervals bootstrap mice then units; only two mice are present.
No full-population recovery or leak simulations are run here; the separate
40-unit trust check remains a pilot. Main models and pilot outputs stay intact.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \
        thesis.ephys.analyses.glm.raw_video_full --workers 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm import kernel_reliability as reliability
from thesis.ephys.analyses.glm import raw_video_kernels as kernels
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.inclusion import holm
from thesis.ephys.analyses.glm.raw_video_test import compare, fingerprint, plot, prepare
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    figure_path,
    pooled_strip,
)


def guard(path, inputs):
    if path.exists() and json.loads(path.read_text()) != inputs:
        raise ValueError(
            f"Inputs changed; preserve the prior run before retrying: {path}"
        )
    fit._write_json_atomic(path, inputs)


def half_unit(job):
    path = Path(job["output"])
    if path.exists():
        return json.loads(path.read_text())
    values = reliability._fit_halves(job)
    result = {
        "unit_id": job["unit_id"],
        "kernels": {k: v.tolist() for k, v in values.items()},
    }
    fit._write_json_atomic(path, result)
    return result


def split_session(root, design, workers):
    destination = root / "raw_video_test/all_split_half"
    destination.mkdir(exist_ok=True)
    records = {
        r["unit_id"]: r
        for r in fit.load_unit_records(
            root / "raw_video_test/all_fit_raw/checkpoints", "unit_*_final.json"
        )
    }
    units = fit.analysis_units(
        dmat._load_windows(root / "stimulus_windows_me.npz"), "all"
    )
    guard(
        destination / "inputs.json",
        {
            str(p.relative_to(root)): fingerprint(p)
            for p in (
                design,
                design.with_suffix(".json"),
                root / "stimulus_windows_me.npz",
                root / "kernel_reliability.npz",
                *(
                    root
                    / f"raw_video_test/all_fit_raw/checkpoints/unit_{u['unit_id']}_final.json"
                    for u in units
                ),
            )
        },
    )
    jobs = [
        {
            **u,
            "components": 20,
            "alpha": records[u["unit_id"]]["alpha"],
            "output": str(destination / f"unit_{u['unit_id']}.json"),
        }
        for u in units
    ]
    results = []
    with Pool(
        workers, reliability._load, (root / "stimulus_windows_me.npz", design)
    ) as pool:
        for r in pool.imap(half_unit, jobs):
            results.append(r)
            print(
                f"{root.name}: split halves {len(results)}/{len(jobs)}, unit {r['unit_id']}",
                flush=True,
            )
    ids = [u["unit_id"] for u in units]
    arrays = {"unit_ids": np.asarray(ids)}
    stats, tests = {}, {}
    with np.load(root / "kernel_reliability.npz", allow_pickle=False) as old:
        positions = [int(np.flatnonzero(old["unit_ids"] == u)[0]) for u in ids]
        for name in kernels.TERMS:
            first, second = [
                np.array([r["kernels"][f"{name}__{half}"] for r in results])
                for half in (0, 1)
            ]
            before = old[f"agreement__{name}"][positions]
            after = reliability.agreement(first, second)
            if not np.isfinite(np.r_[before, after]).all():
                raise ValueError(f"Non-finite split-half agreement: {name}")
            change = after - before
            arrays[f"baseline__{name}"] = before
            arrays[f"raw_video__{name}"] = after
            arrays[f"kernel__{name}"] = np.array([first, second])
            stats[name] = {
                "baseline_median": float(np.median(before)),
                "raw_video_median": float(np.median(after)),
                "median_paired_change": float(np.median(change)),
                "baseline_fraction_above_0.5": float(np.mean(before > 0.5)),
                "raw_video_fraction_above_0.5": float(np.mean(after > 0.5)),
            }
            tests[name] = float(wilcoxon(change).pvalue) if np.any(change) else 1.0
    for name, p in holm(tests).items():
        stats[name]["paired_p_holm"] = p
    np.savez_compressed(destination / "comparison.npz", **arrays)
    report = {"session": root.name, "units": len(ids), "statistics": stats}
    fit._write_json_atomic(destination / "comparison.json", report)
    return report, arrays


def split_plot(sessions, arrays, output):
    subjects = [s["session"].split("_")[0] for s in sessions]
    pooled = {}
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
        for letter, ax, (name, (label, _)) in zip(
            "abcdef", axes.flat, kernels.TERMS.items(), strict=True
        ):
            # The same jitter makes the line endpoints the same paired unit.
            rng = np.random.default_rng(0)
            for data in arrays:
                jitter = rng.uniform(-0.15, 0.15, len(data["unit_ids"]))
                for j, a, b in zip(
                    jitter,
                    data[f"baseline__{name}"],
                    data[f"raw_video__{name}"],
                    strict=True,
                ):
                    ax.plot(
                        [j, 1 + j],
                        [a, b],
                        color=REFERENCE_COLOR,
                        alpha=0.15,
                        lw=0.4,
                        zorder=1,
                    )
            for pos, (prefix, color) in enumerate(
                (("baseline", "black"), ("raw_video", MODEL_COLOR))
            ):
                values = [d[f"{prefix}__{name}"] for d in arrays]
                pooled[f"{prefix}__{name}"] = pooled_strip(
                    ax, pos, values, subjects, color
                )
            ax.axhline(0.5, color=REFERENCE_COLOR, ls="--", lw=0.7)
            ax.text(-0.15, 1.07, letter, weight="bold", transform=ax.transAxes)
            ax.text(
                0.05,
                0.98,
                label.replace(" (uncorrected)", ""),
                va="top",
                transform=ax.transAxes,
            )
            ax.set(
                xticks=[0, 1],
                xticklabels=["current", "+ raw video"],
                xlim=(-0.3, 1.7),
                ylabel="split-half kernel agreement",
            )
            ax.get_xticklabels()[1].set_color(MODEL_COLOR)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.15 * (hi - lo))
        fig.savefig(figure_path(output / "split_half.svg"), bbox_inches="tight")
        plt.close(fig)
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = Path("results/glm/raw_video_full")
    output.mkdir(exist_ok=True)
    if args.plot_only:
        comparisons = json.loads((output / "prediction.json").read_text())["sessions"]
        plot(comparisons, output)
        tasks = json.loads((output / "task_kernels.json").read_text())["sessions"]
        splits = json.loads((output / "split_half.json").read_text())["sessions"]
        task_arrays, split_arrays = [], []
        for session in SESSIONS:
            root = output.parent / session / "raw_video_test"
            with np.load(
                root / "kernel_comparison_all.npz", allow_pickle=False
            ) as data:
                task_arrays.append({key: data[key] for key in data.files})
            with np.load(
                root / "all_split_half/comparison.npz", allow_pickle=False
            ) as data:
                split_arrays.append({key: data[key] for key in data.files})
        kernels.plot(tasks, task_arrays, output)
        split_plot(splits, split_arrays, output)
        return
    comparisons, split_stats, split_arrays, task_stats, task_arrays = [], [], [], [], []
    for session in SESSIONS:
        root = Path("results/glm") / session
        design = prepare(root)
        destination = root / "raw_video_test/all_fit_raw"
        destination.mkdir(exist_ok=True)
        pilot = root / "raw_video_test/sample_fit_raw/checkpoints"
        source = root / "all_fit_me/checkpoints"
        inputs = {
            str(p.relative_to(root)): fingerprint(p)
            for p in (
                design,
                design.with_suffix(".json"),
                root / "stimulus_windows_me.npz",
                *sorted(source.glob("unit_*_folds.json")),
                *sorted(pilot.glob("unit_*.json")),
            )
        }
        guard(destination / "inputs.json", inputs)
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        for p in pilot.glob("unit_*.json"):
            target = checkpoints / p.name
            if not target.exists():
                shutil.copy2(p, target)
        comparisons.append(compare(root, design, args.workers, "all"))
        fit.refit_final(
            root / "stimulus_windows_me.npz",
            design,
            "all",
            destination,
            20,
            args.workers,
        )
        task, values = kernels.session_kernels(root, "all")
        task_stats.append(task)
        task_arrays.append(values)
        np.savez_compressed(root / "raw_video_test/kernel_comparison_all.npz", **values)
        stats, values = split_session(root, design, args.workers)
        split_stats.append(stats)
        split_arrays.append(values)
    fit._write_json_atomic(
        output / "prediction.json",
        {"sessions": comparisons, "pooled_d2_change": plot(comparisons, output)},
    )
    fit._write_json_atomic(
        output / "task_kernels.json",
        {
            "sessions": task_stats,
            "pooled": kernels.plot(task_stats, task_arrays, output),
        },
    )
    fit._write_json_atomic(
        output / "split_half.json",
        {
            "sessions": split_stats,
            "pooled": split_plot(split_stats, split_arrays, output),
        },
    )
    print(
        "Full-unit prediction, task kernels, and split-half comparison complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
