"""Can the task weights of the raw-video pilot be read?

Use the same 20 sampled units per mouse, completed trials, first-flash 1-ms
grid, response-entry mask, pre-entry baseline and exclusions as the main GLM.
Refit the added-video model on all valid trials at the median fold penalty.
Then check collinearity, recovery from known parameters, and leak when the
raw flash × after-exit regressors are zero, with the intercept adjusted for
column centering to keep the in-port baseline fixed. Main-model fit records
are read only.

Collinearity uses the observed unit history and all valid rows. Motion energy
and raw video are separate 10-column groups. The baseline is the identical
correlation submatrix with the raw columns removed, including the same history.
Recovery and leak reuse the simulator and fixed-penalty fitter of the existing
checks: three simulations per unit, seed 0 + unit ID, causal simulated history,
and a cap of 0.5 spikes/bin. The baseline recovery and leak are saved checks,
matched by unit ID; recovery is conditional on each model's own fitted truth,
not a comparison of two fits to the same simulated spikes.

The fixation recovery ratio uses 150–500 ms and only paired units with both
true levels above 0.1 in magnitude. All units enter kernel-recovery and leak
statistics. The flash kernel measure uses 40–100 ms. Matched peaks use only
the existing StimulusResponse excited classification, saved as a read-only
snapshot. The sampling unit is the unit; plots pool mice and bootstrap mice
then units. Tests are within mouse; units share trials. Simulation uncertainty
is conditional on the fitted truth and selected penalty. No split-half or
all-unit validation is performed by this pilot.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \
        thesis.ephys.analyses.glm.raw_video_trust --workers 2
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import collinearity, dmat, fit, leak_test, peaks
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.inclusion import holm
from thesis.ephys.analyses.glm.kernel_reliability import agreement
from thesis.ephys.analyses.glm.parameter_recovery import weight_vectors
from thesis.ephys.analyses.glm.raw_video_test import fingerprint, prepare
from thesis.ephys.analyses.glm.unique_dev import block_slices
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    figure_path,
    hierarchical_interval,
    pooled_strip,
)

SIMULATIONS = 3
_SHARED = {}
TARGETS = (dmat.HOLD, dmat.FLASH_COMMON, dmat.FLASH_AFTER_EXIT, dmat.TURN_DIRECTION)
LABELS = ("fixation", "flash", "flash × exit", "turn")


def groups(metadata, components):
    """Keep raw video separate from motion energy for group diagnostics."""
    result = {
        k: v for k, v in block_slices(metadata, components).items() if k != "task"
    }
    if components == 20:
        start = metadata["base_columns"]
        result["video"] = slice(start, start + 10)
        result["raw_video"] = slice(start + 10, start + 20)
    return result


def vectors(record, metadata, scale):
    values = weight_vectors(record, metadata, scale)
    if int(record["components"]) == 20:
        values["raw_video"] = values["video"][10:]
        values["video"] = values["video"][:10]
    return values


def _load(root):
    root = Path(root)
    design = root / "raw_video_test/common_design_raw.npy"
    leak_test._load(root / "stimulus_windows_me.npz", design)
    width = int(leak_test._SHARED["metadata"]["base_columns"]) + 20
    collinearity._load(root / "stimulus_windows_me.npz", design, width)
    _SHARED["old_metadata"] = json.loads((root / "common_design_me.json").read_text())
    for name in ("parameter_recovery", "leak_test"):
        with np.load(root / f"{name}.npz", allow_pickle=False) as data:
            _SHARED[name] = {key: data[key].copy() for key in data.files}


def _unit(job):
    path = Path(job["output"])
    if path.exists():
        return json.loads(path.read_text())
    unit_id = job["unit_id"]
    shared = leak_test._SHARED
    prepared, valid, metadata = shared["prepared"], shared["valid"], shared["metadata"]
    record = job["record"]
    raw = dmat.build_unit_history(prepared["alignments"], job["spike_times"])
    history, mean, scale = dmat.training_zscore(raw, valid)
    columns = np.column_stack((collinearity._SHARED["common"], history[valid]))
    correlation = columns.T @ columns / len(columns)
    del columns
    old_meta = _SHARED["old_metadata"]
    width = int(metadata["base_columns"]) + 20
    old_width = int(old_meta["base_columns"]) + 10
    old_columns = np.r_[
        np.arange(old_width), np.arange(width, width + dmat.HISTORY_COLUMNS)
    ]
    col = {
        "baseline": collinearity.group_statistics(
            correlation[np.ix_(old_columns, old_columns)], groups(old_meta, 10)
        ),
        "raw_video": collinearity.group_statistics(correlation, groups(metadata, 20)),
    }
    saved = _SHARED["parameter_recovery"]
    old_position = int(np.flatnonzero(saved["unit_ids"] == unit_id)[0])
    old_true = vectors(job["baseline"], old_meta, scale)
    # Do not compare the pilot with a recovery file from an older main fit.
    for name, values in old_true.items():
        np.testing.assert_allclose(
            values, saved[f"true__{name}"][old_position], rtol=1e-8, atol=1e-10
        )
    true = vectors(record, metadata, scale)
    truth = leak_test.measures(record, scale)
    rng = np.random.default_rng(unit_id)
    recovered, measures = [], []
    for _ in range(SIMULATIONS):
        counts, simulated_history, simulated_scale = leak_test.simulate(
            record, np.asarray(record["coefficients"]), mean, scale, rng
        )
        design = dmat._design_with_history(
            shared["common"], simulated_history, width, valid
        )
        model = fit.fit_poisson_at_alpha(design, counts.ravel()[valid], record["alpha"])
        fitted = {"components": 20, "coefficients": model.coef_}
        recovered.append(vectors(fitted, metadata, simulated_scale))
        measures.append(leak_test.measures(fitted, simulated_scale))
    recovery = {}
    for name, values in true.items():
        simulated = np.asarray([r[name] for r in recovered])
        recovery[name] = {
            "agreement": float(
                np.mean([agreement(values[None], v[None])[0] for v in simulated])
            ),
            "size": float(
                np.mean(np.linalg.norm(simulated, axis=1) / np.linalg.norm(values))
            ),
            "true": values.tolist(),
            "recovered": simulated.mean(0).tolist(),
        }
    baseline_recovery = {
        str(name): {
            "agreement": float(saved["agreement"][old_position, i]),
            "size": float(saved["size"][old_position, i]),
        }
        for i, name in enumerate(saved["regressors"])
    }
    hold_times = 1000 * dmat.task_temporal_bases()[dmat.HOLD].basis_time
    window = (hold_times >= leak_test.LEVEL_MS[0]) & (
        hold_times < leak_test.LEVEL_MS[1]
    )
    fixation = {
        "true": float(truth[2]),
        "recovered": np.asarray(measures)[:, 2].tolist(),
        "baseline_true": float(
            saved[f"true__{dmat.HOLD}"][old_position, window].mean()
        ),
        "baseline_recovered": float(
            saved[f"recovered__{dmat.HOLD}"][old_position, window].mean()
        ),
    }
    null = leak_test._unit({**job, "alpha": record["alpha"]})
    saved_leak = _SHARED["leak_test"]
    old_position = int(np.flatnonzero(saved_leak["unit_ids"] == unit_id)[0])
    np.testing.assert_allclose(
        leak_test.measures(job["baseline"], scale),
        saved_leak["real"][old_position],
        atol=1e-10,
    )
    leak = {
        "real": null["real"].tolist(),
        "null": null["leak"].tolist(),
        "real_part": null["real_part"],
        "null_part": null["leak_part"].tolist(),
        "baseline_real": saved_leak["real"][old_position].tolist(),
        "baseline_null": saved_leak["leak"][old_position].tolist(),
        "baseline_real_part": float(saved_leak["real_part"][old_position]),
        "baseline_null_part": saved_leak["leak_part"][old_position].tolist(),
        "null_spikes_over_real": null["spikes_simulated_over_real"],
    }
    result = {
        "unit_id": unit_id,
        "excited": job["excited"],
        "firing_rate": record["fit_mean_count"] / dmat.BINWIDTH_S,
        "collinearity": col,
        "recovery": recovery,
        "baseline_recovery": baseline_recovery,
        "fixation": fixation,
        "leak": leak,
    }
    fit._write_json_atomic(path, result)
    return result


def recovery_ratios(units):
    """Keep ratios paired and exclude a weak planted level in either model.

    >>> units = [
    ... {'unit_id': 1, 'fixation': {'true': -.4, 'recovered': [-.2],
    ...  'baseline_true': -.5, 'baseline_recovered': -.25}},
    ... {'unit_id': 2, 'fixation': {'true': -.05, 'recovered': [-.2],
    ...  'baseline_true': -.5, 'baseline_recovered': -.25}}]
    >>> before, after, ids = recovery_ratios(units)
    >>> before.tolist(), after.tolist(), ids
    ([0.5], [0.5], [1])
    """
    used = [
        u
        for u in units
        if min(abs(u["fixation"]["true"]), abs(u["fixation"]["baseline_true"]))
        > leak_test.MIN_LEVEL
    ]
    before = np.asarray(
        [
            u["fixation"]["baseline_recovered"] / u["fixation"]["baseline_true"]
            for u in used
        ]
    )
    after = np.asarray(
        [np.mean(u["fixation"]["recovered"]) / u["fixation"]["true"] for u in used]
    )
    return before, after, [u["unit_id"] for u in used]


def summarize(units):
    report = {"units": len(units), "simulations_per_unit_per_check": SIMULATIONS}
    for model, source in (("baseline", "baseline_recovery"), ("raw_video", "recovery")):
        entries = [u["collinearity"][model] for u in units]
        maximum = [max(e["gvif"].values()) for e in entries]
        index = [e["condition_indices"][0] for e in entries]
        report[f"{model}_collinearity"] = {
            "gvif": {
                name: {
                    "median": float(np.median([e["gvif"][name] for e in entries])),
                    "max": float(max(e["gvif"][name] for e in entries)),
                }
                for name in entries[0]["gvif"]
            },
            "largest_gvif": float(max(maximum)),
            "units_above_gvif_5": int(np.sum(np.array(maximum) > 5)),
            "largest_condition_index": float(max(index)),
            "median_condition_index": float(np.median(index)),
            "units_above_condition_index_30": int(np.sum(np.array(index) > 30)),
        }
        report[f"{model}_recovery"] = {
            name: {
                key: float(np.median([u[source][name][key] for u in units]))
                for key in ("agreement", "size")
            }
            for name in units[0][source]
        }
        reliable = [u for u in units if u["firing_rate"] >= 5]
        report[f"{model}_history_at_least_5_spikes_s"] = {
            "units": len(reliable),
            **{
                key: float(np.median([u[source]["history"][key] for u in reliable]))
                for key in ("agreement", "size")
            },
        }
    before, after, ids = recovery_ratios(units)
    report["fixation_level_recovery"] = {
        "paired_units": len(ids),
        "unit_ids": ids,
        "baseline_median_ratio": float(np.median(before)),
        "raw_video_median_ratio": float(np.median(after)),
        "median_paired_change": float(np.median(after - before)),
    }
    tests = {}
    for prefix, model in (("baseline_", "baseline"), ("", "raw_video")):
        real = np.array([u["leak"][prefix + "real"][0] for u in units])
        null = np.array([np.mean(u["leak"][prefix + "null"], axis=0)[0] for u in units])
        corrected = real - null
        tests[f"{model}_leak"] = float(wilcoxon(null).pvalue) if np.any(null) else 1.0
        tests[f"{model}_corrected"] = (
            float(wilcoxon(corrected).pvalue) if np.any(corrected) else 1.0
        )
        excited = [u for u in units if u["excited"]]
        peak = np.array(
            [
                u["leak"][prefix + "real_part"]
                - np.mean(u["leak"][prefix + "null_part"])
                for u in excited
            ]
        )
        report[f"{model}_leak"] = {
            "kernel_median_real": float(np.median(real)),
            "kernel_median_null": float(np.median(null)),
            "kernel_median_corrected": float(np.median(corrected)),
            "excited_units": [u["unit_id"] for u in excited],
            "matched_peak_median_corrected": (
                float(np.median(peak)) if len(peak) else None
            ),
            "matched_peak_corrected_by_unit": peak.tolist(),
        }
    report["kernel_leak_tests_holm"] = holm(tests)
    return report


def plot(sessions, output):
    pooled = {}
    subjects = [s["session"].split("_")[0] for s in sessions]
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
        for letter, axis in zip("abcdef", axes.flat, strict=True):
            axis.text(-0.18, 1.07, letter, weight="bold", transform=axis.transAxes)
        for column, (key, text, limit) in enumerate(
            (
                ("gvif", "largest group GVIF^(1/p)", 5),
                ("condition_indices", "largest condition index", 30),
            )
        ):
            ax = axes[0, column]
            for pos, (model, color) in enumerate(
                (("baseline", "black"), ("raw_video", MODEL_COLOR))
            ):
                values = [
                    np.array(
                        [
                            (
                                max(u["collinearity"][model][key].values())
                                if key == "gvif"
                                else u["collinearity"][model][key][0]
                            )
                            for u in s["units"]
                        ]
                    )
                    for s in sessions
                ]
                pooled_strip(ax, pos, values, subjects, color)
            ax.axhline(limit, color=REFERENCE_COLOR, ls="--", lw=0.8)
            ax.set(
                ylabel=text,
                xticks=[0, 1],
                xticklabels=["current", "+ raw video"],
                xlim=(-0.3, 1.7),
            )
            ax.set_ylim(bottom=3.1 if column == 0 else 6.5)
        ax = axes[0, 2]
        for x, label, color in (
            (0.05, "current", "black"),
            (0.48, "+ raw video", MODEL_COLOR),
        ):
            ax.text(x, 1.02, label, color=color, transform=ax.transAxes)
        rng = np.random.default_rng(0)
        for position, name in enumerate(TARGETS):
            for offset, (source, color) in zip(
                (-0.15, 0.15),
                (("baseline_recovery", "black"), ("recovery", MODEL_COLOR)),
                strict=True,
            ):
                values = [
                    np.array([u[source][name]["agreement"] for u in s["units"]])
                    for s in sessions
                ]
                summary = hierarchical_interval(values)
                points = np.concatenate(values)
                ax.scatter(
                    position + offset + rng.uniform(-0.06, 0.06, len(points)),
                    points,
                    s=5,
                    color=color,
                    alpha=0.2,
                    lw=0,
                )
                lo, hi = summary["interval_95"]
                median = summary["median"]
                ax.errorbar(
                    position + offset,
                    median,
                    yerr=[[median - lo], [hi - median]],
                    fmt="o",
                    color=color,
                    ms=3,
                    lw=0.8,
                )
            ax.set(
                xticks=range(len(TARGETS)),
                xticklabels=LABELS,
                ylabel="recovery agreement",
            )
        for col, (name, label) in enumerate(
            (
                ("fixation", "fixation level recovered / true"),
                ("null", "flash × exit leak (Δ log rate)"),
                ("corrected", "real − leak (Δ log rate)"),
            )
        ):
            ax = axes[1, col]
            for pos, (prefix, color) in enumerate(
                (("baseline_", "black"), ("", MODEL_COLOR))
            ):
                if name == "fixation":
                    values = [recovery_ratios(s["units"])[pos] for s in sessions]
                else:
                    values = []
                    for s in sessions:
                        null = np.array(
                            [
                                np.mean(u["leak"][prefix + "null"], axis=0)[0]
                                for u in s["units"]
                            ]
                        )
                        real = np.array(
                            [u["leak"][prefix + "real"][0] for u in s["units"]]
                        )
                        values.append(null if name == "null" else real - null)
                pooled_strip(ax, pos, values, subjects, color)
                pooled[f"{prefix}{name}"] = hierarchical_interval(values)
            ax.axhline(
                1 if name == "fixation" else 0, color=REFERENCE_COLOR, ls="--", lw=0.8
            )
            ax.set(
                ylabel=label,
                xticks=[0, 1],
                xticklabels=["current", "+ raw video"],
                xlim=(-0.3, 1.7),
            )
        for ax in axes.flat:
            for tick in ax.get_xticklabels():
                if tick.get_text() == "+ raw video":
                    tick.set_color(MODEL_COLOR)
        fig.savefig(figure_path(output / "raw_video_trust.svg"), bbox_inches="tight")
        plt.close(fig)
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = Path("results/glm")
    summary_path = output / "raw_video_trust.json"
    if args.plot_only:
        report = json.loads(summary_path.read_text())
        plot(report["sessions"], output)
        return
    classification_path = output / "raw_video_trust_classification.json"
    if not classification_path.exists():
        classes = {}
        for session in SESSIONS:
            subject, stamp = session.split("_", 1)
            classes[session] = sorted(peaks.excited_units(subject, stamp))
        fit._write_json_atomic(classification_path, classes)
    classes = json.loads(classification_path.read_text())
    sessions = []
    for session in SESSIONS:
        root = output / session
        design = prepare(root)
        fit.refit_final(
            root / "stimulus_windows_me.npz",
            design,
            "sample",
            root / "raw_video_test/sample_fit_raw",
            20,
            args.workers,
        )
        prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
        units = fit.analysis_units(prepared, "sample")
        candidate = {
            r["unit_id"]: r
            for r in fit.load_unit_records(
                root / "raw_video_test/sample_fit_raw/checkpoints", "unit_*_final.json"
            )
        }
        baseline = {
            r["unit_id"]: r
            for r in fit.load_unit_records(
                root / "all_fit_me/checkpoints", "unit_*_final.json"
            )
        }
        if set(candidate) != {u["unit_id"] for u in units}:
            raise ValueError("Final candidate models do not match the 20 pilot units.")
        destination = root / "raw_video_test/trust"
        destination.mkdir(exist_ok=True)
        inputs = {
            str(p.relative_to(root)): fingerprint(p)
            for p in (
                design,
                design.with_suffix(".json"),
                root / "stimulus_windows_me.npz",
                root / "parameter_recovery.npz",
                root / "leak_test.npz",
                *(
                    root
                    / "raw_video_test/sample_fit_raw/checkpoints"
                    / f"unit_{u['unit_id']}_final.json"
                    for u in units
                ),
                *(
                    root / "all_fit_me/checkpoints" / f"unit_{u['unit_id']}_final.json"
                    for u in units
                ),
            )
        }
        manifest = {
            "inputs": inputs,
            "classifications": fingerprint(classification_path),
            "simulations": SIMULATIONS,
            "seed": 0,
            "method": 2,
        }
        path = destination / "inputs.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(
                "Trust-check inputs changed; preserve the old folder and use a new run."
            )
        fit._write_json_atomic(path, manifest)
        jobs = [
            {
                **u,
                "record": candidate[u["unit_id"]],
                "baseline": baseline[u["unit_id"]],
                "excited": u["unit_id"] in classes[session],
                "output": str(destination / f"unit_{u['unit_id']}.json"),
            }
            for u in units
        ]
        results = []
        with Pool(args.workers, _load, (root,)) as pool:
            for result in pool.imap(_unit, jobs):
                results.append(result)
                print(
                    f"{session}: trust check {len(results)}/{len(jobs)}, unit {result['unit_id']}",
                    flush=True,
                )
        entry = {"session": session, "statistics": summarize(results), "units": results}
        fit._write_json_atomic(destination / "summary.json", entry["statistics"])
        sessions.append(entry)
    pooled = plot(sessions, output)
    report = {
        "comparison": "40-unit pilot, own-model recovery; cached baseline checks by unit ID",
        "sessions": sessions,
        "pooled": pooled,
    }
    fit._write_json_atomic(summary_path, report)
    for s in sessions:
        print(s["session"], json.dumps(s["statistics"], indent=2))


if __name__ == "__main__":
    main()
