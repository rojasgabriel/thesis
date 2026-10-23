"""Does adding raw video change the task kernels of the V1 GLM?

No new fit. Pair the saved ten trial-fold models of the same 20 pilot units
per mouse, with 10 motion-energy PCs and with those PCs plus 10 raw-video PCs.
Selection, exclusions, first-flash alignment, pre-entry baseline, and
response-entry mask are those of the main GLM. All sampled units are used,
without selection for visual response or regressor importance. Units are the
sampling unit; comparisons are paired and tests are within mouse.
The raw_video_full entry point applies these same comparisons to all eligible
units; this module's own entry point retains the fixed pilot sample.

Undo each design's column scaling and reconstruct task kernels from the
saved coefficients and current bases. Average fold kernels per unit before
comparing models. This compares fold fits, not the baseline final refit with
a candidate fold fit. Windows follow hold_effect.LEVEL_MS (150–500 ms) and
flash_kernels.RESPONSE_MS (40–100 ms); turn direction and response period use
their full 0–1.3 s support, as in kernel_reliability. Signed means can hide
biphasic changes, so also report full-kernel agreement and RMS size ratios.

Scatters show fold SD on both axes, not independent confidence intervals.
Pooled summaries use a hierarchical bootstrap of mice then units,
conditional on the saved fits. Two-sided
Wilcoxon tests compare signed and absolute summary changes, with Holm over
all 12 tests within each mouse. Units share trials; folds share training
trials. There are only two mice. A small change is not evidence of equivalence.

The flash × after-exit modifier in this comparison is uncorrected. Pilot
leak and recovery checks are separate. Parameter changes can reflect shared variance and
new penalties. They do not identify the source of an effect or its cause.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \
        thesis.ephys.analyses.glm.raw_video_kernels
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm.flash_kernels import RESPONSE_MS, SESSIONS
from thesis.ephys.analyses.glm.hold_effect import LEVEL_MS
from thesis.ephys.analyses.glm.inclusion import holm
from thesis.ephys.analyses.glm.kernel_reliability import agreement
from thesis.ephys.analyses.glm.raw_video_test import fingerprint
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    figure_path,
    hierarchical_interval,
    pooled_points,
)

TERMS = {
    dmat.HOLD: ("fixation", LEVEL_MS),
    dmat.FLASH_COMMON: ("flash", RESPONSE_MS),
    dmat.FLASH_JUST_BEFORE: ("flash × short gap", RESPONSE_MS),
    dmat.FLASH_AFTER_EXIT: ("flash × after exit (uncorrected)", RESPONSE_MS),
    dmat.RESPONSE_PERIOD: ("response period", (0, 1300)),
    dmat.TURN_DIRECTION: ("turn direction", (0, 1300)),
}


def task_kernel(coefficients, metadata, name, basis):
    """Undo z-scoring before reconstructing a task kernel.

    >>> basis = np.eye(2)
    >>> meta = {'column_names': ['hold_basis_01', 'hold_basis_02'],
    ...         'training_scale': [2., 4.]}
    >>> task_kernel([6., -8.], meta, 'hold', basis).tolist()
    [3.0, -2.0]
    """
    columns = [
        i
        for i, column in enumerate(metadata["column_names"])
        if column.startswith(f"{name}_basis_")
    ]
    if len(columns) != basis.shape[1]:
        raise ValueError(f"Saved columns do not match the basis: {name}.")
    weights = (
        np.asarray(coefficients)[columns]
        / np.asarray(metadata["training_scale"])[columns]
    )
    values = basis @ weights
    if not np.isfinite(values).all():
        raise ValueError("Task kernels must be finite.")
    return values


def session_kernels(root: Path, unit_set: str = "sample") -> tuple[dict, dict]:
    candidate = root / "raw_video_test"
    old_meta = json.loads((root / "common_design_me.json").read_text())
    new_meta = json.loads((candidate / "common_design_raw.json").read_text())
    for key, path in (
        ("windows", root / "stimulus_windows_me.npz"),
        ("source_design", root / "common_design_me.npy"),
        ("source_metadata", root / "common_design_me.json"),
    ):
        if fingerprint(path) != new_meta["raw_video_test"][key]:
            raise ValueError("Pilot inputs changed since the raw-video comparison.")
    for key in ("column_names", "training_scale", "training_mean"):
        if old_meta[key] != new_meta[key][: len(old_meta[key])]:
            raise ValueError(
                "The paired designs do not share the same original columns."
            )
    records = [
        {
            item["unit_id"]: item
            for item in fit.load_unit_records(path, "unit_*_folds.json")
        }
        for path in (
            root / "all_fit_me/checkpoints",
            candidate / f"{unit_set}_fit_raw/checkpoints",
        )
    ]
    unit_ids = sorted(records[1])
    expected = {
        u["unit_id"]
        for u in fit.analysis_units(
            dmat._load_windows(root / "stimulus_windows_me.npz"), unit_set
        )
    }
    if set(unit_ids) != expected or not expected.issubset(records[0]):
        raise ValueError("Expected the same selected units in both models.")
    bases = dmat.task_temporal_bases()
    bases.update(
        {
            name: dmat.exit_basis()
            for name in (dmat.RESPONSE_PERIOD, dmat.TURN_DIRECTION)
        }
    )
    arrays = {"unit_ids": np.asarray(unit_ids)}
    stats = {}
    for name, (_, window) in TERMS.items():
        basis = bases[name]
        times = 1000 * basis.basis_time
        folds = []
        for unit_id in unit_ids:
            pair = [r[unit_id] for r in records]
            if any(r["folds_scored"] != 10 or len(r["folds"]) != 10 for r in pair):
                raise ValueError("Expected ten matched folds for every unit.")
            folds.append(
                [
                    [
                        task_kernel(fold["coefficients"], meta, name, basis.basis)
                        for fold in record["folds"]
                    ]
                    for record, meta in zip(pair, (old_meta, new_meta), strict=True)
                ]
            )
        folds = np.asarray(folds)  # unit, model, fold, time
        means = folds.mean(2)
        used = (times >= window[0]) & (times < window[1])
        levels = folds[..., used].mean(-1)
        values = levels.mean(2)
        signed_change = values[:, 1] - values[:, 0]
        absolute_change = np.abs(values[:, 1]) - np.abs(values[:, 0])
        with np.errstate(invalid="ignore", divide="ignore"):
            agree = agreement(means[:, 0], means[:, 1])
        both_zero = np.all(means == 0, axis=(1, 2))
        agree[both_zero] = 1
        rms = np.sqrt((means**2).mean(-1))
        ratio = np.divide(
            rms[:, 1],
            rms[:, 0],
            out=np.full(len(unit_ids), np.nan),
            where=rms[:, 0] > 0,
        )
        stats[name] = {
            "window_ms": window,
            "baseline_median": float(np.median(values[:, 0])),
            "raw_video_median": float(np.median(values[:, 1])),
            "median_signed_change": float(np.median(signed_change)),
            "median_absolute_change": float(np.median(absolute_change)),
            "median_full_kernel_agreement": float(np.median(agree)),
            "median_full_kernel_rms_ratio": float(np.nanmedian(ratio)),
            "summary_sign_changes": int((values[:, 0] * values[:, 1] < 0).sum()),
            "signed_p": (
                float(wilcoxon(signed_change).pvalue) if np.any(signed_change) else 1.0
            ),
            "absolute_p": (
                float(wilcoxon(absolute_change).pvalue)
                if np.any(absolute_change)
                else 1.0
            ),
        }
        arrays[f"{name}__times"] = times
        arrays[f"{name}__fold_kernels"] = folds
        arrays[f"{name}__levels"] = values
        arrays[f"{name}__fold_sd"] = levels.std(2, ddof=1)
        arrays[f"{name}__agreement"] = agree
        arrays[f"{name}__rms_ratio"] = ratio
    corrected = holm(
        {
            f"{name}:{kind}": entry[f"{kind}_p"]
            for name, entry in stats.items()
            for kind in ("signed", "absolute")
        }
    )
    for name, entry in stats.items():
        for kind in ("signed", "absolute"):
            entry[f"{kind}_p_holm"] = corrected[f"{name}:{kind}"]
    return {"session": root.name, "unit_ids": unit_ids, "statistics": stats}, arrays


def plot(sessions, arrays, output):
    pooled = {}
    subjects = [s["session"].split("_")[0] for s in sessions]
    with plt.rc_context(FIGURE_STYLE):
        levels, level_axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
        for letter, (name, (label, window)), la in zip(
            "abcdef", TERMS.items(), level_axes.flat, strict=True
        ):
            x = [a[f"{name}__levels"][:, 0] for a in arrays]
            y = [a[f"{name}__levels"][:, 1] for a in arrays]
            for a, first, second in zip(arrays, x, y, strict=True):
                sd = a[f"{name}__fold_sd"]
                la.errorbar(
                    first,
                    second,
                    xerr=sd[:, 0],
                    yerr=sd[:, 1],
                    fmt="none",
                    color=MODEL_COLOR,
                    alpha=0.15,
                    lw=0.5,
                )
            pooled_points(la, x, y, subjects, MODEL_COLOR)
            for text in la.texts:
                if text.get_text() in subjects:
                    text.set_anncoords("axes fraction")
                    text.set_position(
                        (0.65, 0.20 if text.get_text() == "GRB006" else 0.11)
                    )
            low = min(
                np.min(np.r_[a[f"{name}__levels"] - a[f"{name}__fold_sd"]])
                for a in arrays
            )
            high = max(
                np.max(np.r_[a[f"{name}__levels"] + a[f"{name}__fold_sd"]])
                for a in arrays
            )
            pad = max(0.02, 0.1 * (high - low))
            limits = (low - pad, high + pad)
            la.plot(limits, limits, color=REFERENCE_COLOR, ls="--", lw=0.7)
            la.set(
                xlim=limits,
                ylim=limits,
                aspect="equal",
                xlabel="motion energy: mean kernel\n(Δ log rate)",
                ylabel="+ raw video: mean kernel\n(Δ log rate)",
            )
            la.text(-0.18, 1.06, letter, weight="bold", transform=la.transAxes)
            la.text(
                0.03,
                1.06,
                label.replace(" (uncorrected)", "\n(uncorrected)"),
                transform=la.transAxes,
            )
            pooled[name] = {
                "signed_change": hierarchical_interval(
                    [second - first for first, second in zip(x, y, strict=True)]
                ),
                "absolute_change": hierarchical_interval(
                    [
                        np.abs(second) - np.abs(first)
                        for first, second in zip(x, y, strict=True)
                    ]
                ),
            }
        levels.savefig(
            figure_path(output / "raw_video_kernel_levels.svg"), bbox_inches="tight"
        )
        plt.close(levels)
    return pooled


def main():
    output = Path("results/glm")
    results, arrays = zip(
        *(session_kernels(output / name) for name in SESSIONS), strict=True
    )
    pooled = plot(results, arrays, output)
    report = {
        "comparison": "matched fold-mean kernels; no new fit",
        "pooled": pooled,
        "sessions": results,
    }
    (output / "raw_video_kernels.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    for result, values in zip(results, arrays, strict=True):
        np.savez_compressed(
            output / result["session"] / "raw_video_test/kernel_comparison.npz",
            **values,
        )
        print(result["session"], json.dumps(result["statistics"], indent=2))


if __name__ == "__main__":
    main()
