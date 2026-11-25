"""Summarize the paired analysis defined in short_history.py; never refit models.

Require complete real diagnostics for all original fitted neurons and complete
null checkpoints for the original flash-excited subset. Bootstrap neurons in
pairs within mouse; corrected flash intervals additionally resample null draws.
No display trimming or statistical outlier removal. Uncapped results are
reported separately with their available cohort, never mixed with capped fits.
"""

import json

import matplotlib
import numpy as np
from scipy.stats import binomtest, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cap_sensitivity as cap
from thesis.ephys.analyses.glm import fit, leak_test
from thesis.ephys.analyses.glm.short_history import NAMES, REPEATS
from thesis.plotting import FIGURE_STYLE, MODEL_COLOR, save_figure


def paired(values, rng, mc=None):
    """Preserve neuron pairs; optional MC arrays are bootstrap × neuron × variant."""
    values = np.asarray(values)
    keep = np.isfinite(values).all(axis=1)
    y = values[keep]
    if not len(y):
        return {"n": 0, "keep": keep.tolist()}
    indices = rng.integers(len(y), size=(2000, len(y)))
    if mc is None:
        draws = y[indices]
    else:
        draws = mc[:, keep][np.arange(2000)[:, None], indices]
    median = np.median(y, axis=0)
    difference = y[:, 1] - y[:, 0]
    return {
        "n": len(y),
        "keep": keep.tolist(),
        "medians": median.tolist(),
        "median_ci95": np.quantile(
            np.median(draws, axis=1), [0.025, 0.975], axis=0
        ).T.tolist(),
        "paired_change": float(np.median(difference)),
        "paired_change_ci95": np.quantile(
            np.median(draws[:, :, 1] - draws[:, :, 0], axis=1), [0.025, 0.975]
        ).tolist(),
        "paired_change_p": (
            float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0
        ),
        "p_against_zero": [
            float(wilcoxon(v).pvalue) if np.any(v) else 1.0 for v in y.T
        ],
    }


def flash(root, records, source, rng):
    ids = [r["unit_id"] for r in records]
    with np.load(root / "power_test_held_out.npz") as saved:
        zero = saved["alphas"].tolist().index(0.0)
        reference = saved["fit_peaks"][:, zero]
        old = {u: reference[j] for j, u in enumerate(saved["unit_ids"].tolist())}
    real, draws, normal = {}, {}, {}
    missing = []
    for label, name in (
        ("100 ms", "100 ms"),
        ("20 ms", "20 ms"),
        ("uncapped", "20 ms"),
    ):
        actual, nulls, normalized = [], [], []
        for r in records:
            unit = r["unit_id"]
            scale = source["normalization_rate_spikes_s"][
                source["unit_ids"].index(unit)
            ]
            peaks = np.asarray(r["peaks"][name])
            actual.append(leak_test.log_part(peaks)[0])
            if label == "100 ms":
                null = old[unit]
                normalization = 1.0
            elif label == "uncapped" and r["uncapped_unavailable"]:
                missing.append(unit)
                nulls.append(np.full(REPEATS, np.nan))
                normalized.append(np.nan)
                continue
            else:
                key = "uncapped" if label == "uncapped" else "500"
                assert len(r["nulls"]) == REPEATS and all(
                    a[key] is not None for a in r["nulls"]
                )
                null = np.asarray([a[key]["peaks"] for a in r["nulls"]])
                normalization = scale
            nulls.append(leak_test.log_part(null)[:, 0])
            normalized.append(
                leak_test.comparison_part(peaks)[0] / scale
                - np.median(leak_test.comparison_part(null)[:, 0]) / normalization
            )
        real[label] = np.asarray(actual)
        draws[label] = np.asarray(nulls)
        normal[label] = np.asarray(normalized)
    effects, mc, details = {}, {}, {}
    for label in real:
        bias = np.array(
            [
                np.nanmedian(a) if np.isfinite(a).mean() >= 0.5 else np.nan
                for a in draws[label]
            ]
        )
        effects[label] = real[label] - bias
        samples = np.full((2000, len(records)), np.nan)
        for j, a in enumerate(draws[label]):
            a = a[np.isfinite(a)]
            if len(a) and np.isfinite(bias[j]):
                samples[:, j] = real[label][j] - np.median(
                    rng.choice(a, (2000, len(a))), axis=1
                )
        mc[label] = samples
        details[label] = {
            "uncorrected_log_effect": np.where(
                np.isfinite(real[label]), real[label], None
            ).tolist(),
            "bias": np.where(np.isfinite(bias), bias, None).tolist(),
            "corrected_log_effect": np.where(
                np.isfinite(effects[label]), effects[label], None
            ).tolist(),
            "corrected_normalized_effect": np.where(
                np.isfinite(normal[label]), normal[label], None
            ).tolist(),
        }
    comparison = paired(
        np.column_stack([effects[n] for n in NAMES]),
        rng,
        np.stack([mc[n] for n in NAMES], axis=2),
    )
    comparison["unit_ids"] = [
        u for u, k in zip(ids, comparison["keep"], strict=True) if k
    ]
    normalized = paired(np.column_stack([normal[n] for n in NAMES]), rng)
    uncapped = paired(
        np.column_stack([effects["100 ms"], effects["uncapped"]]),
        rng,
        np.stack([mc["100 ms"], mc["uncapped"]], axis=2),
    )
    uncapped["unit_ids"] = [u for u, k in zip(ids, uncapped["keep"], strict=True) if k]
    return {
        "unit_ids": ids,
        "variants": details,
        "comparison": comparison,
        "normalized_comparison": normalized,
        "uncapped_comparison": uncapped,
        "uncapped_unavailable_units": missing,
    }, np.column_stack([effects[n] for n in NAMES])


def draw(ax, values, stats, rng):
    values = values[np.asarray(stats["keep"])]
    if not len(values):
        return
    jitter = rng.uniform(-0.12, 0.12, len(values))
    for offset, row in zip(jitter, values, strict=True):
        ax.plot(np.arange(2) + offset, row, color=".8", alpha=0.4, lw=0.4, zorder=0)
    for i, color in enumerate((".4", MODEL_COLOR)):
        ax.scatter(i + jitter, values[:, i], s=8, alpha=0.45, color=color)
        median = stats["medians"][i]
        lo, hi = stats["median_ci95"][i]
        ax.errorbar(
            i,
            median,
            yerr=[[median - min(lo, median)], [max(hi, median) - median]],
            fmt="o",
            color=color,
            capsize=3,
        )
    ax.set(xticks=[0, 1], xticklabels=NAMES, xlim=(-0.35, 1.35))
    for tick, color in zip(ax.get_xticklabels(), (".4", MODEL_COLOR), strict=True):
        tick.set_color(color)


def summarize(root, out):
    rng = np.random.default_rng(20261119)
    source = json.loads((root / "flash_movement.json").read_text())["sessions"]
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(9, 12), layout="constrained")
        for column, session in enumerate(cap.SESSIONS):
            original = fit.load_unit_records(
                root / session / "all_fit_me/checkpoints", "unit_*_folds.json"
            )
            rows = [
                json.loads(p.read_text())
                for p in sorted((out / session).glob("*/real.json"))
            ]
            assert {r["unit_id"] for r in rows} == {r["unit_id"] for r in original}, (
                "Incomplete real fits"
            )
            with np.load(root / session / "power_test_held_out.npz") as saved:
                excited = set(saved["unit_ids"].tolist())
            completed = [
                json.loads((out / session / f"unit_{u}" / "complete.json").read_text())
                for u in sorted(excited)
            ]
            subject = session.split("_")[0]
            ratios = np.array(
                [[r["diagnostics"][n]["calibration"] for n in NAMES] for r in rows]
            )
            error = np.abs(ratios - 1).mean(axis=2)
            bits = np.array(
                [
                    [
                        np.mean(
                            [a["bits_per_spike"] for a in r["diagnostics"][n]["scores"]]
                        )
                        for n in NAMES
                    ]
                    for r in rows
                ]
            )
            failed = np.array(
                [[r["diagnostics"][n]["failures"] for n in NAMES] for r in rows]
            )
            total = np.array(
                [[r["diagnostics"][n]["trajectories"] for n in NAMES] for r in rows]
            )
            fraction = failed / total
            entry = {
                "unit_ids": [r["unit_id"] for r in rows],
                "calibration": paired(error, rng),
                "prediction": paired(bits, rng),
                "dynamics": paired(fraction, rng),
                "calibration_low_medians": np.median(
                    np.median(ratios[:, :, 1:14], axis=2), axis=0
                ).tolist(),
                "calibration_high_medians": np.median(
                    np.median(ratios[:, :, 15:19], axis=2), axis=0
                ).tolist(),
                "calibration_curves": np.median(ratios, axis=0).tolist(),
                "flagged_units": (failed > 0).sum(axis=0).tolist(),
                "failed_trajectories": failed.sum(axis=0).tolist(),
                "trajectories": total.sum(axis=0).tolist(),
            }
            resolved = int(((failed[:, 0] > 0) & (failed[:, 1] == 0)).sum())
            introduced = int(((failed[:, 0] == 0) & (failed[:, 1] > 0)).sum())
            entry["stability_transitions"] = {
                "resolved": resolved,
                "introduced": introduced,
                "paired_binary_p": (
                    float(binomtest(resolved, resolved + introduced, 0.5).pvalue)
                    if resolved + introduced
                    else 1.0
                ),
            }
            entry["flash"], effects = flash(
                root / session, completed, source[subject], rng
            )
            for row, values, key, ylabel in (
                (0, error, "calibration", "Mean |observed / predicted − 1|"),
                (
                    1,
                    fraction,
                    "dynamics",
                    "Fraction of trajectories crossing threshold",
                ),
                (2, bits, "prediction", "Held-out bits / spike"),
                (3, effects, "flash", "Corrected flash × exit log-ratio contribution"),
            ):
                ax = axes[row, column]
                stats = entry["flash"]["comparison"] if key == "flash" else entry[key]
                draw(ax, values, stats, rng)
                ax.set_ylabel(ylabel)
                if row in (1, 2, 3):
                    ax.axhline(0, color="black", lw=0.6)
                if row == 1:
                    ax.set_yscale("symlog", linthresh=0.0001)
                    ax.set_ylim(bottom=0)
                ax.text(
                    0.02,
                    1.02,
                    f"{subject} · n={stats['n']}",
                    transform=ax.transAxes,
                    va="bottom",
                )
                ax.text(
                    -0.13,
                    1.04,
                    "abcdefgh"[row * 2 + column],
                    transform=ax.transAxes,
                    weight="bold",
                )
            report[subject] = entry
        fig.savefig(out / "short_history.png", dpi=160, bbox_inches="tight")
        save_figure(fig, out / "short_history.svg")
    fit._write_json_atomic(out / "summary.json", report)
    print(
        json.dumps(
            {
                s: {
                    k: v
                    for k, v in e.items()
                    if k not in ("unit_ids", "flash", "calibration_curves")
                }
                for s, e in report.items()
            },
            indent=2,
        )
    )
