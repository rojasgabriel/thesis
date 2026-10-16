"""Test whether the weights of the V1 GLM can be read.

Question: the design is fixed (`design_optimization.py` chose the form of each
term by prediction). Can its weights be trusted? Four tests, each on the
design in use, in both mice (`SESSIONS`; GRB064 took no part in the design
choices). The units of both mice are pooled in the plots; the black marker is
the pooled median with its hierarchical-bootstrap 95% interval (mice, then
units), and the open circles are the median of each mouse (left: GRB006,
right: GRB064). The tests are within each mouse.

- Collinearity (panels a to c, `collinearity.py`): a property of the design
  matrix only, on the rows of the fit (task and motion-energy columns plus the
  spike-history columns of the unit). a, the generalized variance inflation
  factor of each regressor (Fox and Monette 1992), GVIF = det(R_gg) det(R_rest)
  / det(R), with R the correlation matrix of all columns, R_gg that of the
  columns of the regressor, and R_rest that of the others, shown as GVIF^(1/p)
  for p columns: on the scale of a VIF, 1 / (1 - R^2) of a column predicted
  from the others. The usual limits 5 and 10 (R^2 = 0.8 and 0.9) are
  conventions. b, the condition indices of each unit, sqrt(largest
  eigenvalue / each eigenvalue) of R (Belsley, Kuh, and Welsch 1980): a large
  index is a weighted sum of columns that is almost 0; 10 to 30 is a moderate
  and above 30 a strong dependency, from their simulations. c, the first
  canonical correlation of each pair of regressors (lower triangle GRB006,
  upper triangle GRB064; for a pair with spike history, the median over
  units). These are statistics of a fit with no penalty; the L2 penalty makes
  the weights more stable.
- Parameter recovery (d and e, `parameter_recovery.py`): bias of the fit with
  a known truth. The weights of the final refit of each unit are the truth;
  spikes are simulated from them (Poisson, with the history of the simulated
  spikes; 3 simulations), and the model is fit to them with the same design
  and penalty. For each regressor, the true kernel k and the recovered kernel
  k' (its basis times its weights): d, the agreement 1 - |k' - k|^2 / (|k'|^2
  + |k|^2) (1 for equal kernels, 0 for unrelated, below 0 for opposite); e,
  the size |k'| / |k|. A point outside `RECOVERY_LIMITS` is drawn at the
  limit; it is in the statistics.
- Split halves (f and g, `kernel_reliability.py`): variance of the fit on
  real data, with no truth. Each unit is fit on the even and on the odd
  trials, with the penalty of its final refit. f, the agreement of the two
  kernels of each regressor (as in d), for the units that need the regressor
  (unique ΔD² above 0 over the folds; for other units the kernel is near 0
  and the agreement has no meaning). g, the one value of the fixation kernel
  that `hold_effect.py` reads, its mean over `LEVEL_MS`, in the two halves,
  for the units that need the fixation kernel, with the Spearman correlation
  in each mouse. A point outside `LEVEL_LIMITS` is drawn at the limit.
- Leak (h to j, `leak_test.py`): a bias of one result. A flash kernel summed
  over the flash train after the exit is close to a level, so the L2 penalty
  can give part of the rate step at the exit to the flash × move kernels and
  make a flash × move effect from nothing. Spikes are simulated from the
  final refit of each unit with the flash × move weights set to 0, and the
  full model is fit to them. The leak is the flash × move part of the move −
  still peak of that fit, in the matched peak measure of `flash_by_period.py`
  (`peaks.py`; the excited units). h, the real part against the leak; i, real
  minus leak (two-sided Wilcoxon signed-rank test in each mouse, and the
  hierarchical bootstrap); j, the leak against the fixation level of the unit.
  A point outside `PART_LIMITS` is drawn at the limit; it is in the
  statistics.

Units, trials, alignment, and exclusions are those of `fit.py`. The sampling
unit is the unit. This script makes no fit; it writes `model_trust.svg`,
`model_trust_stats.json`, and `model_trust_caption.md` in `--out`.

    uv run python -m thesis.ephys.analyses.glm.model_trust

Run `collinearity`, `parameter_recovery`, `kernel_reliability --refit`, and
`leak_test --refit` first; this script stops if a result file is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import peaks
from thesis.ephys.analyses.glm.collinearity import INDEX_LIMITS, VIF_LIMITS
from thesis.ephys.analyses.glm.flash_by_period import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import LEVEL_LIMITS, LEVEL_MS
from thesis.ephys.analyses.glm.kernel_reliability import kernel_times
from thesis.ephys.analyses.glm.leak_test import PART_LIMITS
from thesis.ephys.analyses.glm.main_figure import REGRESSORS, TASK, VISUAL
from thesis.plotting import (
    FIGURE_STYLE,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    peak_axes,
    pooled_points,
    pooled_strip,
    save_figure,
)

# Axis limits of the recovery panels; a unit outside them is drawn at the limit.
RECOVERY_LIMITS = {"agreement": (-0.5, 1.05), "size": (0, 2)}
CAPTION = """\
**Can the weights of the model be read?** The design in use, in {subject} and
in {held_out} (an animal that took no part in the design choices). Units of
both mice are pooled; the black marker is the pooled median with its
hierarchical-bootstrap 95% interval, and the open circles are the median of
each mouse (left: {subject}; right: {held_out}). **a–c, Collinearity of the
design**, for the design of each unit. **a**, The generalized variance
inflation factor of each regressor, as GVIF^(1/p); the dashed lines are the
usual limits of 5 and 10. The largest value in either animal is {gvif:.2f}.
**b**, The condition indices of each unit, sorted; the dashed lines are 10 and
30. The largest is {index:.1f}. **c**, The first canonical correlation of each
pair of regressors (0, nothing shared; 1, a time course that both can make).
Lower triangle: {subject}; upper triangle: {held_out}. **d–e, Recovery of known
weights**: spikes are simulated from the fitted model of each unit
({simulations} simulations), the model is fit to them, and each recovered
kernel is compared with the true one. **d**, Agreement, 1 − |a − b|² / (|a|² +
|b|²): 1 for equal kernels, 0 for unrelated kernels. **e**, Size of the
recovered kernel over the true size. Every regressor is recovered with a
median agreement of {lowest_agreement:.2f} or more in both animals. The level
of the fixation kernel is recovered at {level_in_use} of its true size: the L2
penalty gives a part of it to the flash × move kernels (h–j). **f–g, The same
weights from independent trials**: each unit is fit on the even and on the
odd trials. **f**, Agreement of the two kernels, for the units that need the
regressor. Every regressor has a median agreement of {lowest_half:.2f} or
more. **g**, The fixation level that the fixation figure reads ({lo} to {hi}
ms after the first flash) in the two halves. {level_halves} **h–j, The flash ×
move effect that the fit makes with none in the data.** **h**, The flash ×
move part of the move − still flash peak (spikes/s; the excited units) in the
real fit against the leak: the same part in fits to spikes simulated with no
flash × move effect. **i**, Real minus leak. {leak} **j**, The leak against the
fixation level of the unit.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    args = parser.parse_args()
    stats = {}
    rng = np.random.default_rng(0)
    roots = [args.out / session for session in SESSIONS]
    measures, recovered, halves, leaks = [], [], [], []
    for root in roots:
        for name, store in (
            ("collinearity", measures),
            ("parameter_recovery", recovered),
            ("kernel_reliability", halves),
            ("leak_test", leaks),
        ):
            path = root / f"{name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path}: run `{name}` first.")
            with np.load(path) as saved:
                store.append({key: saved[key] for key in saved.files})
    subjects = [root.name.split("_")[0] for root in roots]
    with plt.rc_context(FIGURE_STYLE):
        # The dpi is for the rasterized points; the text and the lines are vectors.
        figure = plt.figure(figsize=(7.4, 8.4), layout="constrained", dpi=300)
        bottom, recovery, leak_row = figure.subfigures(
            3, 1, height_ratios=(3.0, 2.6, 2.4)
        )
        gvif_axis, index_axis, pair_axis = bottom.subplots(
            1, 3, width_ratios=(1.35, 0.7, 1.5)
        )
        names = measures[0]["regressors"].tolist()
        order = np.argsort(-np.median(measures[0]["gvif"], 0))

        def by_regressor(axis, values_of, regressors, limits):
            """Units of both mice pooled for each regressor, with the medians.

            Black: the pooled median with its hierarchical-bootstrap 95%
            interval. Open circles: the median of the first mouse (left) and
            of the held-out mouse (right).
            """
            for position, name in enumerate(regressors):
                groups = [values[name] for values in values_of]
                column = np.concatenate(groups)
                axis.scatter(
                    position - 0.1 + rng.uniform(-0.15, 0.15, len(column)),
                    np.clip(column, *limits),
                    s=2,
                    color=REGRESSORS[name][1],
                    alpha=0.45,
                    lw=0,
                    rasterized=True,
                )
                summary = hierarchical_interval(groups)
                low, high = np.clip(summary["interval_95"], *limits)
                median = np.clip(summary["median"], *limits)
                axis.errorbar(
                    position + 0.15,
                    median,
                    yerr=[[median - low], [high - median]],
                    fmt="o",
                    ms=3,
                    color="black",
                    elinewidth=0.8,
                    capsize=1.5,
                    zorder=4,
                )
                for k, mouse in enumerate(summary["median_of_each_mouse"]):
                    axis.scatter(
                        position + 0.27 + 0.1 * k,
                        np.clip(mouse, *limits),
                        s=8,
                        facecolor="white",
                        edgecolor="black",
                        lw=0.6,
                        zorder=4,
                    )
            axis.set_xticks(
                range(len(regressors)),
                [REGRESSORS[name][0] for name in regressors],
                rotation=40,
                ha="right",
                rotation_mode="anchor",
                fontsize=5.5,
            )
            for tick, name in zip(axis.get_xticklabels(), regressors):
                tick.set_color(REGRESSORS[name][1])

        ordered = [names[column] for column in order]
        by_regressor(
            gvif_axis,
            [
                {n: m["gvif"][:, m["regressors"].tolist().index(n)] for n in ordered}
                for m in measures
            ],
            ordered,
            (0.9, 12),
        )
        for value in VIF_LIMITS:
            gvif_axis.axhline(value, color=REFERENCE_COLOR, ls="--", lw=0.8)
        gvif_axis.set_yscale("log")
        gvif_axis.set_ylim(0.9, 12)
        gvif_axis.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
        gvif_axis.set_ylabel("variance inflation,\n" + r"GVIF$^{1/p}$")
        for m in measures:
            for unit in m["condition_indices"]:
                index_axis.plot(unit, color="0.3", alpha=0.12, lw=0.5, rasterized=True)
        index_axis.text(
            0.97,
            0.97,
            f"{sum(len(m['condition_indices']) for m in measures)} units, {len(measures)} mice",
            transform=index_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        for value in INDEX_LIMITS:
            index_axis.axhline(value, color=REFERENCE_COLOR, ls="--", lw=0.8)
        index_axis.set_yscale("log")
        index_axis.set_ylim(0.9, 60)
        index_axis.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        index_axis.set_xlabel("dimension of the\ndesign, sorted")
        index_axis.set_ylabel("condition index")
        # One animal in each triangle of the matrix of shared directions.
        lower = measures[0]["shared_directions"]
        upper = measures[1]["shared_directions"][
            np.ix_(
                [measures[1]["regressors"].tolist().index(n) for n in names],
                [measures[1]["regressors"].tolist().index(n) for n in names],
            )
        ]
        below = np.tri(len(names), k=-1, dtype=bool)
        pairs = np.where(below, lower, np.where(below.T, upper, np.nan))
        pair_axis.imshow(pairs, cmap="Greys", vmin=0, vmax=1)
        for (row, column), value in np.ndenumerate(pairs):
            if np.isfinite(value):
                pair_axis.text(
                    column,
                    row,
                    f"{value:.2f}"[1:],
                    ha="center",
                    va="center",
                    fontsize=3.8,
                    color="white" if value > 0.55 else "black",
                )
        labels = [REGRESSORS[name][0] for name in names]
        pair_axis.set_xticks(
            range(len(labels)),
            labels,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=5.5,
        )
        pair_axis.set_yticks(range(len(labels)), labels, fontsize=5.5)
        for ticks in (pair_axis.get_xticklabels(), pair_axis.get_yticklabels()):
            for tick, name in zip(ticks, names):
                tick.set_color(REGRESSORS[name][1])
        pair_axis.spines[["left", "bottom"]].set_visible(False)
        pair_axis.tick_params(length=0)
        pair_axis.set_title(
            f"lower: {subjects[0]}, upper: {subjects[1]}", fontsize=6, loc="left"
        )
        stats["collinearity"] = {
            subject: {
                "largest_gvif_per_column": float(m["gvif"].max()),
                "largest_condition_index": float(m["condition_indices"].max()),
                "units_outside_the_gvif_axis_limits": int(
                    np.sum((m["gvif"] > 12).any(1))
                ),
            }
            for subject, m in zip(subjects, measures)
        }
        agreement_axis, size_axis, half_axis = recovery.subplots(1, 3)
        kernels = recovered[0]["regressors"].tolist()
        for axis, key, ylabel in (
            (
                agreement_axis,
                "agreement",
                "agreement of the recovered\nand the true kernel",
            ),
            (size_axis, "size", "size of the recovered kernel\nover the true size"),
        ):
            by_regressor(
                axis,
                [
                    {n: r[key][:, r["regressors"].tolist().index(n)] for n in kernels}
                    for r in recovered
                ],
                kernels,
                RECOVERY_LIMITS[key],
            )
            axis.axhline(1, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.set_ylim(*RECOVERY_LIMITS[key])
            axis.set_ylabel(ylabel)
        stats["recovery"] = {
            subject: {
                name: {
                    "median_agreement": float(np.median(r["agreement"][:, i])),
                    "median_size": float(np.median(r["size"][:, i])),
                    "units_outside_the_axis_limits": int(
                        np.sum(
                            (r["agreement"][:, i] < RECOVERY_LIMITS["agreement"][0])
                            | (r["size"][:, i] > RECOVERY_LIMITS["size"][1])
                        )
                    ),
                }
                for i, name in enumerate(r["regressors"].tolist())
            }
            for subject, r in zip(subjects, recovered)
        }
        # Panel l: the agreement of the kernels fit on the even and on the odd
        # trials (`kernel_reliability.py`), for the units that need the regressor.
        stable = [n for n in REGRESSORS if f"agreement__{n}" in halves[0]]
        by_regressor(
            half_axis,
            [
                {n: h[f"agreement__{n}"][h[f"needs__{n}"]] for n in stable}
                for h in halves
            ],
            stable,
            RECOVERY_LIMITS["agreement"],
        )
        gvif_axis.set_title(
            "units of both mice; circles as in d", fontsize=5.5, loc="left"
        )
        agreement_axis.set_title(
            f"open circles: {subjects[0]} (left), {subjects[1]} (right)",
            fontsize=5.5,
            loc="left",
        )
        half_axis.axhline(1, color=REFERENCE_COLOR, ls="--", lw=0.8)
        half_axis.set_ylim(*RECOVERY_LIMITS["agreement"])
        half_axis.set_ylabel("agreement of the kernels\nof even and odd trials")
        stats["split halves, units that need the regressor"] = {
            name: {
                subject: {
                    "units": int(h[f"needs__{name}"].sum()),
                    "median_agreement": float(
                        np.median(h[f"agreement__{name}"][h[f"needs__{name}"]])
                    ),
                    "units_below_the_axis": int(
                        np.sum(
                            h[f"agreement__{name}"][h[f"needs__{name}"]]
                            < RECOVERY_LIMITS["agreement"][0]
                        )
                    ),
                }
                for subject, h in zip(subjects, halves, strict=True)
            }
            for name in stable
        }
        # Panels m to o: the leak (`leak_test.py`), in the matched peak measure
        # of the excited units, as in `flash_by_period.py`.
        halves_axis, part_axis, rest_axis, level_axis = leak_row.subplots(
            1, 4, width_ratios=(1, 1, 0.6, 1)
        )
        # Panel g: the fixation level that `hold_effect.py` reads, in the two
        # halves, for the units that need the fixation kernel.
        hold_times = kernel_times()["hold"]
        window = (hold_times >= LEVEL_MS[0]) & (hold_times <= LEVEL_MS[1])
        level_halves = [
            h["kernel__hold"][:, h["needs__hold"]][:, :, window].mean(2) for h in halves
        ]
        peak_axes(halves_axis, limits=LEVEL_LIMITS)
        halves_axis.set_xticks([-1, 0, 1], ["−1", "0", "1"])
        halves_axis.set_yticks([-1, 0, 1], ["−1", "0", "1"])
        for line in (halves_axis.axhline, halves_axis.axvline):
            line(0, color=REFERENCE_COLOR, lw=0.5)
        pooled_points(
            halves_axis,
            [np.clip(level[0], *LEVEL_LIMITS) for level in level_halves],
            [np.clip(level[1], *LEVEL_LIMITS) for level in level_halves],
            subjects,
            TASK,
        )
        halves_axis.set_xlabel("fixation level,\neven trials (Δ log rate)")
        halves_axis.set_ylabel("fixation level,\nodd trials (Δ log rate)")
        stats["fixation level in the two halves"] = {
            subject: {
                "units": int(level.shape[1]),
                "spearman_rho": float(spearmanr(*level).statistic),
                "spearman_p": float(spearmanr(*level).pvalue),
                "fraction_same_sign": float(
                    np.mean(np.sign(level[0]) == np.sign(level[1]))
                ),
                "median_absolute_difference": float(
                    np.median(np.abs(level[0] - level[1]))
                ),
                "units_outside_the_axis": int(
                    np.sum((level < LEVEL_LIMITS[0]) | (level > LEVEL_LIMITS[1]))
                ),
            }
            for subject, level in zip(subjects, level_halves, strict=True)
        }
        parts = []
        for root, leak in zip(roots, leaks, strict=True):
            subject, _, session = root.name.partition("_")
            excited = np.isin(
                leak["unit_ids"], list(peaks.excited_units(subject, session))
            )
            parts.append(
                {
                    "real": leak["real_part"][excited],
                    "leak": leak["leak_part"][excited].mean(1),
                    "level": leak["simulated_level"][excited],
                }
            )
        part_axis.plot(PART_LIMITS, PART_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--")
        part_axis.set(xlim=PART_LIMITS, ylim=PART_LIMITS, aspect="equal")
        pooled_points(
            part_axis,
            [np.clip(p["leak"], *PART_LIMITS) for p in parts],
            [np.clip(p["real"], *PART_LIMITS) for p in parts],
            subjects,
            VISUAL,
        )
        part_axis.set_xlabel(
            "leak: flash × move part with\nno effect simulated (spikes/s)"
        )
        part_axis.set_ylabel("real flash × move part\n(spikes/s)")
        stats["leak, flash × move part, real minus leak (spikes/s)"] = pooled_strip(
            rest_axis,
            0,
            [np.clip(p["real"] - p["leak"], *PART_LIMITS) for p in parts],
            subjects,
            VISUAL,
        ) | {
            subject: {
                "units": len(p["real"]),
                "median": float(np.median(p["real"] - p["leak"])),
                "wilcoxon_p": float(wilcoxon(p["real"] - p["leak"]).pvalue),
                "units_outside_the_axis": int(
                    np.sum(
                        (np.minimum(p["real"], p["leak"]) < PART_LIMITS[0])
                        | (np.maximum(p["real"], p["leak"]) > PART_LIMITS[1])
                    )
                ),
            }
            for subject, p in zip(subjects, parts, strict=True)
        }
        rest_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        rest_axis.set_xticks([])
        rest_axis.set_xlim(-0.5, 0.9)
        rest_axis.set_ylim(*PART_LIMITS)
        rest_axis.set_ylabel("flash × move part,\nreal minus leak (spikes/s)")
        level_axis.set_ylim(*PART_LIMITS)
        pooled_points(
            level_axis,
            [p["level"] for p in parts],
            [np.clip(p["leak"], *PART_LIMITS) for p in parts],
            subjects,
            TASK,
        )
        level_axis.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        level_axis.set_xlabel("fixation level of the unit\n(Δ log rate)")
        level_axis.set_ylabel("leak (spikes/s)")
        label_panels(
            [gvif_axis, index_axis, pair_axis, agreement_axis, size_axis, half_axis],
            x=-0.25,
            y=1.04,
        )
        label_panels(
            [None] * 6 + [halves_axis, part_axis, rest_axis, level_axis],
            x=-0.35,
            y=1.04,
        )
        print(save_figure(figure, args.out / "model_trust"))
    recovery_stats = json.loads((args.out / "parameter_recovery.json").read_text())
    (args.out / "model_trust_stats.json").write_text(json.dumps(stats, indent=1))
    (args.out / "model_trust_caption.md").write_text(
        CAPTION.format(
            subject=subjects[0],
            held_out=subjects[1],
            gvif=max(
                v["largest_gvif_per_column"] for v in stats["collinearity"].values()
            ),
            index=max(
                v["largest_condition_index"] for v in stats["collinearity"].values()
            ),
            simulations=recovery_stats[SESSIONS[0]]["simulations"],
            lowest_agreement=min(
                v["median_agreement"]
                for animal in stats["recovery"].values()
                for v in animal.values()
            ),
            level_in_use=" and ".join(
                f"{recovery_stats[s]['fixation level, recovered over true (median)']['in_use']:.2f}"
                for s in SESSIONS
            ),
            lowest_half=min(
                mouse["median_agreement"]
                for entry in stats[
                    "split halves, units that need the regressor"
                ].values()
                for mouse in entry.values()
            ),
            lo=LEVEL_MS[0],
            hi=LEVEL_MS[1],
            level_halves="Spearman correlation of the two halves: "
            + "; ".join(
                f"ρ = {v['spearman_rho']:.2f} in {subject} ({v['units']} units; the"
                f" same sign in {v['fraction_same_sign']:.0%})"
                for subject, v in stats["fixation level in the two halves"].items()
            )
            + ".",
            leak="; ".join(
                f"{subject}: median {entry['median']:+.1f} spikes/s"
                f" ({entry['units']} units, p = {entry['wilcoxon_p']:.2g})"
                for subject, entry in stats[
                    "leak, flash × move part, real minus leak (spikes/s)"
                ].items()
                if subject in subjects
            )
            + ".",
        )
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
