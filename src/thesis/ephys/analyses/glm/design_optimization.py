"""Combine the design tests of the V1 GLM in one figure: optimizing the design matrix.

Question: why does the design matrix have these regressors, with these
parameters, and not others? This script makes no fits and no new analysis. It
draws results of the other design modules in one figure and writes the caption
next to it. Each panel answers one question, written above it, and the panels
follow one regressor at a time. A dotted circle marks the value that the model
uses.

- Panel a (the starting point): regressors of the first, nine-regressor
  design. Median and interquartile range of the unique ΔD² (167 units). Gray
  regressors were removed; the arrow next to a kept regressor names the panels
  that test it.
- Visual flash kernel. Panel b: number of bases (150 ms window). Panel c:
  window (8 bases). `flash_kernel_grid.py`.
- Adaptation term. Panel d: no term, a synaptic-depression resource with a
  slow and with a fast recovery, and "a flash in the last 50 ms".
  `flash_kernel_grid.py` and `adaptation_grid.py`.
- Withdrawal. Panel e: width of the withdrawal kernel, without the in-port
  column. Panel f: observed firing rate and the rate that the model with the
  0 to 100 ms kernel and no in-port column predicts, against time from the
  first flash; the shaded time is the median time in the center port. Panel g:
  the lower rate in the center port is not stillness (`port_vs_stillness.py`,
  all units, raw rates). Panel h: the 0 to 100 ms kernel, the in-port column
  alone, and both; the dashed line is the ±300 ms kernel. `withdrawal_grid.py`.
- Motion energy. Panel i: number of PCs (first design, 167 units). Panel j:
  each unit's unique ΔD² of motion energy against its trial-specific ΔD²
  (motion energy taken from another trial) in the final design; bars are the
  SEM over folds.
- Panel k (what is left out): the ΔD² that a term not in the final design
  would add (`residual_checks.py`, in-sample; median and interquartile range
  over units), and the median with the labels shuffled over trials (gray
  line).

Panels b to e, h, and i show the mean D² over units on the validation trials
for each choice. Panels a and k have a logarithmic axis that starts at
`FLOOR`; a smaller value is drawn at the floor. Firing rates are divided by
the mean rate of that unit over all bins of the analysis. The choice panels
have no error bars: the units differ much more in D² than the options do, so
the paired test carries the uncertainty of each choice. The mark over an
option is a two-sided Wilcoxon signed-rank test against the circled option,
paired over units (panel e: against the ±300 ms kernel); the exact values go
to `design_optimization_stats.json`.
Each module docstring gives the comparison, units, baseline, alignment,
exclusions, and sampling unit of its panels. The first design is the model at
commit `fcd34fa`; its fit is in the directory next to the session directory
with the suffix `FULL_SUFFIX`.

    uv run python -m thesis.ephys.analyses.glm.design_optimization --units all

Run the other modules first; this script stops if a result file is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from thesis.ephys.analyses.glm import (
    cli,
    dmat,
    port_vs_stillness,
    residual_checks,
    withdrawal_grid,
)
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    mark_selected,
    save_figure,
)

FLOOR = 1e-5
REMOVED_COLOR = "0.6"
VISUAL, TASK, MOTION = (
    CONCEPT_COLORS["visual"],
    CONCEPT_COLORS["task"],
    CONCEPT_COLORS["motion energy"],
)
FULL_SUFFIX = "_full_design"
# Synaptic-depression models of panel d: (fraction used, recovery in s).
DEPRESSION = {"slow\ndepression": (0.5, 0.2), "fast\ndepression": (0.5, 0.025)}
# Regressors of the full design: label, color (gray = removed), note.
FULL_DESIGN = {
    "history": ("spike history", CONCEPT_COLORS["history"], "kept"),
    "visual_flash": ("visual flash", VISUAL, "→ b, c"),
    "withdrawal": ("withdrawal, ±300 ms", TASK, "→ e–h"),
    "video": ("motion energy, 25 PCs", MOTION, "→ i, j"),
    "flash_adaptation": ("flash adaptation", VISUAL, "→ d"),
    "flash_pre_post_withdrawal": ("flash: post − pre withdrawal", REMOVED_COLOR, ""),
    "flash_motion_energy": ("flash × motion energy", REMOVED_COLOR, ""),
    "response_entry": ("response entry", REMOVED_COLOR, ""),
    "initiation": ("initiation", REMOVED_COLOR, ""),
}
# Terms that are not in the final design (panel k), in `residual_checks` names.
LEFT_OUT = {
    "time in trial": "time in the trial",
    "session drift": "drift over the session",
    "side, after withdrawal": "response side, after exit",
    "flash rate": "flash rate",
    "side, before withdrawal": "response side, before exit",
}
CAPTION = """\
**Optimizing the GLM design matrix.** The question that a panel answers is
above it. A dotted circle marks the value that the model uses. D² is the
deviance explained on trials that were not used for the fit, as the mean over
units. Where two options do not differ reliably, the model uses the simpler one.
The Friedman test in each panel compares all options at once, paired over units.
The mark over each other option is a two-sided Wilcoxon signed-rank test
of that option against the circled one, paired over units (n.s., p ≥ 0.05;
\*, p < 0.05; \*\*, p < 0.01; \*\*\*, p < 0.001; in **e**, against the
±300 ms kernel). The exact values are in `design_optimization_stats.json`.
**a**, The starting point: the unique ΔD² of each regressor of the first,
nine-regressor design (median and interquartile range of {full_units} units).
Gray regressors explained almost nothing and were removed. The arrows name the
panels that test each kept regressor.
**b, c, Visual flash kernel** ({sample_units} units). **b**, D² against the
number of raised cosines. **c**, D² against the length of the kernel.
**d, Adaptation term**: a second flash kernel for a flash that follows another
flash. D² with no such term, with a synaptic-depression resource that recovers
slowly (200 ms) or fast (25 ms), and with a term that is 1 for a flash that
follows another within 50 ms (recent flash). The depression model is best when
it recovers faster than the shortest interval between flashes (40 ms), where
it is the same as the recent-flash term.
**e–h, Withdrawal.** **e**, D² against the width of the withdrawal kernel. A
kernel shorter than ±300 ms predicts worse. **f**, What the short kernel
misses: observed firing rate (black) and the rate that the model with the 0 to
100 ms kernel predicts (purple, dashed), against time from the first flash
(median of {sample_units} units). Each rate is divided by the mean rate of that
unit, so 1 is the average rate of the unit. The model predicts too many spikes
while the animal is in the center port (shaded: median time in the port). With
a column for the time in the port (purple, solid), the prediction is closer. **g**, This lower rate is not because the animal
is still: firing rate against the motion energy of the video frame, by the
state of the animal (median and interquartile range of {port_units} units). At
a side port, with the same motion energy and no stimulus, the rate is higher.
**h**, D² with the 0 to 100 ms kernel, with a column that is 1 while the animal
is in the center port (port only), and with both (dashed line: the ±300 ms
kernel, which is not better than both; p = {wide_p:.2f}).
**i, j, Motion energy.** **i**, D² against the number of motion-energy PCs in
the first design ({full_units} units); 25 and 50 are not reliably better than
10. **j**, Does motion energy predict spikes
through the movement of that trial, or only because movement and firing both
change with the time in the trial? The D² that each unit loses when motion
energy is shuffled in time within the trial, against the loss when it is
replaced with the motion energy of another trial, which has the same average
time course (final design; bars are the SEM over folds). The two losses are
equal, so the movement of that trial is what predicts the spikes.
**k**, Terms that are not in the final design: the ΔD² that each one would add
(median and interquartile range of {residual_units} units). Gray line: the same
value after the labels of that term are shuffled 30 times (order of the time
bins for the time in the trial, trial order for the drift, rate group for the
flash rate, response side for the two side terms), which is what a term with no
relation to the spikes gains by chance.
"""


def _load(path: Path, command: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `{command}` first.")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return json.loads(path.read_text())


def _plot_full_design(axis, groups: dict) -> None:
    order = sorted(FULL_DESIGN, key=lambda name: groups[name]["median"])
    for position, name in enumerate(order):
        _, color, note = FULL_DESIGN[name]
        low, median, high = (
            max(groups[name][key], FLOOR) for key in ("q25", "median", "q75")
        )
        axis.plot([low, high], [position] * 2, color=color, linewidth=1.2)
        axis.scatter(median, position, color=color, s=22, zorder=3)
        axis.text(high * 1.3, position, note or "removed", color=color, va="center")
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 3)
    axis.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    axis.set_yticks(range(len(order)), [FULL_DESIGN[name][0] for name in order])
    for tick, name in zip(axis.get_yticklabels(), order, strict=True):
        tick.set_color(FULL_DESIGN[name][1])
    axis.set_xlabel(r"unique $\Delta D^2$ in the first design")


def _stars(p: float) -> str:
    return "n.s." if p >= 0.05 else "*" * (1 + (p < 0.01) + (p < 0.001))


def _plot_choice(
    axis, scores: dict, color, xlabel, selected=None, reference=None, line=True
) -> dict:
    """Mean D² over units for each choice, with a paired test against one choice.

    `scores` maps the label of each choice to the D² of each unit (same unit
    order). A dotted circle marks `selected`. Each other choice gets the result
    of a two-sided Wilcoxon signed-rank test against `reference` (default:
    `selected`) over units. Returns the statistics.
    """
    labels = list(scores)
    reference = selected if reference is None else reference
    means = [float(np.mean(scores[label])) for label in labels]
    positions = np.arange(len(labels))
    base = np.asarray(scores[reference])
    axis.plot(
        positions,
        means,
        color=color,
        marker="o",
        markerfacecolor="white",
        linestyle="-" if line else "none",
    )
    if selected is not None:
        mark_selected(axis, labels.index(selected), means[labels.index(selected)])
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.margins(y=0.25)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"mean $D^2$")
    # Omnibus test over all choices, paired over units, before the pairwise tests.
    omnibus = float(friedmanchisquare(*(scores[label] for label in labels)).pvalue)
    stats = {"friedman_p_all_choices": omnibus}
    axis.set_xlabel(f"{xlabel}\n(Friedman p = {omnibus:.2g})")
    for position, label in enumerate(labels):
        if label == reference:
            continue
        other = np.asarray(scores[label])
        p_value = float(wilcoxon(base, other).pvalue)
        stats[str(label).replace("\n", " ")] = {
            "reference": str(reference).replace("\n", " "),
            "mean_difference_reference_minus_choice": float(np.mean(base - other)),
            "units_reference_higher": int(np.sum(base > other)),
            "units": len(base),
            "wilcoxon_p": p_value,
        }
        axis.annotate(
            _stars(p_value),
            (position, means[position]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    return stats


def _plot_rates(axis, without: dict, with_port: dict) -> None:
    """Observed rate and the rates that the model predicts without and with the port column."""
    edges = residual_checks.ONSET_EDGES_S
    x = 1000 * (edges[:-1] + edges[1:]) / 2
    median = lambda curve: np.nanmedian(np.asarray(curve), axis=0)  # noqa: E731
    entry, leave = (1000 * value for value in with_port["hold_s"])
    axis.axvspan(max(entry, x[0]), leave, color=TASK, alpha=0.15, linewidth=0)
    axis.text(
        (max(entry, x[0]) + leave) / 2,
        0.93,
        "in center port",
        color=TASK,
        ha="center",
        transform=axis.get_xaxis_transform(),
    )
    axis.plot(x, median(with_port["curves"]["observed"]), color="black", linewidth=1.4)
    axis.plot(
        x, median(without["curves"]["predicted"]), color=MODEL_COLOR, linestyle="--"
    )
    axis.plot(x, median(with_port["curves"]["predicted"]), color=MODEL_COLOR)
    axis.set_ylim(bottom=0)
    top = axis.get_ylim()[1]
    for height, text, color in (
        (0.36, "observed", "black"),
        (0.26, "predicted, no port column", MODEL_COLOR),
        (0.16, "predicted, with port column", MODEL_COLOR),
    ):
        axis.text(
            x[-1], height * top, text, color=color, ha="right", va="top", fontsize=6.5
        )
    axis.set_xlabel("time from first flash (ms)")
    axis.set_ylabel("rate / mean rate of the unit")


def _plot_motion_control(axis, records: list[dict]) -> None:
    video = [item["groups"]["video"] for item in records]
    unique, unique_sem, specific, specific_sem = (
        np.asarray([item[key] for item in video])
        for key in (
            "unique_test_deviance_explained",
            "unique_test_deviance_explained_sem",
            "trial_specific_test_deviance_explained",
            "trial_specific_test_deviance_explained_sem",
        )
    )
    high = 1.08 * max(np.max(unique + unique_sem), np.max(specific + specific_sem))
    low = min(
        0, 1.08 * min(np.min(unique - unique_sem), np.min(specific - specific_sem))
    )
    axis.plot([low, high], [low, high], color=REFERENCE_COLOR, linewidth=0.8)
    axis.errorbar(
        unique,
        specific,
        xerr=unique_sem,
        yerr=specific_sem,
        fmt="o",
        color=MOTION,
        markersize=2.5,
        elinewidth=0.5,
        alpha=0.8,
    )
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel(r"$D^2$ lost: motion energy" + "\nshuffled in the trial")
    axis.set_ylabel(r"$D^2$ lost: motion energy" + "\nfrom another trial")


def _plot_left_out(axis, checks: list[dict]) -> None:
    values = lambda key: np.asarray([item[key] for item in checks])  # noqa: E731
    order = sorted(LEFT_OUT, key=lambda name: np.median(values(name)))
    for position, name in enumerate(order):
        low, median, high = np.maximum(np.percentile(values(name), [25, 50, 75]), FLOOR)
        axis.plot([low, high], [position] * 2, color="black", linewidth=1.2)
        axis.scatter(median, position, color="black", s=22, zorder=3)
        if f"{name} shuffle" in checks[0]:
            shuffled = max(np.median(values(f"{name} shuffle")), FLOOR)
            axis.plot(
                [shuffled] * 2,
                [position - 0.3, position + 0.3],
                color=REFERENCE_COLOR,
                linewidth=2,
            )
    axis.text(
        0.98,
        0.04,
        "shuffled labels",
        color=REFERENCE_COLOR,
        ha="right",
        transform=axis.transAxes,
    )
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 1e-1)
    axis.set_ylim(-0.7, len(order) - 0.4)
    axis.set_yticks(range(len(order)), [LEFT_OUT[name] for name in order])
    axis.set_xlabel(r"$\Delta D^2$ that the term would add")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    root = args.root
    module = "uv run python -m thesis.ephys.analyses.glm"
    sibling = lambda suffix: root.with_name(root.name + suffix)  # noqa: E731
    full = sibling(FULL_SUFFIX) / "all_fit_me"
    pc_summary = _load(full / "validation_summary.json", "glm fit (full design)")
    full_unique = _load(full / "unique_deviance" / "summary.json", "glm unique")
    per_unit = lambda units: np.asarray([units[u] for u in sorted(units)])  # noqa: E731
    flash = {
        (row["bases"], row["window"], row["second"]): per_unit(row["units"])
        for row in _load(
            root / "flash_kernel_grid" / "flash_kernel_grid.json",
            f"{module}.flash_kernel_grid",
        )
    }
    adaptation = {
        (row["u"], row["tau"]): per_unit(row["units"])
        for row in _load(
            root / "adaptation_grid" / "adaptation_grid.json",
            f"{module}.adaptation_grid",
        )
    }
    port = _load(root / "port_vs_stillness.npz", f"{module}.port_vs_stillness")
    kernel = {
        name: per_unit(units)
        for name, units in _load(
            root / "withdrawal_grid" / "withdrawal_grid.json",
            f"{module}.withdrawal_grid",
        ).items()
    }
    no_port = _load(
        sibling(withdrawal_grid.NO_PORT_SUFFIX)
        / "sample_fit_me"
        / "residual_checks.json",
        f"{module}.withdrawal_grid",
    )
    fit_dir = cli.PoissonGLM(root=root).fit_dir(args.units)
    residual = _load(
        fit_dir / "residual_checks.json",
        f"{module}.residual_checks --units {args.units}",
    )
    unique_records = load_unit_records(
        fit_dir / "checkpoints" / "unique", "unit_*.json"
    )
    bases, window = dmat.FLASH_BASES, dmat.FLASH_WINDOW_S
    base_counts = sorted({key[0] for key in flash})
    windows = sorted({key[1] for key in flash})
    selections = load_unit_records(full / "checkpoints", "unit_*_validation.json")
    pcs = {
        count: np.asarray(
            [
                item["plus_video"][count]["validation"]["deviance_explained"]
                for item in selections
            ]
        )
        for count in sorted(selections[0]["plus_video"], key=int)
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 11.2))

        # Fixed positions: the panels carry labels outside their axes.
        def row(top, bottom, widths, left=0.1, right=0.98, wspace=0.62):
            grid = figure.add_gridspec(
                1,
                len(widths),
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                width_ratios=widths,
                wspace=wspace,
            )
            return [figure.add_subplot(cell) for cell in grid]

        (a,) = row(0.962, 0.84, (1,), left=0.36, right=0.8)
        b, c, d = row(0.755, 0.655, (1, 0.8, 1.5))
        e, f, g, h = row(0.535, 0.43, (0.85, 1.1, 1.75, 0.9), wspace=0.8)
        i, j = row(0.3, 0.17, (1, 0.9), right=0.5, wspace=0.95)
        (k,) = row(0.3, 0.17, (1,), left=0.76)
        spare_figure = plt.figure()
        spare = spare_figure.subplots(1, 2)

        _plot_full_design(a, full_unique["groups"])
        stats = {}
        stats["b flash bases"] = _plot_choice(
            b,
            {count: flash[(count, window, True)] for count in base_counts},
            VISUAL,
            "raised cosines in the flash kernel",
            selected=bases,
        )
        stats["c flash window"] = _plot_choice(
            c,
            {f"{1000 * value:.0f}": flash[(bases, value, True)] for value in windows},
            VISUAL,
            "length of the flash kernel (ms)",
            selected=f"{1000 * window:.0f}",
        )
        second = {"none": flash[(bases, window, False)]}
        second |= {name: adaptation[pair] for name, pair in DEPRESSION.items()}
        second["recent\nflash"] = adaptation[(None, None)]
        stats["d adaptation term"] = _plot_choice(
            d, second, VISUAL, "adaptation term", selected="recent\nflash", line=False
        )
        d.tick_params(axis="x", labelsize=7)
        stats["e withdrawal width"] = _plot_choice(
            e,
            {name: kernel[name] for name in ("±300", "±100", "0–100")},
            TASK,
            "withdrawal kernel (ms)",
            reference="±300",
        )
        e.tick_params(axis="x", labelsize=7)
        in_use = withdrawal_grid.IN_USE
        stats["h in-port column"] = _plot_choice(
            h,
            {
                "0–100": kernel["0–100"],
                "port\nonly": kernel["in port\nonly"],
                "both": kernel[in_use],
            },
            TASK,
            "withdrawal terms",
            selected="both",
            line=False,
        )
        wide = float(kernel["±300"].mean())
        h.axhline(wide, color=TASK, linestyle="--", linewidth=0.8)
        h.text(-0.45, wide, "±300 ms", color=TASK, va="bottom", fontsize=7)
        h.tick_params(axis="x", labelsize=7)
        stats["h in-port column"]["±300 ms kernel"] = {
            "reference": "both",
            "mean_difference_reference_minus_choice": float(
                np.mean(kernel[in_use] - kernel["±300"])
            ),
            "units_reference_higher": int(np.sum(kernel[in_use] > kernel["±300"])),
            "units": len(kernel[in_use]),
            "wilcoxon_p": float(wilcoxon(kernel[in_use], kernel["±300"]).pvalue),
        }
        _plot_rates(f, no_port, residual)
        port_vs_stillness.plot(port, None, axes=(g, *spare))
        g.set_ylabel("")  # the same axis as panel f
        g.set_xlabel("motion energy of the video frame")
        plt.close(spare_figure)
        stats["i motion-energy PCs"] = _plot_choice(
            i,
            pcs,
            MOTION,
            "motion-energy PCs",
            selected=str(max(dmat.VIDEO_COMPONENT_COUNTS)),
        )
        i.tick_params(axis="x", labelsize=7)
        _plot_motion_control(j, unique_records)
        _plot_left_out(k, residual["units"])

        # The question that each panel answers, above it.
        questions = {
            a: "how much does each regressor of the first design explain?",
            b: "how many bases?",
            c: "how long a kernel?",
            d: "which term for a flash that follows a flash?",
            e: "how wide a kernel?",
            f: "what does the short kernel miss?",
            g: "is the lower rate in the port stillness?",
            h: "does a column for the port fix it?",
            i: "how many PCs?",
            j: "movement, or time in the trial?",
            k: "how much would each term add?",
        }
        for axis, question in questions.items():
            axis.text(
                0.5 if axis not in (a, k) else 0.3,
                1.03,
                question,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                style="italic",
            )
        # Letters at one distance from each panel; the panels differ in width.
        wide = {a: 0.27, k: 0.26}
        panels = (a, b, c, d, e, f, g, h, i, j, k)
        for letter, axis in zip("abcdefghijk", panels, strict=True):
            box = axis.get_position()
            figure.text(
                box.x0 - wide.get(axis, 0.075),
                box.y1 + 0.03,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        # One line above the first panel of each group names its regressor.
        for axis, text, color in (
            (a, "the first design", "black"),
            (b, "visual flash kernel", VISUAL),
            (d, "adaptation term", VISUAL),
            (e, "withdrawal", TASK),
            (i, "motion energy", MOTION),
            (k, "what is left out", "black"),
        ):
            box = axis.get_position()
            figure.text(
                box.x0 - wide.get(axis, 0.075) + 0.025,
                box.y1 + 0.0305,
                text,
                color=color,
                fontsize=9,
            )
        output = save_figure(figure, root / "design_optimization")

    output.with_name("design_optimization_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False)
    )
    caption = CAPTION.format(
        full_units=pc_summary["units"],
        sample_units=len(adaptation[(None, None)]),
        port_units=len(port["rate"]),
        residual_units=len(residual["units"]),
        wide_p=stats["h in-port column"]["±300 ms kernel"]["wilcoxon_p"],
    )
    output.with_name("design_optimization_caption.md").write_text(caption)
    print(output)


if __name__ == "__main__":
    main()
