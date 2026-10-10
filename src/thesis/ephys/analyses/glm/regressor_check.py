"""Find regressor blocks that add columns to the V1 GLM but no held-out deviance.

Question: which regressor blocks can be removed from the model? A removal
candidate adds design columns and does not improve cross-validated deviance,
and this must hold in repeated draws of units, not in one draw only.

Comparison: this script makes no new fits. It reads the per-unit records that
`glm unique` wrote. For each block and unit these give the unique ΔD² (the
complete-model D² minus the D² with that block shuffled) and the maximal ΔD²
(the D² with only that block unshuffled minus the D² with all blocks shuffled).
A block with a low unique ΔD² and a high maximal ΔD² is redundant with other
blocks. A block with both low does not predict spikes.

Units: the fit sets `sample`, `sample1`, `sample2`, ... of one session. Each is
a random set of 20 eligible units (quality criterion set 1, stability set 0,
unit 212 excluded), and no unit is in two sets. Each set was fit separately, so
each selected its own motion-energy PC count and penalties. Only sets with a
completed `glm unique` run are read.

Baseline, alignment, and exclusions are those of `fit.py` and `unique_dev.py`:
windows aligned to the first measured flash, bins after response entry
excluded, D² against a constant-rate null, ten whole-trial folds.

Sampling unit: the unit (its mean over scored folds). Each sample is one
replicate of the unit draw. Per sample and block the script reports the median
over units, a one-sided Wilcoxon signed-rank test of unique ΔD² > 0, and the
fraction of units with a reliable gain (unique ΔD² > 2 fold SEM).

Removal guide: a block is flagged when its pooled median unique ΔD² is below
1% of the pooled median complete-model D² and, in every sample, fewer than half
of the units have a reliable gain. The flag is a guide for the researcher, not
a decision.

Panels: (a) unique and (b) maximal ΔD² of every unit, one column of points per
sample within each block, with each sample's median. (c) Pooled median unique
ΔD² against the column count of the block; the vertical line is the range of
the sample medians, and the horizontal line is the range of the column count
(only motion energy varies, because each sample selects its PC count). (d)
Fraction of units with a reliable gain, one point per sample, with the mean.

Display: the ΔD² axes are symmetric-log (linear below 1e-3) because spike
history is about 100 times larger than the smallest blocks. Every unit is
plotted; no value is clipped.

    uv run python -m thesis.ephys.analyses.glm.regressor_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm.figures import (
    DEVIANCE_COLORS,
    DEVIANCE_LABELS,
    FIGURE_STYLE,
    _label_panels,
    _save_figure,
)
from thesis.ephys.analyses.glm.fit import _write_json_atomic, load_unit_records
from thesis.ephys.analyses.glm.unique_dev import DETAILED_GROUPS

SHARE_THRESHOLD = 0.01
RELIABLE_SEM = 2.0
LINEAR_THRESHOLD = 1e-3
UNIQUE = "unique_test_deviance_explained"
MAXIMAL = "maximal_test_deviance_explained"


def load_samples(root: Path) -> dict[str, list[dict]]:
    """Return the unique-deviance unit records of each completed sample fit."""
    samples = {}
    for directory in sorted(
        root.glob("sample*_fit_me"),
        key=lambda path: int(path.name.removesuffix("_fit_me")[6:] or 0),
    ):
        if (directory / "unique_deviance" / "summary.json").exists():
            samples[directory.name.removesuffix("_fit_me")] = load_unit_records(
                directory / "checkpoints" / "unique", "unit_*.json"
            )
    unit_ids = [record["unit_id"] for records in samples.values() for record in records]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("A unit is in more than one sample.")
    return samples


def block_table(samples: dict[str, list[dict]]) -> dict:
    """Summarize each block within each sample and over all units."""
    complete = np.concatenate(
        [
            [record["complete_test_deviance_explained"] for record in records]
            for records in samples.values()
        ]
    )
    table = {
        "samples": list(samples),
        "units": int(len(complete)),
        "complete_d2_median": float(np.median(complete)),
        "share_threshold": SHARE_THRESHOLD,
        "reliable_sem": RELIABLE_SEM,
        "blocks": {},
    }
    for group in DETAILED_GROUPS:
        per_sample = []
        for records in samples.values():
            entries = [record["groups"][group] for record in records]
            unique = np.asarray([entry[UNIQUE] for entry in entries])
            sem = np.asarray([entry[f"{UNIQUE}_sem"] for entry in entries])
            per_sample.append(
                {
                    "columns": int(entries[0]["columns"]),
                    "unique": unique,
                    "maximal": np.asarray([entry[MAXIMAL] for entry in entries]),
                    "reliable_fraction": float(np.mean(unique > RELIABLE_SEM * sem)),
                    "wilcoxon_p": float(wilcoxon(unique, alternative="greater").pvalue),
                }
            )
        pooled_unique = np.concatenate([item["unique"] for item in per_sample])
        share = float(np.median(pooled_unique)) / table["complete_d2_median"]
        reliable = [item["reliable_fraction"] for item in per_sample]
        table["blocks"][group] = {
            "columns": [item["columns"] for item in per_sample],
            "unique_median_by_sample": [
                float(np.median(item["unique"])) for item in per_sample
            ],
            "maximal_median_by_sample": [
                float(np.median(item["maximal"])) for item in per_sample
            ],
            "reliable_fraction_by_sample": reliable,
            "wilcoxon_p_by_sample": [item["wilcoxon_p"] for item in per_sample],
            "unique_median": float(np.median(pooled_unique)),
            "unique_q25_q75": np.quantile(pooled_unique, [0.25, 0.75]).tolist(),
            "maximal_median": float(
                np.median(np.concatenate([item["maximal"] for item in per_sample]))
            ),
            "share_of_complete_d2": share,
            "removal_candidate": bool(share < SHARE_THRESHOLD and max(reliable) < 0.5),
            "_unique": [item["unique"] for item in per_sample],
            "_maximal": [item["maximal"] for item in per_sample],
        }
    return table


def _block_ticks(axis) -> None:
    axis.set_xticks(
        np.arange(len(DETAILED_GROUPS)),
        [DEVIANCE_LABELS[group] for group in DETAILED_GROUPS],
    )
    axis.tick_params(axis="x", labelrotation=48)
    for label, group in zip(axis.get_xticklabels(), DETAILED_GROUPS, strict=True):
        label.set_ha("right")
        label.set_color(_color(group))


def _color(group: str) -> str:
    return DEVIANCE_COLORS.get(group, DEVIANCE_COLORS["task"])


def _plot_units(axis, table: dict, key: str) -> None:
    """Plot every unit, one column per sample, with each sample's median."""
    offsets = np.linspace(-0.32, 0.32, len(table["samples"]))
    rng = np.random.default_rng(0)
    for position, group in enumerate(DETAILED_GROUPS):
        for offset, values in zip(offsets, table["blocks"][group][key], strict=True):
            x = position + offset
            axis.scatter(
                x + rng.uniform(-0.04, 0.04, len(values)),
                values,
                s=5,
                alpha=0.55,
                linewidths=0,
                color=_color(group),
            )
            axis.hlines(
                np.median(values), x - 0.07, x + 0.07, color="black", linewidth=1.1
            )
    axis.axhline(0, color="0.35", linestyle="--", linewidth=0.7, zorder=0)
    axis.set_yscale("symlog", linthresh=LINEAR_THRESHOLD)
    _block_ticks(axis)


def plot_regressor_check(table: dict, output: Path) -> Path:
    """Draw the per-unit ΔD², the cost against the gain, and the consistency."""
    threshold = SHARE_THRESHOLD * table["complete_d2_median"]
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
        axes = axes.ravel()

        _plot_units(axes[0], table, "_unique")
        axes[0].axhline(threshold, color="0.35", linestyle=":", linewidth=0.8)
        axes[0].text(
            1.01,
            threshold,
            f"{SHARE_THRESHOLD:.0%} of full-\nmodel $D^2$",
            transform=axes[0].get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6,
            color="0.35",
        )
        axes[0].set_ylabel(r"unique deviance explained ($\Delta D^2$)")
        _plot_units(axes[1], table, "_maximal")
        axes[1].set_ylabel(r"maximal deviance explained ($\Delta D^2$)")

        for group in DETAILED_GROUPS:
            block = table["blocks"][group]
            medians = block["unique_median_by_sample"]
            columns = float(np.mean(block["columns"]))
            axes[2].vlines(
                columns, min(medians), max(medians), color=_color(group), linewidth=1
            )
            axes[2].hlines(
                block["unique_median"],
                min(block["columns"]),
                max(block["columns"]),
                color=_color(group),
                linewidth=1,
            )
            left = columns <= 6
            axes[2].scatter(
                columns, block["unique_median"], s=22, color=_color(group), zorder=3
            )
            axes[2].annotate(
                DEVIANCE_LABELS[group],
                (columns if left else max(block["columns"]), block["unique_median"]),
                xytext=(-5 if left else 5, 0),
                textcoords="offset points",
                ha="right" if left else "left",
                va="center",
                fontsize=6.5,
                color=_color(group),
            )
        axes[2].axhline(threshold, color="0.35", linestyle=":", linewidth=0.8)
        axes[2].text(
            0.99,
            threshold,
            f"{SHARE_THRESHOLD:.0%} of full-model $D^2$",
            transform=axes[2].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.35",
        )
        axes[2].set_xscale("log")
        axes[2].set_yscale("symlog", linthresh=1e-4)
        axes[2].set_xlim(1, 120)
        axes[2].set_ylim(0, 0.06)
        axes[2].set_xlabel("design columns in the block")
        axes[2].set_ylabel("median unique deviance\n" r"explained ($\Delta D^2$)")

        offsets = np.linspace(-0.32, 0.32, len(table["samples"]))
        for position, group in enumerate(DETAILED_GROUPS):
            fractions = table["blocks"][group]["reliable_fraction_by_sample"]
            axes[3].scatter(
                position + offsets, fractions, s=14, color=_color(group), zorder=3
            )
            axes[3].hlines(
                np.mean(fractions),
                position - 0.4,
                position + 0.4,
                color="black",
                linewidth=1.1,
            )
        axes[3].axhline(0.5, color="0.35", linestyle=":", linewidth=0.8)
        axes[3].set_ylim(-0.03, 1.03)
        axes[3].set_ylabel(
            f"fraction of units with unique\n$\\Delta D^2$ > {RELIABLE_SEM:g} fold SEM"
        )
        _block_ticks(axes[3])

        _label_panels(axes, -0.13, 1.04)
        figure.tight_layout(h_pad=1.5, w_pad=2.0)
        return _save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("figures/glm/GRB006_20240821_121447"),
        help="Session artifact directory that holds the sample fits.",
    )
    args = parser.parse_args()
    samples = load_samples(args.root)
    if len(samples) < 2:
        raise ValueError("Run `glm fit` and `glm unique` on at least two samples.")
    table = block_table(samples)
    figure = plot_regressor_check(table, args.root / "regressor_check")
    for block in table["blocks"].values():
        del block["_unique"], block["_maximal"]
    _write_json_atomic(args.root / "regressor_check.json", table)
    print(json.dumps({**table, "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
