"""Compare the visual flash and motion-energy weights of units in the V1 GLM.

Question: do units that the visual flash predicts differ from units that
motion energy predicts, and how do their fitted weights look?

Comparison: this script makes no new fits. It reads the final refit (all valid
trials) and the `glm unique` records of one fit set of one session (`--units`,
`all` by default).

- Panel a: one point per unit. x is the size of its visual flash weights and y
  is the size of its motion-energy weights. The size is the Euclidean norm of
  the block's coefficients. All design columns are z-scored on training bins,
  so a coefficient is the change in log rate per SD of its column. The norm is
  not exactly comparable between units, because each unit has its own L2
  penalty.
- Panels b and c: three example units. Panel b is the common visual flash
  kernel (the response to a flash averaged over the pre/post-withdrawal and
  adaptation terms). Panel c is the motion-energy weight map. Motion energy has
  no temporal kernel in this model: each PC has one same-bin coefficient. The
  map is the sum of the PC images weighted by the coefficients, so it shows
  where in the back-view video frame motion goes with more (red) or fewer
  (blue) spikes. Each map has its own color range, given above it, because the
  weights of the three units differ by a factor of more than 10; a map with a
  small range is mostly noise. Gray contours mark where the mean motion energy
  is highest.

Example units: selected by the deviance per second that a block explains
uniquely (`block_deviance_rate`; unique ΔD² alone selects units with few
spikes). With each unit ranked on both blocks: "flash" is the unit with the
most flash deviance among units in the lower half for motion energy, "motion
energy" is the reverse. "Both" is a unit near the unity line of panel a (weight
norms within a factor of 1.2); among those units it is the one whose lower rank
is highest. The units with the largest norms near the unity line are not used:
they fire at less than 2 spikes/s, and their motion-energy block explains
little deviance.

Units, alignment, baseline, and exclusions are those of `fit.py`. The sampling
unit is the unit. No statistic is computed; the figure is descriptive.

    uv run python -m thesis.ephys.analyses.glm.kernel_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from scipy.stats import rankdata

from thesis.ephys.analyses.glm.dmat import HISTORY_COLUMNS
from thesis.ephys.analyses.glm.figures import FIGURE_STYLE, _save_figure
from thesis.ephys.analyses.glm.fit import _write_json_atomic, load_unit_records
from thesis.ephys.analyses.glm.results import (
    FLASH_COMMON,
    block_deviance_rate,
    fitted_kernels,
)

EXAMPLE_COLORS = {"flash": "C0", "motion energy": "C1", "both": "C4"}
# A "both" unit has flash and motion-energy weight norms within this factor.
UNITY_FACTOR = 1.2


def load_units(root: Path, unit_set: str) -> list[dict]:
    """Return each unit's final fit, block norms, and block deviance rates."""
    with (root / "common_design_me.json").open() as handle:
        design = json.load(handle)
    flash = next(
        index
        for index, item in enumerate(design["task_manifest"])
        if item["name"] == FLASH_COMMON
    )
    start = sum(int(item["columns"]) for item in design["task_manifest"][:flash])
    flash_columns = slice(start, start + design["task_manifest"][flash]["columns"])
    base = int(design["base_columns"])
    checkpoints = root / f"{unit_set}_fit_me" / "checkpoints"
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_final.json")
    }
    units = []
    for record in load_unit_records(checkpoints / "unique", "unit_*.json"):
        fit = final[int(record["unit_id"])]
        coefficients = np.asarray(fit["coefficients"], dtype=float)
        video = coefficients[base : base + int(fit["components"])]
        units.append(
            {
                "unit_id": int(record["unit_id"]),
                "fit": fit,
                "video_coefficients": video,
                "flash_norm": float(np.linalg.norm(coefficients[flash_columns])),
                "video_norm": float(np.linalg.norm(video)),
                "flash_rate": block_deviance_rate(record, fit, FLASH_COMMON),
                "video_rate": block_deviance_rate(record, fit, "video"),
            }
        )
    return units


def select_examples(units: list[dict]) -> dict[str, dict]:
    """Return the flash, motion-energy, and both example units."""
    flash = np.asarray([unit["flash_rate"] for unit in units])
    video = np.asarray([unit["video_rate"] for unit in units])
    flash_rank = rankdata(flash) / len(units)
    video_rank = rankdata(video) / len(units)
    ratio = np.asarray([unit["video_norm"] / unit["flash_norm"] for unit in units])
    near = np.abs(np.log(ratio)) < np.log(UNITY_FACTOR)
    picks = {
        "flash": np.argmax(np.where(video_rank < 0.5, flash, -np.inf)),
        "motion energy": np.argmax(np.where(flash_rank < 0.5, video, -np.inf)),
        "both": np.argmax(np.where(near, np.minimum(flash_rank, video_rank), -np.inf)),
    }
    return {name: units[int(index)] for name, index in picks.items()}


def motion_weight_map(unit: dict, design: dict, video: dict) -> np.ndarray:
    """Return the change in log rate per unit of motion energy at each pixel."""
    count = len(unit["video_coefficients"])
    base = int(design["base_columns"])
    design_scale = np.asarray(design["training_scale"])[base : base + count]
    weights = unit["video_coefficients"] / (design_scale * video["score_scale"][:count])
    return (weights @ video["components"][:count]).reshape(video["shape"])


def plot_kernel_comparison(
    units: list[dict],
    examples: dict[str, dict],
    design: dict,
    video: dict,
    output: Path,
) -> Path:
    kernels = {
        name: fitted_kernels(unit["fit"], design, np.ones(HISTORY_COLUMNS))[
            FLASH_COMMON
        ]
        for name, unit in examples.items()
    }
    maps = {
        name: motion_weight_map(unit, design, video) for name, unit in examples.items()
    }
    kernel_limit = 1.05 * max(np.abs(values).max() for _, values in kernels.values())

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(10.5, 5.6))
        outer = figure.add_gridspec(1, 2, width_ratios=(1, 1.45), wspace=0.3)
        scatter = figure.add_subplot(outer[0])
        grid = outer[1].subgridspec(3, 2, hspace=0.5, wspace=0.25)

        chosen = {unit["unit_id"] for unit in examples.values()}
        rest = [unit for unit in units if unit["unit_id"] not in chosen]
        scatter.scatter(
            [unit["flash_norm"] for unit in rest],
            [unit["video_norm"] for unit in rest],
            s=12,
            color="0.6",
            alpha=0.7,
            linewidths=0,
        )
        for name, unit in examples.items():
            scatter.scatter(
                unit["flash_norm"], unit["video_norm"], s=34, color=EXAMPLE_COLORS[name]
            )
            scatter.annotate(
                name,
                (unit["flash_norm"], unit["video_norm"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7,
                color=EXAMPLE_COLORS[name],
                # A white outline keeps the label legible over the other units.
                path_effects=[withStroke(linewidth=2, foreground="white")],
            )
        scatter.set_xscale("log")
        scatter.set_yscale("log")
        low = min(scatter.get_xlim()[0], scatter.get_ylim()[0])
        high = max(scatter.get_xlim()[1], scatter.get_ylim()[1])
        scatter.plot(
            [low, high], [low, high], color="0.35", linestyle="--", linewidth=0.7
        )
        scatter.set_xlim(low, high)
        scatter.set_ylim(low, high)
        scatter.set_box_aspect(1)
        scatter.set_xlabel(r"visual flash weights ($\|\beta\|$)")
        scatter.set_ylabel(r"motion-energy weights ($\|\beta\|$)")

        kernel_axes, map_axes = [], []
        for row, name in enumerate(examples):
            color = EXAMPLE_COLORS[name]
            axis = figure.add_subplot(grid[row, 0])
            times_ms, values = kernels[name]
            axis.plot(times_ms, values, color=color, linewidth=1.2)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.5)
            axis.set_ylim(-kernel_limit, kernel_limit)
            axis.set_title(name, fontsize=8, color=color, loc="left")
            axis.set_ylabel(r"$\Delta$ log rate")
            kernel_axes.append(axis)

            axis = figure.add_subplot(grid[row, 1])
            # Each map has its own color range; the title gives that range.
            limit = float(np.abs(maps[name]).max())
            image = axis.imshow(maps[name] / limit, cmap="RdBu_r", vmin=-1, vmax=1)
            axis.set_title(
                rf"$\pm${limit:.2f} $\Delta$ log rate", fontsize=7, loc="left"
            )
            axis.contour(
                video["mean"],
                levels=np.quantile(video["mean"], [0.9, 0.98]),
                colors="0.3",
                linewidths=0.4,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            map_axes.append(axis)
        for axis in kernel_axes[:-1]:
            axis.tick_params(axis="x", labelbottom=False)
        kernel_axes[-1].set_xlabel("time from flash (ms)")
        map_axes[-1].set_xlabel("back-view video frame")
        bar = figure.colorbar(image, ax=map_axes, shrink=0.6, pad=0.04)
        bar.set_label(
            "weight per unit of motion energy\n(relative to the largest of the unit)"
        )

        for letter, axis, x, y in (
            ("a", scatter, -0.2, 1.04),
            ("b", kernel_axes[0], -0.3, 1.2),
            ("c", map_axes[0], -0.12, 1.2),
        ):
            axis.text(
                x,
                y,
                letter,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        return _save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("figures/glm/GRB006_20240821_121447"),
        help="Session artifact directory that holds the fit.",
    )
    parser.add_argument(
        "--units",
        default="all",
        help="Fit set to read: all, sample, or sample1 to sample9.",
    )
    args = parser.parse_args()
    with (args.root / "common_design_me.json").open() as handle:
        design = json.load(handle)
    with np.load(args.root / "video_me_features.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        shape = (int(metadata["height"]), int(metadata["width"]))
        video = {
            "components": data["components"],
            "score_scale": data["score_scale"],
            "mean": data["mean"].reshape(shape),
            "shape": shape,
        }
    units = load_units(args.root, args.units)
    examples = select_examples(units)
    figure = plot_kernel_comparison(
        units, examples, design, video, args.root / "kernel_comparison"
    )
    summary = {
        "units": len(units),
        "examples": {
            name: {
                key: unit[key]
                for key in (
                    "unit_id",
                    "flash_norm",
                    "video_norm",
                    "flash_rate",
                    "video_rate",
                )
            }
            for name, unit in examples.items()
        },
        "figure": str(figure),
    }
    _write_json_atomic(args.root / "kernel_comparison.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
