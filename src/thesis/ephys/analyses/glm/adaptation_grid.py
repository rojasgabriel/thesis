"""Show how the two flash-adaptation parameters of the V1 GLM were selected.

Question: do other values of the adaptation parameters predict held-out spikes
better than the values in `dmat.py`? The adaptation term scales the flash
kernel by a synaptic resource: each flash uses the fraction `ADAPT_U` of the
resource, which recovers with the time constant `ADAPT_TAU_RECOVER_S`.

Comparison: for each pair on the grid (`U_VALUES` x `TAU_VALUES_S`), build the
full design with that pair and run the model-selection stage of `glm fit`: fit
on the training trials and score D² on the validation trials. Nothing else in
the model changes. The comparison is paired: each unit is scored with each
pair.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample (`--units sample`), because the grid needs one design and
one fit set for each pair. The sampling unit is the unit.

- Panel a: mean validation D² over units against the recovery time constant,
  one line for each fraction used. The filled point is the pair in `dmat.py`.
- Panel b: for each pair, the validation D² of each unit minus its value with
  the pair in `dmat.py` (small points), and the median over units (large
  points).

The grid is not a fit of the parameters, and the pair is not selected again
inside the cross-validation folds. The y range of panel a is small; read the
size of the differences from the axis.

    uv run python -m thesis.ephys.analyses.glm.adaptation_grid

A pair that has saved scores in `adaptation_grid/adaptation_grid.json` is not
fit again. Each new pair takes about 2 minutes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

U_VALUES = (0.25, 0.5, 0.75)
TAU_VALUES_S = (0.1, 0.2, 0.4, 0.8)
UNIT_SET = "sample"
INPUTS = ("stimulus_windows_me.npz", "video_me_features.npz", "frame_times.npy")


def score_pair(source: Path, work: Path, u: float, tau_s: float) -> dict[str, float]:
    """Return each unit's validation D² with one pair of adaptation values."""
    model = cli.PoissonGLM(root=work / f"u{u}_tau{tau_s}")
    model.root.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copy(source / name, model.root / name)
    # ponytail: the design reads these module constants; set them for one pair
    # and restore them. Pass them as arguments if a second caller needs this.
    saved = dmat.ADAPT_U, dmat.ADAPT_TAU_RECOVER_S
    dmat.ADAPT_U, dmat.ADAPT_TAU_RECOVER_S = u, tau_s
    try:
        dmat.prepare_common_design(model.windows, model.video, model.design)
        fit.fit_models(model.windows, model.design, UNIT_SET, model.fit_dir(UNIT_SET))
    finally:
        dmat.ADAPT_U, dmat.ADAPT_TAU_RECOVER_S = saved
    count = str(max(dmat.VIDEO_COMPONENT_COUNTS))
    scores = {}
    for path in (model.fit_dir(UNIT_SET) / "checkpoints").glob(
        "unit_*_validation.json"
    ):
        record = json.loads(path.read_text())
        scores[str(record["unit_id"])] = record["plus_video"][count]["validation"][
            "deviance_explained"
        ]
    shutil.rmtree(model.root)  # the design of one pair is about 80 MB
    return scores


def run_grid(source: Path, output: Path) -> list[dict]:
    """Score every pair that has no saved scores; return all pairs."""
    path = output / "adaptation_grid.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(row["u"], row["tau"]) for row in rows}
    for u in U_VALUES:
        for tau_s in TAU_VALUES_S:
            if (u, tau_s) in done:
                continue
            rows.append(
                {
                    "u": u,
                    "tau": tau_s,
                    "units": score_pair(source, output / "work", u, tau_s),
                }
            )
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=1))
            print(f"Scored u={u}, tau={tau_s} s.", flush=True)
    return rows


def plot_grid(rows: list[dict], output: Path) -> Path:
    """Plot mean validation D² and each unit's change from the pair in use."""
    scores = {(row["u"], row["tau"]): row["units"] for row in rows}
    current = (dmat.ADAPT_U, dmat.ADAPT_TAU_RECOVER_S)
    if current not in scores:
        raise ValueError("The grid must contain the pair that dmat.py uses.")
    unit_ids = sorted(scores[current])
    if any(sorted(units) != unit_ids for units in scores.values()):
        raise ValueError("Every pair must score the same units.")
    values = {
        pair: np.asarray([units[unit] for unit in unit_ids])
        for pair, units in scores.items()
    }
    positions = np.arange(len(TAU_VALUES_S))
    # One concept, the visual response; a darker line uses more of the resource.
    shades = dict(zip(U_VALUES, (0.45, 0.7, 1.0), strict=True))
    rng = np.random.default_rng(0)

    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
        for index, u in enumerate(U_VALUES):
            style = {"color": CONCEPT_COLORS["visual"], "alpha": shades[u]}
            means = [values[(u, tau)].mean() for tau in TAU_VALUES_S]
            axes[0].plot(positions, means, marker="o", markerfacecolor="white", **style)
            axes[0].text(
                positions[-1] + 0.12,
                means[-1],
                f"{u:.0%} used per flash",
                va="center",
                **style,
            )
            offset = 0.22 * (index - 1)
            for position, tau in zip(positions, TAU_VALUES_S, strict=True):
                change = values[(u, tau)] - values[current]
                if (u, tau) == current:
                    continue
                axes[1].scatter(
                    position + offset + rng.uniform(-0.05, 0.05, len(change)),
                    change,
                    s=5,
                    linewidths=0,
                    color=CONCEPT_COLORS["visual"],
                    alpha=0.45 * shades[u],
                )
                axes[1].scatter(
                    position + offset,
                    np.median(change),
                    s=34,
                    edgecolors="white",
                    zorder=3,
                    **style,
                )
        axes[0].scatter(
            TAU_VALUES_S.index(current[1]),
            values[current].mean(),
            color="black",
            s=75,
            zorder=3,
        )
        axes[0].set_xlim(-0.3, positions[-1] + 1.9)
        axes[0].set_ylabel(r"mean validation deviance explained ($D^2$)")
        axes[1].axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axes[1].scatter(
            TAU_VALUES_S.index(current[1]) + 0.22 * (U_VALUES.index(current[0]) - 1),
            0,
            color="black",
            s=34,
            zorder=3,
        )
        axes[1].set_ylabel(r"validation $D^2$ minus the pair in use")
        for axis in axes:
            axis.set_xticks(positions, [f"{1000 * tau:.0f}" for tau in TAU_VALUES_S])
            axis.set_xlabel("recovery time constant (ms)")
        label_panels(axes, -0.2)
        figure.tight_layout(w_pad=2.0)
        return save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=cli.PoissonGLM().root,
        help="Session artifact directory with the prepared windows and video.",
    )
    args = parser.parse_args()
    output = args.root / "adaptation_grid"
    rows = run_grid(args.root, output)
    print(plot_grid(rows, output / "adaptation_grid"))


if __name__ == "__main__":
    main()
