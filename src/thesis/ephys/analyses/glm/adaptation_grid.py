"""Compare a synaptic-depression flash term with the term in the V1 GLM.

Question: the model scales the flash kernel by 1 for a flash that follows
another within 50 ms (`dmat.flash_recent`). Does a synaptic-depression resource
in its place predict held-out spikes better? In the depression model each flash
uses the fraction u of a resource, which recovers with the time constant tau
(`flash_resource`).

Comparison: for each pair on the grid (`U_VALUES` x `TAU_VALUES_S`), build the
full design with the depression resource as the second flash term and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. The term in use is scored in the same way. Nothing else
in the model changes. The comparison is paired: each unit is scored with each
design.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample (`--units sample`), because the grid needs one design and
one fit set for each pair. The sampling unit is the unit.

- Panel a: mean validation D² over units against the recovery time constant,
  one line for each fraction of the resource used per flash. The black line is the term in use.
- Panel b: for each pair, the validation D² of each unit minus its value with
  the term in use (small points), and the median over units (large points).

The shortest interval between flashes is 40 ms. With a recovery much faster
than that, the resource is full again at each flash unless the last flash was
40 ms before, so the depression model becomes the term in use. The y range of
panel a is small; read the size of the differences from the axis.

    uv run python -m thesis.ephys.analyses.glm.adaptation_grid

A design that has saved scores in `adaptation_grid/adaptation_grid.json` is not
fit again. Each new design takes about 20 seconds.
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

U_VALUES = (0.25, 0.5, 0.75, 0.9)
TAU_VALUES_S = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
UNIT_SET = "sample"
INPUTS = ("stimulus_windows_me.npz", "video_me_features.npz", "frame_times.npy")


def flash_resource(modeled: np.ndarray, u: float, tau_recover_s: float) -> np.ndarray:
    """Available synaptic resource at each flash, from short-term depression.

    A resource pool starts full (R = 1). Each flash spends fraction `u` of what
    is available, then the pool recovers toward 1 with time constant
    `tau_recover_s` before the next flash. Returns R just before each flash: the
    first flash sees a full pool (R = 1, the strongest response) and later
    flashes in a dense train see a depleted pool (graded adaptation).

    This is the Tsodyks-Markram depression model (Tsodyks & Markram, 1997, PNAS),
    driven by stimulus flashes.
    """
    resource = np.ones(len(modeled))
    for position in range(1, len(modeled)):
        recovery = np.exp(-(modeled[position] - modeled[position - 1]) / tau_recover_s)
        resource[position] = 1 - (1 - resource[position - 1] * (1 - u)) * recovery
    return resource


def score_design(
    source: Path, work: Path, name: str, settings: dict
) -> dict[str, float]:
    """Return each unit's validation D² with some `dmat` attributes replaced."""
    model = cli.PoissonGLM(root=work / name)
    model.root.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copy(source / name, model.root / name)
    # ponytail: the design reads module constants; set them for one design and
    # restore them. Make them arguments if the design gets more callers.
    saved = {name: getattr(dmat, name) for name in settings}
    for name, value in settings.items():
        setattr(dmat, name, value)
    try:
        dmat.prepare_common_design(model.windows, model.video, model.design)
        fit.fit_models(model.windows, model.design, UNIT_SET, model.fit_dir(UNIT_SET))
    finally:
        for name, value in saved.items():
            setattr(dmat, name, value)
    count = str(max(dmat.VIDEO_COMPONENT_COUNTS))
    scores = {}
    for path in (model.fit_dir(UNIT_SET) / "checkpoints").glob(
        "unit_*_validation.json"
    ):
        record = json.loads(path.read_text())
        scores[str(record["unit_id"])] = record["plus_video"][count]["validation"][
            "deviance_explained"
        ]
    shutil.rmtree(model.root)  # one design is about 80 MB
    return scores


def run_grid(source: Path, output: Path) -> list[dict]:
    """Score every pair that has no saved scores; return all pairs."""
    path = output / "adaptation_grid.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(row["u"], row["tau"]) for row in rows}
    if (None, None) not in done:  # the term in use
        rows.append(
            {
                "u": None,
                "tau": None,
                "units": score_design(source, output / "work", "in_use", {}),
            }
        )
    for u in U_VALUES:
        for tau_s in TAU_VALUES_S:
            if (u, tau_s) in done:
                continue
            rows.append(
                {
                    "u": u,
                    "tau": tau_s,
                    "units": score_design(
                        source,
                        output / "work",
                        f"u{u}_tau{tau_s}",
                        {
                            "flash_recent": lambda modeled, u=u, tau_s=tau_s: (
                                flash_resource(modeled, u, tau_s)
                            )
                        },
                    ),
                }
            )
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=1))
            print(f"Scored u={u}, tau={tau_s} s.", flush=True)
    return rows


def plot_grid(rows: list[dict], output: Path | None, axes=None) -> Path | None:
    """Plot mean validation D² and each unit's change from the term in use.

    With `axes`, draw into them and do not label or save (for a combined figure).
    """
    scores = {(row["u"], row["tau"]): row["units"] for row in rows}
    current = (None, None)
    unit_ids = sorted(scores[current])
    if any(sorted(units) != unit_ids for units in scores.values()):
        raise ValueError("Every pair must score the same units.")
    values = {
        pair: np.asarray([units[unit] for unit in unit_ids])
        for pair, units in scores.items()
    }
    positions = np.arange(len(TAU_VALUES_S))
    # One concept, the visual response; a darker line uses more of the resource.
    shades = dict(zip(U_VALUES, np.linspace(0.3, 1.0, len(U_VALUES)), strict=True))
    dodge = 0.18 * (np.arange(len(U_VALUES)) - (len(U_VALUES) - 1) / 2)
    rng = np.random.default_rng(0)

    with plt.rc_context(FIGURE_STYLE):
        figure = None
        if axes is None:
            figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
        for index, u in enumerate(U_VALUES):
            style = {"color": CONCEPT_COLORS["visual"], "alpha": shades[u]}
            means = [values[(u, tau)].mean() for tau in TAU_VALUES_S]
            axes[0].plot(positions, means, marker="o", markerfacecolor="white", **style)
            axes[0].text(
                positions[-1] + 0.25,
                means[-1],
                f"{u:.0%} used",
                va="center",
                **style,
            )
            offset = dodge[index]
            for position, tau in zip(positions, TAU_VALUES_S, strict=True):
                change = values[(u, tau)] - values[current]
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
        axes[0].axhline(values[current].mean(), color="black", linewidth=1.2)
        axes[0].text(
            positions[-1] + 1.6,
            values[current].mean(),
            "term in use:\nflash in the last 50 ms",
            ha="right",
            va="bottom",
        )
        axes[0].set_xlim(-0.3, positions[-1] + 1.6)
        axes[0].set_ylabel(r"mean validation $D^2$")
        axes[1].axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axes[1].set_ylabel(r"validation $D^2$ minus the term in use")
        for axis in axes:
            axis.set_xticks(positions, [f"{1000 * tau:.0f}" for tau in TAU_VALUES_S])
            axis.set_xlabel("recovery time constant (ms)")
        if figure is None:
            return None
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
