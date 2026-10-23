"""Test whether the predicted rate of the V1 GLM is correct where it is low and high.

Question: a model can predict well on average and still be too high at low
rates and too low at high rates. Is the held-out predicted rate calibrated
over its range?

Comparison: no new fit. For each unit, its held-out 1-ms bins (each bin
predicted by the cross-validation fold that did not train on its trial, with
the observed spike history scaled on the fitting rows of that fold) are sorted
by predicted rate and cut into `GROUPS` groups of equal size. In each group,
the observed spike count over the predicted count; 1 is a calibrated model.

- Units, trials, alignment, valid bins and exclusions: those of `fit.py`
  (all units of both mice). A fold with no test spikes has no saved model and
  is not used.
- Statistics: in each group, the pooled median of the log ratio over units
  with a hierarchical bootstrap (mice, then units), drawn as a ratio; the line
  of each mouse is its median. The sampling unit is the unit.
- The fit with a 100 times weaker penalty on the fixation kernel
  (`penalty_sensitivity.py`, in `penalty_sensitivity.OUT`) is drawn too when
  its fits exist.

The ratios of each fit are saved in `calibration.npz` in its fit directory and
not computed again. The script writes `calibration.svg` and `calibration.json`
in `results/glm/`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.penalty_sensitivity import MAIN, OUT
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    hierarchical_interval,
    save_figure,
)

GROUPS = 20
WEAKER_COLOR = "0.45"


def unit_ratios(root: Path) -> np.ndarray:
    """Return observed / predicted count by group of predicted rate (unit, group)."""
    model = cli.PoissonGLM(root=root)
    cache = model.fit_dir("all") / "calibration.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return data["ratio"]
    prepared = dmat._load_windows(model.windows)
    alignments = prepared["alignments"]
    valid = dmat._valid_bin_mask(prepared)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    partitions = fit.cross_validation_partitions(
        len(alignments), len(valid) // len(alignments), valid
    )
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")
    ratios = []
    for record in fit.load_unit_records(
        model.fit_dir("all") / "checkpoints", "unit_*_folds.json"
    ):
        spikes = np.asarray(units.loc[record["unit_id"], "spike_times_s"], dtype=float)
        counts = fit.build_unit_counts(alignments, spikes)
        history = dmat.build_unit_history(alignments, spikes)
        width = int(metadata["base_columns"]) + int(record["components"])
        scored = [part for part in partitions if counts[part["test"]].sum() > 0]
        observed, predicted = [], []
        for part, fold in zip(scored, record["folds"], strict=True):
            scaled, _, _ = dmat.training_zscore(history, part["fit"])
            design = dmat._design_with_history(common, scaled, width, part["test"])
            predicted.append(
                np.exp(fold["intercept"] + design @ np.asarray(fold["coefficients"]))
            )
            observed.append(counts[part["test"]])
        observed, predicted = np.concatenate(observed), np.concatenate(predicted)
        groups = np.array_split(np.argsort(predicted), GROUPS)
        ratios.append([observed[g].sum() / predicted[g].sum() for g in groups])
        print(f"{root.name}: calibration of unit {record['unit_id']}", flush=True)
    ratio = np.asarray(ratios)
    np.savez(cache, ratio=ratio)
    return ratio


def complete(root: Path) -> bool:
    """Return whether every unit of a fit has its final refit."""
    checkpoints = cli.PoissonGLM(root=root).fit_dir("all") / "checkpoints"
    finals = len(list(checkpoints.glob("unit_*_final.json")))
    return finals > 0 and finals == len(
        list(checkpoints.glob("unit_*_validation.json"))
    )


def pooled(ratios: list[np.ndarray]) -> dict:
    """Return the pooled median ratio of each group, its interval, and each mouse's."""
    logs = [np.log(np.maximum(r, 1e-6)) for r in ratios]
    stats = [hierarchical_interval([g[:, k] for g in logs]) for k in range(GROUPS)]
    return {
        "median": np.exp([s["median"] for s in stats]),
        "interval": np.exp([s["interval_95"] for s in stats]),
        "p_ratio_is_1": [s["p_bootstrap"] for s in stats],
        "mouse": np.exp([s["median_of_each_mouse"] for s in stats]).T,
    }


def main() -> None:
    fits = {"model in use": MAIN}
    if all(complete(OUT / session) for session in SESSIONS):
        fits["fixation penalty 100× weaker"] = OUT
    subjects = [session.split("_")[0] for session in SESSIONS]
    results = {
        name: pooled([unit_ratios(out / session) for session in SESSIONS])
        for name, out in fits.items()
    }
    centers = (np.arange(GROUPS) + 0.5) * 100 / GROUPS
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(3.6, 2.8), layout="constrained")
        axis.axhline(1, color=REFERENCE_COLOR, lw=0.8)
        for row, (name, color) in enumerate(
            zip(results, (MODEL_COLOR, WEAKER_COLOR), strict=False)
        ):
            data = results[name]
            if row == 0:
                axis.fill_between(
                    centers, *data["interval"].T, color=color, alpha=0.2, lw=0
                )
                for mouse, style in zip(data["mouse"], (":", "--"), strict=True):
                    axis.plot(centers, mouse, color=color, lw=0.8, ls=style)
            axis.plot(centers, data["median"], color=color, lw=1.6 if row == 0 else 1.0)
            axis.text(
                0.97,
                0.97 - 0.09 * row,
                name,
                color=color,
                transform=axis.transAxes,
                va="top",
                ha="right",
            )
        axis.text(
            0.97,
            0.97 - 0.09 * len(results),
            f"dotted {subjects[0]}, dashed {subjects[1]}",
            color=MODEL_COLOR,
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=6,
        )
        axis.set_xlabel("predicted rate (percentile of held-out bins)")
        axis.set_ylabel("observed / predicted count")
        axis.set_xlim(0, 100)
        print(save_figure(figure, MAIN / "calibration"))
    report = {
        name: {
            "group_centers_percentile": centers.tolist(),
            **{
                key: np.asarray(value).tolist()
                for key, value in data.items()
                if key != "mouse"
            },
            **{
                f"median_{subject}": data["mouse"][index].tolist()
                for index, subject in enumerate(subjects)
            },
        }
        for name, data in results.items()
    }
    (MAIN / "calibration.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps({k: np.round(v["median"], 2).tolist() for k, v in results.items()})
    )


if __name__ == "__main__":
    main()
