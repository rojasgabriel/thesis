"""Give the held-out performance of the V1 GLM in the measures that other papers use.

Question: how well does the model predict, in measures that can be set next
to other encoding models? The fit is scored with the Poisson deviance in 1-ms
bins (D², bits/spike), which few papers report. This script makes no fit. It
predicts each trial with the cross-validation fold that did not train on it,
and compares the prediction with the observed spikes at four bin widths and
as a mean over trials.

- Prediction with no spike history: the rate from the task and motion-energy
  terms of the fold. This is the fair measure at bins wider than 1 ms. The
  spike-history term uses the observed spikes, so in a wide bin the full
  prediction has the spikes of that same bin in it.
- Single trials: the Pearson correlation of the observed and the predicted
  spike count over all held-out bins of `WIDTHS_MS`, and its square. Only
  bins before the response entry are used.
- Mean over trials (PSTH): the same correlation for the mean over the
  held-out trials in `PSTH_MS` bins, for the bins where half of the trials or
  more have not reached the response port.
- There is no comparison between conditions. The sampling unit is the unit:
  the figure gives the median and the interquartile range over units, for
  each session. Units, trials, alignment, and exclusions are those of
  `fit.py`.

The script writes `performance.svg` and `performance.json` in `figures/glm/`.

    uv run python -m thesis.ephys.analyses.glm.performance
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

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.analyses.glm.fit import build_unit_counts, load_unit_records
from thesis.ephys.analyses.glm.flash_by_period import SESSIONS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, MODEL_COLOR, label_panels, save_figure

WIDTHS_MS = (1, 10, 33, 100)
PSTH_MS = 20
_SHARED: dict = {}


def _load(windows: Path, design: Path) -> None:
    prepared = dmat._load_windows(windows)
    _SHARED.update(
        prepared=prepared,
        valid=dmat._valid_bin_mask(prepared),
        common=np.load(design, mmap_mode="r", allow_pickle=False),
        metadata=json.loads(design.with_suffix(".json").read_text()),
    )


def _unit(job: dict) -> dict:
    """Return the correlations of one unit, from its held-out predictions."""
    prepared, valid, common, metadata = (
        _SHARED[key] for key in ("prepared", "valid", "common", "metadata")
    )
    alignments = prepared["alignments"]
    grid = (len(alignments), len(prepared["relative_centers"]))
    counts = build_unit_counts(alignments, job["spike_times"])
    width = int(metadata["base_columns"]) + int(job["folds"]["components"])
    predicted = np.full(len(counts), np.nan)
    partitions = fit.cross_validation_partitions(grid[0], grid[1], valid)
    scored = [part for part in partitions if counts[part["test"]].sum() > 0]
    for part, fold in zip(scored, job["folds"]["folds"]):
        weights = np.asarray(fold["coefficients"], dtype=float)[:width]
        rows = np.asarray(common[part["test"], :width], dtype=float)
        predicted[part["test"]] = np.exp(fold["intercept"] + rows @ weights)
    used = (valid & np.isfinite(predicted)).reshape(grid)
    observed = counts.astype(float).reshape(grid)
    predicted = np.nan_to_num(predicted).reshape(grid)

    def cut(values, width_ms):
        bins = grid[1] // width_ms
        return values[:, : bins * width_ms].reshape(grid[0], bins, width_ms)

    out = {}
    for width_ms in WIDTHS_MS:
        full = cut(used, width_ms).all(2)
        out[f"{width_ms}"] = float(
            np.corrcoef(
                cut(observed, width_ms).sum(2)[full],
                cut(predicted, width_ms).sum(2)[full],
            )[0, 1]
        )
    full = cut(used, PSTH_MS).all(2)
    keep = full.mean(0) >= 0.5
    mean = lambda values: (  # noqa: E731
        (cut(values, PSTH_MS).sum(2) * full).sum(0)[keep] / full.sum(0)[keep]
    )
    out["psth"] = float(np.corrcoef(mean(observed), mean(predicted))[0, 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stats, values = {}, {}
    for session in SESSIONS:
        model = cli.PoissonGLM(root=args.out / session)
        subject, _, name = session.partition("_")
        folds = load_unit_records(
            model.fit_dir(args.units) / "checkpoints", "unit_*_folds.json"
        )
        table = fetch_unit_table(
            subject,
            name,
            unit_criteria_id=1,
            stability_param_id=0,
            include_metrics=False,
        )
        spikes = dict(zip(table["unit_id"].astype(int), table["spike_times_s"]))
        jobs = [
            {"folds": item, "spike_times": np.asarray(spikes[int(item["unit_id"])])}
            for item in folds
        ]
        with Pool(args.workers, _load, (model.windows, model.design)) as pool:
            units = pool.map(_unit, jobs)
        values[subject] = {key: np.asarray([u[key] for u in units]) for key in units[0]}
        stats[session] = {"units": len(units)} | {
            ("mean over trials" if key == "psth" else f"single trials, {key} ms"): {
                "median_r": float(np.median(r)),
                "r_quartiles": np.percentile(r, [25, 75]).tolist(),
                "median_r_squared": float(np.median(r**2)),
            }
            for key, r in values[subject].items()
        }
    rng = np.random.default_rng(0)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), layout="constrained")
        keys = [*map(str, WIDTHS_MS), "psth"]
        for axis, (subject, by_key) in zip(axes, values.items()):
            for position, key in enumerate(keys):
                r = by_key[key]
                axis.scatter(
                    position + rng.uniform(-0.25, 0.25, len(r)),
                    r,
                    s=4,
                    color=MODEL_COLOR,
                    alpha=0.4,
                    lw=0,
                )
                axis.hlines(np.median(r), position - 0.35, position + 0.35, "black")
            axis.set_xticks(
                range(len(keys)),
                [f"{w} ms" for w in WIDTHS_MS] + [f"mean over\ntrials, {PSTH_MS} ms"],
            )
            axis.set_ylim(-0.2, 1)
            axis.set_xlabel(f"{subject}: single trials by bin width")
        axes[0].set_ylabel("correlation of held-out\nprediction and spikes")
        label_panels(axes, x=-0.2, y=1.03)
        print(save_figure(figure, args.out / "performance"))
    (args.out / "performance.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
