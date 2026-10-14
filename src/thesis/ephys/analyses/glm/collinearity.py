"""Measure the collinearity of the regressors of the V1 GLM.

Question: are the regressors so correlated that the weights (the kernels)
cannot be read? Collinearity does not change the prediction, but it makes the
weights of the collinear regressors uncertain, and the goal of this model is
to read the weights.

There is no comparison between conditions. The statistics are properties of
the design matrix, on the rows that the fit uses (the valid bins of all
trials). The design of a unit is the common columns (task and motion energy)
plus the spike-history columns of that unit, so each statistic has one value
for each unit.

- Generalized variance inflation factor of a regressor (Fox and Monette,
  1992): GVIF = det(R_gg) det(R_rest) / det(R), with R the correlation matrix
  of all columns, R_gg that of the columns of the regressor, and R_rest that
  of all other columns. For a regressor with one column it is the usual VIF.
  For a regressor with p columns the script gives GVIF^(1/p), which is on the
  scale of a VIF: the factor by which the collinearity with the OTHER
  regressors multiplies the variance of its weights. The usual limits are 5
  (some concern) and 10 (a problem).
- The VIF of one column is also given (the largest in each regressor). For a
  kernel this value is not a problem by itself: the basis functions of one
  kernel overlap, so their single weights are correlated, but the kernel is
  their weighted sum and is defined. The GVIF is the correct measure for a
  kernel.
- First canonical correlation of a regressor with all the other columns: the
  largest correlation that a weighted sum of its columns has with a weighted
  sum of the other columns.
- Condition indices (Belsley, Kuh, and Welsch, 1980): sqrt(largest eigenvalue
  / each eigenvalue) of R. An index above 30 is a sign of a strong dependency,
  10 to 30 of a moderate one. For each index above `MODERATE_INDEX`, the
  variance-decomposition proportions say which weights that dependency makes
  uncertain: the script gives, for each regressor, the largest proportion of
  one of its columns. A dependency is a problem between regressors only when
  two or more regressors have a high proportion (above 0.5) in the same index.
- These are statistics of an ordinary fit. The model has an L2 penalty, which
  makes the weights more stable than these values say. The direct test of the
  weights is `kernel_reliability.py`.
- Figure: one row for each session. a, GVIF^(1/p) of each regressor in each
  unit (points; the line is the median). b, the condition indices of each
  unit, sorted. The dashed lines are the usual limits.

Units (`--units`), trials, alignment, and exclusions are those of `fit.py`.
The sampling unit is the unit. The script writes `collinearity.svg` and
`collinearity.json` in `figures/glm/`.

    uv run python -m thesis.ephys.analyses.glm.collinearity
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

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_by_period import SESSIONS
from thesis.ephys.analyses.glm.main_figure import REGRESSORS
from thesis.ephys.analyses.glm.unique_dev import block_slices
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, label_panels, save_figure

VIF_LIMITS = (5, 10)
INDEX_LIMITS = (10, 30)
MODERATE_INDEX = 10
HIGH_PROPORTION = 0.5
_SHARED: dict = {}


def _load(windows: Path, design: Path, width: int) -> None:
    prepared = dmat._load_windows(windows)
    valid = dmat._valid_bin_mask(prepared)
    common = np.asarray(np.load(design, mmap_mode="r")[:, :width], dtype=float)[valid]
    _SHARED.update(
        alignments=prepared["alignments"],
        valid=valid,
        common=(common - common.mean(0)) / common.std(0),
    )


def _unit_correlation(spike_times: np.ndarray) -> np.ndarray:
    """Return the correlation matrix of the design columns of one unit."""
    history = dmat.build_unit_history(_SHARED["alignments"], spike_times)
    history = history[_SHARED["valid"]]
    history = (history - history.mean(0)) / history.std(0)
    columns = np.hstack((_SHARED["common"], history))
    return columns.T @ columns / len(columns)


def largest_shared_direction(
    correlation: np.ndarray, groups: dict[str, slice]
) -> tuple[float, str, str]:
    """Return the largest first canonical correlation between two regressors.

    It is the strongest direction that two regressors of a design share: 0 for
    regressors that share nothing, 1 for two regressors where a weighted sum of
    the columns of one is a weighted sum of the columns of the other.
    """
    best = (0.0, "", "")
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            own = correlation[groups[first], groups[first]]
            other = correlation[groups[second], groups[second]]
            cross = correlation[groups[first], groups[second]]
            shared = np.linalg.solve(own, cross) @ np.linalg.solve(other, cross.T)
            value = float(np.sqrt(max(np.linalg.eigvals(shared).real.max(), 0)))
            best = max(best, (value, first, second))
    return best


def group_statistics(correlation: np.ndarray, groups: dict[str, slice]) -> dict:
    """Return the collinearity statistics of one design."""
    size = len(correlation)
    inverse = np.linalg.inv(correlation)
    vif = np.diag(inverse)
    log_det = np.linalg.slogdet(correlation)[1]
    out = {"gvif": {}, "largest_column_vif": {}, "canonical_r": {}}
    for name, columns in groups.items():
        inside = np.zeros(size, dtype=bool)
        inside[columns] = True
        own = correlation[np.ix_(inside, inside)]
        rest = correlation[np.ix_(~inside, ~inside)]
        cross = correlation[np.ix_(inside, ~inside)]
        gvif = np.exp(np.linalg.slogdet(own)[1] + np.linalg.slogdet(rest)[1] - log_det)
        out["gvif"][name] = float(gvif ** (1 / inside.sum()))
        out["largest_column_vif"][name] = float(vif[inside].max())
        shared = np.linalg.solve(own, cross) @ np.linalg.solve(rest, cross.T)
        out["canonical_r"][name] = float(np.sqrt(np.linalg.eigvals(shared).real.max()))
    values, vectors = np.linalg.eigh(correlation)
    index = np.sqrt(values.max() / values)
    # Variance-decomposition proportions: (column, eigenvalue).
    parts = vectors**2 / values
    proportions = parts / parts.sum(1, keepdims=True)
    order = np.argsort(index)[::-1]
    out["condition_indices"] = index[order].tolist()
    out["dependencies"] = [
        {
            "condition_index": float(index[position]),
            "regressors_with_a_high_proportion": {
                name: float(proportions[columns, position].max())
                for name, columns in groups.items()
                if proportions[columns, position].max() > HIGH_PROPORTION
            },
        }
        for position in order
        if index[position] > MODERATE_INDEX
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stats = {}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(SESSIONS),
            2,
            figsize=(9.5, 3.1 * len(SESSIONS)),
            width_ratios=(2.2, 1),
            layout="constrained",
        )
        for (gvif_axis, index_axis), session in zip(axes, SESSIONS):
            model = cli.PoissonGLM(root=args.out / session)
            metadata = json.loads(model.design.with_suffix(".json").read_text())
            final = load_unit_records(
                model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
            )
            components = int(final[0]["components"])
            groups = {
                name: columns
                for name, columns in block_slices(metadata, components).items()
                if name != "task"
            }
            width = int(metadata["base_columns"]) + components
            subject, _, name = session.partition("_")
            table = fetch_unit_table(
                subject,
                name,
                unit_criteria_id=1,
                stability_param_id=0,
                include_metrics=False,
            )
            spikes = dict(zip(table["unit_id"].astype(int), table["spike_times_s"]))
            jobs = [
                np.asarray(spikes[int(item["unit_id"])], dtype=float) for item in final
            ]
            with Pool(
                args.workers, _load, (model.windows, model.design, width)
            ) as pool:
                correlations = pool.map(_unit_correlation, jobs)
            units = [group_statistics(matrix, groups) for matrix in correlations]
            names = sorted(
                groups, key=lambda name: -np.median([u["gvif"][name] for u in units])
            )
            entry = {"units": len(units), "columns": len(correlations[0])}
            for name in names:
                columns = groups[name]
                entry[name] = {"columns": columns.stop - columns.start}
                for key in ("gvif", "largest_column_vif", "canonical_r"):
                    values = np.asarray([unit[key][name] for unit in units])
                    label = "gvif_per_column" if key == "gvif" else key
                    entry[name][label] = {
                        "median": float(np.median(values)),
                        "max": float(values.max()),
                    }
            largest = np.asarray([unit["condition_indices"][0] for unit in units])
            entry["largest_condition_index"] = {
                "median": float(np.median(largest)),
                "max": float(largest.max()),
                "units_above_30": int(np.sum(largest > INDEX_LIMITS[1])),
            }
            shared = [
                sorted(item["regressors_with_a_high_proportion"])
                for unit in units
                for item in unit["dependencies"]
                if len(item["regressors_with_a_high_proportion"]) > 1
            ]
            entry["dependencies_above_10_with_two_or_more_regressors"] = {
                " + ".join(pair): shared.count(pair)
                for pair in sorted({tuple(pair) for pair in shared})
            }
            entry["dependencies_above_10_per_unit_median"] = float(
                np.median([len(unit["dependencies"]) for unit in units])
            )
            stats[session] = entry
            rng = np.random.default_rng(0)
            for position, name in enumerate(names):
                values = np.asarray([unit["gvif"][name] for unit in units])
                color = REGRESSORS[name][1]
                gvif_axis.scatter(
                    position + rng.uniform(-0.25, 0.25, len(values)),
                    values,
                    s=5,
                    color=color,
                    alpha=0.5,
                    lw=0,
                )
                gvif_axis.hlines(
                    np.median(values), position - 0.35, position + 0.35, color="black"
                )
            for limit in VIF_LIMITS:
                gvif_axis.axhline(limit, color=REFERENCE_COLOR, ls="--", lw=0.8)
            gvif_axis.set_ylim(0.9, 12)
            gvif_axis.set_yscale("log")
            gvif_axis.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
            gvif_axis.set_xticks(
                range(len(names)),
                [REGRESSORS[name][0] for name in names],
                rotation=35,
                ha="right",
            )
            for tick, name in zip(gvif_axis.get_xticklabels(), names):
                tick.set_color(REGRESSORS[name][1])
            gvif_axis.set_ylabel(f"{subject}\nvariance inflation,\n" + r"GVIF$^{1/p}$")
            entry["units_outside_the_axis_limits"] = int(
                sum(max(unit["gvif"].values()) > 12 for unit in units)
            )
            for unit in units:
                index_axis.plot(
                    unit["condition_indices"], color="0.3", alpha=0.15, lw=0.6
                )
            for limit in INDEX_LIMITS:
                index_axis.axhline(limit, color=REFERENCE_COLOR, ls="--", lw=0.8)
            index_axis.set_yscale("log")
            index_axis.set_ylim(0.9, 60)
            index_axis.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
            index_axis.set_xlabel("dimension of the design, sorted")
            index_axis.set_ylabel("condition index")
        label_panels(axes[0], x=-0.12, y=1.05)
        print(save_figure(figure, args.out / "collinearity"))
    (args.out / "collinearity.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
