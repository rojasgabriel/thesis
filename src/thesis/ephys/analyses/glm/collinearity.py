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
- Shared direction of two regressors (`shared_directions`): their first
  canonical correlation, the largest correlation that a weighted sum of the
  columns of one has with a weighted sum of the columns of the other. It is
  the worst case for one pair, where the GVIF is an average over all the
  directions of a regressor. The pairs of task and motion-energy regressors are
  the same for all units; for a pair with spike history, the value is the
  median over units.
- This script makes no figure. `design_optimization.py` draws the three
  measures for a session.

Units (`--units`), trials, alignment, and exclusions are those of `fit.py`.
The sampling unit is the unit. The script writes `collinearity.json` in
`figures/glm/` and `collinearity.npz` in each session directory.

    uv run python -m thesis.ephys.analyses.glm.collinearity
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.unique_dev import block_slices
from thesis.ephys.units import fetch_unit_table

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


def shared_directions(correlation: np.ndarray, groups: dict[str, slice]) -> np.ndarray:
    """Return the first canonical correlation of each pair of regressors.

    It is the strongest direction that two regressors share: the largest
    correlation that a weighted sum of the columns of one has with a weighted
    sum of the columns of the other. 0 for regressors that share nothing, 1
    when a time course can be made by both. The diagonal is NaN.
    """
    names = list(groups)
    out = np.full((len(names), len(names)), np.nan)
    for index, first in enumerate(names):
        for other in range(index + 1, len(names)):
            second = names[other]
            own = correlation[groups[first], groups[first]]
            rest = correlation[groups[second], groups[second]]
            cross = correlation[groups[first], groups[second]]
            shared = np.linalg.solve(own, cross) @ np.linalg.solve(rest, cross.T)
            value = np.sqrt(max(np.linalg.eigvals(shared).real.max(), 0))
            out[index, other] = out[other, index] = value
    return out


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
    for session in SESSIONS:
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
        jobs = [np.asarray(spikes[int(item["unit_id"])], dtype=float) for item in final]
        with Pool(args.workers, _load, (model.windows, model.design, width)) as pool:
            correlations = pool.map(_unit_correlation, jobs)
        units = [group_statistics(matrix, groups) for matrix in correlations]
        names = list(groups)
        # The pairs with spike history differ between units: the median over units.
        pairs = np.median(
            [shared_directions(matrix, groups) for matrix in correlations], axis=0
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
        order = np.argsort(np.nan_to_num(pairs).ravel())[::-1][::2][:5]
        entry["largest_shared_directions"] = [
            {
                "between": [names[index // len(names)], names[index % len(names)]],
                "value": float(pairs.ravel()[index]),
            }
            for index in order
        ]
        stats[session] = entry
        np.savez(
            model.root / "collinearity.npz",
            regressors=np.asarray(names),
            gvif=np.asarray([[unit["gvif"][name] for name in names] for unit in units]),
            condition_indices=np.asarray([u["condition_indices"] for u in units]),
            shared_directions=pairs,
        )
    (args.out / "collinearity.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
