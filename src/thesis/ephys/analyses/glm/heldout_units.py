"""Check that the V1 GLM design is not tuned to the units that selected it.

Question: every design choice was made on the fixed 20-unit sample. Does the
final design predict the units outside that sample as well as it predicts the
sample?

Comparison: no new fits. From the full fit (`--units all`) of one session, the
cross-validated D², bits/spike, and the unique ΔD² of each regressor are
summarized for all units, for the selection units (the units in
`sample_fit_me`), and for the held-out units (all others). The two groups are
compared with a two-sided Mann-Whitney U test; the groups are not paired.

Units, trials, alignment, baseline, and exclusions are those of `fit.py`. The
sampling unit is the unit. The script writes `heldout_units.json` in the fit
directory and makes no figure.

    uv run python -m thesis.ephys.analyses.glm.heldout_units
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.unique_dev import DETAILED_GROUPS, UNIQUE


def _describe(values: np.ndarray) -> dict:
    low, median, high = np.percentile(values, [25, 50, 75])
    return {
        "units": len(values),
        "median": float(median),
        "q25": float(low),
        "q75": float(high),
    }


def summarize(root: Path) -> dict:
    """Return each measure for all, selection, and held-out units."""
    model = cli.PoissonGLM(root=root)
    checkpoints = model.fit_dir("all") / "checkpoints"
    final = load_unit_records(checkpoints, "unit_*_final.json")
    unique = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints / "unique", "unit_*.json")
    }
    selection = {
        int(item["unit_id"])
        for item in load_unit_records(
            model.fit_dir("sample") / "checkpoints", "unit_*_final.json"
        )
    }
    if not final or not selection:
        raise FileNotFoundError("Run `glm fit` with --units all and --units sample.")
    chosen = np.asarray([int(item["unit_id"]) in selection for item in final])
    measures = {
        "cross-validated D2": [
            item["cross_validated_deviance_explained"] for item in final
        ],
        "bits/spike": [item["cross_validated_bits_per_spike"] for item in final],
    }
    if len(unique) == len(final):  # `glm unique` is complete
        for group in DETAILED_GROUPS:
            measures[f"unique dD2, {group}"] = [
                unique[int(item["unit_id"])]["groups"][group][UNIQUE] for item in final
            ]
    summary = {}
    for name, values in measures.items():
        values = np.asarray(values, dtype=float)
        summary[name] = {
            "all": _describe(values),
            "selection": _describe(values[chosen]),
            "held out": _describe(values[~chosen]),
            "mann_whitney_p": float(
                mannwhitneyu(values[chosen], values[~chosen]).pvalue
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    summary = summarize(args.root)
    output = cli.PoissonGLM(root=args.root).fit_dir("all") / "heldout_units.json"
    output.write_text(json.dumps(summary, indent=1))
    for name, item in summary.items():
        print(
            f"{name:<34} all {item['all']['median']:.4f} | "
            f"selection {item['selection']['median']:.4f} "
            f"(n={item['selection']['units']}) | "
            f"held out {item['held out']['median']:.4f} "
            f"(n={item['held out']['units']}) | p = {item['mann_whitney_p']:.2g}"
        )


if __name__ == "__main__":
    main()
