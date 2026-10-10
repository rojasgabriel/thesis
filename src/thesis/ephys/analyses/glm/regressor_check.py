"""Find regressor blocks that add columns to the V1 GLM but no held-out deviance.

Question: which regressor blocks do not explain neural activity? Such a block
adds design columns and does not improve cross-validated deviance, and this
must hold in repeated draws of units, not in one draw only. The script
identifies these blocks; it does not remove them.

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
fraction of units with a reliable gain. A unit has a reliable gain when a
one-sided one-sample t-test of its fold values of unique ΔD² against zero gives
p < 0.05. Zero is the value when the shuffled block predicts as well as the real
block. With ten folds this is mean > 1.83 fold SEM. The folds share training
trials, so the test is somewhat too easy to pass; a block with no effect gives
about 5% of units.

Figure: the same five panels that `glm unique` draws for one fit
(`plot_deviance_explained`), here for the units of all samples. (a) Full-model
D², (b) maximal and (c) unique ΔD² of all units, one box per block, without
unit points or outlier markers. (d) Pooled median unique ΔD² against the column
count of the block, on logarithmic axes; the vertical line is the range of the
sample medians, and the horizontal line is the range of the column count (only
motion energy varies, because each sample selects its PC count). A sample
median at or below zero cannot be drawn, so its range line stops at the bottom
of the axis. (e) Fraction of units with a reliable gain, one point per sample,
with the mean. The dotted line is the chance level, 0.05: the fraction of units
that pass the test when the block has no effect. The medians and tests in
`regressor_check.json` use every unit.

    uv run python -m thesis.ephys.analyses.glm.regressor_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.ephys.analyses.glm.figures import plot_deviance_explained
from thesis.ephys.analyses.glm.fit import _write_json_atomic, load_unit_records
from thesis.ephys.analyses.glm.unique_dev import block_table


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
    figure = plot_deviance_explained(samples, args.root / "regressor_check")
    _write_json_atomic(args.root / "regressor_check.json", table)
    print(json.dumps({**table, "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
