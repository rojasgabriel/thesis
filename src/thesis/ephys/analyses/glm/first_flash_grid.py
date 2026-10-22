"""Test the form of the kernel for the first flash after the exit.

Question: the candidate kernel for the first flash after the center exit
(`dmat.FIRST_FLASH_AFTER_EXIT_TERM`) passes the inclusion rule in both mice,
but its fitted response peaks about 10 ms after the observed one
(`first_flash_after_exit.py`). Its basis is that of the other flash kernels:
8 raised cosines over 150 ms, about 21 ms apart. Does a finer or a shorter
basis predict better?

Comparison: the design in use plus the kernel, with its basis changed
(`DESIGNS`; window in ms and number of raised cosines), and the design in use
with no such kernel ("none"). Only this kernel changes; the other flash
kernels keep their basis. Validation D² of each unit on the fixed sets of 20
units of the grids (`--samples 4` gives 80 units of GRB006; no unit is in two
sets). The comparison with `IN_USE` is paired over units: the median change
of D² and a two-sided Wilcoxon signed-rank test. By the rule of the grids,
another form replaces `IN_USE` only when it predicts reliably better. Units,
trials, alignment, baseline, and exclusions are those of `fit.py`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.first_flash_grid --samples 4

A design with saved scores in `first_flash_grid/first_flash_grid.json` is not
fit again. The comparison goes to `first_flash_grid/comparison.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, grid

ON = {"FIRST_FLASH_AFTER_EXIT_TERM": True}
DESIGNS = {
    "none": {"FIRST_FLASH_AFTER_EXIT_TERM": False},
    "150 ms, 8": {**ON, "FIRST_FLASH_WINDOW_S": 0.15, "FIRST_FLASH_BASES": 8},
    "150 ms, 12": {**ON, "FIRST_FLASH_WINDOW_S": 0.15, "FIRST_FLASH_BASES": 12},
    "150 ms, 16": {**ON, "FIRST_FLASH_WINDOW_S": 0.15, "FIRST_FLASH_BASES": 16},
    "100 ms, 8": {**ON, "FIRST_FLASH_WINDOW_S": 0.10, "FIRST_FLASH_BASES": 8},
}
IN_USE = "150 ms, 8"
NAME = "first_flash_grid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    output = args.root / NAME
    scores = grid.score_designs(
        args.root, output, f"{NAME}.json", DESIGNS, args.samples
    )
    units = sorted(set.intersection(*(set(s) for s in scores.values())))
    reference = np.asarray([scores[IN_USE][unit] for unit in units])
    comparison = {"units": len(units), "in_use": IN_USE}
    for name in DESIGNS:
        if name == IN_USE:
            continue
        change = np.asarray([scores[name][unit] for unit in units]) - reference
        comparison[name] = {
            "median_change": float(np.median(change)),
            "mean_change": float(np.mean(change)),
            "units_better": int(np.sum(change > 0)),
            "wilcoxon_p": float(wilcoxon(change).pvalue),
        }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=1))
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()
