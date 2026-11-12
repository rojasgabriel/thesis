"""Show two category-selective example units per mouse with the population map.

Question: as `condition_responses.py --condition category`, with two example
units per mouse in place of one, so the slow and the fast preference can
both be seen.

Comparison, units, trials, rates, preference and statistics: those of
`condition_responses.py` (`CONDITIONS["category"]`, `BIN_S` bins, no baseline
subtracted). Examples: among the units with a mean rate of at least
`MIN_RATE_HZ`, the two with the largest |d'| of the trial rates in the
preference window. The heatmap limit and its rule are those of
`condition_responses.py`. The sampling unit is the trial in the example
panels and the unit in the heatmaps.

Writes `figures/condition_responses/category_examples.svg`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.category_examples
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses import condition_responses as cr
from thesis.plotting import label_panels, save_figure

EXAMPLES = 2


def main() -> None:
    spec = cr.CONDITIONS["category"]
    sessions = [cr.session_responses(s, spec) for s in cr.SESSIONS]
    with plt.rc_context(cr.FIGURE_STYLE):
        fig = plt.figure(figsize=(13.0, 6.4), layout="constrained")
        grid = fig.add_gridspec(2, EXAMPLES + 1, width_ratios=(*[1] * EXAMPLES, 0.95))
        letters, image = [], None
        for row, d in enumerate(sessions):
            active = np.flatnonzero(np.nanmean(d["rates"], (1, 2)) >= cr.MIN_RATE_HZ)
            top = active[np.argsort(-np.abs(d["dprime"][active]))][:EXAMPLES]
            print(
                d["subject"],
                [d["unit_ids"][u] for u in top],
                [round(float(d["dprime"][u]), 2) for u in top],
            )
            for col, unit in enumerate(top):
                cell = grid[row, col].subgridspec(
                    2, 1, height_ratios=(1.3, 1.0), hspace=0.05
                )
                raster = fig.add_subplot(cell[0])
                rate = fig.add_subplot(cell[1], sharex=raster)
                raster.tick_params(labelbottom=False)
                d["example"] = int(unit)
                cr.draw_example(raster, rate, d, spec)
                letters.append(raster)
            heat = fig.add_subplot(grid[row, EXAMPLES])
            image = cr.draw_population(heat, d, spec)
            letters.append(heat)
        fig.colorbar(
            image,
            ax=letters[EXAMPLES :: EXAMPLES + 1],
            label="fast − slow (/ mean rate)",
        )
        label_panels(letters)
        print(save_figure(fig, cr.OUT / "category_examples"))


if __name__ == "__main__":
    main()
