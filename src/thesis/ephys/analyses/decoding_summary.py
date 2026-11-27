"""Show the decoder results of all recorded mice in one panel per decoder.

Question: can the V1 population of each mouse tell the stimulus category, the
choice and the stimulus rate? `category_decoding.py` and `rate_decoding.py`
answer it for one session; this module only draws their saved results for
the sessions of `behavior.trials.SESSIONS`. No new analysis.

- Comparison: the held-out score against the score with shuffled labels,
  paired in each mouse.
- Scores: category and choice, accuracy with random cross-validation folds
  (`random_cv.accuracy_mean`); rate, R² with random folds
  (`random_cv_r2_mean`). Units, trials, window, balancing and exclusions are
  those of the two modules.
- Sampling unit: the mouse (one session each). A small open marker is one
  mouse (the shape is the mouse), the large marker is the mean ± SEM of the
  mice; filled is the data, open is the shuffle.

Writes `figures/decoding_summary/decoding_<category|choice|rate>.svg`. Run the
two decoder modules for each session first.

    uv run python -m thesis.ephys.analyses.decoding_summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.behavior.trials import SESSIONS
from thesis.plotting import CONCEPT_COLORS, FIGURE_STYLE, mouse_points, save_figure

OUT = Path("results/decoding_summary")
# name: (module, color concept, y label, chance, getter of (data, shuffle))
PANELS = {
    "category": (
        "category_decoding",
        "visual",
        "category accuracy (held out)",
        0.5,
        lambda r: (
            r["category"]["random_cv"]["accuracy_mean"],
            r["category"]["random_cv"]["shuffle_mean"],
        ),
    ),
    "choice": (
        "category_decoding",
        "task",
        "choice accuracy (held out)",
        0.5,
        lambda r: (
            r["choice"]["random_cv"]["accuracy_mean"],
            r["choice"]["random_cv"]["shuffle_mean"],
        ),
    ),
    "rate": (
        "rate_decoding",
        "visual",
        "stimulus rate R² (held out)",
        0.0,
        lambda r: (
            r["continuous_rate"]["random_cv_r2_mean"],
            r["shuffle"]["random_cv_r2_mean"],
        ),
    ),
}


def main() -> None:
    subjects = [session.split("_")[0] for session in SESSIONS]
    report = {}
    for name, (module, concept, label, chance, get) in PANELS.items():
        values = [
            get(json.loads(Path(f"figures/{module}/{s}/results.json").read_text()))
            for s in SESSIONS
        ]
        report[name] = dict(zip(subjects, values, strict=True))
        color = CONCEPT_COLORS[concept]
        with plt.rc_context(FIGURE_STYLE):
            figure, axis = plt.subplots(figsize=(2.4, 3.0), layout="constrained")
            mouse_points(axis, [0], [[v[0]] for v in values], subjects, color)
            mouse_points(
                axis, [1], [[v[1]] for v in values], subjects, "0.5", hollow=True
            )
            axis.axhline(chance, color="0.6", ls=":", lw=0.8)
            axis.set_xticks([0, 1], ["data", "shuffle"])
            axis.set_xlim(-0.6, 1.6)
            axis.set_ylabel(label)
            print(save_figure(figure, OUT / f"decoding_{name}"))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
