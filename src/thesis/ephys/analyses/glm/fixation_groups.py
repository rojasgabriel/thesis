"""Do units with a different fixation effect differ in their movement effects?

Question: V1 units fire less (most), more, or the same during the fixation.
Do these groups differ in what happens when the mouse moves: the flash
response after the center exit, the rate change at the exit, and how much the
movement terms of the GLM explain? Exploratory; no new fit.

- Fixation groups (`hold_effect.fixation_groups`): the fixation kernel level
  below 0 by more than 2 fold SDs (suppressed), above 0 by more than 2 fold SDs
  (excited), else no effect. All GLM units with a final fit.
- a, flash increase after the exit (flash-excited units only,
  `peaks.load_excited_units`): window peak of the first after-exit flash minus
  that of the last in-port flash, each trial paired (`exit_flash_pairs.peaks`),
  ÷ the unit's mean rate.
- b, rate change at the exit (all units): mean rate from 100 to 400 ms after
  the center exit minus the last 200 ms before it (the windows of
  `hold_effect.py`), all selected GLM trials, ÷ the unit's mean rate (0.5 s
  before to 1.5 s after the first flash, as `exit_flash_pairs.py`).
- c, d, unique gain of the motion energy (10 video PCs) and of the turn
  direction (all units): held-out bits per spike lost when that term is
  shuffled (`unique_dev.py`).

Sampling unit: the unit. Each dot is a unit; the mouse is the marker shape;
big open markers: the median of each mouse and group. c and d use a
symmetric log axis, linear within ±0.01 bits/spike. Statistics, each mouse
alone: Kruskal-Wallis over the three groups (groups with fewer than 3 units
are left out of the test).

Writes `figures/glm/fixation_groups.svg` and `results/glm/fixation_groups.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.fixation_groups
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import kruskal

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm import exit_flash_pairs as pairs
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import FIXATION_GROUPS, fixation_groups
from thesis.ephys.analyses.glm.peaks import load_excited_units
from thesis.ephys.analyses.glm.unique_dev import UNIQUE_BITS, add_bits_per_spike
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
BEFORE_S, AFTER_S = (-0.2, 0.0), (0.1, 0.4)
MEASURES = {
    "flash": "flash peak increase after exit\n(÷ mean rate)",
    "exit": "rate change at the exit\n(÷ mean rate)",
    "video": "unique gain, motion energy\n(bits/spike)",
    "turn_direction": "unique gain, turn direction\n(bits/spike)",
}
SHORT = ("suppressed", "no effect", "excited")


def session_values(session: str) -> dict:
    """Return each measure per unit id, and the fixation group of each unit."""
    root = OUT / session
    groups = fixation_groups(root)
    d = pairs.session(session, set(load_excited_units()[session]))
    p = pairs.peaks(d)
    values = {
        "flash": {
            int(d["ids"][u]): float(b - a)
            for u, a, b in zip(d["exc"], p["in port"], p["after exit"], strict=True)
        }
    }
    trials = dmat._load_selected_trials(
        dmat._load_windows(root / "stimulus_windows_me.npz")
    )
    exits = trials["center_exit_s"].to_numpy(float)

    def rate(spikes, window):
        edges = np.c_[exits + window[0], exits + window[1]]
        n = np.diff(np.searchsorted(spikes, edges), axis=1)
        return n.mean() / (window[1] - window[0])

    values["exit"] = {
        int(i): (rate(s, AFTER_S) - rate(s, BEFORE_S)) / max(m, 1e-9)
        for i, s, m in zip(d["ids"], d["spikes"], d["mean_rate"], strict=True)
        if int(i) in groups
    }
    checkpoints = cli.PoissonGLM(root=root).fit_dir("all") / "checkpoints"
    unique = add_bits_per_spike(
        load_unit_records(checkpoints / "unique", "unit_*.json"), checkpoints
    )
    for term in ("video", "turn_direction"):
        values[term] = {
            int(item["unit_id"]): float(item["groups"][term][UNIQUE_BITS])
            for item in unique
        }
    return {"subject": session.split("_")[0], "groups": groups, "values": values}


def main() -> None:
    data = [session_values(s) for s in SESSIONS]
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout="constrained")
        rng = np.random.default_rng(0)
        for axis, (key, ylabel) in zip(axes, MEASURES.items(), strict=True):
            for m, d in enumerate(data):
                marker = mouse_marker(d["subject"], m)
                by_group = []
                for g, name in enumerate(FIXATION_GROUPS):
                    v = np.array(
                        [
                            x
                            for u, x in d["values"][key].items()
                            if d["groups"].get(u) == name
                        ]
                    )
                    by_group.append(v)
                    x = g + (m - 0.5) * 0.36
                    axis.scatter(
                        x + rng.uniform(-0.1, 0.1, len(v)),
                        v,
                        s=9,
                        marker=marker,
                        color=OBSERVED_COLOR,
                        alpha=0.35,
                        lw=0,
                    )
                    if len(v):
                        axis.plot(
                            x,
                            np.median(v),
                            marker,
                            ms=10,
                            color="white",
                            mec=OBSERVED_COLOR,
                            mew=1.6,
                            zorder=5,
                        )
                tested = [v for v in by_group if len(v) >= 3]
                report.setdefault(key, {})[d["subject"]] = {
                    "units": {n: len(v) for n, v in zip(SHORT, by_group, strict=True)},
                    "median": {
                        n: float(np.median(v)) if len(v) else None
                        for n, v in zip(SHORT, by_group, strict=True)
                    },
                    "kruskal_p": float(kruskal(*tested).pvalue)
                    if len(tested) >= 2
                    else None,
                }
            axis.axhline(0, color="0.6", ls="--", lw=0.8, zorder=0)
            if key in ("video", "turn_direction"):
                # Most gains are near 0 and a few are large: linear within ±0.01.
                axis.set_yscale("symlog", linthresh=0.01)
            axis.set_xticks(range(3), SHORT)
            axis.set_xlabel("fixation effect")
            axis.set_ylabel(ylabel)
        label_panels(axes, x=-0.25, y=1.04)
        print(save_figure(fig, OUT / "fixation_groups"))
    (OUT / "fixation_groups.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
