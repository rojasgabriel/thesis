"""Is there a flash response during the poke, against the poke with no flash?

Question (Izquierdo lab meeting, 2026-11-23: "poke no flash versus poke with
flash"): the mouse is in the center port 10 to 100 ms before the first flash
(the pre-stimulus delay). If V1 responds to the flash, the rate after the
first flash must rise above the rate in that delay, with the head already in
the same place. The task has no catch trials, so this delay is the only
"poke, no flash" period in the data.

Comparison, no model, paired over units in each mouse:

- poke, no flash: the rate of each unit from the center entry to the first
  flash, pooled over trials (spikes over time);
- flash: the window peak (`peaks.window_peak`, the mean within 20 ms of the
  largest 10-ms bin, 30 to 120 ms after the first flash) of the mean rate
  after the first flash.

- Trials: the selected GLM trials with a pre-stimulus delay of at least
  `MIN_DELAY_S` (center entry to first flash, from the trial table).
- Alignment: the first flash. In the figure, the bins before the center entry
  of a trial are left out of that trial (not set to 0), so the trace before
  0 is the poke period only.
- Units: the flash-excited units (`peaks.load_excited_units`) in the
  figure; the report also gives all GLM units, which were not chosen by
  their response to the first flash. Values are
  divided by the mean rate of the unit (`exit_flash_pairs.mean_rate`). The
  sampling unit is the unit; Wilcoxon signed-rank of flash minus poke in each
  mouse.
- Limit: the delay is short and varies. The flash-excited units were chosen
  by their response to the first flash, so the report also has a split-half
  test (`split_half`): units chosen on even trials, tested on odd trials,
  and the other way.
- Mice: GRB006 and GRB064 by default, or the mice given as arguments.

Writes `figures/glm/poke_before_flash.svg` and
`results/glm/poke_before_flash.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.poke_before_flash
"""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.exit_flash_pairs import OUT, mean_rate
from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import load_excited_units, window_peak
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    POINT_ALPHA,
    REFERENCE_COLOR,
    label_panels,
    median_marker,
    mouse_marker,
    save_figure,
)

MIN_DELAY_S = 0.04
EDGES_MS = np.arange(-100, 151, 10)
CENTERS_MS = (EDGES_MS[:-1] + EDGES_MS[1:]) / 2
POKE = CONCEPT_COLORS["task"]
FLASH = CONCEPT_COLORS["visual"]


def session(name: str, excited: set[int]) -> dict:
    subject, stamp = name.split("_", 1)
    prepared = dmat._load_windows(OUT / name / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    entry = trials["center_entry_s"].to_numpy(float)
    first = np.array([float(t[0]) for t in trials["stim_pulse_times_s"]])
    keep = first - entry >= MIN_DELAY_S
    units = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False)
    units = units.iloc[analysis_unit_indices(units, "all")].reset_index(drop=True)
    edges = first[keep, None] + EDGES_MS[None, :] / 1000
    # A bin counts in a trial only when it starts after the center entry.
    inside = edges[:, :-1] >= entry[keep, None]
    out = {"subject": subject, "trials": int(keep.sum()), "delay_ms": [], "units": []}
    out["delay_ms"] = (1000 * (first - entry)[keep]).tolist()
    for spikes, unit in zip(units["spike_times_s"], units["unit_id"], strict=True):
        spikes = np.sort(np.asarray(spikes, float))
        scale = max(mean_rate(spikes, first), 1e-9)
        poke = np.sum(
            np.searchsorted(spikes, first[keep]) - np.searchsorted(spikes, entry[keep])
        ) / np.sum((first - entry)[keep])
        counts = np.diff(np.searchsorted(spikes, edges), axis=1).astype(float)
        counts[~inside] = np.nan
        trace = np.nanmean(counts, axis=0) / 0.01
        peak = window_peak(np.nan_to_num(trace), CENTERS_MS)
        # Split halves (even and odd trials): poke rate and flash peak of each.
        halves = []
        for h in (0, 1):
            rows = np.arange(keep.sum()) % 2 == h
            ph = np.sum(
                np.searchsorted(spikes, first[keep][rows])
                - np.searchsorted(spikes, entry[keep][rows])
            ) / np.sum((first - entry)[keep][rows])
            th = np.nanmean(counts[rows], axis=0) / 0.01
            halves.append(
                [ph / scale, float(window_peak(np.nan_to_num(th), CENTERS_MS)) / scale]
            )
        out["units"].append(
            {
                "halves": halves,
                "excited": int(unit) in excited,
                "poke": poke / scale,
                "flash": float(peak) / scale,
                "trace": (trace / scale).tolist(),
            }
        )
    return out


def split_half(d: dict) -> dict:
    """Choose units on one half of the trials, test them on the other half.

    The units with the largest flash peak minus poke rate in one half, as many
    as the flash-excited units, are tested in the other half (Wilcoxon), both
    ways. This avoids choosing and testing on the same trials.
    """
    n = sum(u["excited"] for u in d["units"])
    h = np.array([u["halves"] for u in d["units"]])  # (unit, half, poke/peak)
    out = {}
    for pick, test in ((0, 1), (1, 0)):
        chosen = np.argsort(h[:, pick, 1] - h[:, pick, 0])[-n:]
        diff = h[chosen, test, 1] - h[chosen, test, 0]
        out[f"chosen_on_half_{pick}"] = {
            "units": int(n),
            "units_higher_after_flash": int(np.sum(diff > 0)),
            "median_difference": float(np.median(diff)),
            "p": float(wilcoxon(diff).pvalue),
        }
    return out


def plot(data: list[dict], report: dict):
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(data) + 1, figsize=(4.0 * (len(data) + 1), 3.6), layout="constrained"
        )
        for j, d in enumerate(data):
            a = axes[j]
            traces = np.array([u["trace"] for u in d["units"] if u["excited"]])
            m = np.nanmean(traces, 0)
            se = np.nanstd(traces, 0, ddof=1) / np.sqrt(len(traces))
            before = CENTERS_MS < 0
            for part, color in ((before, POKE), (~before, FLASH)):
                a.plot(CENTERS_MS[part], m[part], color=color)
                a.fill_between(
                    CENTERS_MS[part],
                    (m - se)[part],
                    (m + se)[part],
                    color=color,
                    alpha=0.2,
                    lw=0,
                )
            a.axvline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            a.set_xlabel("time from the first flash (ms)")
            a.text(
                0,
                1.02,
                f"{d['subject']}: {len(traces)} units, {d['trials']} trials",
                transform=a.transAxes,
            )
            if j == 0:
                a.set_ylabel("rate ÷ mean rate")
                a.text(
                    -95,
                    m[before].max() * 1.15,
                    "poke, no flash",
                    color=POKE,
                    fontsize=8,
                )
                k = int(np.nanargmax(m))
                a.text(
                    CENTERS_MS[k] + 8,
                    m[k],
                    "after the flash",
                    color=FLASH,
                    fontsize=8,
                    va="center",
                )
        for a in axes[1:-1]:
            a.sharey(axes[0])  # the same scale for every mouse
        axes[0].autoscale()
        s = axes[-1]
        lim = 0
        for j, d in enumerate(data):
            x = np.array([u["poke"] for u in d["units"] if u["excited"]])
            y = np.array([u["flash"] for u in d["units"] if u["excited"]])
            xa = np.array([u["poke"] for u in d["units"]])
            ya = np.array([u["flash"] for u in d["units"]])
            mk = mouse_marker(d["subject"], j)
            s.scatter(x, y, marker=mk, color=FLASH, alpha=POINT_ALPHA, s=20, lw=0)
            median_marker(s, np.median(x), np.median(y), mk, FLASH)
            lim = max(lim, x.max(), y.max())
            w = wilcoxon(y - x)
            report[d["subject"]] = {
                "units": int(len(x)),
                "trials": d["trials"],
                "delay_ms_median": float(np.median(d["delay_ms"])),
                "median_poke": float(np.median(x)),
                "median_flash_peak": float(np.median(y)),
                "units_higher_after_flash": int(np.sum(y > x)),
                "p": float(w.pvalue),
                "split_half": split_half(d),
                "all_units": {
                    "units": int(len(xa)),
                    "units_higher_after_flash": int(np.sum(ya > xa)),
                    "median_poke": float(np.median(xa)),
                    "median_flash_peak": float(np.median(ya)),
                    "p": float(wilcoxon(ya - xa).pvalue),
                },
            }
        lim *= 1.05
        s.plot([0, lim], [0, lim], color=REFERENCE_COLOR, ls="--", lw=0.8)
        s.set(xlim=(0, lim), ylim=(0, lim), aspect="equal")
        s.set_xlabel("rate during the poke, no flash\n(÷ mean rate)")
        s.set_ylabel("peak after the first flash (÷ mean rate)")
        label_panels(axes, x=-0.15, y=1.08)
        return save_figure(figure, OUT / "poke_before_flash")


def main() -> None:
    mice = sys.argv[1:] or ["GRB006", "GRB064"]
    excited = load_excited_units()
    data = [session(s, set(excited[s])) for s in SESSIONS if s.split("_")[0] in mice]
    report = {}
    print(plot(data, report))
    (OUT / "poke_before_flash.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
