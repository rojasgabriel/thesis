"""Does movement change flash suppression more than flash excitation?

Question: after the center exit, is the flash dip of flash-suppressed units
deeper than in the port, while the flash peak of flash-excited units stays the
same? And is a deeper dip only a floor or penalty artifact of low port rates?

Comparison, per unit (paired): the flash kernel of the final refit at a gap of
80 ms or more, after the exit minus in the port (`exit_kernel_units.py`,
`flash_kernels.py`). Measure: window peak (flash-excited units,
`peaks.load_excited_units`) or window dip (flash-suppressed units,
`peaks.load_suppressed_units`), Δ log rate. A negative change of the dip is a
deeper dip.

- a, change after exit for both unit classes. One point per unit, open
  markers: the median of each mouse.
- b, floor check (flash-suppressed units): change of the dip against the rate
  rise at the exit, log((after + 0.5) / (before + 0.5)) with the rates of
  `fixation_groups.py` (100 to 400 ms after against the last 200 ms before the
  exit, all selected trials). A floor artifact predicts a negative slope; the
  intercept is the change at no rate change. Shaded: rates within ±25%.
- c, penalty check (flash-suppressed units): change of the dip against the
  spike count in the port (the same 200 ms before the exit, summed over
  trials). Shrinkage of poorly constrained port dips predicts a positive slope.

Sessions: `flash_kernels.SESSIONS`. The sampling unit is the unit; units are
pooled over mice. Uncertainty: hierarchical bootstrap (mice, then units in each
mouse, `N_BOOT`), 95% percentile intervals; each mouse alone with units
resampled. No display limit: all units are drawn.

Needs `flash_kernels.npz` with `dip` and the saved unit files of `peaks.py`.
Writes `figures/glm/suppression_gain.svg` and
`results/glm/suppression_gain.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.suppression_gain
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm import exit_flash_pairs as pairs
from thesis.ephys.analyses.glm.exit_kernel_units import EXIT, PORT
from thesis.ephys.analyses.glm.fixation_groups import AFTER_S, BEFORE_S
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import load_excited_units, load_suppressed_units
from thesis.plotting import (
    FIGURE_STYLE,
    POINT_ALPHA,
    label_panels,
    median_marker,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
N_BOOT = 2000
RATE_FLOOR_HZ = 0.5
MATCHED = np.log(1.25)
CLASSES = {"excited": ("peak", "#E69F00"), "suppressed": ("dip", "#332288")}


def load() -> dict:
    """Per class: mouse, port and after-exit kernel value, exit rates, port spikes."""
    units = {"excited": load_excited_units(), "suppressed": load_suppressed_units()}
    rows = {kind: [] for kind in CLASSES}
    for session in SESSIONS:
        root = OUT / session
        k = np.load(root / "flash_kernels.npz")
        cond = list(k["conditions"])
        i, j = cond.index(PORT), cond.index(EXIT)
        d = pairs.session(session, set(units["excited"][session]))
        exits = dmat._load_selected_trials(
            dmat._load_windows(root / "stimulus_windows_me.npz")
        )["center_exit_s"].to_numpy(float)

        def count(spikes, w):
            edges = np.c_[exits + w[0], exits + w[1]]
            return np.diff(np.searchsorted(spikes, edges), axis=1).sum()

        spikes = dict(zip(map(int, d["ids"]), d["spikes"], strict=True))
        for kind, (measure, _) in CLASSES.items():
            keep = set(map(int, units[kind][session]))
            for n, u in enumerate(map(int, k["unit_ids"])):
                if u not in keep or u not in spikes:
                    continue
                before = count(spikes[u], BEFORE_S)
                after = count(spikes[u], AFTER_S)
                rows[kind].append(
                    (
                        session.split("_")[0],
                        k[measure][n, i],
                        k[measure][n, j],
                        before / (np.diff(BEFORE_S)[0] * len(exits)),
                        after / (np.diff(AFTER_S)[0] * len(exits)),
                        before,
                    )
                )
    out = {}
    for kind, r in rows.items():
        mouse = np.array([x[0] for x in r])
        port, exit_, rb, ra, cnt = (
            np.array([x[q] for x in r], float) for q in range(1, 6)
        )
        out[kind] = dict(
            mouse=mouse,
            change=exit_ - port,
            rate_rise=np.log((ra + RATE_FLOOR_HZ) / (rb + RATE_FLOOR_HZ)),
            port_spikes=cnt,
        )
    return out


def boot_index(mouse: np.ndarray, rng, hierarchical: bool) -> np.ndarray:
    """Resample mice (if hierarchical), then units within each drawn mouse."""
    names = np.unique(mouse)
    drawn = rng.choice(names, len(names)) if hierarchical else names
    return np.concatenate(
        [rng.choice(np.flatnonzero(mouse == m), (mouse == m).sum()) for m in drawn]
    )


def ci(values) -> list[float]:
    return [round(float(v), 3) for v in np.nanpercentile(values, [2.5, 97.5])]


def line(x, y) -> np.ndarray:
    return np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]


def stats(d: dict, rng, hierarchical: bool) -> dict:
    """Median change; for suppressed units also the floor and penalty fits."""
    mouse, ch = d["mouse"], d["change"]
    lc = np.log10(d["port_spikes"] + 1)
    lc0 = np.median(lc)
    boots = []
    for _ in range(N_BOOT):
        b = boot_index(mouse, rng, hierarchical)
        boots.append(
            [
                np.median(ch[b]),
                *line(d["rate_rise"][b], ch[b]),
                line(lc[b] - lc0, ch[b])[1],
            ]
        )
    boots = np.array(boots)
    matched = np.abs(d["rate_rise"]) < MATCHED
    floor = line(d["rate_rise"], ch)
    return {
        "units": int(len(ch)),
        "median_change": round(float(np.median(ch)), 3),
        "median_change_ci": ci(boots[:, 0]),
        "fraction_more_negative": round(float(np.mean(ch < 0)), 2),
        "floor_slope": round(float(floor[1]), 3),
        "floor_slope_ci": ci(boots[:, 2]),
        "change_at_no_rate_change": round(float(floor[0]), 3),
        "change_at_no_rate_change_ci": ci(boots[:, 1]),
        "rate_matched_units": int(matched.sum()),
        "rate_matched_median_change": round(float(np.median(ch[matched])), 3)
        if matched.any()
        else None,
        "penalty_slope_per_decade": round(float(line(lc - lc0, ch)[1]), 3),
        "penalty_slope_ci": ci(boots[:, 3]),
        "median_port_spikes": int(np.median(d["port_spikes"])),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    data = load()
    report = {"rate_floor_hz": RATE_FLOOR_HZ, "n_boot": N_BOOT}
    for kind, d in data.items():
        report[kind] = {"pooled": stats(d, rng, hierarchical=True)}
        for m in np.unique(d["mouse"]):
            sub = {key: v[d["mouse"] == m] for key, v in d.items()}
            report[kind][str(m)] = stats(sub, rng, hierarchical=False)
    mice = list(dict.fromkeys(s.split("_")[0] for s in SESSIONS))
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(
            1, 3, figsize=(11, 3.8), layout="constrained", width_ratios=(0.8, 1, 1)
        )
        for c, (kind, (measure, color)) in enumerate(CLASSES.items()):
            d = data[kind]
            for m, name in enumerate(mice):
                q = d["mouse"] == name
                x = c + (m - 1) * 0.22 + rng.uniform(-0.06, 0.06, q.sum())
                ax[0].scatter(
                    x,
                    d["change"][q],
                    marker=mouse_marker(name, m),
                    s=10,
                    color=color,
                    alpha=POINT_ALPHA,
                    lw=0,
                )
                median_marker(
                    ax[0],
                    c + (m - 1) * 0.22,
                    np.median(d["change"][q]),
                    mouse_marker(name, m),
                    color,
                )
        ax[0].axhline(0, color="0.6", ls="--", lw=0.8)
        ax[0].set_xticks([0, 1], ["flash-excited\n(peak)", "flash-suppressed\n(dip)"])
        ax[0].set_xlim(-0.6, 1.6)
        ax[0].set_ylabel("after exit − in port (Δ log rate)")
        s = data["suppressed"]
        color = CLASSES["suppressed"][1]
        for panel, key, xlabel in (
            (ax[1], "rate_rise", "rate rise at the exit (log after / before)"),
            (ax[2], "port_spikes", "spikes in the port before exit"),
        ):
            for m, name in enumerate(mice):
                q = s["mouse"] == name
                panel.scatter(
                    s[key][q],
                    s["change"][q],
                    marker=mouse_marker(name, m),
                    s=12,
                    color=color,
                    alpha=POINT_ALPHA,
                    lw=0,
                )
            panel.axhline(0, color="0.6", ls="--", lw=0.8)
            panel.set_xlabel(xlabel)
            panel.set_ylabel("dip after exit − in port (Δ log rate)")
        xx = np.linspace(s["rate_rise"].min(), s["rate_rise"].max(), 50)
        b0, b1 = line(s["rate_rise"], s["change"])
        ax[1].plot(xx, b0 + b1 * xx, color="0.3", lw=1.2)
        ax[1].axvspan(-MATCHED, MATCHED, color="0.92", zorder=-1)
        ax[1].axvline(0, color="0.85", lw=0.6, zorder=0)
        lc = np.log10(s["port_spikes"] + 1)
        b0, b1 = line(lc, s["change"])
        xx = np.logspace(0, lc.max(), 50)
        ax[2].plot(xx, b0 + b1 * np.log10(xx + 1), color="0.3", lw=1.2)
        ax[2].set_xscale("symlog", linthresh=1)
        label_panels(ax, x=-0.2, y=1.02)
        print(save_figure(fig, OUT / "suppression_gain"))
    (OUT / "suppression_gain.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
