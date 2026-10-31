"""Do V1 neurons follow the fixation without their flash responses?

Question: during the wait in the center port (the fixation), most units fire
less, and some fire more (`hold_effect.py`, panel a). Flashes come during the
whole wait, so the fixation pattern could be built from flash responses. Is it
still there when the flash responses are taken out?

Comparison, per unit, in the same trials and bins:

- observed: the spike counts;
- without the flash part: the counts divided by exp(flash part), the flash
  part being all flash columns of the design (`FLASH_TERMS`, the five flash
  terms) times the fitted weights. Each trial uses the fold model of the GLM
  cross-validation that held that trial out (`fit.trial_folds`).

The fixation kernel, the other task terms, the motion energy and the spike
history stay in. A pattern that is there in both versions does not come from
the modelled flash responses. Limits: the fixation kernel starts at the first
flash, so the first ~100 ms after it hold both; a flash response that the
model misses stays in (the first flash after the exit is the main one,
`first_flash_after_exit.py`).

- Units: all GLM units with a final fit (172 / 79); examples: the units of
  `hold_effect.py` panel a (the lowest and the highest fixation level), read
  from `hold_effect_stats.json`.
- Alignment: the first flash (bins before the center exit only) and the center
  exit (bins after the first flash only), `hold_effect.FLASH_WINDOW_MS` and
  `EXIT_WINDOW_MS`; all bins before the response-port entry.
- PSTH: `BIN_MS` bins, mean over trials in spikes/s. Population: each unit
  divided by its observed mean rate over all valid bins, then mean ± SEM over
  units, each mouse alone.

Writes `figures/glm/fixation_without_flash.svg`; no new fit.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.fixation_without_flash
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm.decoding_without_terms import FLASH_TERMS
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import EXIT_WINDOW_MS, FLASH_WINDOW_MS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
BIN_MS = 20
VERSIONS = {"observed": OBSERVED_COLOR, "without the flash part": MODEL_COLOR}
ALIGNMENTS = {
    "flash": (
        "time from the first flash (ms)",
        FLASH_WINDOW_MS,
        CONCEPT_COLORS["visual"],
    ),
    "exit": ("time from the center exit (ms)", EXIT_WINDOW_MS, CONCEPT_COLORS["task"]),
}


def session_psths(session: str, examples: list[int]) -> dict:
    """Return the PSTHs of each version: examples (spikes/s) and all units (normalized)."""
    root = OUT / session
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    metadata = json.loads((root / "common_design_me.json").read_text())
    design = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    flash = [
        i for i, n in enumerate(metadata["column_names"]) if n.startswith(FLASH_TERMS)
    ]
    # Raw flash columns: a raw zero gives no contribution (`results.raw_contribution`).
    offset = (
        np.asarray(metadata["training_mean"])[flash]
        / np.asarray(metadata["training_scale"])[flash]
    )
    x = (np.asarray(design[:, flash]) + offset).astype(np.float32)
    relative = prepared["relative_centers"]
    n_trials, n_bins = len(prepared["alignments"]), len(relative)
    x = x.reshape(n_trials, n_bins, len(flash))
    valid = dmat._valid_bin_mask(prepared).reshape(n_trials, n_bins)
    leave = trials["center_exit_s"].to_numpy(float) - prepared["alignments"]
    times_ms = {
        "flash": np.broadcast_to(1000 * relative, (n_trials, n_bins)),
        "exit": 1000 * (relative[None, :] - leave[:, None]),
    }
    used = {
        "flash": valid & (relative[None, :] < leave[:, None]),
        "exit": valid & (relative[None, :] >= 0),
    }
    edges = {
        name: np.arange(limits[0], limits[1] + 1e-9, BIN_MS)
        for name, (_, limits, _) in ALIGNMENTS.items()
    }
    index = {name: np.digitize(times_ms[name], edges[name]) - 1 for name in ALIGNMENTS}
    folds = fit.trial_folds(n_trials, fit.CV_FOLDS, fit.CV_SEED)
    records = fit.load_unit_records(
        root / "all_fit_me" / "checkpoints", "unit_*_folds.json"
    )
    subject, stamp = session.split("_", 1)
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )

    def psth(values, name):
        """Mean over trials of each bin, spikes/s (every used 1-ms bin counts once)."""
        keep = used[name] & (index[name] >= 0) & (index[name] < len(edges[name]) - 1)
        sums = np.bincount(index[name][keep], values[keep], len(edges[name]) - 1)
        exposure = np.bincount(index[name][keep], None, len(edges[name]) - 1)
        return sums / np.maximum(exposure, 1) / dmat.BINWIDTH_S

    out = {
        "examples": {},
        "population": {v: {a: [] for a in ALIGNMENTS} for v in VERSIONS},
    }
    for record in records:
        unit_id = int(record["unit_id"])
        counts = fit.build_unit_counts(
            prepared["alignments"], table.loc[unit_id, "spike_times_s"]
        ).reshape(n_trials, n_bins)
        coefficients = np.asarray([f["coefficients"] for f in record["folds"]])[
            :, flash
        ]
        part = np.empty(counts.shape, dtype=np.float32)
        for fold in range(fit.CV_FOLDS):
            held = folds == fold
            part[held] = x[held] @ coefficients[fold].astype(np.float32)
        versions = {
            "observed": counts,
            "without the flash part": counts * np.exp(-part),
        }
        mean_rate = counts[valid].mean() / dmat.BINWIDTH_S
        if mean_rate <= 0:
            continue
        for version, values in versions.items():
            for name in ALIGNMENTS:
                curve = psth(values, name)
                out["population"][version][name].append(curve / mean_rate)
                if unit_id in examples:
                    out["examples"].setdefault(unit_id, {}).setdefault(version, {})[
                        name
                    ] = curve
    out["population"] = {
        v: {a: np.asarray(c) for a, c in by.items()}
        for v, by in out["population"].items()
    }
    out["centers"] = {a: (e[:-1] + e[1:]) / 2 for a, e in edges.items()}
    out["subject"] = subject
    out["units"] = len(out["population"]["observed"]["flash"])
    return out


def main() -> None:
    stats = json.loads((OUT / "hold_effect_stats.json").read_text())
    sessions = [
        session_psths(s, stats[s]["example_units_suppressed_and_excited"])
        for s in SESSIONS
    ]
    rows = [
        (s, unit, label)
        for name, s in zip(SESSIONS, sessions, strict=True)
        for unit, label in zip(
            stats[name]["example_units_suppressed_and_excited"],
            ("lower rate in fixation", "higher rate in fixation"),
            strict=True,
        )
    ]
    with plt.rc_context(FIGURE_STYLE):
        n_rows = len(rows) + len(sessions)
        figure, axes = plt.subplots(
            n_rows, 2, figsize=(6.4, 1.45 * n_rows), sharex="col", layout="constrained"
        )
        panels = [(s, unit, label) for s, unit, label in rows] + [
            (s, None, f"all {s['units']} units") for s in sessions
        ]
        for row, (session, unit, label) in enumerate(panels):
            for column, (name, (xlabel, limits, event)) in enumerate(
                ALIGNMENTS.items()
            ):
                axis = axes[row, column]
                centers = session["centers"][name]
                for version, color in VERSIONS.items():
                    if unit is None:
                        values = session["population"][version][name]
                        mean = values.mean(0)
                        sem = values.std(0, ddof=1) / np.sqrt(len(values))
                        axis.fill_between(
                            centers,
                            mean - sem,
                            mean + sem,
                            color=color,
                            alpha=0.2,
                            lw=0,
                        )
                    else:
                        mean = session["examples"][unit][version][name]
                    axis.plot(centers, mean, color=color, lw=1.1)
                axis.axvline(0, color=event, lw=0.8)
                axis.set_xlim(*limits)
                axis.set_ylim(bottom=0)
                if column == 0:
                    axis.set_ylabel("rate / mean" if unit is None else "spikes/s")
                    title = f"{session['subject']}, " + (
                        f"unit {unit}, {label}" if unit is not None else label
                    )
                    axis.set_title(title, fontsize=7, loc="left")
                if row == n_rows - 1:
                    axis.set_xlabel(xlabel)
        for y, (version, color) in zip((0.95, 0.8), VERSIONS.items(), strict=True):
            axes[0, 0].text(
                0.98,
                y,
                version,
                color=color,
                ha="right",
                va="top",
                fontsize=7,
                transform=axes[0, 0].transAxes,
            )
        # The population rows of the mice share one scale in each column.
        for column in range(2):
            top = max(axes[r, column].get_ylim()[1] for r in range(len(rows), n_rows))
            for r in range(len(rows), n_rows):
                axes[r, column].set_ylim(0, top)
        label_panels(axes[:, 0], x=-0.2, y=1.12)
        print(save_figure(figure, OUT / "fixation_without_flash"))


if __name__ == "__main__":
    main()
