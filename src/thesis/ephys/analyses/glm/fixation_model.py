"""How well does the GLM follow the fixation pattern of V1 units?

Question: during the wait in the center port (the fixation), most units fire
less, and some fire more (`hold_effect.py`, panel a); at the center exit the
rate goes up. Does the model predict this time course in the population?

Comparison, per unit, in the same trials and bins:

- observed: the spike counts;
- model: a closed-loop simulation of the held-out model. Each trial uses the
  fold fit of the GLM cross-validation that held it out (`fit.trial_folds`)
  and is run bin by bin: the rate of a bin comes from the task, flash and
  movement terms of that fit plus its spike-history filter (lags 1 to
  100 ms) applied to the spikes simulated so far in the trial, never the
  recorded ones (`DRAWS` Poisson trains per trial, no past spikes at the
  trial start, 500 ms before the first flash). Mean over draws.

Rate limit: with no limit, the history filter runs away in single trials (a
sample of every 6th unit: 38% of GRB006 and 15% of GRB064 units over 2× their
observed mean rate). So a simulated rate is limited to the unit's own
99.9th-percentile rate in 10-ms bins of single trials (`rate_limit`). With
the limit, every sampled unit stays within 0.8-1.25× its observed mean rate;
over all units 0.15% (GRB006) and 0.39% (GRB064) of bins reach it (printed
at each run).

Until 2026-11-13 the model trace used the recorded spikes as history (a
one-step-ahead prediction that can copy the unit's recent firing).
Until 2026-11-12 the second trace was the counts with the model's flash part
divided out (Anne asked for observed against the model prediction).

- Units: top row, all GLM units with a final fit (172 / 115); then one row
  per fixation group (`hold_effect.fixation_groups`): fixation level below 0
  (suppressed), above 0 (excited) or neither (no effect) by more than 2 fold
  SDs.
- Alignment: the first flash (bins before the center exit only) and the center
  exit (bins after the first flash only), `hold_effect.FLASH_WINDOW_MS` and
  `EXIT_WINDOW_MS`; all bins before the response-port entry.
- PSTH: `BIN_MS` bins, mean over trials in spikes/s. Population: each unit
  divided by its observed mean rate over all valid bins, then mean ± SEM over
  units, each mouse alone. The sampling unit is the unit.

Writes `figures/glm/fixation_model.svg`; no new fit.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.fixation_model
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import (
    EXIT_WINDOW_MS,
    FIXATION_GROUPS,
    FLASH_WINDOW_MS,
    fixation_groups,
)
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
BIN_MS = 10
VERSIONS = {"observed": OBSERVED_COLOR, "simulated model": MODEL_COLOR}
DRAWS = 10  # simulated spike trains per held-out trial
SEED = 20261113
ALIGNMENTS = {
    "flash": (
        "time from the first flash (ms)",
        FLASH_WINDOW_MS,
        CONCEPT_COLORS["visual"],
    ),
    "exit": ("time from the center exit (ms)", EXIT_WINDOW_MS, CONCEPT_COLORS["task"]),
}


def history_filter(prepared: dict) -> np.ndarray:
    """Return the raw history columns at lags 1..100 ms of one spike (lag, column)."""
    relative = prepared["relative_centers"]
    at = int(np.argmin(np.abs(relative)))
    impulse = dmat.build_unit_history(np.array([0.0]), np.array([relative[at]]))
    if np.abs(impulse[: at + 1]).sum() > 0:
        raise ValueError("The history columns are not strictly causal.")
    lags = impulse[at + 1 :]
    return lags[: np.flatnonzero(np.abs(lags).sum(1))[-1] + 1]


def rate_limit(counts: np.ndarray, valid: np.ndarray) -> float:
    """Return the 99.9th-percentile single-trial count of 10-ms bins, per 1-ms bin."""
    n = counts.shape[1] // 10 * 10
    tens = counts[:, :n].reshape(len(counts), -1, 10)
    whole = valid[:, :n].reshape(len(valid), -1, 10).all(-1)
    return float(np.percentile(tens.sum(-1)[whole], 99.9)) / 10


def simulated_rates(
    counts, raw_history, common, metadata, record, partitions, lags, limit, rng
):
    """Return the closed-loop held-out rate (trial, bin) and the limited bins."""
    flat = counts.ravel()
    width = int(metadata["base_columns"]) + int(record["components"])
    out = np.full(counts.shape, np.nan)
    limited = 0
    folds = iter(record["folds"])
    for part in partitions:
        if flat[part["test"]].sum() <= 0:
            continue  # no saved fold for a test set without spikes
        fold = next(folds)
        weights = np.asarray(fold["coefficients"], dtype=float)
        _, mean, scale = dmat.training_zscore(raw_history, part["fit"])
        w_history = weights[width:] / scale
        rows = np.flatnonzero(part["all"])  # whole held-out trials, in order
        base = float(fold["intercept"]) - mean @ w_history
        base = base + np.asarray(common[rows, :width]) @ weights[:width]
        trials = np.unique(rows // counts.shape[1])
        out[trials], n = _run(
            base.reshape(len(trials), -1), (lags @ w_history)[::-1], limit, rng
        )
        limited += n
    if next(folds, None) is not None:
        raise ValueError("Too many saved folds for the reconstructed trial split.")
    return out, limited


def _run(base, kernel, limit, rng):
    """Simulate trials (rows of `base`, the log rate without history)."""
    n_trials, n_bins = base.shape
    n_lags = len(kernel)
    drive = np.repeat(base, DRAWS, axis=0)
    spikes = np.zeros((len(drive), n_lags + n_bins))
    rates = np.empty((len(drive), n_bins))
    limited = 0
    for t in range(n_bins):
        rate = np.exp(drive[:, t] + spikes[:, t : t + n_lags] @ kernel)
        limited += int(np.sum(rate > limit))
        rates[:, t] = np.minimum(rate, limit)
        spikes[:, n_lags + t] = rng.poisson(rates[:, t])
    return rates.reshape(n_trials, DRAWS, n_bins).mean(1), limited / DRAWS


def session_psths(session: str) -> dict:
    """Return the population PSTHs of each version (each unit ÷ its mean rate)."""
    root = OUT / session
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    metadata = json.loads((root / "common_design_me.json").read_text())
    design = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    relative = prepared["relative_centers"]
    n_trials, n_bins = len(prepared["alignments"]), len(relative)
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
    partitions = fit.cross_validation_partitions(n_trials, n_bins, valid.ravel())
    lags = history_filter(prepared)
    rng = np.random.default_rng(SEED)
    limited = 0
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
        "population": {v: {a: [] for a in ALIGNMENTS} for v in VERSIONS},
        "ids": [],
    }
    for record in records:
        unit_id = int(record["unit_id"])
        spikes = np.asarray(table.loc[unit_id, "spike_times_s"], float)
        flat = fit.build_unit_counts(prepared["alignments"], spikes)
        counts = flat.reshape(n_trials, n_bins)
        history = dmat.build_unit_history(prepared["alignments"], spikes)
        model, n = simulated_rates(
            counts,
            history,
            design,
            metadata,
            record,
            partitions,
            lags,
            rate_limit(counts, valid),
            rng,
        )
        limited += n
        versions = {"observed": counts, "simulated model": model}
        mean_rate = counts[valid].mean() / dmat.BINWIDTH_S
        if mean_rate <= 0:
            continue
        out["ids"].append(unit_id)
        for version, values in versions.items():
            for name in ALIGNMENTS:
                curve = psth(values, name)
                out["population"][version][name].append(curve / mean_rate)
    out["population"] = {
        v: {a: np.asarray(c) for a, c in by.items()}
        for v, by in out["population"].items()
    }
    out["centers"] = {a: (e[:-1] + e[1:]) / 2 for a, e in edges.items()}
    out["subject"] = subject
    out["units"] = len(out["population"]["observed"]["flash"])
    out["limited_fraction"] = limited / (len(out["ids"]) * n_trials * n_bins)
    return out


def main() -> None:
    sessions = [session_psths(s) for s in SESSIONS]
    for s in sessions:
        print(s["subject"], f"bins at the rate limit: {s['limited_fraction']:.5f}")
    groups = [fixation_groups(OUT / name) for name in SESSIONS]
    rows = {"all units": [None, None]}
    for group in FIXATION_GROUPS:
        rows[f"{group} units"] = [
            np.array([g.get(u) == group for u in s["ids"]])
            for s, g in zip(sessions, groups, strict=True)
        ]
    with plt.rc_context(FIGURE_STYLE):
        # Rows: all units, then each fixation group. Columns: per mouse, the
        # first flash and the center exit.
        figure, axes = plt.subplots(
            len(rows),
            4,
            figsize=(11.0, 2.5 * len(rows) + 0.4),
            sharex="col",
            sharey=True,
            layout="constrained",
        )
        for row, (group, keeps) in enumerate(rows.items()):
            for m, (session, keep) in enumerate(zip(sessions, keeps, strict=True)):
                for offset, (name, (xlabel, limits, _)) in enumerate(
                    ALIGNMENTS.items()
                ):
                    axis = axes[row, 2 * m + offset]
                    centers = session["centers"][name]
                    for version, color in VERSIONS.items():
                        values = session["population"][version][name]
                        values = values if keep is None else values[keep]
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
                        axis.plot(centers, mean, color=color, lw=1.2)
                    axis.axvline(0, color="0.6", ls="--", lw=0.8)
                    axis.set_xlim(*limits)
                    axis.set_xticks([0, 200, 400])
                    if row == len(rows) - 1:
                        axis.set_xlabel(xlabel)
                    if offset == 0:
                        n = len(values)
                        axis.set_title(
                            f"{session['subject']}, "
                            + (f"all {n} units" if keep is None else f"{n} {group}"),
                            fontsize=8,
                            loc="left",
                        )
            axes[row, 0].set_ylabel("rate ÷ mean rate")
        axes[0, 0].set_ylim(bottom=0)
        # Direct labels at the right end of the first exit panel.
        for y, (label, color) in zip((0.16, 0.04), VERSIONS.items(), strict=True):
            axes[0, 1].text(
                0.97,
                y,
                label,
                color=color,
                ha="right",
                va="bottom",
                fontsize=8,
                transform=axes[0, 1].transAxes,
            )
        label_panels(
            [axes[r, c] for r in range(len(rows)) for c in (0, 2)], x=-0.3, y=1.1
        )
        print(save_figure(figure, OUT / "fixation_model"))


if __name__ == "__main__":
    main()
