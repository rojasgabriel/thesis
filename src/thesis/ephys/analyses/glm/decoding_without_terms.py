"""Are the category and choice signals around the center exit visual?

Questions (`--target`):

- category: the stimulus category (fast or slow train) is clearest just
  after the mouse leaves the center port (`decoding_time_course.py`). The
  mouse moves then, and the flashes go on. Does the category signal come from
  the flash responses, or from the other signals that the GLM gives to task
  and movement?
- choice: the choice is decoded almost perfectly after the exit. Is that a
  choice signal, or the flash responses changed by the turn of the head,
  which moves the LED panel to another part of the visual field?

Comparison: a decoder of the target on three versions of the same rates, in
the same trials and resamples, so the versions are paired:

- observed: the spike counts;
- without the flash terms: the counts divided by exp(flash part), the flash
  part being all flash columns of the design (`FLASH_TERMS`) times the fitted
  weights;
- without the other terms: the counts divided by exp(the part of every other
  column of the common design: the fixation kernel (`hold`) and level,
  response period, turn direction, previous side, session drift and
  motion-energy PCs).

Spike history is in neither part, so it stays in all versions. A signal that
is visual should fall when the flash part is divided out and stay when the
other part is divided out.

A fourth version, observed with single units and multi-units (the default of
the decoders, `_decoding._units_with_mua`), shows how much of the decoder's
signal the GLM units carry; the multi-units have no GLM fit, so the
model parts exist only for the GLM units.

Limits:

- Spike history stays, so a flash response that the model gives to the
  history kernel stays in the version without the flash terms.
- The model misses most of the response to the first flash after the exit
  (`first_flash_after_exit.py`); that part stays in the after-exit window.
  So the AUC without the flash terms is an upper limit of a category signal
  that is not visual, not evidence for one.
- A weak observed signal gives a small change: GRB006 in the port is only
  just above chance (0.61 against 0.42–0.57 on 2026-10-29).
- Choice: the flash terms are the same for both sides (the after-exit flash
  kernel has no side), so a flash response that changes with the side of the
  turn is not in the flash part; it stays in the version without the flash
  terms. A choice signal that survives does not exclude the visual
  hypothesis; only a large fall supports it. The other part holds the turn
  term, which is the side of the response after the exit, so the choice
  signal must fall without the other terms; that version is only a check.
- The division by exp(part) does not let a few trials dominate: on
  2026-10-29 the top 1% of unit-trials held the same share of the spikes as
  in the observed version (6–11%).

- Sessions and units: the GLM sessions (`flash_kernels.SESSIONS`) and the GLM
  units with a final fit (unit criteria 1, stability 0, exclusions of
  `fit.py`): 172 and 79. These are not the decoder defaults
  (single units and multi-units, and another GRB064 session).
- Trials: the selected GLM trials that are not at the category boundary.
  Each resample takes equal numbers of the four category × choice groups, so
  the category decoder cannot use the choice, and the choice decoder cannot
  use the category.
- Periods: `WINDOW_S` before the center exit (in the port) and `WINDOW_S`
  after it; the rate of each unit in each window, spikes/s. All bins are
  before the response-port entry.
- Model parts: each trial uses the fold model of the GLM cross-validation
  that held that trial out (`fit.trial_folds`), so the model that changes a
  trial did not see it. The GLM gets no category label; it sees the flash
  times. It sees the side of the response after the exit (the turn term). The decoder training trials of one fold are changed by GLM models
  that saw its test trials.
- Decoder: z-score and logistic regression (`_decoding.decoder`),
  `CV_FOLDS` stratified folds, the AUC of the held-out output, mean over
  `N_RESAMPLES` balanced resamples. Chance: the target shuffled within each
  level of the other variable, `N_SHUFFLES` times; each shuffle gives the mean AUC over
  `NULL_RESAMPLES` resamples, so the gray band (2.5–97.5 percentiles) is the
  chance range of a mean AUC, as the points are, not of one decoder run.
- Statistic: each version minus observed, per resample: mean and 2.5–97.5
  percentiles. The interval shows how much the score depends on which trials
  balance the groups; it is not a bootstrap over trials. The sampling unit is
  the trial; each mouse is one session.

The script writes `figures/glm/<target>_without_terms.svg` and
`results/glm/<target>_without_terms.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.decoding_without_terms --target choice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses._decoding import (
    _units_with_mua,
    balanced_indices,
    decoder,
    window_rates,
)
from thesis.ephys.analyses.glm import dmat, fit
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
TARGETS = ("category", "choice")
WINDOW_S = 0.3
CV_FOLDS = 10
N_RESAMPLES = 50
N_SHUFFLES = 50
# Resamples of each shuffle: the chance band is that of a mean AUC, as the
# points are. Fewer than N_RESAMPLES; the resamples share most trials.
NULL_RESAMPLES = 10
SEED = 0
FLASH_TERMS = (
    "visual_flash",
    "flash_just_before",
    "flash_after_exit",
    "flash_recovery",
)
VERSIONS = ("observed", "without the flash terms", "without the other terms")
ALL_UNITS = "observed, single and multi-units"
COLORS = {
    "observed": OBSERVED_COLOR,
    "without the flash terms": CONCEPT_COLORS["visual"],
    "without the other terms": CONCEPT_COLORS["task"],
    ALL_UNITS: OBSERVED_COLOR,  # observed too; an open marker tells it apart
}
PERIODS = ("in port", "after exit")
ALL_VERSIONS = (*VERSIONS, ALL_UNITS)
AUC_LIMITS = (0.1, 1.02)


def session_rates(session: str) -> dict:
    """Return the window rates of each version, (period, version) -> (trial, unit)."""
    root = OUT / session
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    metadata = json.loads((root / "common_design_me.json").read_text())
    design = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    names = metadata["column_names"]
    flash = [i for i, n in enumerate(names) if n.startswith(FLASH_TERMS)]
    other = [i for i in range(len(names)) if i not in flash]
    centers = prepared["relative_centers"]
    n_trials, n_bins = len(prepared["alignments"]), len(centers)
    exit_s = trials["center_exit_s"].to_numpy(float) - prepared["alignments"]
    # The 1-ms bins of each window, (trial, bin of the window).
    width = int(round(WINDOW_S / dmat.BINWIDTH_S))
    first_after = np.searchsorted(centers, exit_s)
    if np.any(first_after - width < 0) or np.any(first_after + width > n_bins):
        raise ValueError(f"A window is outside the trial grid in {session}.")
    valid = dmat._valid_bin_mask(prepared).reshape(n_trials, n_bins)
    windows = {
        "in port": first_after[:, None] + np.arange(-width, 0),
        "after exit": first_after[:, None] + np.arange(width),
    }
    for period, bins in windows.items():
        if not np.take_along_axis(valid, bins, 1).all():
            raise ValueError(f"The {period} window has bins after the response.")
    rows = {p: (np.arange(n_trials)[:, None] * n_bins + b) for p, b in windows.items()}
    parts = {"flash": flash, "other": other}
    designs = {
        (p, k): np.asarray(design[r.ravel()][:, cols]).reshape(*r.shape, len(cols))
        for p, r in rows.items()
        for k, cols in parts.items()
    }
    scaling = {
        k: {
            "training_mean": np.asarray(metadata["training_mean"])[cols],
            "training_scale": np.asarray(metadata["training_scale"])[cols],
        }
        for k, cols in parts.items()
    }
    folds = fit.trial_folds(n_trials, fit.CV_FOLDS, fit.CV_SEED)
    checkpoints = root / "all_fit_me" / "checkpoints"
    records = fit.load_unit_records(checkpoints, "unit_*_folds.json")
    subject, stamp = session.split("_", 1)
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False)
    table = table.set_index("unit_id")
    rates = {(p, v): [] for p in PERIODS for v in VERSIONS}
    for record in records:
        counts = fit.build_unit_counts(
            prepared["alignments"], table.loc[record["unit_id"], "spike_times_s"]
        ).reshape(n_trials, n_bins)
        if len(record["folds"]) != fit.CV_FOLDS:
            raise ValueError(f"Unit {record['unit_id']} has not all folds.")
        coefficients = np.asarray([f["coefficients"] for f in record["folds"]])
        for period, bins in windows.items():
            window = np.take_along_axis(counts, bins, 1)
            log_parts = {}
            for k, cols in parts.items():
                part = np.empty(window.shape)
                for fold in range(fit.CV_FOLDS):
                    held = folds == fold
                    x = designs[(period, k)][held]
                    part[held] = raw_contribution(
                        x.reshape(-1, len(cols)),
                        coefficients[fold][cols],
                        scaling[k],
                        np.arange(len(cols)),
                    ).reshape(x.shape[:2])
                log_parts[k] = part
            for version, values in zip(
                VERSIONS,
                (
                    window,
                    window * np.exp(-log_parts["flash"]),
                    window * np.exp(-log_parts["other"]),
                ),
                strict=True,
            ):
                rates[(period, version)].append(values.sum(1) / WINDOW_S)
    keep = trials["stim_category"].ne("boundary").to_numpy()
    rates = {k: np.asarray(v).T[keep] for k, v in rates.items()}
    every_unit = _units_with_mua(subject, stamp, 1)
    exits = trials["center_exit_s"].to_numpy(float)[keep]
    rates[("in port", ALL_UNITS)] = window_rates(every_unit, exits - WINDOW_S, exits)
    rates[("after exit", ALL_UNITS)] = window_rates(every_unit, exits, exits + WINDOW_S)
    return {
        "subject": subject,
        "units": len(records),
        "units_with_multi_units": len(every_unit),
        "rates": rates,
        "category": trials["stim_category"].eq("high_rate").to_numpy(int)[keep],
        "choice": trials["response"].to_numpy(int)[keep],
    }


def target_labels(data: dict, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the labels (0/1) of the target and the other variable (-1/1)."""
    if target == "category":
        return data["category"], data["choice"]
    return (data["choice"] == 1).astype(int), np.where(data["category"] == 1, 1, -1)


def scores(data: dict, labels, other, rng, n_resamples: int) -> np.ndarray:
    """Return the AUC of `labels` per resample, (resample, period, version)."""
    out = np.empty((n_resamples, len(PERIODS), len(ALL_VERSIONS)))
    for resample in range(n_resamples):
        index = balanced_indices(labels, other, rng)
        y = labels[index]
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + resample)
        for p, period in enumerate(PERIODS):
            for v, version in enumerate(ALL_VERSIONS):
                x = data["rates"][(period, version)][index]
                output = np.empty(len(index))
                for train, test in folds.split(x, y):
                    model = decoder().fit(x[train], y[train])
                    output[test] = model.decision_function(x[test])
                out[resample, p, v] = roc_auc_score(y, output)
    return out


def shuffled(labels, other, rng) -> np.ndarray:
    """Return the labels shuffled within each level of the other variable."""
    labels = labels.copy()
    for level in (-1, 1):
        mask = other == level
        labels[mask] = rng.permutation(labels[mask])
    return labels


def summary(real: np.ndarray, null: np.ndarray, data: dict, target: str) -> dict:
    report = {
        "subject": data["subject"],
        "units": data["units"],
        "units_with_multi_units": data["units_with_multi_units"],
        "trials_per_resample": int(
            len(
                balanced_indices(*target_labels(data, target), np.random.default_rng(0))
            )
        ),
    }
    for p, period in enumerate(PERIODS):
        report[period] = {
            version: {
                "auc_mean": float(real[:, p, v].mean()),
                "auc_95": np.percentile(real[:, p, v], [2.5, 97.5]).tolist(),
                "shuffle_auc_95": np.percentile(null[:, p, v], [2.5, 97.5]).tolist(),
                **(
                    {}
                    if v == 0
                    else {
                        "minus_observed_mean": float(
                            (real[:, p, v] - real[:, p, 0]).mean()
                        ),
                        "minus_observed_95": np.percentile(
                            real[:, p, v] - real[:, p, 0], [2.5, 97.5]
                        ).tolist(),
                    }
                ),
            }
            for v, version in enumerate(ALL_VERSIONS)
        }
    return report


def plot(reports: list[dict], target: str) -> Path:
    offsets = (np.arange(len(ALL_VERSIONS)) - 1.5) * 0.2
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(reports), figsize=(3.4 * len(reports), 3.2), sharey=True
        )
        for index, (axis, report) in enumerate(zip(axes, reports, strict=True)):
            for p, period in enumerate(PERIODS):
                for v, version in enumerate(ALL_VERSIONS):
                    values = report[period][version]
                    x = p + offsets[v]
                    low, high = values["shuffle_auc_95"]
                    axis.fill_between(
                        [x - 0.08, x + 0.08], low, high, color="0.85", linewidth=0
                    )
                    mean = values["auc_mean"]
                    lo, hi = values["auc_95"]
                    axis.errorbar(
                        x,
                        mean,
                        yerr=[[mean - lo], [hi - mean]],
                        fmt=mouse_marker(report["subject"], index),
                        color=COLORS[version],
                        mfc="white" if version == ALL_UNITS else COLORS[version],
                        capsize=3,
                        ms=6,
                    )
            axis.axhline(0.5, color="0.6", linestyle=":", linewidth=0.8)
            axis.set_xticks(range(len(PERIODS)), PERIODS)
            axis.set_xlim(-0.5, len(PERIODS) - 0.5)
            axis.set_ylim(*AUC_LIMITS)
            # Above the plot: an AUC near 1 reaches the top of the axes.
            axis.text(
                0.02,
                1.02,
                f"{report['subject']}: {report['units']} neurons"
                f" ({report['units_with_multi_units']} with multi-units)",
                color=OBSERVED_COLOR,
                transform=axis.transAxes,
                va="bottom",
            )
            axis.text(
                0.98,
                0.01,
                "shuffle",
                color="0.5",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
            )
        # Below the lowest shuffle band edge of the in-port group (0.33).
        for v, version in enumerate(ALL_VERSIONS):
            axes[0].text(
                0.02,
                0.20 - 0.055 * v,
                version,
                color=COLORS[version],
                transform=axes[0].transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
            )
        axes[0].set_ylabel(f"{target} AUC (held out)")
        label_panels(axes, x=-0.18, y=1.03)
        figure.tight_layout()
        return save_figure(figure, OUT / f"{target}_without_terms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=TARGETS, default="category")
    target = parser.parse_args().target
    reports = []
    for session in SESSIONS:
        data = session_rates(session)
        labels, other = target_labels(data, target)
        rng = np.random.default_rng(SEED)
        real = scores(data, labels, other, rng, N_RESAMPLES)
        null = np.asarray(
            [
                scores(
                    data, shuffled(labels, other, rng), other, rng, NULL_RESAMPLES
                ).mean(0)
                for _ in range(N_SHUFFLES)
            ]
        )
        reports.append(summary(real, null, data, target))
        print(data["subject"], "done", flush=True)
    (OUT / f"{target}_without_terms.json").write_text(json.dumps(reports, indent=1))
    print(json.dumps(reports, indent=1))
    print(plot(reports, target))


if __name__ == "__main__":
    main()
