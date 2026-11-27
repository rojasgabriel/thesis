"""When do the stimulus category and the choice appear in the V1 population?

Question: the category and choice decoders (`category_decoding.py`) use one
window, 0 to 1 s after the first flash, which includes the turn to the chosen
side. When in the trial can each be read?

Comparison: the same decoders in a sliding window. Category (fast or slow) is
decoded with equal numbers of left and right choices in each category, and the
choice with equal numbers of fast and slow trains on each side, so neither
decoder can use the other variable.

- Windows: `WIDTH_S` wide, every `STEP_S`, aligned to the first flash
  (`FIRST_FLASH_CENTERS_S`) and to the center exit (`EXIT_CENTERS_S`).
- Features: each unit's rate in the window; by default single units and
  multi-units of unit criteria 1, no stability filter; `--units sua`: the
  stable single units of the GLM (outputs with `_sua`).
- Trials: the decoder trials (completed, non-boundary, with flashes).
- Score: held-out balanced accuracy, `CV_FOLDS` random folds, mean over
  `N_RESAMPLES` balanced resamples. The colored band is ± the SEM across the
  folds of each resample, averaged over resamples (fold scores share training
  trials, so this SEM is a little too small).
- Chance: the same with the label shuffled within each level of the other
  variable, `N_SHUFFLES` times for each decoder; the gray band spans the 2.5
  to 97.5 percentiles of both decoders. The sampling unit is the trial.
- Sessions: `SESSIONS`, one per mouse (the behavior sessions).

The script writes `figures/decoding_time_course/decoding_time_course.svg` and
`results/decoding_time_course/decoding_time_course.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.decoding_time_course
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.behavior.trials import SESSIONS
from thesis.ephys.analyses._decoding import (
    add_units_argument,
    balanced_indices,
    decoder,
    load_decoding_data,
    window_rates,
)
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    mouse_curves,
    save_figure,
)

OUT = Path("results/decoding_time_course")
WIDTH_S = 0.2
STEP_S = 0.05
FIRST_FLASH_CENTERS_S = np.arange(-0.1, 1.0001, STEP_S)
EXIT_CENTERS_S = np.arange(-0.5, 0.5001, STEP_S)
CV_FOLDS = 10
N_RESAMPLES = 50
N_SHUFFLES = 20
SEED = 0
COLORS = {"category": CONCEPT_COLORS["visual"], "choice": CONCEPT_COLORS["task"]}
ACCURACY_LIMITS = (0.25, 1.02)  # the lowest band edge is 0.295


def accuracy(
    rates, target, other, rng, shuffle: bool, n_resamples: int = N_RESAMPLES
) -> np.ndarray:
    """Return the held-out balanced accuracy of each fold, (resample, fold)."""
    scores = []
    for resample in range(n_resamples):
        labels = target.copy()
        if shuffle:
            for level in np.unique(other):
                mask = other == level
                labels[mask] = rng.permutation(labels[mask])
        index = balanced_indices(labels, other, rng)
        folds = StratifiedKFold(
            n_splits=min(CV_FOLDS, int(np.bincount(labels[index]).min())),
            shuffle=True,
            random_state=SEED + resample,
        )
        scores.append(
            cross_val_score(
                decoder(),
                rates[index],
                labels[index],
                cv=folds,
                scoring="balanced_accuracy",
            )
        )
    return np.asarray(scores)


def session_course(session: str, units_kind: str) -> dict:
    subject, stamp = session.split("_", 1)
    units, trials, _ = load_decoding_data(
        subject,
        stamp,
        unit_criteria_id=1,
        stability_param_id=0 if units_kind == "sua" else None,
        window_start_s=0.0,
        window_stop_s=1.0,
        include_mua=units_kind == "all",
    )
    first = trials["stim_pulse_times_s"].str[0].to_numpy(float)
    exit_s = trials["center_exit_s"].to_numpy(float)
    category = trials["stim_category"].eq("high_rate").to_numpy(int)
    choice = trials["response"].to_numpy(int)
    # Choice as 0/1 with the category as the balancing variable (-1/1).
    targets = {
        "category": (category, choice),
        "choice": ((choice == 1).astype(int), np.where(category == 1, 1, -1)),
    }
    rng = np.random.default_rng(SEED)
    out = {"subject": subject, "units": len(units), "trials": len(trials)}
    for name, events, centers in (
        ("first_flash", first, FIRST_FLASH_CENTERS_S),
        ("exit", exit_s, EXIT_CENTERS_S),
    ):
        curves = {}
        for label, (target, other) in targets.items():
            real, spread, null = [], [], []
            for center in centers:
                rates = window_rates(
                    units, events + center - WIDTH_S / 2, events + center + WIDTH_S / 2
                )
                scores = accuracy(rates, target, other, rng, shuffle=False)
                real.append(scores.mean())
                spread.append(
                    float((scores.std(1, ddof=1) / np.sqrt(scores.shape[1])).mean())
                )
                shuffles = [
                    accuracy(
                        rates, target, other, rng, shuffle=True, n_resamples=1
                    ).mean()
                    for _ in range(N_SHUFFLES)
                ]
                null.append(np.percentile(shuffles, [2.5, 50, 97.5]).tolist())
            curves[label] = {
                "accuracy": real,
                "sem_across_folds": spread,
                "shuffle_2.5_50_97.5": null,
            }
        out[name] = {"centers_s": centers.tolist(), **curves}
        print(subject, name, "done", flush=True)
    out["median_exit_after_first_flash_s"] = float(np.median(exit_s - first))
    return out


def draw_courses(axis, parts: list[dict]) -> None:
    """Draw the accuracy of each decoder over time for all mice on one axis.

    A thin line for each mouse and the mean ± SEM of the mice
    (`plotting.mouse_curves`); the gray band holds the shuffle intervals of
    all mice and decoders.
    """
    centers = np.asarray(parts[0]["centers_s"])
    nulls = np.stack(
        [
            np.asarray(part[label]["shuffle_2.5_50_97.5"])
            for part in parts
            for label in COLORS
        ]
    )
    axis.fill_between(
        centers, nulls[:, :, 0].min(0), nulls[:, :, 2].max(0), color="0.88", linewidth=0
    )
    for label, color in COLORS.items():
        mouse_curves(axis, centers, [part[label]["accuracy"] for part in parts], color)
    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
    axis.axvline(0, color="0.4", linewidth=0.6)
    axis.set_ylim(*ACCURACY_LIMITS)


def plot(results: list[dict], name: str) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
        for axis, (alignment, xlabel) in zip(
            axes,
            (
                ("first_flash", "time from the first flash (s)"),
                ("exit", "time from the center exit (s)"),
            ),
            strict=True,
        ):
            draw_courses(axis, [data[alignment] for data in results])
            axis.set_xlabel(xlabel)
        exits = [data["median_exit_after_first_flash_s"] for data in results]
        for exit_s in exits:
            axes[0].axvline(
                exit_s, color=CONCEPT_COLORS["task"], linestyle="--", linewidth=0.8
            )
        axes[0].text(
            max(exits) + 0.02,
            0.26,  # below the lowest band edge (0.295)
            "median exit of each mouse",
            color=CONCEPT_COLORS["task"],
            fontsize=7,
        )
        axes[0].set_ylabel("held-out accuracy")
        axes[0].text(
            0,
            1.02,
            f"{len(results)} mice (thin: each mouse; thick: mean ± SEM of the mice)",
            transform=axes[0].transAxes,
            va="bottom",
            fontsize=7.5,
        )
        for label, y in (("category", 0.96), ("choice", 0.86)):
            axes[1].text(
                0.02,
                y,
                label,
                color=COLORS[label],
                transform=axes[1].transAxes,
                va="top",
            )
        axes[1].text(
            0.98,
            0.005,
            "shuffle",
            color="0.5",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
        )
        label_panels(axes, x=-0.2, y=1.03)
        figure.tight_layout()
        return save_figure(figure, OUT / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_units_argument(parser)
    units_kind = parser.parse_args().units
    name = "decoding_time_course" + ("_sua" if units_kind == "sua" else "")
    results = [session_course(s, units_kind) for s in SESSIONS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(results, indent=1))
    print(plot(results, name))


if __name__ == "__main__":
    main()
