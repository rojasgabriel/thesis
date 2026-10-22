"""Find the flash response that the V1 GLM misses after the center exit.

Question: the model predicts a smaller increase of the flash peak after the
exit than the one observed (result 3 of `docs/glm.md`). Which flashes does it
miss, and is the missing part a response to the flash or an exit transient
that is there with no flash?

Comparison: no new fit. For each flash-excited unit, the observed peak against
the peak of spikes drawn from its held-out model rate (the "model + noise"
peak). The observed peak is the maximum of a noisy trial mean, so it is
larger than the peak of the smooth model rate even when the model is correct;
spikes drawn from the model (`DRAWS` Poisson draws of the held-out rate of
the same bins, `SEED`) have the same bias. The miss is observed minus model
+ noise.

- Trials, alignment and bins: the completed trials of the GLM, on its native
  1-ms grid; a window must lie fully inside the valid interval (center entry
  to response entry). Held-out rates are those of `flash_movement.py` (the
  saved trial folds, observed spike history).
- Events: flashes with a preceding gap above `dmat.short_gap_s()` (the
  80+ ms gap class). Panel a: the flashes in the port, and the 1st, 2nd and
  later flashes after the exit. Panels b and c: "flash", the first flash after
  the exit when it comes `LATENCY_S` after the exit; "no flash", a pseudo
  flash in completed trials with no flash in the first `NO_FLASH_S` after the
  exit, placed at the median latency of the "flash" events after the exit,
  and more than the short gap after the last flash.
- Peak: 10-ms bins, the mean from -40 to 0 ms before the event is the
  baseline, the peak is the largest bin from 30 to 120 ms (as in
  `flash_movement.py`). Each unit's values are divided by its mean observed
  rate over all valid bins.
- Units: the flash-excited units of `raw_video_trust_classification.json`
  (the units of `flash_movement.py`). The sampling unit is the unit; the
  comparisons are paired over units, within each mouse (two-sided Wilcoxon
  signed-rank test), and pooled with the hierarchical bootstrap.
- Panels: a, the miss for each kind of flash; b, the mean rate around the
  event, observed and model, for "flash" and "no flash"; c, the miss for
  "flash" and "no flash". All values enter the statistics; no axis limit
  hides a point.
- Limits: "no flash" trials are few, and the pseudo flash is later in the
  stimulus train. Two mice.

The script writes `first_flash_after_exit.svg` and
`first_flash_after_exit.json` in `figures/glm/`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.first_flash_after_exit
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
from scipy.stats import sem, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, flash_movement
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    cross_validation_partitions,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import LAGS_MS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    pooled_strip,
    save_figure,
)

OUTPUT = flash_movement.OUTPUT
DRAWS = 50
SEED = 3
LATENCY_S = (0.02, 0.12)
NO_FLASH_S = 0.25
RANKS = ("in port", "1st after exit", "2nd after exit", "3rd+ after exit")
CONTROLS = ("flash", "no flash")
MISS_COLOR = "0.25"
# Distance between strip columns, so that the mouse labels clear the next column.
SPACING = 1.8
CENTERS_S = (LAGS_MS.reshape(-1, 10).mean(-1) + 0.5) / 1000


def binned(rate: np.ndarray) -> np.ndarray:
    """Return the mean over events of (event, 1-ms bin) rates, in 10-ms bins."""
    return rate.mean(0).reshape(-1, 10).mean(-1)


def peak(rate: np.ndarray) -> float:
    """Return the peak of (event, 1-ms bin) rates over their own baseline."""
    trace = binned(rate)
    baseline = trace[(CENTERS_S >= -0.04) & (CENTERS_S < 0)].mean()
    return float(trace[(CENTERS_S >= 0.03) & (CENTERS_S < 0.12)].max() - baseline)


def events(trials, prepared) -> tuple[dict, dict]:
    """Return the window bins of each kind of event, (event, bin), and their counts."""
    n_bins = len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared).reshape(-1, n_bins)
    short = dmat.short_gap_s()
    found = {name: [] for name in (*RANKS, *CONTROLS)}
    quiet = []  # (row, exit, last flash before exit) of the no-flash trials

    def add(name, row, time):
        start = time - prepared["alignments"][row] - prepared["relative_centers"][0]
        columns = int(np.rint(start / dmat.BINWIDTH_S)) + LAGS_MS
        if columns[0] >= 0 and columns[-1] < n_bins and valid[row, columns].all():
            found[name].append(row * n_bins + columns)

    for row, (_, trial) in enumerate(trials.iterrows()):
        times = np.asarray(trial["stim_pulse_times_s"], dtype=float)
        leave = trial["center_exit_s"]
        gaps = np.diff(times, prepend=-np.inf)
        rank = np.cumsum(times >= leave)
        for time, gap, after in zip(times, gaps, rank, strict=True):
            if gap <= short:
                continue
            add(RANKS[min(after, 3)] if time >= leave else RANKS[0], row, time)
            latency = time - leave
            if after == 1 and time >= leave and LATENCY_S[0] <= latency < LATENCY_S[1]:
                add("flash", row, time)
        if not np.any((times >= leave) & (times < leave + NO_FLASH_S)):
            before = times[times < leave]
            quiet.append((row, leave, before[-1] if len(before) else -np.inf))
    latencies = []
    for row, (_, trial) in enumerate(trials.iterrows()):
        times = np.asarray(trial["stim_pulse_times_s"], dtype=float)
        after = times[times >= trial["center_exit_s"]]
        if len(after):
            latency = after[0] - trial["center_exit_s"]
            if LATENCY_S[0] <= latency < LATENCY_S[1]:
                latencies.append(latency)
    delay = float(np.median(latencies))
    for row, leave, last in quiet:
        if leave + delay - last > short:
            add("no flash", row, leave + delay)
    windows = {name: np.asarray(value) for name, value in found.items()}
    return windows, {
        "events": {name: len(value) for name, value in windows.items()},
        "pseudo_flash_after_exit_s": delay,
    }


def session_values(session: str, excited: list[int]) -> dict:
    """Return the observed and model + noise peaks and traces of the excited units."""
    root = OUTPUT / session
    metadata = json.loads((root / "common_design_me.json").read_text())
    records = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            root / "all_fit_me/checkpoints", "unit_*_folds.json"
        )
    }
    subject, stamp = session.split("_", 1)
    units = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    windows, counts_of = events(trials, prepared)
    common = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    valid = dmat._valid_bin_mask(prepared)
    partitions = cross_validation_partitions(
        len(trials), len(prepared["relative_centers"]), valid
    )
    requested = np.zeros(len(valid), dtype=bool)
    requested[np.concatenate([w.ravel() for w in windows.values()])] = True
    rng = np.random.default_rng(SEED)
    ids = sorted(set(excited) & set(records))
    observed, model, traces = [], [], []
    for unit_id in ids:
        spikes = units.loc[unit_id, "spike_times_s"]
        counts = build_unit_counts(prepared["alignments"], spikes)
        history = dmat.build_unit_history(prepared["alignments"], spikes)
        predicted = flash_movement.held_out_rates(
            counts, history, common, metadata, records[unit_id], partitions, requested
        )[0]
        scale = (counts * valid).sum() / (valid.sum() * dmat.BINWIDTH_S)
        to_rate = 1 / (dmat.BINWIDTH_S * scale)
        observed.append([peak(counts[w] * to_rate) for w in windows.values()])
        model.append(
            [
                np.mean(
                    [peak(rng.poisson(predicted[w]) * to_rate) for _ in range(DRAWS)]
                )
                for w in windows.values()
            ]
        )
        traces.append(
            [
                [
                    binned(counts[windows[name]] * to_rate),
                    binned(predicted[windows[name]] * to_rate),
                ]
                for name in CONTROLS
            ]
        )
        print(f"{session}: {len(observed)}/{len(ids)} units", flush=True)
    return {
        "subject": subject,
        "unit_ids": ids,
        "names": list(windows),
        "observed": np.asarray(observed),  # (unit, event kind)
        "model": np.asarray(model),
        "traces": np.asarray(traces),  # (unit, control, observed or model, bin)
        **counts_of,
    }


def summary(groups: list[np.ndarray], subjects: list[str]) -> dict:
    """Return the mean, median and Wilcoxon p of each mouse, and the pooled median."""
    return {
        **{
            subject: {
                "units": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "wilcoxon_p": float(wilcoxon(values).pvalue),
            }
            for subject, values in zip(subjects, groups, strict=True)
        },
        "pooled": hierarchical_interval(groups),
    }


def main() -> None:
    classes = json.loads((OUTPUT / "raw_video_trust_classification.json").read_text())
    sessions = [session_values(session, classes[session]) for session in SESSIONS]
    subjects = [data["subject"] for data in sessions]
    names = sessions[0]["names"]
    miss = [data["observed"] - data["model"] for data in sessions]
    stats = {
        "events": {d["subject"]: d["events"] for d in sessions},
        "pseudo_flash_after_exit_s": {
            d["subject"]: d["pseudo_flash_after_exit_s"] for d in sessions
        },
        "draws": DRAWS,
    }
    for index, name in enumerate(names):
        stats[name] = {
            "observed": summary([d["observed"][:, index] for d in sessions], subjects),
            "model + noise": summary(
                [d["model"][:, index] for d in sessions], subjects
            ),
            "miss": summary([m[:, index] for m in miss], subjects),
        }
    flash, quiet = names.index("flash"), names.index("no flash")
    stats["miss, flash minus no flash"] = summary(
        [m[:, flash] - m[:, quiet] for m in miss], subjects
    )
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 2.8), layout="constrained")
        rank_axis, flash_axis, quiet_axis, control_axis = figure.subplots(
            1, 4, width_ratios=(1.5, 1, 1, 0.9)
        )
        for position, name in enumerate(RANKS):
            pooled_strip(
                rank_axis,
                SPACING * position,
                [m[:, names.index(name)] for m in miss],
                subjects,
                MISS_COLOR,
            )
        rank_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        rank_axis.set_xticks(
            SPACING * np.arange(len(RANKS)), ["in port", "1st", "2nd", "3rd+"]
        )
        rank_axis.set_xlim(-0.5, SPACING * (len(RANKS) - 1) + 1.1)
        rank_axis.set_xlabel("long-gap flash (rank after exit)")
        rank_axis.set_ylabel("observed minus model + noise\npeak (/ mean rate)")
        times = 1000 * CENTERS_S
        pooled = np.concatenate([d["traces"] for d in sessions])
        for axis, index, title in (
            (flash_axis, 0, "first flash after exit"),
            (quiet_axis, 1, "no flash (same time)"),
        ):
            for source, color, label in (
                (0, OBSERVED_COLOR, "observed"),
                (1, MODEL_COLOR, "model"),
            ):
                values = pooled[:, index, source]
                mean, error = values.mean(0), sem(values, axis=0)
                axis.fill_between(
                    times, mean - error, mean + error, color=color, alpha=0.15, lw=0
                )
                axis.plot(times, mean, color=color, lw=1.2)
                if index == 1:  # one key, in the empty top of the no-flash axis
                    axis.text(
                        0.97,
                        0.97 - 0.1 * source,
                        label,
                        color=color,
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                    )
            axis.axvline(0, color=REFERENCE_COLOR, lw=0.6, ls="--")
            axis.set_xlabel(
                "time from flash (ms)" if index == 0 else "time from pseudo flash (ms)"
            )
            axis.set_title(title, fontsize=7, loc="left")
        flash_axis.set_ylabel("rate (/ mean rate)")
        quiet_axis.sharey(flash_axis)
        for position, index in enumerate((flash, quiet)):
            pooled_strip(
                control_axis,
                SPACING * position,
                [m[:, index] for m in miss],
                subjects,
                MISS_COLOR,
            )
        control_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        control_axis.set_xticks([0, SPACING], ["flash", "no flash"])
        control_axis.set_xlim(-0.5, SPACING + 1.1)
        control_axis.set_ylabel("observed minus model + noise\npeak (/ mean rate)")
        # Name the mice only at the last column, where the labels have room.
        for axis in (rank_axis, control_axis):
            names_drawn = [text for text in axis.texts if text.get_text() in subjects]
            for text in names_drawn[: -len(subjects)]:
                text.remove()
        label_panels([rank_axis, flash_axis, control_axis], x=-0.3, y=1.08)
        print(save_figure(figure, OUTPUT / "first_flash_after_exit"))
    (OUTPUT / "first_flash_after_exit.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
