"""Compare the recorded and the GLM-predicted change of the flash response after the exit.

Question: the flash kernel of flash-suppressed units has a deeper dip after
the center exit (`suppression_gain.py`). Is the recorded response deeper too,
does the model predict the recorded change, and which model term holds it?

Comparison: for each tracked unit, the response to a flash after the exit
minus the response to a flash in the port, in the same trials, recorded and
predicted. The comparison is paired by trial and by unit, and it is split by
the side of the turn (the response side of the trial).

- Units: StimulusResponse class "suppressed" with stability criterion 0
  (`peaks.excited_units`) and saved fold fits, in the fitted days of one
  mouse. A unit is one UnitMatch track (`<mouse>_ids.csv`, column `track`;
  tracks with two units in one session are excluded). The trials of all the
  days of a unit are put together. A unit needs 20 or more paired trials of
  each turn side.
- Flashes: the selected trials of the GLM. In each trial, the last flash
  whose window (-40 to 120 ms) is in the port and the first flash whose window
  starts at or after the center exit and ends before the response entry. Both
  flashes are 75 ms or more after the flash before. Only trials with both
  flashes, and only windows where every bin is valid.
- Response (alignment: flash onset): mean rate 30 to 120 ms minus the mean
  rate -40 to 0 ms of the same flash (baseline), divided by the mean rate of
  the unit on that day. A negative change means a deeper dip after the exit.
- Predicted: the held-out rate of the full model from the saved fold weights
  of that day (as `flash_movement.held_out_rates`), through the same measure.
- Without a term: the same held-out rate with the raw contribution of one
  term removed (`results.raw_contribution`; the other terms and the intercept
  stay). There is no refit, and the model is exponential, so the effects of
  the terms do not add.
- Summary: median over tracked units; 95% interval from a bootstrap over
  tracked units (the sampling unit). One mouse per run.
- Days: is the result the same on each day? For each day, the mean over units
  with a bootstrap that resamples the paired trials inside each turn side
  (one draw for all units, because the units of a day share the trials) and
  the units; Cochran's Q across days on these standard errors.

Figure: (a) the change of each tracked unit after a left and after a right
turn, recorded and predicted, with the medians; (b) the predicted change after
a left turn with each term removed, against the full model and the data.

Writes `results/glm/observed_vs_predicted/<mouse>_<session>.npz` (values of
each unit and trial; computed once, needs the database),
`results/glm/observed_vs_predicted_<mouse>.json` and
`figures/glm/observed_vs_predicted_<mouse>.svg`. Each argument is the fit root
of one day (the directory with `common_design_me.json`):

    uv run python -m thesis.ephys.analyses.glm.observed_vs_predicted GRB064 <root> ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.exit_flash_responses import BASELINE, PEAK
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    cross_validation_partitions,
    load_unit_records,
)
from thesis.ephys.analyses.glm.kernel_retest import TERMS as KERNEL_TERMS
from thesis.ephys.analyses.glm.peaks import LAGS_MS, excited_units
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    BOOTSTRAPS,
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    POINT_ALPHA,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

MATCH = Path("results/unit_match")
OUT = Path("results/glm")
GAP_MIN_S = 0.075
MIN_TRIALS_PER_SIDE = 20
SIDES = {"left": -1, "right": 1}
LABELS = {name: label for name, (label, _) in KERNEL_TERMS.items()} | {
    "history": "spike history",
    "video_pc": "video components",
    dmat.DRIFT: "session drift",
}


def paired_windows(prepared, trials, valid) -> tuple[np.ndarray, np.ndarray]:
    """Return the design rows (trial, period, lag) of the paired flashes and the side."""
    n_bins = len(prepared["relative_centers"])
    mask = valid.reshape(len(trials), -1)
    low, high = LAGS_MS[0] / 1000, (LAGS_MS[-1] + 1) / 1000

    def rows(trial, time):
        start = time - prepared["alignments"][trial] - prepared["relative_centers"][0]
        columns = int(np.rint(start / dmat.BINWIDTH_S)) + LAGS_MS
        if columns[0] < 0 or columns[-1] >= n_bins or not mask[trial, columns].all():
            return None
        return trial * n_bins + columns

    windows, side = [], []
    for index, (_, trial) in enumerate(trials.iterrows()):
        flashes = np.asarray(trial["stim_pulse_times_s"], float)
        entry, leave, response = (
            trial[key]
            for key in ("center_entry_s", "center_exit_s", "response_port_entry_s")
        )
        if (
            not len(flashes)
            or not np.isfinite([entry, leave, response]).all()
            or trial["response"] not in (-1, 1)
        ):
            continue
        ok = np.diff(flashes, prepend=-np.inf) >= GAP_MIN_S
        inside = np.flatnonzero(
            ok & (flashes + low >= entry) & (flashes + high <= leave)
        )
        outside = np.flatnonzero(
            ok & (flashes + low >= leave) & (flashes + high <= response)
        )
        if not len(inside) or not len(outside):
            continue
        pair = rows(index, flashes[inside[-1]]), rows(index, flashes[outside[0]])
        if pair[0] is None or pair[1] is None:
            continue
        windows.append(pair)
        side.append(int(trial["response"]))
    return np.asarray(windows), np.asarray(side)


def term_columns(metadata: dict) -> dict[str, list[int]]:
    """Return the design columns of each term.

    >>> term_columns({"column_names": ["hold_basis_01", "hold_basis_02", "fixation"]})
    {'hold': [0, 1], 'fixation': [2]}
    """
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(metadata["column_names"]):
        groups.setdefault(re.sub(r"_?(basis)?_?\d+$", "", name), []).append(index)
    return groups


def held_out_rates(counts, raw_history, common, metadata, record, partitions, rows):
    """Return the held-out rate of the full model and without each term."""
    width = int(metadata["base_columns"]) + int(record["components"])
    groups = term_columns(metadata)
    out = {name: np.full(len(counts), np.nan) for name in ("full", "history", *groups)}
    folds = iter(record["folds"])
    for part in partitions:
        spikes = counts[part["test"]].sum()
        if spikes <= 0:
            continue
        fold = next(folds)
        if not np.isclose(spikes, fold["test"]["observed_spikes"]):
            raise ValueError("Saved fold and observed test spikes differ.")
        weights = np.asarray(fold["coefficients"], dtype=float)
        if len(weights) != width + dmat.HISTORY_COLUMNS:
            raise ValueError("Saved fold and design dimensions differ.")
        test = part["test"] & rows
        if not test.any():
            continue
        history, mean, scale = dmat.training_zscore(raw_history, part["fit"])
        design = np.asarray(common[test, :width])
        log_rate = (
            float(fold["intercept"])
            + design @ weights[:width]
            + history[test] @ weights[width:]
        )
        out["full"][test] = np.exp(log_rate)
        out["history"][test] = np.exp(
            log_rate - (history[test] + mean / scale) @ weights[width:]
        )
        for name, columns in groups.items():
            columns = [c for c in columns if c < width]
            out[name][test] = np.exp(
                log_rate - raw_contribution(design, weights, metadata, columns)
            )
    return out


def response(rate: np.ndarray) -> np.ndarray:
    """Return the response of each flash from its (flash, 1-ms lag) rates."""
    bins = rate.reshape(*rate.shape[:-1], -1, 10).mean(-1)
    return bins[..., PEAK].mean(-1) - bins[..., BASELINE].mean(-1)


def session_values(subject: str, session: str, root: Path) -> dict:
    """Return the change after the exit of each unit and paired trial of one day."""
    cache = OUT / "observed_vs_predicted" / f"{subject}_{session}.npz"
    if cache.exists():
        return dict(np.load(cache))
    metadata = json.loads((root / "common_design_me.json").read_text())
    records = {
        int(r["unit_id"]): r
        for r in load_unit_records(root / "all_fit_me/checkpoints", "unit_*_folds.json")
    }
    units = sorted(
        u for u in excited_units(subject, session, "suppressed") if u in records
    )
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    common = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    valid = dmat._valid_bin_mask(prepared)
    partitions = cross_validation_partitions(
        len(trials), len(prepared["relative_centers"]), valid
    )
    windows, side = paired_windows(prepared, trials, valid)
    requested = np.zeros(len(valid), bool)
    requested[windows.ravel()] = True
    table = fetch_unit_table(subject, session, 1, 0, include_metrics=False)
    table = table.set_index("unit_id")
    observed, predicted, terms = [], [], None
    for unit in units:
        spikes = table.loc[unit, "spike_times_s"]
        counts = build_unit_counts(prepared["alignments"], spikes)
        history = dmat.build_unit_history(prepared["alignments"], spikes)
        rates = held_out_rates(
            counts, history, common, metadata, records[unit], partitions, requested
        )
        terms = list(rates)
        # Counts per bin over the mean count per valid bin = rate ÷ mean rate.
        scale = valid.sum() / (counts * valid).sum()

        def change(rate):
            port, after = response(rate[windows] * scale).T
            return after - port

        observed.append(change(counts))
        predicted.append([change(rates[name]) for name in terms])
    values = {
        "unit_ids": np.asarray(units),
        "side": side,
        "terms": np.asarray(terms),
        "observed": np.asarray(observed),  # (unit, trial)
        "predicted": np.asarray(predicted),  # (unit, term, trial)
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **values)
    return values


def by_track(days: dict[str, dict], ids: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Return one row per tracked unit: the change by side, recorded and predicted."""
    pooled: dict[int, dict] = {}
    for session, values in days.items():
        tracks = ids[ids["session"] == session].set_index("unit_id")["track"]
        for row, unit in enumerate(values["unit_ids"]):
            if unit not in tracks.index:
                continue
            entry = pooled.setdefault(
                int(tracks[unit]), {"side": [], "obs": [], "pred": []}
            )
            entry["side"].append(values["side"])
            entry["obs"].append(values["observed"][row])
            entry["pred"].append(values["predicted"][row])
    terms = [str(t) for t in next(iter(days.values()))["terms"]]
    rows = []
    for track, entry in pooled.items():
        side = np.concatenate(entry["side"])
        if min((side == s).sum() for s in SIDES.values()) < MIN_TRIALS_PER_SIDE:
            continue
        observed = np.concatenate(entry["obs"])
        predicted = np.concatenate(entry["pred"], axis=1)
        row = {"track": track, "days": len(entry["side"]), "trials": len(side)}
        for name, value in SIDES.items():
            row[f"observed {name}"] = observed[side == value].mean()
            for term, values in zip(terms, predicted):
                row[f"{term} {name}"] = values[side == value].mean()
        rows.append(row)
    return pd.DataFrame(rows), terms


def interval(values, rng) -> dict:
    """Return the median and its 95% bootstrap interval over tracked units."""
    values = np.asarray(values)
    draws = np.median(
        values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))], 1
    )
    return {
        "median": float(np.median(values)),
        "interval_95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
    }


def cochran_q(values, errors) -> float:
    """Return the p-value of the test that all days have the same value.

    >>> round(cochran_q([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 2)
    1.0
    """
    values, weights = np.asarray(values), 1 / np.asarray(errors) ** 2
    mean = (weights * values).sum() / weights.sum()
    return float(chi2.sf((weights * (values - mean) ** 2).sum(), len(values) - 1))


def day_test(days: dict[str, dict], rng) -> dict:
    """Test, for the change after a left turn, that the days are the same."""
    full = {s: list(v["terms"]).index("full") for s, v in days.items()}
    stats: dict[str, list] = {
        "observed": [],
        "predicted": [],
        "observed − predicted": [],
    }
    errors = {key: [] for key in stats}
    for session, values in days.items():
        left = np.flatnonzero(values["side"] == SIDES["left"])
        observed = values["observed"]
        predicted = values["predicted"][:, full[session]]
        sources = {
            "observed": observed,
            "predicted": predicted,
            "observed − predicted": observed - predicted,
        }
        draws = {key: [] for key in stats}
        for _ in range(BOOTSTRAPS):
            trials = rng.choice(left, len(left))
            units = rng.integers(0, len(observed), len(observed))
            for key, source in sources.items():
                draws[key].append(source[np.ix_(units, trials)].mean())
        for key, source in sources.items():
            stats[key].append(float(source[:, left].mean()))
            errors[key].append(float(np.std(draws[key], ddof=1)))
    return {
        key: {
            "mean_of_each_day": dict(zip(days, stats[key])),
            "standard_error_of_each_day": dict(zip(days, errors[key])),
            "p_equal_days": cochran_q(stats[key], errors[key]),
        }
        for key in stats
    }


def plot(mouse: str, table: pd.DataFrame, terms: list, report: dict) -> Path:
    rng = np.random.default_rng(0)
    with plt.rc_context(FIGURE_STYLE):
        fig, (ax, ax_terms) = plt.subplots(
            1, 2, figsize=(7.2, 3.4), layout="constrained", width_ratios=(1, 1.2)
        )
        sources = (
            ("observed", OBSERVED_COLOR, "recorded"),
            ("full", MODEL_COLOR, "model"),
        )
        for x, side in enumerate(SIDES):
            for offset, (source, color, label) in zip((-0.18, 0.18), sources):
                values = table[f"{source} {side}"]
                jitter = rng.uniform(-0.1, 0.1, len(values))
                ax.plot(
                    x + offset + jitter,
                    values,
                    ".",
                    color=color,
                    alpha=POINT_ALPHA,
                    ms=4,
                )
                entry = report["change"][side][
                    "observed" if source == "observed" else "predicted"
                ]
                ax.errorbar(
                    x + offset,
                    entry["median"],
                    yerr=np.abs(np.subtract(entry["interval_95"], entry["median"]))[
                        :, None
                    ],
                    fmt="_",
                    color=color,
                    ms=14,
                    mew=2,
                    capsize=0,
                    zorder=3,
                )
                if x == 0:
                    ax.text(
                        x + offset,
                        1.02,
                        label,
                        color=color,
                        ha="center",
                        transform=ax.get_xaxis_transform(),
                    )
        ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--")
        ax.set_xticks(range(len(SIDES)), [f"{side} turn" for side in SIDES])
        limit = float(
            np.percentile(np.abs(table[[f"observed {s}" for s in SIDES]]), 97)
        )
        ax.set_ylim(-limit, limit)
        hidden = int(
            (
                table[
                    [f"{src} {s}" for src in ("observed", "full") for s in SIDES]
                ].abs()
                > limit
            )
            .to_numpy()
            .sum()
        )
        ax.set_ylabel(
            "flash response after exit − in port\n(÷ mean rate; below 0 = deeper dip)"
        )
        ax.set_xlabel(
            f"{len(table)} tracked units, {report['days']} days, 1 mouse"
            + (f"; {hidden} points past the axis" if hidden else ""),
            fontsize=6,
        )
        removed = sorted(
            (t for t in terms if t != "full"),
            key=lambda t: report["left_turn_without_term"][t]["median"],
        )
        for row, term in enumerate(removed):
            entry = report["left_turn_without_term"][term]
            ax_terms.plot(entry["interval_95"], [row, row], color=MODEL_COLOR, lw=1.2)
            ax_terms.plot(entry["median"], row, "o", color=MODEL_COLOR, ms=4)
        ax_terms.set_yticks(range(len(removed)), [LABELS.get(t, t) for t in removed])
        left = report["change"]["left"]
        for key, color, label in (
            ("predicted", MODEL_COLOR, "full model"),
            ("observed", OBSERVED_COLOR, "recorded"),
        ):
            ax_terms.axvline(left[key]["median"], color=color, lw=1, ls=":")
            ax_terms.text(
                left[key]["median"],
                1.01,
                label,
                color=color,
                ha="center",
                va="bottom",
                transform=ax_terms.get_xaxis_transform(),
            )
        ax_terms.set_xlabel(
            "predicted change after a left turn,\nmodel without one term"
        )
        label_panels([ax, ax_terms], x=-0.3, y=1.08)
        return save_figure(fig, OUT / f"observed_vs_predicted_{mouse}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mouse")
    parser.add_argument("roots", nargs="+", type=Path, help="fit root of each day")
    args = parser.parse_args()
    ids = pd.read_csv(MATCH / f"{args.mouse}_ids.csv", dtype={"session": str})
    ids = ids[
        ~ids.groupby("track")["session"].transform(lambda s: s.duplicated().any())
    ]
    days = {}
    for root in args.roots:
        session = re.search(rf"{args.mouse}_(\d+_\d+)", str(root)).group(1)
        days[session] = session_values(args.mouse, session, root)
    table, terms = by_track(days, ids)
    rng = np.random.default_rng(0)
    report = {
        "mouse": args.mouse,
        "days": len(days),
        "tracked_units": len(table),
        "paired_trials": {s: int(len(v["side"])) for s, v in days.items()},
        "change": {},
    }
    columns = {
        "left": lambda source: table[f"{source} left"],
        "right": lambda source: table[f"{source} right"],
        "mean of turns": lambda source: (
            (table[f"{source} left"] + table[f"{source} right"]) / 2
        ),
        "left − right": lambda source: (
            table[f"{source} left"] - table[f"{source} right"]
        ),
    }
    for name, column in columns.items():
        report["change"][name] = {
            "observed": interval(column("observed"), rng),
            "predicted": interval(column("full"), rng),
            "observed − predicted": interval(column("observed") - column("full"), rng),
        }
    report["left_turn_without_term"] = {
        term: interval(table[f"{term} left"], rng) for term in terms if term != "full"
    }
    report["left_minus_right_without_term"] = {
        term: interval(columns["left − right"](term), rng)
        for term in terms
        if term != "full"
    }
    report["days_left_turn"] = day_test(days, rng)
    (OUT / f"observed_vs_predicted_{args.mouse}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    for name, entry in report["change"].items():
        text = "  ".join(
            f"{key} {v['median']:+.2f} [{v['interval_95'][0]:+.2f}, {v['interval_95'][1]:+.2f}]"
            for key, v in entry.items()
        )
        print(f"{name:14s} {text}")
    for term, entry in report["left_turn_without_term"].items():
        print(f"without {LABELS.get(term, term):20s} left {entry['median']:+.2f}")
    for key, entry in report["days_left_turn"].items():
        print(f"equal days, {key}: p = {entry['p_equal_days']:.2f}")
    print(plot(args.mouse, table, terms, report))


if __name__ == "__main__":
    main()
