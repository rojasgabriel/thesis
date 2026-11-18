"""Do the mice differ because their sessions have different trial types?

Question: the flash response after the center exit is much larger than in the
port in GRB006 and not in GRB064. The sessions differ in the flash rates, the
choice side, the errors and the wait time. Does this trial mix make the
difference between the mice?

Sessions and trials: `flash_kernels.SESSIONS`, the selected GLM trials. Units:
the flash-excited GLM units (`peaks.load_excited_units`). The sampling unit is
the unit; every comparison is paired within the unit.

Measure (the observation of `flash_movement.py`): in each trial and gap class
(gap from the flash before of 80 ms or more; 40 ms), the last in-port flash
and the first after-exit flash of that class at least 90 ms later
(`flash_movement.paired_flashes_by_gap`). Rate in 10-ms bins aligned to the
flash; baseline = the in-port rate from -40 to 0 ms, taken from both periods;
peak = `peaks.window_peak` (±20 ms around the largest bin from 30 to 120 ms;
the in-port search ends at 90 ms); divided by the unit's mean rate
(`exit_flash_pairs.mean_rate`, all GLM trials, so that it does not change with
the trial subset). Increase = peak after the exit minus peak in the port.
A subset of trials with fewer than `MIN_PAIRS` pairs gives no value.

1. Audit (`trial_balance.json`, `audit`): trials by rate, choice side and
   outcome; wait time (first flash to center exit); flashes in the port; and,
   for the flash pairs, their number by side and category, the gap of each
   flash and its time from the first flash. Task settings
   (`ephys.trials.trial_settings`): required wait, brightness, reward volume.
2. Splits within a mouse (`splits`): the increase of each unit in the trials
   of one level against the other level of `SPLITS`. Wilcoxon signed-rank
   over units, Holm over the splits of a mouse and gap class. "wait time" is
   the measured wait, "required wait" the setting of the trial (no split in a
   session with one value). If a variable
   does not change the increase within a mouse, its mix cannot make the
   difference between the mice.
3. Matched trials (`matched`): trials are drawn so that each mouse has the
   same number in every cell of `MATCHES` (`DRAWS` draws, without
   replacement). "random, same size" draws the same number of trials with no
   matching: it shows what fewer trials alone do. The value of a unit is its
   mean over the draws; the spread over the draws is in the report.
   - two mice (GRB006, GRB064), flash rate × choice side;
   - the same, also × wait time (below or above the median of both mice);
   - all mice, the rates that every session has, category × choice side
     (rate × side leaves too few trials).
4. Refit on matched trials (`refit`, only if `results/glm_matched` has the
   fits): the first draw of the two-mouse rate × side match is written to
   `results/glm_matched/<session>/trials.json`. The GLM is fit again on those
   trials only (`glm prepare --trials`, `glm fit`, `flash_kernels --out`).
   Per unit: the change of the flash kernel after the exit (window peak for
   flash-excited units, window dip for flash-suppressed units, gap of 80 ms
   or more, Δ log rate) in the fit of all trials against the matched fit.
   Error bars: SD over the cross-validation folds, the two periods taken as
   independent (an upper bound, they come from the same fits).

No display limit: all units are drawn. Color is not used for the mice (marker
shape); in panel a the shade is the choice side.

Writes `figures/glm/trial_balance.svg` and `results/glm/trial_balance.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.trial_balance
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm import exit_flash_pairs as pairs
from thesis.ephys.analyses.glm import flash_movement as fm
from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import load_excited_units, load_suppressed_units
from thesis.ephys.trials import trial_settings
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    POINT_ALPHA,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
MATCHED = Path("results/glm_matched")
DRAWS = 200
MIN_PAIRS = 10
SEED = 20261115
GAP_NAMES = ("80+ ms", "40 ms")  # the order of `flash_movement.GAPS_MS`
# The mice whose sessions have many flash rates: matched on rate and refit.
TWO_MICE = ("GRB006", "GRB064")
# name: (column, first level, second level); the difference is second - first.
SPLITS = {
    "choice side": ("side", "left", "right"),
    "category": ("category", "low", "high"),
    "outcome": ("outcome", "correct", "error"),
    "wait time": ("wait", "short", "long"),
    "required wait": ("required", "short", "long"),
    "session half": ("half", "first", "second"),
    "trial before": ("previous", "reward", "no reward"),
}
# name: (subjects, columns of the cells, only the rates of every session)
MATCHES = {
    "rate × side": (TWO_MICE, ("visual_stim_rate_hz", "side"), False),
    "rate × side × wait": (TWO_MICE, ("visual_stim_rate_hz", "side", "wait"), False),
    "category × side, all mice": (None, ("category", "side"), True),
}
SIDE_SHADES = {"left": "0.25", "right": "0.7"}
PORT, EXIT = "in port, 80+ ms gap", "after exit, 80+ ms gap"
PERIODS = ("stationary", "movement")  # the keys of a `flash_movement` pair group


def load(session: str) -> dict:
    """Return the GLM trials, the flash pairs and their counts for the excited units."""
    subject, stamp = session.split("_", 1)
    prepared = dmat._load_windows(OUT / session / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared).merge(
        trial_settings(subject, stamp), on="trial_num", how="left", validate="1:1"
    )
    first = trials["stim_pulse_times_s"].str[0].to_numpy(float)
    rate = trials["visual_stim_rate_hz"].to_numpy(float)
    boundary = trials["category_boundary_hz"].to_numpy(float)
    wait = trials["center_exit_s"].to_numpy(float) - first
    trials = trials.assign(
        side=np.where(trials["response"] == 1, "right", "left"),
        category=np.select([rate < boundary, rate > boundary], ["low", "high"], ""),
        outcome=np.where(trials["rewarded"] == 1, "correct", "error"),
        wait_s=wait,
        wait=np.where(wait < np.median(wait), "short", "long"),
        # No level in a session with one required wait time.
        required=np.where(
            trials["required_wait_s"].nunique() < 2,
            "",
            np.where(
                trials["required_wait_s"] < trials["required_wait_s"].median(),
                "short",
                "long",
            ),
        ),
        half=np.where(np.arange(len(trials)) < len(trials) / 2, "first", "second"),
        previous=np.select(
            [trials["last_outcome"] == 1, trials["last_outcome"] == -1],
            ["reward", "no reward"],
            "",
        ),
        flashes_in_port=[
            int(np.sum((np.asarray(t) >= a) & (np.asarray(t) < b)))
            for t, a, b in zip(
                trials["stim_pulse_times_s"],
                trials["center_entry_s"],
                trials["center_exit_s"],
                strict=True,
            )
        ],
        first_flash_s=first,
    )
    units = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False)
    units = units.iloc[analysis_unit_indices(units, "all")]
    units = units[units["unit_id"].astype(int).isin(load_excited_units()[session])]
    spikes = [np.sort(np.asarray(s, float)) for s in units["spike_times_s"]]
    groups, _ = fm.paired_flashes_by_gap(trials)
    return dict(
        session=session,
        subject=subject,
        trials=trials,
        unit_ids=units["unit_id"].astype(int).to_numpy(),
        mean_rate=np.array([pairs.mean_rate(s, first) for s in spikes]),
        pair_trials=[np.asarray(g["trial_ids"]) for g in groups],
        pair_times=[np.array([g["stationary"], g["movement"]]) for g in groups],
        pair_gaps=[np.asarray(g["gaps_ms"]) for g in groups],
        # (unit, pair, period, bin)
        counts=[
            np.array(
                [
                    np.stack([pairs.counts(s, np.asarray(g[k])) for k in PERIODS], 1)
                    for s in spikes
                ]
            )
            for g in groups
        ],
    )


def peaks(d: dict, gap: int, trial_numbers=None) -> np.ndarray | None:
    """Return (unit, period) peaks ÷ mean rate from the pairs of some trials."""
    keep = (
        np.ones(len(d["pair_trials"][gap]), bool)
        if trial_numbers is None
        else np.isin(d["pair_trials"][gap], trial_numbers)
    )
    if keep.sum() < MIN_PAIRS:
        return None
    rate = d["counts"][gap][:, keep].mean(1) / 0.01
    return fm.observed_peaks(rate, pairs.CENTERS) / d["mean_rate"][:, None]


def summary(increase: np.ndarray) -> dict:
    """Return the mean, SEM and sign test of a per-unit value."""
    return dict(
        units=int(len(increase)),
        mean=float(increase.mean()),
        sem=float(increase.std(ddof=1) / np.sqrt(len(increase))),
        median=float(np.median(increase)),
        above_0=int(np.sum(increase > 0)),
        p=float(wilcoxon(increase).pvalue) if np.any(increase) else 1.0,
    )


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * np.sort(p))
    out = np.empty(len(p))
    out[order] = np.minimum(adjusted, 1)
    return out.tolist()


def audit(d: dict) -> dict:
    t = d["trials"]
    by = t.groupby(["visual_stim_rate_hz", "side", "outcome"]).size()
    report = dict(
        trials=int(len(t)),
        rates_hz=sorted(float(r) for r in t["visual_stim_rate_hz"].unique()),
        trials_by_rate_side_outcome={
            f"{rate:g} Hz, {side}, {outcome}": int(n)
            for (rate, side, outcome), n in by.items()
        },
        right_choices=float((t["side"] == "right").mean()),
        errors=int((t["outcome"] == "error").sum()),
        correct_by_category={
            c: float((g["outcome"] == "correct").mean())
            for c, g in t.groupby("category")
            if c
        },
        rewarded_trial_before=float((t["previous"] == "reward").mean()),
        wait_s_quartiles=np.percentile(t["wait_s"], [25, 50, 75]).tolist(),
        required_wait_s_quartiles=np.percentile(
            t["required_wait_s"], [25, 50, 75]
        ).tolist(),
        wait_above_required_s_median=float(
            (t["center_exit_s"] - t["center_entry_s"] - t["required_wait_s"]).median()
        ),
        stim_brightness=sorted(float(v) for v in t["stim_brightness"].unique()),
        reward_volume_ul=sorted(float(v) for v in t["reward_volume_ul"].unique()),
        flashes_in_port_median_by_category={
            c: float(g["flashes_in_port"].median())
            for c, g in t.groupby("category")
            if c
        },
        pairs={},
    )
    index = t.set_index("trial_num")
    for gap, name in enumerate(GAP_NAMES):
        rows = index.loc[d["pair_trials"][gap]]
        since = d["pair_times"][gap] - rows["first_flash_s"].to_numpy()
        report["pairs"][name] = dict(
            pairs=int(len(rows)),
            by_side=rows["side"].value_counts().to_dict(),
            by_category=rows["category"]
            .replace("", "boundary")
            .value_counts()
            .to_dict(),
            by_outcome=rows["outcome"].value_counts().to_dict(),
            median_gap_ms_in_port_after_exit=np.median(d["pair_gaps"][gap], 0).tolist(),
            median_s_from_first_flash_in_port_after_exit=np.median(since, 1).tolist(),
        )
    return report


def splits(d: dict) -> dict:
    """Return, per gap class and split, the increase in each level and its difference."""
    t = d["trials"]
    report = {}
    for gap, gap_name in enumerate(GAP_NAMES):
        rows = {}
        for name, (column, first, second) in SPLITS.items():
            levels = [
                peaks(d, gap, t.loc[t[column] == level, "trial_num"].to_numpy())
                for level in (first, second)
            ]
            n = [
                int(
                    np.isin(
                        d["pair_trials"][gap], t.loc[t[column] == level, "trial_num"]
                    ).sum()
                )
                for level in (first, second)
            ]
            rows[name] = dict(levels=[first, second], pairs=n)
            if any(level is None for level in levels):
                rows[name]["note"] = f"fewer than {MIN_PAIRS} pairs in a level"
                continue
            increase = [level[:, 1] - level[:, 0] for level in levels]
            rows[name].update(
                mean_increase=[float(v.mean()) for v in increase],
                mean_peak_in_port=[float(level[:, 0].mean()) for level in levels],
                mean_peak_after_exit=[float(level[:, 1].mean()) for level in levels],
                difference=summary(increase[1] - increase[0]),
                difference_per_unit=(increase[1] - increase[0]).tolist(),
            )
        tested = [name for name in rows if "difference" in rows[name]]
        for name, p in zip(
            tested, holm([rows[n]["difference"]["p"] for n in tested]), strict=True
        ):
            rows[name]["difference"]["p_holm"] = p
        report[gap_name] = rows
    return report


def draw_matched(tables: dict, columns, rng) -> dict:
    """Draw trial numbers so that every session has the same count in each cell."""
    cells = {
        s: t.groupby(list(columns))["trial_num"].apply(list) for s, t in tables.items()
    }
    common = set.intersection(*(set(c.index) for c in cells.values()))
    picked = {s: [] for s in tables}
    for cell in sorted(common):
        n = min(len(c[cell]) for c in cells.values())
        for s, c in cells.items():
            picked[s].extend(rng.choice(c[cell], n, replace=False).tolist())
    return {s: np.sort(v) for s, v in picked.items()}


def matched(data: list[dict], report: dict) -> dict:
    """Return the per-unit increase (mean over draws) of each match and mouse."""
    values = {}
    for name, (subjects, columns, common_rates) in MATCHES.items():
        used = [d for d in data if subjects is None or d["subject"] in subjects]
        tables = {d["session"]: d["trials"] for d in used}
        if common_rates:
            shared = set.intersection(
                *(set(t["visual_stim_rate_hz"]) for t in tables.values())
            )
            tables = {
                s: t[t["visual_stim_rate_hz"].isin(shared)] for s, t in tables.items()
            }
        if "wait" in columns:  # one wait-time boundary for the mice of the match
            boundary = np.median(np.concatenate([t["wait_s"] for t in tables.values()]))
            tables = {
                s: t.assign(wait=np.where(t["wait_s"] < boundary, "short", "long"))
                for s, t in tables.items()
            }
        rng = np.random.default_rng(SEED)
        draws = [draw_matched(tables, columns, rng) for _ in range(DRAWS)]
        if name == "rate × side":  # the trials of the refit: the first draw
            for session, numbers in draws[0].items():
                path = MATCHED / session / "trials.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and json.loads(path.read_text()) != numbers.tolist():
                    raise ValueError(f"{path} differs from the first draw.")
                path.write_text(json.dumps(numbers.tolist()))
        n_trials = len(next(iter(draws[0].values())))
        entry = report.setdefault(name, {"trials_per_mouse": n_trials, "mice": {}})
        for kind in (name, f"random, same size as {name}"):
            for d in used:
                all_numbers = d["trials"]["trial_num"].to_numpy()
                per_gap = {}
                for gap, gap_name in enumerate(GAP_NAMES):
                    stack, n_pairs = [], []
                    for draw in draws:
                        numbers = (
                            draw[d["session"]]
                            if kind == name
                            else rng.choice(all_numbers, n_trials, replace=False)
                        )
                        p = peaks(d, gap, numbers)
                        n_pairs.append(
                            int(np.isin(d["pair_trials"][gap], numbers).sum())
                        )
                        if p is not None:
                            stack.append(p[:, 1] - p[:, 0])
                    if len(stack) < DRAWS / 2:
                        per_gap[gap_name] = dict(
                            note=f"fewer than {MIN_PAIRS} pairs in most draws",
                            median_pairs=float(np.median(n_pairs)),
                        )
                        continue
                    stack = np.array(stack)  # (draw, unit)
                    per_unit = stack.mean(0)
                    values[(kind, d["subject"], gap_name)] = per_unit
                    per_gap[gap_name] = dict(
                        median_pairs=float(np.median(n_pairs)),
                        draws_with_a_value=len(stack),
                        mean_increase_over_draws=dict(
                            median=float(np.median(stack.mean(1))),
                            interval_95=np.percentile(
                                stack.mean(1), [2.5, 97.5]
                            ).tolist(),
                        ),
                        per_unit_mean_over_draws=summary(per_unit),
                    )
                entry["mice"].setdefault(d["subject"], {})[
                    "matched" if kind == name else "random, same size"
                ] = per_gap
    return values


def kernel_change(root: Path, session: str, kind: str) -> dict:
    """Return {unit id: (after exit - in port, SD)} of the kernel peak or dip."""
    units = (load_excited_units() if kind == "peak" else load_suppressed_units())[
        session
    ]
    k = np.load(root / session / "flash_kernels.npz")
    cond = list(k["conditions"])
    i, j = cond.index(PORT), cond.index(EXIT)
    change = k[kind][:, j] - k[kind][:, i]
    sd = np.hypot(k[f"{kind}_sd"][:, i], k[f"{kind}_sd"][:, j])
    return {
        int(u): (float(c), float(s))
        for u, c, s in zip(k["unit_ids"], change, sd, strict=True)
        if int(u) in units
    }


def refit(report: dict) -> dict | None:
    """Compare the kernel change of the fit of all trials and of the matched fit."""
    sessions = [s for s in SESSIONS if s.split("_")[0] in TWO_MICE]
    if not all((MATCHED / s / "flash_kernels.npz").exists() for s in sessions):
        return None
    values = {}
    for session in sessions:
        subject = session.split("_")[0]
        fit_summary = json.loads(
            (MATCHED / session / "all_fit_me/cross_validation_summary.json").read_text()
        )
        report.setdefault(subject, {})["matched_fit"] = {
            key: value
            for key, value in fit_summary.items()
            if isinstance(value, (int, float, str))
        }
        for kind in ("peak", "dip"):
            full = kernel_change(OUT, session, kind)
            match = kernel_change(MATCHED, session, kind)
            ids = sorted(set(full) & set(match))
            x = np.array([full[u] for u in ids])
            y = np.array([match[u] for u in ids])
            values[(subject, kind)] = (x, y)
            report[subject][kind] = dict(
                units=len(ids),
                units_only_in_the_fit_of_all_trials=len(set(full) - set(match)),
                all_trials=summary(x[:, 0]),
                matched_trials=summary(y[:, 0]),
                matched_minus_all=summary(y[:, 0] - x[:, 0]),
                spearman=float(spearmanr(x[:, 0], y[:, 0]).statistic),
            )
    return values


def strip(axis, x, values, marker, rng):
    """Draw the units of one mouse as light points with their mean and SEM."""
    axis.scatter(
        x + rng.uniform(-0.05, 0.05, len(values)),
        values,
        s=7,
        marker=marker,
        color="black",
        alpha=POINT_ALPHA,
        lw=0,
    )
    axis.errorbar(
        x,
        values.mean(),
        yerr=values.std(ddof=1) / np.sqrt(len(values)),
        fmt=marker,
        ms=6,
        color="white",
        mec="black",
        mew=1.3,
        ecolor="black",
        capsize=2,
        zorder=5,
    )


def plot(data, report, match_values, refit_values):
    rng = np.random.default_rng(0)
    n = len(data)
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(
            figsize=(12, 9.2 if refit_values else 6.4), layout="constrained"
        )
        grid = fig.add_gridspec(3 if refit_values else 2, 12)
        top = grid[0, :].subgridspec(1, n + 1)
        letters = []
        # a: trials by rate and choice side, one axis per mouse.
        first = None
        for m, d in enumerate(data):
            axis = fig.add_subplot(top[0, m], sharey=first)
            first = first or axis
            t = d["trials"]
            rates = np.sort(t["visual_stim_rate_hz"].unique())
            bottom = np.zeros(len(rates))
            for side, shade in SIDE_SHADES.items():
                count = np.array(
                    [
                        np.sum((t["visual_stim_rate_hz"] == r) & (t["side"] == side))
                        for r in rates
                    ]
                )
                axis.bar(rates, count, bottom=bottom, width=0.8, color=shade)
                bottom += count
            axis.axvline(12, color="0.6", ls="--", lw=0.7)
            axis.set_xticks([4, 8, 12, 16, 20])
            axis.set_xlabel("flash rate (Hz)")
            axis.plot([], [], mouse_marker(d["subject"], m), color="black")
            axis.text(
                0,
                1.02,
                f"{d['subject']}: {len(t)} trials",
                transform=axis.transAxes,
                va="bottom",
            )
            if m == 0:
                axis.set_ylabel("trials")
                letters.append(axis)
                # The shade legend, once, above the low bars of the first rates.
                for k, (side, shade) in enumerate(SIDE_SHADES.items()):
                    axis.text(
                        0.03,
                        0.97 - 0.08 * k,
                        f"{side} choice",
                        color=shade if side == "left" else "0.5",
                        transform=axis.transAxes,
                        va="top",
                    )
        # b: wait time of each trial.
        axis = fig.add_subplot(top[0, n])
        for m, d in enumerate(data):
            wait = d["trials"]["wait_s"].to_numpy()
            marker = mouse_marker(d["subject"], m)
            axis.scatter(
                m + rng.uniform(-0.2, 0.2, len(wait)),
                wait,
                s=7,
                marker=marker,
                color="black",
                alpha=POINT_ALPHA,
                lw=0,
            )
            axis.plot(
                m, np.median(wait), marker, ms=8, color="white", mec="black", mew=1.4
            )
        axis.set_xticks(range(n), [d["subject"] for d in data])
        axis.set_ylabel("wait time, first flash to exit (s)")
        letters.append(axis)
        # c: splits within each mouse, 80+ ms gap.
        axis = fig.add_subplot(grid[1, :6])  # c
        for m, d in enumerate(data):
            rows = report["splits"][d["subject"]][GAP_NAMES[0]]
            for k, name in enumerate(SPLITS):
                if "difference_per_unit" in rows[name]:
                    strip(
                        axis,
                        k + (m - (n - 1) / 2) * 0.24,
                        np.asarray(rows[name]["difference_per_unit"]),
                        mouse_marker(d["subject"], m),
                        rng,
                    )
        axis.axhline(0, color="0.6", ls="--", lw=0.8, zorder=0)
        axis.set_xticks(
            range(len(SPLITS)),
            [f"{name}\n{b} −\n{a}" for name, (_, a, b) in SPLITS.items()],
            fontsize=6.5,
        )
        axis.set_ylabel("difference of the increase after exit\n(÷ mean rate)")
        letters.append(axis)
        # d: all trials, random subsets and matched trials, 80+ ms gap.
        axis = fig.add_subplot(grid[1, 6:])
        columns = ["all trials"]
        for name in MATCHES:
            columns += [f"random, same size as {name}", name]
        for m, d in enumerate(data):
            for k, column in enumerate(columns):
                if column == "all trials":
                    p = peaks(d, 0)
                    values = p[:, 1] - p[:, 0]
                else:
                    values = match_values.get((column, d["subject"], GAP_NAMES[0]))
                if values is not None:
                    strip(
                        axis,
                        k + (m - (n - 1) / 2) * 0.24,
                        values,
                        mouse_marker(d["subject"], m),
                        rng,
                    )
        axis.axhline(0, color="0.6", ls="--", lw=0.8, zorder=0)
        ticks = ["all\ntrials"]
        for name in MATCHES:
            size = report["matched"][name]["trials_per_mouse"]
            label = name.replace(", ", "\n")
            ticks += [f"random\n{size} trials", f"matched\n{label}"]
        axis.set_xticks(range(len(columns)), ticks, fontsize=6.5)
        axis.set_ylabel("flash peak increase after exit\n(÷ mean rate)")
        letters.append(axis)
        # e, f: kernel change in the fit of all trials against the matched fit.
        if refit_values:
            for k, (kind, title) in enumerate(
                (("peak", "flash-excited units"), ("dip", "flash-suppressed units"))
            ):
                axis = fig.add_subplot(grid[2, 6 * k + 1 : 6 * k + 5])
                for m, d in enumerate(data):
                    if (d["subject"], kind) not in refit_values:
                        continue
                    x, y = refit_values[(d["subject"], kind)]
                    marker = mouse_marker(d["subject"], m)
                    axis.errorbar(
                        x[:, 0],
                        y[:, 0],
                        xerr=x[:, 1],
                        yerr=y[:, 1],
                        fmt=marker,
                        ms=3.5,
                        color="black",
                        elinewidth=0.4,
                        alpha=POINT_ALPHA,
                        lw=0,
                    )
                    axis.plot(
                        np.median(x[:, 0]),
                        np.median(y[:, 0]),
                        marker,
                        ms=9,
                        color="white",
                        mec="black",
                        mew=1.6,
                        zorder=5,
                    )
                low = min(axis.get_xlim()[0], axis.get_ylim()[0])
                high = max(axis.get_xlim()[1], axis.get_ylim()[1])
                axis.plot([low, high], [low, high], color="0.6", ls="--", lw=0.8)
                axis.axhline(0, color="0.85", lw=0.6, zorder=0)
                axis.axvline(0, color="0.85", lw=0.6, zorder=0)
                axis.set_aspect("equal")
                axis.set_xlabel(
                    f"kernel {kind}, after exit − in port\nfit of all trials (Δ log rate)"
                )
                axis.set_ylabel(
                    f"kernel {kind}, after exit − in port\nfit of matched trials (Δ log rate)"
                )
                axis.text(0, 1.02, title, transform=axis.transAxes, va="bottom")
                letters.append(axis)
        label_panels(letters, x=-0.12, y=1.06)
        return save_figure(fig, OUT / "trial_balance")


def main() -> None:
    data = [load(s) for s in SESSIONS]
    report = {
        "min_pairs": MIN_PAIRS,
        "draws": DRAWS,
        "audit": {d["subject"]: audit(d) for d in data},
        "all_trials": {
            d["subject"]: {
                name: summary(p[:, 1] - p[:, 0])
                for gap, name in enumerate(GAP_NAMES)
                if (p := peaks(d, gap)) is not None
            }
            for d in data
        },
        "splits": {d["subject"]: splits(d) for d in data},
        "matched": {},
        "refit": {},
    }
    match_values = matched(data, report["matched"])
    refit_values = refit(report["refit"])
    print(plot(data, report, match_values, refit_values))
    for subject in report["splits"]:  # the per-unit values are only for the figure
        for rows in report["splits"][subject].values():
            for row in rows.values():
                row.pop("difference_per_unit", None)
    (OUT / "trial_balance.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
