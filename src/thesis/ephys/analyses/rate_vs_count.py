"""Do V1 neurons follow the flash rate, or only the number of flashes?

Question: in the port, many neurons fire more on fast trains than on slow trains
(`condition_responses.py`). But in this task the rate, the number of flashes
and the total light change together. A neuron tuned to the rate fires
differently on fast and slow trains even when the same number of flashes falls
in the window; a neuron that only responds to each flash (or to the total
light) does not.

Comparison: for each unit, the fast-minus-slow difference of its rate in
`SPIKE_WINDOW_S`, (1) over all trials (unmatched) and (2) only between trials
with the same number of flashes in `FLASH_WINDOW_S` (matched). The flash slots
are random, so a fast and a slow train can put the same count in the window.
The matched difference is the mean over counts of fast − slow, weighted by
n_fast·n_slow/(n_fast + n_slow); a count is used when it has at least
`MIN_TRIALS` trials of each category. Each difference is divided by the unit's
mean rate in the window. Unmatched and matched are paired by unit.

- Sessions and units: the decoder sessions (`decoding_time_course.SESSIONS`),
  stable single units (criteria 1, stability 0).
- Trials: decoder trials (completed, non-boundary) whose center exit comes at
  least `EXIT_MARGIN_S` after the spike window, so the window is in the port.
- Windows from the first flash: flashes counted in `FLASH_WINDOW_S` (not the
  first flash); spikes in the same window shifted by `LATENCY_S`.
- Units shown: the units that fire more on fast trains (Mann–Whitney
  p < `ALPHA`), selected on the trials whose count is not used in the matched
  difference, so the selection does not bias it; all units are in the json.
- Test: Wilcoxon signed-rank over units of the matched difference against 0,
  and of matched against unmatched. The sampling unit is the unit; each mouse
  alone.
- Limit: at a matched count in the window, a fast train has more flashes before
  the window, so more adaptation; that pushes the matched difference down.

Writes `figures/rate_vs_count/rate_vs_count.svg` and
`results/rate_vs_count/rate_vs_count.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m thesis.ephys.analyses.rate_vs_count
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses._decoding import load_decoding_data, window_rates
from thesis.ephys.analyses.decoding_time_course import SESSIONS
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/rate_vs_count")
FLASH_WINDOW_S = (0.08, 0.40)
LATENCY_S = 0.03
SPIKE_WINDOW_S = (FLASH_WINDOW_S[0] + LATENCY_S, FLASH_WINDOW_S[1] + LATENCY_S)
EXIT_MARGIN_S = 0.02
MIN_TRIALS = 3
ALPHA = 0.05


def session_values(session: str) -> dict:
    subject, stamp = session.split("_", 1)
    units, trials, _ = load_decoding_data(
        subject,
        stamp,
        unit_criteria_id=1,
        stability_param_id=0,
        window_start_s=SPIKE_WINDOW_S[0],
        window_stop_s=SPIKE_WINDOW_S[1],
        include_mua=False,
    )
    first = trials["stim_pulse_times_s"].str[0].to_numpy(float)
    in_port = trials["center_exit_s"].to_numpy(float) - first >= (
        SPIKE_WINDOW_S[1] + EXIT_MARGIN_S
    )
    trials, first = trials[in_port].reset_index(drop=True), first[in_port]
    counts = np.array(
        [
            np.sum(
                (np.asarray(p) - p[0] >= FLASH_WINDOW_S[0] - 1e-6)
                & (np.asarray(p) - p[0] < FLASH_WINDOW_S[1] - 1e-6)
            )
            for p in trials["stim_pulse_times_s"]
        ]
    )
    fast = trials["stim_category"].eq("high_rate").to_numpy()
    rates = window_rates(units, first + SPIKE_WINDOW_S[0], first + SPIKE_WINDOW_S[1])
    mean = rates.mean(0)
    keep = mean > 0
    rates, mean = rates[:, keep], mean[keep]
    unmatched = (rates[fast].mean(0) - rates[~fast].mean(0)) / mean
    numerator, weight, used = np.zeros(rates.shape[1]), 0.0, []
    for k in np.unique(counts):
        f, s = (counts == k) & fast, (counts == k) & ~fast
        if f.sum() < MIN_TRIALS or s.sum() < MIN_TRIALS:
            continue
        w = f.sum() * s.sum() / (f.sum() + s.sum())
        numerator += w * (rates[f].mean(0) - rates[s].mean(0))
        weight += w
        used.append({"count": int(k), "fast": int(f.sum()), "slow": int(s.sum())})
    matched = numerator / weight / mean
    # Units are selected on the trials that the matched difference does not
    # use, so that selecting on noise does not inflate the matched difference.
    other = ~np.isin(counts, [u["count"] for u in used])
    p_select = np.array(
        [mannwhitneyu(r[other & fast], r[other & ~fast]).pvalue for r in rates.T]
    )
    direction = rates[other & fast].mean(0) > rates[other & ~fast].mean(0)
    tuned = (p_select < ALPHA) & direction

    def test(values):
        return float(wilcoxon(values).pvalue) if len(values) > 5 else float("nan")

    return {
        "subject": subject,
        "units": int(rates.shape[1]),
        "trials": {"fast": int(fast.sum()), "slow": int((~fast).sum())},
        "counts_used": used,
        "unmatched": unmatched.tolist(),
        "matched": matched.tolist(),
        "tuned": tuned.tolist(),
        "tuned_units": int(tuned.sum()),
        "median_unmatched_tuned": float(np.median(unmatched[tuned])),
        "median_matched_tuned": float(np.median(matched[tuned])),
        "matched_vs_zero_p_tuned": test(matched[tuned]),
        "matched_vs_unmatched_p_tuned": test(matched[tuned] - unmatched[tuned]),
        "median_unmatched_all": float(np.median(unmatched)),
        "median_matched_all": float(np.median(matched)),
        "matched_vs_zero_p_all": test(matched),
    }


def plot(results: list[dict]) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(results), figsize=(6.4, 3.4), sharey=True, layout="constrained"
        )
        for axis, r in zip(np.atleast_1d(axes), results, strict=True):
            tuned = np.asarray(r["tuned"])
            u = np.asarray(r["unmatched"])[tuned]
            m = np.asarray(r["matched"])[tuned]
            for a, b in zip(u, m, strict=True):
                axis.plot([0, 1], [a, b], color="0.75", lw=0.7)
            axis.scatter(np.zeros_like(u), u, s=10, color=OBSERVED_COLOR, zorder=3)
            axis.scatter(np.ones_like(m), m, s=10, color=OBSERVED_COLOR, zorder=3)
            axis.plot(
                [0, 1],
                [np.median(u), np.median(m)],
                color=OBSERVED_COLOR,
                lw=2.2,
                marker="_",
                ms=22,
            )
            axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
            axis.set_xticks((0, 1), ("all trials", "same flash\ncount"))
            axis.set_xlim(-0.4, 1.4)
            axis.set_title(
                f"{r['subject']}: {r['tuned_units']} of {r['units']} units\n"
                f"same count: median {r['median_matched_tuned']:+.2f}, "
                f"p = {r['matched_vs_zero_p_tuned']:.2g}",
                fontsize=7.5,
            )
        np.atleast_1d(axes)[0].set_ylabel("fast − slow rate\n(/ unit's mean rate)")
        label_panels(np.atleast_1d(axes))
        return save_figure(figure, OUT / "rate_vs_count")


def main() -> None:
    results = [session_values(s) for s in SESSIONS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rate_vs_count.json").write_text(json.dumps(results, indent=1))
    print(plot(results))
    for r in results:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("unmatched", "matched", "tuned")
                }
            )
        )


if __name__ == "__main__":
    main()
