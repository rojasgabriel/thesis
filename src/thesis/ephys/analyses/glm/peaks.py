"""The one measure of a flash response, for the observed rate and the model alike.

Every comparison of flash responses (in port against after exit, one gap against
another) uses this measure, as `stim_responses_by_locomotion.py` does on raw
spikes:

- the trial-mean rate around the flashes of a condition, in `PSTH_BIN_MS` bins;
- the baseline is the rate of the reference condition (the first of the pair)
  from `BASELINE_MS`, and the same baseline is taken from both conditions;
- the peak is the largest baseline-corrected rate with a bin center in
  `PEAK_MS`, found in each condition apart (spikes/s).

The same function gives the peak of the observed counts, of the rate of the
fitted model, and of that rate with some terms set to 0. The units are those
with the "excited" classification of StimulusResponse: a peak has no meaning
for a unit with no response to the flash.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import build_unit_counts, load_unit_records
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table

PSTH_BIN_MS = 10
BASELINE_MS = (-40, 0)
PEAK_MS = (30, 120)
LAGS_MS = np.arange(BASELINE_MS[0], PEAK_MS[1])
STIM_RESPONSE_PARAM_ID = 0


def excited_units(subject: str, session: str) -> set[int]:
    """Return the stable units with the "excited" classification of StimulusResponse.

    Stable: the units of `fetch_unit_table` with quality criteria 1 and
    stability 0, the unit rule of the GLM. StimulusResponse itself has no
    stability filter (28 and 21 excited units in GRB006 and GRB064, against 23
    and 15 stable ones).
    """
    from labdata_plugin.schema import StimulusResponse

    query = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": 1,
        "stim_response_param_id": STIM_RESPONSE_PARAM_ID,
        "response_type": "excited",
    }
    stable = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )["unit_id"].astype(int)
    excited = {int(unit) for unit in (StimulusResponse.Unit & query).fetch("unit_id")}
    return excited & set(stable)


def peaks(trace: np.ndarray) -> np.ndarray:
    """Return the peak of each condition; trace is (condition, 1-ms bin) in spikes/s.

    The baseline comes from the first condition and is taken from all.
    """
    binned = trace.reshape(len(trace), -1, PSTH_BIN_MS).mean(-1)
    centers = LAGS_MS.reshape(-1, PSTH_BIN_MS).mean(-1) + 0.5
    baseline = binned[
        0, (centers >= BASELINE_MS[0]) & (centers < BASELINE_MS[1])
    ].mean()
    window = (centers >= PEAK_MS[0]) & (centers < PEAK_MS[1])
    return (binned[:, window] - baseline).max(1)


def session_peaks(
    root: Path, unit_set: str, conditions, zeroed: tuple[str, ...] = ()
) -> dict:
    """Return the peak of each excited unit in each condition, for three rates.

    `conditions` gives, for each condition, the trial row and the time of each
    flash of it (two arrays). The rates are the observed counts, the rate of
    the final refit (all terms, with the observed spike history; in-sample),
    and that rate with the raw regressors in `zeroed` set to 0. Removing a
    raw contribution also adjusts for column centering, so the rate where
    that regressor was already zero stays fixed.
    Returns `peaks` (unit, rate, condition) in spikes/s.
    """
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared)
    windows = []
    for rows, times in conditions:
        first = np.rint((times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S)
        columns = first.astype(int)[:, None] + LAGS_MS[None, :]
        if columns.min() < 0 or columns.max() >= grid[1]:
            raise ValueError("A response window is outside the model grid.")
        windows.append((rows, columns))
    subject, session = (
        prepared["metadata"][key] for key in ("subject_name", "session_name")
    )
    units = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    excited = excited_units(subject, session)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    removed = [
        index
        for index, name in enumerate(metadata["column_names"])
        if name.startswith(tuple(f"{term}_basis" for term in zeroed))
    ]
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
        )
    }
    out, unit_ids = [], []
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        if int(unit_id) not in final or int(unit_id) not in excited:
            continue
        record = final[int(unit_id)]
        spikes = np.asarray(spikes, dtype=float)
        observed = build_unit_counts(alignments, spikes).reshape(grid)
        width = int(metadata["base_columns"]) + int(record["components"])
        weights = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        log_rate = (
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ weights[:width]
            + history @ weights[width:]
        )
        without = log_rate - raw_contribution(common, weights, metadata, removed)
        sources = (
            observed,
            np.exp(log_rate).reshape(grid),
            np.exp(without).reshape(grid),
        )
        out.append(
            [
                peaks(
                    np.stack(
                        [
                            source[rows[:, None], columns].mean(0)
                            for rows, columns in windows
                        ]
                    )
                    / dmat.BINWIDTH_S
                )
                for source in sources
            ]
        )
        unit_ids.append(int(unit_id))
    return {
        "subject": subject,
        "peaks": np.asarray(out),  # (unit, rate, condition)
        "unit_ids": unit_ids,
        "events": [len(rows) for rows, _ in windows],
    }


def paired_flashes(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the trial row and the times of the still and the move flash.

    In each trial: the last flash whose window (`BASELINE_MS[0]` to
    `PEAK_MS[1]`) is in the fixation, and the first flash at or after the
    center exit whose window ends before the response entry. Only trials with
    both, so the comparison is paired over the same trials.
    """
    before_s, after_s = -BASELINE_MS[0] / 1000, PEAK_MS[1] / 1000
    rows, still, move = [], [], []
    for row, (flashes, entry, leave, response) in enumerate(
        zip(
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            trials["response_port_entry_s"],
            strict=True,
        )
    ):
        flashes = np.asarray(flashes, dtype=float)
        inside = flashes[(flashes - before_s >= entry) & (flashes + after_s <= leave)]
        outside = flashes[(flashes >= leave) & (flashes + after_s <= response)]
        if len(inside) and len(outside):
            rows.append(row)
            still.append(inside[-1])
            move.append(outside[0])
    return np.asarray(rows), np.asarray(still), np.asarray(move)
