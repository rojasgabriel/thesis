"""The one measure of a flash response, for the observed rate and the model alike.

Every comparison of flash responses (in port against after exit, one gap against
another) uses this measure:

- the trial-mean rate around the flashes of a condition, in `PSTH_BIN_MS` bins;
- the baseline is the rate of the reference condition (the first of the pair)
  from `BASELINE_MS`, and the same baseline is taken from both conditions;
- the peak (`window_peak`) is the mean baseline-corrected rate over the bins
  within `PEAK_HALF_MS` of the largest bin with a center in `PEAK_MS`, found
  in each condition apart (spikes/s). Since 2026-11-15 every analysis uses
  it; the single-largest-bin `peaks` is removed.

The same function gives the peak of the observed counts, of the rate of the
fitted model, and of that rate with some terms set to 0. The units are those
with the "excited" classification of StimulusResponse: a peak has no meaning
for a unit with no response to the flash.

The flash-excited units of every analysis are saved in `EXCITED_UNITS` and
read with `load_excited_units`. This module writes that file from
`excited_units` (it needs the database; run it where the database is
reachable). The flash-suppressed units ("suppressed" classification, the same
rule) are saved the same way in `SUPPRESSED_UNITS`:

    uv run python -m thesis.ephys.analyses.glm.peaks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.behavior.trials import SESSIONS
from thesis.ephys.analyses.glm.fit import (
    _write_json_atomic,
)
from thesis.ephys.units import fetch_unit_table

PSTH_BIN_MS = 10
BASELINE_MS = (-40, 0)
PEAK_MS = (30, 120)
# Half-width of the window around the peak bin that gives the peak (2026-11-12,
# Anne: a window, not the single largest bin). 20 ms = 5 bins of 10 ms.
PEAK_HALF_MS = 20
LAGS_MS = np.arange(BASELINE_MS[0], PEAK_MS[1])
STIM_RESPONSE_PARAM_ID = 0
# The flash-excited units of each session, {session: [unit id]} (`main`).
# Until 2026-11-07: results/glm/raw_video_trust_classification.json.
EXCITED_UNITS = Path("results/glm/flash_excited_units.json")
SUPPRESSED_UNITS = Path("results/glm/flash_suppressed_units.json")


def load_excited_units() -> dict[str, list[int]]:
    """Return the saved flash-excited units of each session."""
    return json.loads(EXCITED_UNITS.read_text())


def load_suppressed_units() -> dict[str, list[int]]:
    """Return the saved flash-suppressed units of each session."""
    return json.loads(SUPPRESSED_UNITS.read_text())


def excited_units(
    subject: str, session: str, response_type: str = "excited"
) -> set[int]:
    """Return the stable units with a classification of StimulusResponse.

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
        "response_type": response_type,
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


def window_peak(
    binned: np.ndarray,
    centers: np.ndarray,
    latency_from: np.ndarray | None = None,
    end_ms: float = PEAK_MS[1],
) -> np.ndarray:
    """Return the mean over `PEAK_HALF_MS` around the peak bin, along the last axis.

    The peak bin is the largest bin of `latency_from` (default: `binned`) with
    its center from `PEAK_MS[0]` to `end_ms` (default `PEAK_MS[1]`); the mean
    is over the bins with centers within `PEAK_HALF_MS` of it, clipped to the
    trace and to `end_ms`. Give the observed trace as `latency_from` to
    measure a model trace in the same window.
    """
    source = binned if latency_from is None else latency_from
    # The tolerance (here and below) handles centers such as 1000 * 0.035 =
    # 34.999...; without it the window lost 1-2 of its 5 bins in
    # `exit_flash_pairs.py` (until 2026-11-13).
    before_end = centers < end_ms - 1e-6
    window = (centers >= PEAK_MS[0] - 1e-6) & before_end
    at = centers[window][np.argmax(source[..., window], axis=-1)]
    near = (
        np.abs(centers - np.expand_dims(at, -1)) <= PEAK_HALF_MS + 1e-6
    ) & before_end
    return (binned * near).sum(-1) / near.sum(-1)


def window_peaks(trace: np.ndarray, latency: np.ndarray | None = None) -> np.ndarray:
    """Return the window peak of each condition; trace is (condition, 1-ms bin), spikes/s.

    The baseline comes from the first condition and is taken from all. `latency`
    (same shape, default `trace`) gives the window of each condition: give
    the observed trace to measure a model trace in the observed window.
    """
    binned = trace.reshape(len(trace), -1, PSTH_BIN_MS).mean(-1)
    source = (
        binned
        if latency is None
        else latency.reshape(binned.shape[0], -1, PSTH_BIN_MS).mean(-1)
    )
    centers = LAGS_MS.reshape(-1, PSTH_BIN_MS).mean(-1) + 0.5
    baseline = binned[
        0, (centers >= BASELINE_MS[0]) & (centers < BASELINE_MS[1])
    ].mean()
    return window_peak(binned, centers, source) - baseline


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


def main() -> None:
    """Write the flash-excited and flash-suppressed units of each session of `SESSIONS`.

    Entries of other sessions stay. A saved list that differs from the rule is
    reported and kept, unless `--replace`: the saved results used it.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    for response_type, path in (
        ("excited", EXCITED_UNITS),
        ("suppressed", SUPPRESSED_UNITS),
    ):
        saved = json.loads(path.read_text()) if path.exists() else {}
        for session in SESSIONS:
            subject, stamp = session.split("_", 1)
            units = sorted(excited_units(subject, stamp, response_type))
            old = saved.get(session)
            name = f"{session} ({response_type})"
            if old is not None and old != units:
                print(
                    f"{name}: the rule gives {len(units)} units, the file has "
                    f"{len(old)}; only the rule: {sorted(set(units) - set(old))}, "
                    f"only the file: {sorted(set(old) - set(units))}"
                    + ("; replaced" if args.replace else "; kept (use --replace)")
                )
                if not args.replace:
                    continue
            elif old == units:
                print(f"{name}: {len(units)} units, as saved")
            else:
                print(f"{name}: {len(units)} units, new entry: {units}")
            saved[session] = units
        _write_json_atomic(path, saved)


if __name__ == "__main__":
    main()
