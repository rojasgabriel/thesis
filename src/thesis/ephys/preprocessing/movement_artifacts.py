"""Count the stable units of a session that look like movement artifacts.

A session enters the neural analyses only with good cortical units. A unit
that fires only while the mouse moves, with a broad spike, is likely not a
neuron but probe or muscle noise from movement. This read-only check scores a
session by the share of such units.

Units: the stable units the analyses use (unit quality criteria 1, stability
parameter set 0; `thesis.ephys.units.fetch_unit_table`).

In-port ratio: a unit's firing rate in the center port (hardware center entry
to center exit, every trial that has both) divided by its rate over the
recording (its spike count over the time from the first to the last spike of
any unit). The mouse holds still in the port and sees the flashes, so a V1
neuron keeps firing there: session medians are 0.5-1.0 in GRB006, GRB064 and
GRB061.

Artifact unit: in-port ratio below `MAX_RATIO` AND spike width at three
quarters of the trough (`fw3m`, ms) above `MIN_FW3M_MS`. Both are needed. In
GRB006 20240821 and GRB064 20261008, 11 units are silent in the port, but
their spikes are as narrow as the others (median fw3m 0.13 ms), so they are
kept as movement-driven neurons. In GRB068 20261113, 46 of 100 units are silent
in the port, with broad spikes (median 0.42 ms) on few channels (median span
15 µm against about 100 µm): 45 pass both rules. Only 4 units of the four
sessions are broad without being silent in the port, all in GRB068 20261113.
The thresholds were set by eye on these four sessions (2026-11-15); the 95th
percentile of fw3m is at most 0.19 ms in the clean ones.

Scored on 2026-11-15 (artifact units / stable units): GRB068 20261113 45 / 100,
GRB006 20240821 0 / 173, GRB064 20261008 0 / 116, GRB061 20261113 0 / 75.

    uv run movement-artifacts GRB068 20261113_143230
    uv run movement-artifacts GRB068 20261113_143230 --units
"""

from __future__ import annotations

import argparse

import numpy as np

MAX_RATIO = 0.1
MIN_FW3M_MS = 0.25


def in_port_ratio(
    spike_times_s: np.ndarray, windows_s: np.ndarray, recording_s: float
) -> float:
    """Return the in-port rate over the recording rate of one unit."""
    spikes = np.sort(np.asarray(spike_times_s, dtype=float))
    windows = np.asarray(windows_s, dtype=float).reshape(-1, 2)
    in_port_s = (windows[:, 1] - windows[:, 0]).sum()
    if in_port_s <= 0 or recording_s <= 0 or spikes.size == 0:
        raise ValueError("need spikes, in-port time and a recording length")
    n_in = (
        np.searchsorted(spikes, windows[:, 1]) - np.searchsorted(spikes, windows[:, 0])
    ).sum()
    return float((n_in / in_port_s) / (spikes.size / recording_s))


def is_artifact(ratio, fw3m_ms, max_ratio=MAX_RATIO, min_fw3m_ms=MIN_FW3M_MS):
    """Return True where a unit is silent in the port and has a broad spike."""
    return (np.asarray(ratio) < max_ratio) & (np.asarray(fw3m_ms) > min_fw3m_ms)


def score_session(
    subject: str, session: str, unit_criteria_id: int = 1, stability_param_id: int = 0
):
    """Return one row per stable unit with its ratio, waveform and verdict."""
    from thesis.ephys.trials import build_trial_table
    from thesis.ephys.units import fetch_unit_table

    units = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=unit_criteria_id,
        stability_param_id=stability_param_id,
    )
    trials = build_trial_table(subject, session, include_frames=False)
    windows = trials[["center_entry_s", "center_exit_s"]].to_numpy(dtype=float)
    windows = windows[
        np.isfinite(windows).all(axis=1) & (windows[:, 1] > windows[:, 0])
    ]
    first = min(times.min() for times in units["spike_times_s"])
    last = max(times.max() for times in units["spike_times_s"])
    table = units[["unit_id", "depth", "fw3m", "spike_amplitude", "span_um"]].copy()
    table["in_port_ratio"] = [
        in_port_ratio(times, windows, last - first) for times in units["spike_times_s"]
    ]
    table["artifact"] = is_artifact(table["in_port_ratio"], table["fw3m"])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument(
        "--units", action="store_true", help="also list every artifact unit"
    )
    args = parser.parse_args()
    table = score_session(args.subject, args.session)
    n_artifact = int(table["artifact"].sum())
    print(
        f"{args.subject} {args.session}: {n_artifact} / {len(table)} stable units "
        f"look like movement artifacts ({n_artifact / len(table):.0%}); "
        f"median in-port ratio {table['in_port_ratio'].median():.2f}"
    )
    silent = table["in_port_ratio"] < MAX_RATIO
    print(
        f"Silent in the port: {int(silent.sum())}; of those with a narrow spike "
        f"(kept as neurons): {int((silent & ~table['artifact']).sum())}"
    )
    if args.units and n_artifact:
        print(
            table[table["artifact"]]
            .drop(columns="artifact")
            .round(3)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
