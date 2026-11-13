"""Offline task-population tests; fake every plugin/database boundary.

The reused fixtures import only events/trials and scipy, with database imports
inside functions. No schema or analysis module is imported by these tests.
"""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_replay_trials import Relation, schema_module

from thesis.ephys import events, trials
from thesis.ephys.task_events import fetch_task_stimulus_pulses


class TaskPulsesTest(unittest.TestCase):
    def select(
        self,
        starts,
        numbers,
        replay_sources=(),
        trial_numbers=(0, 1),
        first_flags=None,
        pre_task=False,
    ):
        chipmunk = types.ModuleType("chipmunk")
        chipmunk.Chipmunk = types.SimpleNamespace(
            trial_query=lambda **key: Relation(
                [dict(trial_num=n) for n in trial_numbers]
            )
        )
        chipmunk.ChipmunkPassive = types.SimpleNamespace(
            Trial=Relation(
                [
                    dict(passive_trial_num=i, source_trial_num=n)
                    for i, n in enumerate(replay_sources)
                ]
            )
        )
        modules = schema_module(
            [dict(event_values=numbers)] if numbers is not None else []
        )
        modules["chipmunk"] = chipmunk
        starts = np.asarray(starts, dtype=float)
        # Pulses at starts test half-open windows. Distinct indices, widths,
        # extra columns, and false first flags detect reconstruction/regrouping.
        timestamps = np.repeat(starts, 2) + np.tile([0.0, 0.2], len(starts))
        pulses = pd.DataFrame(
            dict(
                timestamp=timestamps,
                width_ms=np.resize([15.0, 30.0, np.nan], len(timestamps)),
                first_in_train=(
                    np.tile([True, False], len(starts))
                    if first_flags is None
                    else first_flags
                ),
                provenance=np.arange(len(timestamps)),
            ),
            index=np.arange(len(timestamps)) * 3,
        )
        if pre_task:
            before = pulses.iloc[[0]].copy()
            before.index = [-1]
            before["timestamp"] = starts[0] - 5
            pulses = pd.concat([before, pulses])
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                events,
                "fetch_session_events",
                return_value=({"trial_start": starts}, pulses),
            ) as fetch,
            patch.object(
                trials,
                "build_trial_table",
                side_effect=AssertionError(
                    "Classification must not call the cached trial builder"
                ),
            ),
        ):
            selected = fetch_task_stimulus_pulses("offline", "test")
            fetch.assert_called_once_with("offline", "test")
        return selected, pulses

    def test_mixed_task_and_replay_keeps_only_task_pulses(self):
        selected, original = self.select([10, 20, 30, 40], [0, 1, 2, 3], [0, 1])
        pd.testing.assert_frame_equal(selected, original.iloc[:4])

    def test_interrupted_task_then_replay_bounds_final_task(self):
        selected, original = self.select([10, 20, 30, 40, 50], [0, 1, 3], [0])
        pd.testing.assert_frame_equal(selected, original.iloc[:4])


if __name__ == "__main__":
    unittest.main()
