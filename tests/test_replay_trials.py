"""Offline replay alignment regressions: all database/plugin access is faked."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.interpolate  # Load extensions before patching sys.modules.

from thesis.ephys import events, trials

assert scipy.interpolate.CubicSpline


class Relation:
    def __init__(self, rows=(), frame=None):
        self.rows, self.frame = rows, frame

    def __call__(self):
        return self

    def __and__(self, key):
        return Relation(
            [
                row
                for row in self.rows
                if all(row.get(k, v) == v for k, v in key.items())
            ],
            self.frame,
        )

    def __len__(self):
        return len(self.rows)

    def proj(self, *args):
        return self

    def fetch(self, *fields, **kwargs):
        if kwargs.get("format") == "frame":
            return self.frame.copy()
        if kwargs.get("as_dict"):
            return [{k: row[k] for k in fields} for row in self.rows]
        if len(fields) == 1:
            result = np.empty(len(self.rows), dtype=object)
            result[:] = [row[fields[0]] for row in self.rows]
            return result
        raise AssertionError("Unexpected database operation")

    def fetch1(self):
        assert len(self.rows) == 1
        return self.rows[0]


def schema_module(rows=(), sync=()):
    schema = types.ModuleType("labdata.schema")
    schema.DatasetEvents = types.SimpleNamespace(Digital=Relation(rows))
    schema.StreamSync = Relation(sync)
    return {"labdata": types.ModuleType("labdata"), "labdata.schema": schema}


class PulseNumbersTest(unittest.TestCase):
    def test_unexplained_extra_pulses_raise(self):
        with self.assertRaises(ValueError):
            events.saved_pulse_indices(6, 3, [0, 1, 2])


class TrialTablesTest(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory(prefix="thesis-offline-cache-")
        self.addCleanup(self.cache.cleanup)
        self.env = patch.dict(os.environ, THESIS_CACHE_DIR=self.cache.name)
        self.env.start()
        self.addCleanup(self.env.stop)

    def tables(self, starts, numbers, replay_sources=()):
        index = pd.MultiIndex.from_tuples(
            [("offline", "test", "chipmunk", i) for i in range(2)],
            names=["subject_name", "session_name", "dataset_name", "trial_num"],
        )
        frame = pd.DataFrame(
            dict(
                response=[0, 0],
                rewarded=[0, 0],
                early_withdrawal=[1, 1],
                t_react=[np.nan, np.nan],
                t_initiate=[np.nan, np.nan],
                left_poke=[[], []],
                center_poke=[[], []],
                right_poke=[[], []],
                stim_duration=[1.0, 1.0],
                stim_rate_vision=[4.0, 20.0],
                category_boundary=[12.0, 12.0],
            ),
            index=index,
        )
        chipmunk = types.ModuleType("chipmunk")
        chipmunk.Chipmunk = types.SimpleNamespace(
            trial_query=lambda **key: Relation(frame=frame)
        )
        chipmunk.ChipmunkPassive = types.SimpleNamespace(
            Trial=Relation(
                [
                    dict(passive_trial_num=i, source_trial_num=source)
                    for i, source in enumerate(replay_sources)
                ]
            )
        )
        modules = schema_module(
            [dict(event_values=numbers)] if numbers is not None else []
        )
        modules["chipmunk"] = chipmunk
        starts = np.asarray(starts, dtype=float)
        pulses = starts + 0.5
        sess_ev = dict(
            trial_start=starts, stim=pulses, left_port=[], center_port=[], right_port=[]
        )
        stim = pd.DataFrame(dict(timestamp=pulses, width_ms=15.0, first_in_train=True))
        with (
            patch.dict(sys.modules, modules),
            patch.object(events, "fetch_session_events", return_value=(sess_ev, stim)),
        ):
            task = trials.build_trial_table("offline", "test", include_frames=False)
            replay = trials.build_replay_table("offline", "test")
        return task, replay

    def test_replay_source_must_exist(self):
        with self.assertRaisesRegex(ValueError, "missing task trials"):
            self.tables([10, 20, 30], [0, 1, 2], [99])

    def test_historical_task_only(self):
        task, replay = self.tables([10, 20], [0, 1])
        self.assertEqual(task.stim_pulse_times_s.tolist(), [[10.5], [20.5]])
        self.assertTrue(replay.empty)

    def test_normal_replay_and_source_trial_join(self):
        task, replay = self.tables([10, 20, 30, 40], [0, 1, 2, 3], [1, 0])
        self.assertEqual(task.stim_pulse_times_s.tolist(), [[10.5], [20.5]])
        self.assertEqual(replay.stim_pulse_times_s.tolist(), [[30.5], [40.5]])
        self.assertEqual(replay.source_trial_num.tolist(), [1, 0])
        self.assertEqual(replay.visual_stim_rate_hz.tolist(), [20.0, 4.0])

    def test_stop_after_sync_and_final_interrupted_replay(self):
        task, replay = self.tables([10, 20, 30, 40, 50, 60], [0, 1, 3, 4], [1, 0])
        self.assertEqual(task.stim_pulse_times_s.tolist(), [[10.5], [20.5]])
        self.assertEqual(replay.stim_pulse_times_s.tolist(), [[40.5], [50.5]])
        self.assertEqual(replay.trial_start_s.tolist(), [40.0, 50.0])


if __name__ == "__main__":
    unittest.main()
