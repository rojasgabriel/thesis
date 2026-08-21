import importlib
import sys
import types
import unittest
from typing import Any

import numpy as np


class FakeRelation:
    def __init__(self, rows=None):
        self.rows = rows or []

    def __and__(self, _other):
        return self

    def proj(self):
        return self

    def fetch_synced(self):
        return self.rows

    def fetch(self, *args, **kwargs):
        if kwargs.get("as_dict"):
            return self.rows
        return self.rows


def load_utils_io(events_rows, mapping_rows):
    fake_schema: Any = types.ModuleType("labdata.schema")
    fake_schema.Dataset = lambda: FakeRelation()
    fake_schema.DatasetEvents = types.SimpleNamespace(
        Digital=lambda: FakeRelation(events_rows)
    )
    fake_schema.SpikeSorting = lambda: FakeRelation()
    fake_schema.UnitMetrics = object()
    fake_schema.EphysRecording = types.SimpleNamespace(
        ProbeSetting=lambda: FakeRelation()
    )
    fake_schema.UnitCount = types.SimpleNamespace(Unit=FakeRelation())

    fake_labdata: Any = types.ModuleType("labdata")
    fake_labdata.schema = fake_schema

    fake_analysisschema: Any = types.ModuleType("labdata_plugin.schema")
    fake_analysisschema.EventMapping = lambda: FakeRelation(mapping_rows)

    fake_labdata_plugin: Any = types.ModuleType("labdata_plugin")
    fake_labdata_plugin.schema = fake_analysisschema

    sys.modules["labdata"] = fake_labdata
    sys.modules["labdata.schema"] = fake_schema
    sys.modules["labdata_plugin"] = fake_labdata_plugin
    sys.modules["labdata_plugin.schema"] = fake_analysisschema

    for name in ("thesis.ephys.utils.io_digital_events",):
        sys.modules.pop(name, None)
    import thesis.ephys.utils.io_digital_events as utils_io

    return importlib.reload(utils_io)


class FetchSessionEventsTests(unittest.TestCase):
    def test_fetch_session_events_uses_semantic_mapping_for_onset_only_grb006_stim(
        self,
    ):
        events_rows = [
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "ai0",
                "event_timestamps": np.array([0.1, 0.4, 1.5]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "2",
                "event_timestamps": np.array([0.0, 0.1, 1.0, 1.1]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "1",
                "event_timestamps": np.array([0.0, 1.0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "3",
                "event_timestamps": np.array([0.2]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "4",
                "event_timestamps": np.array([0.3]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "5",
                "event_timestamps": np.array([0.5]),
            },
        ]
        mapping_rows = [
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "visual_stim",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "ai0",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "trial_start",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "2",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "frames",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "1",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "left_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "3",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "center_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "4",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "right_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "5",
            },
        ]

        utils_io = load_utils_io(events_rows, mapping_rows)
        align_ev = utils_io.fetch_session_events("GRB006", "20240821_121447")

        np.testing.assert_allclose(align_ev["stim"], [0.1, 0.4, 1.5])
        np.testing.assert_allclose(align_ev["trial_start"], [0.0, 1.0])
        np.testing.assert_allclose(align_ev["stim_ev_15ms"], [0.1, 0.4, 1.5])
        np.testing.assert_allclose(align_ev["first_stim_ev_15ms"], [0.1, 1.5])

    def test_fetch_session_events_preserves_raw_ttl_width_classification(self):
        events_rows = [
            {
                "dataset_name": "ephys_g0",
                "stream_name": "obx",
                "event_name": "io0",
                "event_timestamps": np.array([0.1, 0.115, 1.2, 1.21, 1.22, 1.23]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "obx",
                "event_name": "io2",
                "event_timestamps": np.array([0.0, 0.1, 1.0, 1.1]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "obx",
                "event_name": "io3",
                "event_timestamps": np.array([0.0, 1.0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "obx",
                "event_name": "io4",
                "event_timestamps": np.array([0.2]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "obx",
                "event_name": "io5",
                "event_timestamps": np.array([0.3]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "obx",
                "event_name": "io6",
                "event_timestamps": np.array([0.5]),
            },
        ]
        mapping_rows = [
            {
                "subject_name": "GRB058",
                "session_name": "20260312_134952",
                "event_name": "visual_stim",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "obx",
                "source_event_name": "io0",
            },
            {
                "subject_name": "GRB058",
                "session_name": "20260312_134952",
                "event_name": "trial_start",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "obx",
                "source_event_name": "io2",
            },
            {
                "subject_name": "GRB058",
                "session_name": "20260312_134952",
                "event_name": "frames",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "obx",
                "source_event_name": "io3",
            },
            {
                "subject_name": "GRB058",
                "session_name": "20260312_134952",
                "event_name": "left_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "obx",
                "source_event_name": "io4",
            },
            {
                "subject_name": "GRB058",
                "session_name": "20260312_134952",
                "event_name": "center_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "obx",
                "source_event_name": "io5",
            },
            {
                "subject_name": "GRB058",
                "session_name": "20260312_134952",
                "event_name": "right_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "obx",
                "source_event_name": "io6",
            },
        ]

        utils_io = load_utils_io(events_rows, mapping_rows)
        align_ev = utils_io.fetch_session_events("GRB058", "20260312_134952")

        np.testing.assert_allclose(align_ev["stim_ev_15ms"], [0.1])
        np.testing.assert_allclose(align_ev["stim_ev_30ms"], [1.2])

    def test_fetch_session_events_uses_rising_edges_for_ports_when_values_exist(self):
        events_rows = [
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "ai0",
                "event_timestamps": np.array([0.2, 1.2]),
                "event_values": np.array([1, 1]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "2",
                "event_timestamps": np.array([0.0, 0.1, 1.0, 1.1]),
                "event_values": np.array([1, 0, 1, 0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "1",
                "event_timestamps": np.array([0.05, 0.15]),
                "event_values": np.array([1, 0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "3",
                "event_timestamps": np.array([0.3, 0.4]),
                "event_values": np.array([1, 0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "4",
                "event_timestamps": np.array([0.15, 0.25, 0.7, 0.8]),
                "event_values": np.array([1, 0, 1, 0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "5",
                "event_timestamps": np.array([0.5, 0.6]),
                "event_values": np.array([1, 0]),
            },
        ]
        mapping_rows = [
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "visual_stim",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "ai0",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "trial_start",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "2",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "frames",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "1",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "left_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "3",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "center_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "4",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "right_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "5",
            },
        ]

        utils_io = load_utils_io(events_rows, mapping_rows)
        align_ev = utils_io.fetch_session_events("GRB006", "20240821_121447")

        np.testing.assert_allclose(align_ev["trial_start"], [0.0, 1.0])
        np.testing.assert_allclose(align_ev["center_port"], [0.15, 0.7])
        np.testing.assert_allclose(align_ev["center_port_exit"], [0.25, 0.8])
        np.testing.assert_allclose(align_ev["left_port"], [0.3])
        np.testing.assert_allclose(align_ev["left_port_exit"], [0.4])
        np.testing.assert_allclose(align_ev["right_port"], [0.5])
        np.testing.assert_allclose(align_ev["right_port_exit"], [0.6])

    def test_fetch_session_events_raises_for_missing_logical_mapping(self):
        events_rows = [
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "ai0",
                "event_timestamps": np.array([0.1]),
            }
        ]
        mapping_rows = [
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "visual_stim",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "ai0",
            }
        ]

        utils_io = load_utils_io(events_rows, mapping_rows)
        with self.assertRaisesRegex(ValueError, "Missing EventMapping rows"):
            utils_io.fetch_session_events("GRB006", "20240821_121447")

    def test_fetch_session_events_raises_when_source_row_is_missing(self):
        events_rows = [
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "ai0",
                "event_timestamps": np.array([0.1]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "io2",
                "event_timestamps": np.array([0.0, 0.1]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "io3",
                "event_timestamps": np.array([0.0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "io4",
                "event_timestamps": np.array([0.0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "io5",
                "event_timestamps": np.array([0.0]),
            },
            {
                "dataset_name": "ephys_g0",
                "stream_name": "nidq",
                "event_name": "io6",
                "event_timestamps": np.array([0.0]),
            },
        ]
        mapping_rows = [
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "visual_stim",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "io0",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "trial_start",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "io2",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "frames",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "io3",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "left_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "io4",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "center_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "io5",
            },
            {
                "subject_name": "GRB006",
                "session_name": "20240821_121447",
                "event_name": "right_port",
                "source_dataset_name": "ephys_g0",
                "source_stream_name": "nidq",
                "source_event_name": "io6",
            },
        ]

        utils_io = load_utils_io(events_rows, mapping_rows)
        with self.assertRaisesRegex(ValueError, "Mapped source row is missing"):
            utils_io.fetch_session_events("GRB006", "20240821_121447")


if __name__ == "__main__":
    unittest.main()
