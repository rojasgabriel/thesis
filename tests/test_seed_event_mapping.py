import importlib
import sys
import types
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np


class FakeDigitalRelation:
    rows: list[dict] = []
    inserted_rows: list[dict] = []
    delete_calls = 0

    def __and__(self, restriction):
        if isinstance(restriction, dict):
            rows = [
                row
                for row in self.rows
                if all(row.get(key) == value for key, value in restriction.items())
            ]
            relation = FakeDigitalRelation()
            relation.rows = rows
            return relation
        return self

    def __len__(self):
        return len(self.rows)

    def fetch1(self):
        return self.rows[0]

    def insert1(self, row, allow_direct_insert=False):
        self.inserted_rows.append(row)

    def delete(self, force=False):
        type(self).delete_calls += 1


def load_seed_module(existing_rows=None):
    FakeDigitalRelation.rows = existing_rows or []
    FakeDigitalRelation.inserted_rows = []
    FakeDigitalRelation.delete_calls = 0

    fake_schema: Any = types.ModuleType("labdata.schema")
    fake_schema.DatasetEvents = types.SimpleNamespace(Digital=FakeDigitalRelation)

    fake_labdata: Any = types.ModuleType("labdata")
    fake_labdata.schema = fake_schema

    fake_analysisschema: Any = types.ModuleType("labdata_plugin.schema")
    fake_analysisschema.EventMapping = lambda: object()

    fake_labdata_plugin: Any = types.ModuleType("labdata_plugin")
    fake_labdata_plugin.schema = fake_analysisschema

    with patch.dict(
        sys.modules,
        {
            "labdata": fake_labdata,
            "labdata.schema": fake_schema,
            "labdata_plugin": fake_labdata_plugin,
            "labdata_plugin.schema": fake_analysisschema,
        },
    ):
        sys.modules.pop("scripts.diagnostics.seed_event_mapping", None)
        import scripts.diagnostics.seed_event_mapping as seed_event_mapping

        return importlib.reload(seed_event_mapping)


class Grb006Ai0InsertTests(unittest.TestCase):
    def test_ai0_absent_inserts_when_apply_is_true(self):
        module = load_seed_module()

        with patch.object(
            module,
            "load_grb006_visual_onsets",
            return_value=np.array([0.1, 0.2]),
        ):
            row = module.insert_grb006_ai0_if_missing(apply=True)

        self.assertEqual(FakeDigitalRelation.delete_calls, 0)
        self.assertEqual(len(FakeDigitalRelation.inserted_rows), 1)
        self.assertEqual(row["event_name"], "ai0")
        np.testing.assert_allclose(row["event_timestamps"], [0.1, 0.2])

    def test_ai0_present_is_kept_by_default(self):
        existing = {
            "subject_name": "GRB006",
            "session_name": "20240821_121447",
            "dataset_name": "ephys_g0",
            "stream_name": "nidq",
            "event_name": "ai0",
            "event_timestamps": np.array([9.0]),
            "event_values": np.array([1], dtype=np.uint8),
        }
        module = load_seed_module(existing_rows=[existing])

        with patch.object(
            module,
            "load_grb006_visual_onsets",
            return_value=np.array([0.1, 0.2]),
        ):
            row = module.insert_grb006_ai0_if_missing(apply=True)

        self.assertEqual(FakeDigitalRelation.delete_calls, 0)
        self.assertEqual(len(FakeDigitalRelation.inserted_rows), 0)
        np.testing.assert_allclose(row["event_timestamps"], [9.0])

    def test_ai0_present_can_be_replaced_explicitly(self):
        existing = {
            "subject_name": "GRB006",
            "session_name": "20240821_121447",
            "dataset_name": "ephys_g0",
            "stream_name": "nidq",
            "event_name": "ai0",
            "event_timestamps": np.array([9.0]),
            "event_values": np.array([1], dtype=np.uint8),
        }
        module = load_seed_module(existing_rows=[existing])

        with patch.object(
            module,
            "load_grb006_visual_onsets",
            return_value=np.array([0.1, 0.2]),
        ):
            row = module.insert_grb006_ai0_if_missing(
                apply=True,
                replace_existing_ai0=True,
            )

        self.assertEqual(FakeDigitalRelation.delete_calls, 1)
        self.assertEqual(len(FakeDigitalRelation.inserted_rows), 1)
        np.testing.assert_allclose(row["event_timestamps"], [0.1, 0.2])

    def test_replace_dry_run_does_not_write(self):
        existing = {
            "subject_name": "GRB006",
            "session_name": "20240821_121447",
            "dataset_name": "ephys_g0",
            "stream_name": "nidq",
            "event_name": "ai0",
            "event_timestamps": np.array([9.0]),
            "event_values": np.array([1], dtype=np.uint8),
        }
        module = load_seed_module(existing_rows=[existing])

        with patch.object(
            module,
            "load_grb006_visual_onsets",
            return_value=np.array([0.1, 0.2]),
        ):
            row = module.insert_grb006_ai0_if_missing(
                apply=False,
                replace_existing_ai0=True,
            )

        self.assertEqual(FakeDigitalRelation.delete_calls, 0)
        self.assertEqual(len(FakeDigitalRelation.inserted_rows), 0)
        np.testing.assert_allclose(row["event_timestamps"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
