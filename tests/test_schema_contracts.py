import importlib
import sys
import types
import unittest
from typing import Any
from unittest.mock import patch


class FakeRelation:
    def __init__(self, restrictions=()):
        self.restrictions = restrictions

    def __and__(self, restriction):
        return FakeRelation((*self.restrictions, restriction))

    def proj(self, **_renames):
        return self


def load_schema():
    fake_dj: Any = types.ModuleType("datajoint")
    fake_dj.Manual = type("Manual", (), {})
    fake_dj.Computed = type("Computed", (), {})

    fake_schema: Any = types.ModuleType("labdata.schema")
    fake_schema.DatasetEvents = types.SimpleNamespace(Digital=FakeRelation())
    fake_schema.EphysRecording = types.SimpleNamespace()
    fake_schema.Session = FakeRelation()
    fake_schema.SpikeSorting = FakeRelation()
    fake_schema.UnitCount = types.SimpleNamespace(Unit=FakeRelation())
    fake_schema.get_user_schema = lambda: lambda cls: cls

    fake_labdata: Any = types.ModuleType("labdata")
    fake_labdata.schema = fake_schema

    sys.modules.pop("labdata_plugin.schema", None)
    with patch.dict(
        sys.modules,
        {
            "datajoint": fake_dj,
            "labdata": fake_labdata,
            "labdata.schema": fake_schema,
        },
    ):
        return importlib.import_module("labdata_plugin.schema")


class SchemaContractTests(unittest.TestCase):
    def test_package_import_does_not_register_schema(self):
        sys.modules.pop("labdata_plugin.schema", None)
        package = importlib.reload(importlib.import_module("labdata_plugin"))

        self.assertEqual(package.__all__ if hasattr(package, "__all__") else [], [])
        self.assertNotIn("labdata_plugin.schema", sys.modules)

    def test_protected_table_definitions_are_unchanged(self):
        module = load_schema()

        self.assertEqual(
            module.EventMapping.definition.strip(),
            """-> Session
    event_name                           : varchar(54)   # shared logical event role
    ---
    -> DatasetEvents.Digital.proj(source_dataset_name='dataset_name', source_stream_name='stream_name', source_event_name='event_name')""",
        )
        self.assertEqual(
            module.LocomotionPeaks.definition.strip(),
            """-> UnitCount.Unit
    ---
    stat_peak       : float  # peak amplitude of stat event (sp/s)
    stat_latency    : float  # latency of stat event (s)
    move_peak       : float  # peak amplitude of move event (sp/s)
    move_latency    : float  # latency of move event (s)""",
        )
        self.assertEqual(
            module.LocomotionPeaks.key_source.restrictions,
            ("unit_criteria_id = 1", "passes = 1"),
        )
        self.assertFalse(hasattr(module.LocomotionPeaks, "plot"))


if __name__ == "__main__":
    unittest.main()
