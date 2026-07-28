from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT))


class FakeRelation:
    def __and__(self, _other):
        return self

    def __sub__(self, _other):
        return self


class FakeTable(FakeRelation):
    key_source = FakeRelation()

    def __call__(self):
        return self


class FakeSchema:
    def __call__(self, cls):
        return cls


class SchemaImportTests(unittest.TestCase):
    def test_analysis_schema_imports_with_fake_labdata(self):
        fake_dj = types.ModuleType("datajoint")
        fake_dj.Manual = FakeTable
        fake_dj.Computed = FakeTable
        fake_dj.Part = FakeTable

        fake_schema = types.ModuleType("labdata.schema")
        fake_schema.DecisionTask = types.SimpleNamespace(TrialSet=FakeTable())
        fake_schema.Session = FakeTable
        fake_schema.Subject = FakeTable
        fake_schema.get_user_schema = lambda: FakeSchema()

        fake_labdata = types.ModuleType("labdata")
        fake_labdata.schema = fake_schema

        with patch.dict(
            sys.modules,
            {
                "datajoint": fake_dj,
                "labdata": fake_labdata,
                "labdata.schema": fake_schema,
            },
        ):
            sys.modules.pop("labdata_plugin.analysisschema", None)
            module = importlib.import_module("labdata_plugin.analysisschema")
            module = importlib.reload(module)

        self.assertTrue(hasattr(module, "BehaviorSessionSet"))
        self.assertTrue(hasattr(module, "PsychophysicalKernel"))


if __name__ == "__main__":
    unittest.main()
