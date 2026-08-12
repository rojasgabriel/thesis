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
    def test_locked_datajoint_imports(self):
        import datajoint

        self.assertTrue(hasattr(datajoint, "schema"))

    def test_analysis_schema_imports_with_fake_labdata(self):
        fake_dj = types.ModuleType("datajoint")
        fake_dj.Manual = FakeTable
        fake_dj.Lookup = FakeTable
        fake_dj.Computed = FakeTable
        fake_dj.Part = FakeTable

        fake_schema = types.ModuleType("labdata.schema")
        fake_schema.DecisionTask = types.SimpleNamespace(TrialSet=FakeTable())
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

        self.assertTrue(hasattr(module, "BehaviorAnalysisSet"))
        self.assertTrue(hasattr(module, "PsychometricFitConfig"))
        self.assertTrue(hasattr(module, "PsychophysicalKernelFitConfig"))
        self.assertTrue(hasattr(module, "PsychophysicalKernel"))
        self.assertIn(
            "kernel_fit_config_id                 : int",
            module.PsychophysicalKernelFitConfig.definition,
        )
        self.assertNotIn(
            "analysis_version",
            module.PsychophysicalKernelFitConfig.definition,
        )
        self.assertNotIn(
            "kernel_method",
            module.PsychophysicalKernelFitConfig.definition,
        )
        self.assertNotIn(
            "evidence_encoding",
            module.PsychophysicalKernelFitConfig.definition,
        )
        self.assertEqual(
            [row[0] for row in module.PsychophysicalKernelFitConfig.contents],
            [0, 1],
        )
        self.assertIn(
            "timing_source                        : enum('nidq', 'bpod')",
            module.PsychophysicalKernel.definition.split("---")[0],
        )
        self.assertNotIn("mixed", module.PsychophysicalKernel.definition)
        self.assertFalse(hasattr(module, "LearningSessionMetrics"))


if __name__ == "__main__":
    unittest.main()
