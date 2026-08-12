from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


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

    def test_kernel_key_source_uses_relation_union_for_nidq_rows(self):
        module = importlib.import_module("labdata_plugin.analysisschema")
        source = MagicMock()
        base = MagicMock()
        bpod = MagicMock()
        nidq = MagicMock()
        union = MagicMock()
        source.aggr.return_value.proj.return_value = base
        base.__mul__.return_value = base
        base.fetch.return_value = [
            {
                "analysis_set_id": "set",
                "subject_name": "GRB006",
                "trialset_description": "visual",
                "kernel_fit_config_id": 0,
            }
        ]
        base.proj.return_value = bpod
        base.__and__.return_value.proj.return_value = nidq
        key_relation = MagicMock()
        key_relation.__and__.side_effect = [bpod, nidq]
        bpod.__add__.return_value = union
        universal_set = MagicMock(side_effect=[source, key_relation])

        with (
            patch.object(module.dj, "U", universal_set),
            patch.object(
                module,
                "BehaviorAnalysisSet",
                types.SimpleNamespace(TrialSet=MagicMock()),
            ),
            patch.object(module, "PsychophysicalKernelFitConfig", MagicMock()),
            patch.object(
                module,
                "_selected_trialset_keys",
                return_value=[
                    {
                        "subject_name": "GRB006",
                        "session_name": "session",
                        "dataset_name": "chipmunk",
                        "trialset_description": "visual",
                    }
                ],
            ),
            patch(
                "behavior_analyses.kernel_timing.available_timing_sources",
                return_value=["nidq", "bpod"],
            ),
        ):
            descriptor = module.PsychophysicalKernel.key_source
            key_source = (
                descriptor.fget(object.__new__(module.PsychophysicalKernel))
                if isinstance(descriptor, property)
                else descriptor
            )

        self.assertTrue(
            all(
                isinstance(arg, str)
                for call in universal_set.call_args_list
                for arg in call.args
            )
        )
        self.assertIn("timing_source", universal_set.call_args_list[-1].args)
        bpod.__add__.assert_called_once_with(nidq)
        self.assertIs(key_source, union)


if __name__ == "__main__":
    unittest.main()
