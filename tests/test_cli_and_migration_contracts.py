from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "analyses"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS))


class _Andable:
    def __and__(self, _other):
        return self

    def __sub__(self, _other):
        return [1, 2]


class FakeTrialSet(_Andable):
    def __init__(self, rows):
        self._rows = rows

    def __call__(self):
        return self

    def fetch(self, *_args, **_kwargs):
        return self._rows


class FakeComputed:
    key_source = _Andable()

    def __call__(self):
        return self

    @staticmethod
    def populate(*_args, **_kwargs):
        raise AssertionError("populate should not run in dry-run")


class CliContractTests(unittest.TestCase):
    def test_seed_script_dry_run_prints_counts_without_insert(self):
        fake_rows = [
            {
                "subject_name": "GRB001",
                "session_name": "20240101_120000",
                "trialset_description": "visual",
            },
            {
                "subject_name": "GRB001",
                "session_name": "20240102_120000",
                "trialset_description": "visual",
            },
        ]

        fake_schema = types.ModuleType("labdata.schema")
        fake_schema.DecisionTask = types.SimpleNamespace(
            TrialSet=FakeTrialSet(fake_rows)
        )

        fake_labdata = types.ModuleType("labdata")
        fake_labdata.schema = fake_schema

        def _boom(*_args, **_kwargs):
            raise AssertionError("database writes should not run in dry-run")

        fake_plugin = types.ModuleType("labdata_plugin.analysisschema")
        fake_plugin.BehaviorSessionSet = types.SimpleNamespace(
            insert1=_boom,
            Session=types.SimpleNamespace(insert=_boom),
            TrialSet=types.SimpleNamespace(insert=_boom),
            SubjectTrialSet=types.SimpleNamespace(insert=_boom),
        )

        argv = [
            "seed_behavior_session_set.py",
            "--session-set-id",
            "test_set",
            "--name",
            "Test",
            "--subjects",
            "GRB001",
            "--dry-run",
        ]
        with (
            patch.dict(
                sys.modules,
                {
                    "labdata": fake_labdata,
                    "labdata.schema": fake_schema,
                    "labdata_plugin": types.ModuleType("labdata_plugin"),
                    "labdata_plugin.analysisschema": fake_plugin,
                },
            ),
            patch.object(sys, "argv", argv),
        ):
            runpy.run_path(
                str(SCRIPTS / "seed_behavior_session_set.py"), run_name="__main__"
            )

    def test_populate_script_dry_run_reports_pending(self):
        fake_plugin = types.ModuleType("labdata_plugin.analysisschema")
        for name in [
            "LearningSessionMetrics",
            "PsychometricSessionFit",
            "PsychometricSubjectFit",
            "PsychophysicalKernel",
        ]:
            setattr(fake_plugin, name, type(name, (FakeComputed,), {}))

        argv = [
            "populate_behavior_tables.py",
            "--session-set-id",
            "test_set",
            "--dry-run",
        ]
        with (
            patch.dict(
                sys.modules,
                {
                    "labdata_plugin": types.ModuleType("labdata_plugin"),
                    "labdata_plugin.analysisschema": fake_plugin,
                },
            ),
            patch.object(sys, "argv", argv),
        ):
            runpy.run_path(
                str(SCRIPTS / "populate_behavior_tables.py"), run_name="__main__"
            )


class IoConfigTests(unittest.TestCase):
    def test_configured_path_reads_env(self):
        from behavior_analyses import io as io_mod

        with patch.dict("os.environ", {"CHIPMUNK_PLUGIN_PATH": "/tmp/chipmunk-plugin"}):
            path = io_mod._configured_chipmunk_plugin_path()
        self.assertEqual(path, Path("/tmp/chipmunk-plugin"))


class PsychometricPlotHelperTests(unittest.TestCase):
    def test_fetch_uses_labdata_rates_and_session_names(self):
        from psychometric_curves import utils

        relation = MagicMock()
        relation.__mul__.return_value = relation
        relation.__and__.return_value = relation
        relation.fetch.return_value = (
            np.array([1, -1, 1]),
            np.array(["visual", "audio", "visual+audio"]),
            np.array([20.0, 8.0, 20.0]),
            np.array([14.0, 20.0, 13.0]),
            np.array([12.0, 10.0, 12.0]),
        )
        part = MagicMock(return_value=relation)
        chipmunk = types.SimpleNamespace(Trial=part, TrialParameters=part)
        with patch.object(utils, "get_chipmunk_table", return_value=chipmunk):
            responses, intensity = utils._fetch_choice_and_stim(
                "GRB006",
                None,
                [datetime(2024, 8, 26, 11, 33, 7)],
            )

        np.testing.assert_array_equal(responses, [1, -1, 1])
        np.testing.assert_allclose(intensity, [2, -2, 1])
        self.assertIn(
            [{"session_name": "20240826_113307"}],
            [call.args[0] for call in relation.__and__.call_args_list],
        )
        self.assertNotIn("stim_rate", relation.fetch.call_args.args)


class NoDjchurchlandImportsTests(unittest.TestCase):
    def test_maintained_python_sources_do_not_import_djchurchland(self):
        import ast

        roots = [
            REPO_ROOT / "src",
            REPO_ROOT / "scripts",
            REPO_ROOT / "labdata_plugin",
            REPO_ROOT / "psychometric_curves" / "utils.py",
            REPO_ROOT / "tests",
        ]
        offenders = []
        for root in roots:
            paths = [root] if root.is_file() else root.rglob("*.py")
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    else:
                        continue
                    if any(
                        name == "djchurchland" or name.startswith("djchurchland.")
                        for name in names
                    ):
                        offenders.append(str(path.relative_to(REPO_ROOT)))
                        break

        for path in [
            REPO_ROOT / "behavioral_metrics" / "plot_learning_curves.ipynb",
            REPO_ROOT / "psychometric_curves" / "plot_psychometric_fits.ipynb",
            REPO_ROOT / "psychophysical_kernels" / "plot_kernels.ipynb",
            REPO_ROOT / "sess.ipynb",
        ]:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            if "djchurchland" in code:
                offenders.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
