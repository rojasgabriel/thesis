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
    def test_kernel_timing_migration_only_adds_missing_columns(self):
        module = runpy.run_path(str(SCRIPTS / "migrate_kernel_timing_source.py"))

        statements = module["_missing_column_statements"](
            "labdata_user",
            module["KERNEL_TABLE"],
            module["KERNEL_COLUMNS"],
            {"timing_source", "n_trials_fit", "fit_message"},
        )

        self.assertEqual(len(statements), len(module["KERNEL_COLUMNS"]) - 1)
        self.assertTrue(all("timing_source" not in sql for sql in statements))
        self.assertTrue(all(sql.startswith("ALTER TABLE") for sql in statements))

    def test_schema_migration_archive_names_leave_room_for_foreign_keys(self):
        module = runpy.run_path(str(SCRIPTS / "migrate_behavior_analysis_schema.py"))

        self.assertTrue(
            all(
                len(f"{name}_ibfk_99") <= 64
                for name in module["ARCHIVE_TABLES"].values()
            )
        )

    def test_schema_migration_rejects_occupied_target_tables(self):
        module = runpy.run_path(str(SCRIPTS / "migrate_behavior_analysis_schema.py"))
        connection = MagicMock()
        connection.query.return_value.fetchall.return_value = [
            *[(name,) for name in module["ARCHIVE_TABLES"]],
            ("behavior_analysis_set",),
        ]

        with self.assertRaisesRegex(RuntimeError, "occupied_targets"):
            module["_validate_table_state"](connection, "labdata_user")

        expected = MagicMock()
        expected.fetchall.return_value = [(10, 10, 0)]
        incompatible = MagicMock()
        incompatible.fetchall.return_value = [(8, 10, 0)]
        connection.query.side_effect = [expected, incompatible]
        with self.assertRaisesRegex(RuntimeError, "Incompatible legacy kernel"):
            module["_validate_kernel_configs"](connection, "labdata_user")

    def test_schema_migration_accepts_expected_resume_state(self):
        module = runpy.run_path(str(SCRIPTS / "migrate_behavior_analysis_schema.py"))
        connection = MagicMock()
        existing = (
            set(module["ARCHIVE_TABLES"].values())
            | module["NEW_TABLES"]
            | module["COMPUTED_TABLES"]
        )
        connection.query.return_value.fetchall.return_value = [
            (name,) for name in existing
        ]

        module["_validate_resume_state"](connection, "labdata_user")

    def test_schema_migration_deduplicates_direct_trialset_keys(self):
        module = runpy.run_path(str(SCRIPTS / "migrate_behavior_analysis_schema.py"))
        rows = [
            {"subject_name": "GRB001", "session_name": "s1", "set": "a"},
            {"subject_name": "GRB001", "session_name": "s1", "set": "b"},
            {"subject_name": "GRB001", "session_name": "s2", "set": "a"},
        ]

        unique = module["_deduplicate_by_fields"](
            rows, ("subject_name", "session_name")
        )

        self.assertEqual([row["session_name"] for row in unique], ["s1", "s2"])

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
        fake_plugin.BehaviorAnalysisSet = types.SimpleNamespace(
            insert1=_boom,
            TrialSet=types.SimpleNamespace(insert=_boom),
        )

        argv = [
            "seed_behavior_analysis_set.py",
            "--analysis-set-id",
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
                str(SCRIPTS / "seed_behavior_analysis_set.py"), run_name="__main__"
            )

    def test_populate_script_dry_run_reports_pending(self):
        fake_plugin = types.ModuleType("labdata_plugin.analysisschema")
        fake_plugin.BehaviorAnalysisSet = types.SimpleNamespace(
            TrialSet=FakeTrialSet([])
        )
        for name in [
            "PsychometricSessionFit",
            "PsychometricSubjectFit",
            "PsychophysicalKernel",
        ]:
            setattr(fake_plugin, name, type(name, (FakeComputed,), {}))

        argv = [
            "populate_behavior_tables.py",
            "--analysis-set-id",
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

    def test_registered_chipmunk_plugin_is_used(self):
        from behavior_analyses import io as io_mod

        table = object()
        fake_labdata = types.ModuleType("labdata")
        fake_labdata.plugins = {"chipmunk": types.SimpleNamespace(Chipmunk=table)}
        with patch.dict(sys.modules, {"labdata": fake_labdata}):
            registered = io_mod._registered_chipmunk_table()

        self.assertIs(registered, table)


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
