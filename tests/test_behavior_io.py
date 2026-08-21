import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from thesis.behavior import io


class BehaviorIoTests(unittest.TestCase):
    def test_plugin_path_prefers_environment(self):
        with patch.dict("os.environ", {"CHIPMUNK_PLUGIN_PATH": "/tmp/chipmunk"}):
            self.assertEqual(
                io._configured_chipmunk_plugin_path(), Path("/tmp/chipmunk")
            )

    def test_registered_plugin_is_used(self):
        table = object()
        fake_labdata: Any = types.ModuleType("labdata")
        fake_labdata.plugins = {"chipmunk": types.SimpleNamespace(Chipmunk=table)}
        with patch.dict(sys.modules, {"labdata": fake_labdata}):
            self.assertIs(io._registered_chipmunk_table(), table)


if __name__ == "__main__":
    unittest.main()
