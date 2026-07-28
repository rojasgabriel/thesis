from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Any


CHIPMUNK_PLUGIN_PATH = Path("/Users/gabriel/labdata/plugins/chipmunk")


def get_chipmunk_table() -> Any:
    """Return the labdata Chipmunk plugin table.

    The preferred runtime path is the plugin alias `from chipmunk import
    Chipmunk`. A local fallback loads Gabriel's checked-out labdata plugin
    without importing this repo's `labdata_plugin` package by mistake.
    """
    try:
        from chipmunk import Chipmunk

        return Chipmunk
    except ModuleNotFoundError:
        return _load_local_chipmunk_plugin().Chipmunk


def _load_local_chipmunk_plugin() -> Any:
    module_name = "_behavior_analyses_chipmunk_plugin"
    if module_name in sys.modules:
        return sys.modules[module_name]

    init_path = CHIPMUNK_PLUGIN_PATH / "labdata_plugin" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load Chipmunk plugin from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
