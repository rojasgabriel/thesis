from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configured_chipmunk_plugin_path() -> Path | None:
    env_path = os.environ.get("CHIPMUNK_PLUGIN_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()

    pyproject = _repo_root() / "pyproject.toml"
    if not pyproject.exists():
        return None

    in_section = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == "[tool.behavior_analyses]"
            continue
        if not in_section or not stripped.startswith("chipmunk_plugin_path"):
            continue
        _, _, value = stripped.partition("=")
        value = value.strip().strip("\"'")
        if value:
            return Path(value).expanduser()
    return None


def get_chipmunk_table() -> Any:
    """Return the labdata Chipmunk plugin table.

    Preferred path: ``from chipmunk import Chipmunk`` (plugin entry point).
    Optional fallback: load a local checkout via ``CHIPMUNK_PLUGIN_PATH`` or
    ``tool.behavior_analyses.chipmunk_plugin_path`` in ``pyproject.toml``.
    """
    try:
        from chipmunk import Chipmunk

        return Chipmunk
    except ModuleNotFoundError:
        plugin_path = _configured_chipmunk_plugin_path()
        if plugin_path is None:
            raise ModuleNotFoundError(
                "Chipmunk plugin not importable as `chipmunk`, and no "
                "CHIPMUNK_PLUGIN_PATH / tool.behavior_analyses.chipmunk_plugin_path "
                "is configured."
            ) from None
        return _load_local_chipmunk_plugin(plugin_path).Chipmunk


def _load_local_chipmunk_plugin(plugin_root: Path) -> Any:
    module_name = "_behavior_analyses_chipmunk_plugin"
    if module_name in sys.modules:
        return sys.modules[module_name]

    init_path = plugin_root / "labdata_plugin" / "__init__.py"
    if not init_path.exists():
        alt = plugin_root / "__init__.py"
        if alt.exists() and plugin_root.name == "labdata_plugin":
            init_path = alt
            search_locations = [str(plugin_root)]
        else:
            raise ModuleNotFoundError(
                f"Could not find Chipmunk plugin at {plugin_root}; expected "
                f"{plugin_root / 'labdata_plugin' / '__init__.py'}"
            )
    else:
        search_locations = [str(init_path.parent)]

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load Chipmunk plugin from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
