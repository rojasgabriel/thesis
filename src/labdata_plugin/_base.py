"""Shared header of the thesis plugin: the user schema and the labdata tables it uses."""

import importlib
import sys
import warnings
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "thesis"


def _restore_thesis() -> None:
    """Make `thesis` this package again if a labdata plugin took its name.

    labdata puts each plugin in sys.modules under its preference name. A plugin
    named "thesis" hides this package from imports such as thesis.plotting,
    before or after this module imports labdata. Loaded submodules stay as they are.
    """
    module = sys.modules.get("thesis")
    if (
        module
        and Path(getattr(module, "__file__", "") or ".").resolve().parent == _PACKAGE
    ):
        return
    sys.modules.pop("thesis", None)
    importlib.import_module("thesis")


_restore_thesis()

from thesis.ephys.preprocessing._database import (  # noqa: E402
    read_only_database_imports,
)

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

import datajoint as dj  # noqa: E402
import numpy as np  # noqa: E402

with read_only_database_imports():
    from labdata.schema import (
        AnalysisFile,
        EphysRecording,
        Session,
        SpikeSorting,
        UnitCount,
        dataschema,
    )
_restore_thesis()

_DATABASE = f"{dataschema.database}_user"
rojasbowe_schema = dj.Schema(
    connection=dataschema.connection, create_schema=False, create_tables=False
)

__all__ = [
    "AnalysisFile",
    "EphysRecording",
    "Session",
    "SpikeSorting",
    "UnitCount",
    "dataschema",
    "dj",
    "np",
    "rojasbowe_schema",
]
