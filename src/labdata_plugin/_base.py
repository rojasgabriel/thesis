"""Shared header of the thesis plugin: the user schema and the labdata tables it uses."""

import sys
import warnings

import thesis
from thesis.ephys.preprocessing._database import read_only_database_imports

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
# labdata puts each plugin in sys.modules under its preference name. A plugin
# named "thesis" would hide this package from later imports such as
# thesis.plotting, so put the package back.
sys.modules["thesis"] = thesis

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
