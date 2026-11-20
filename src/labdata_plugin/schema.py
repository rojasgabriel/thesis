"""All thesis plugin tables in one schema; import tables from here.

The tables live in one module per analysis; each module imports the tables it
depends on. This module binds them to the user schema and creates them on request.
"""

from ._base import _DATABASE, dataschema, rojasbowe_schema
from .behavior import Psychometric
from .events import EventMapping
from .glm import GLMDesignParam, GLMFit
from .sessions import ThesisSession
from .stimulus_response import StimulusResponse, StimulusResponseParam
from .units import (
    MovementArtifact,
    MovementArtifactParam,
    UnitStability,
    UnitStabilityParam,
)

__all__ = [
    "EventMapping",
    "GLMDesignParam",
    "GLMFit",
    "MovementArtifact",
    "MovementArtifactParam",
    "Psychometric",
    "StimulusResponse",
    "StimulusResponseParam",
    "ThesisSession",
    "UnitStability",
    "UnitStabilityParam",
    "initialize_schema",
]


def initialize_schema() -> None:
    """Create plugin tables and seed defaults only for an explicit apply command."""
    was_active = rojasbowe_schema.is_activated()
    rojasbowe_schema.create_schema = True
    rojasbowe_schema.create_tables = True
    rojasbowe_schema.activate(_DATABASE)
    if was_active:
        # activate() returns early for a schema already bound by a reader.
        for table, context in rojasbowe_schema.declare_list:
            rojasbowe_schema._decorate_master(table, context)
    for table in (
        EventMapping,
        UnitStabilityParam,
        StimulusResponseParam,
        MovementArtifactParam,
        ThesisSession,
        GLMDesignParam,
    ):
        table().insert(table.default_contents, skip_duplicates=True)


# Bind existing tables for readers. A new user schema stays dormant until apply.
if dataschema.connection.query(
    "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=%s",
    args=(_DATABASE,),
).fetchone():
    rojasbowe_schema.activate(_DATABASE)
