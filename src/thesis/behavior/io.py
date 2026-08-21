from __future__ import annotations

from typing import Any


def get_chipmunk_table() -> Any:
    """Return the installed or registered labdata Chipmunk plugin table."""
    try:
        from chipmunk import Chipmunk

        return Chipmunk
    except ModuleNotFoundError:
        registered = _registered_chipmunk_table()
        if registered is not None:
            return registered
        raise ModuleNotFoundError(
            "Chipmunk plugin is not importable or registered with labdata."
        ) from None


def _registered_chipmunk_table() -> Any | None:
    import labdata

    try:
        return labdata.plugins["chipmunk"].Chipmunk
    except KeyError:
        return None
