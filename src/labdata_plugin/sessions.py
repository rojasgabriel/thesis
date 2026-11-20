"""The sessions of the thesis tables."""

from ._base import Session, dj, rojasbowe_schema  # noqa: F401


@rojasbowe_schema
class ThesisSession(dj.Lookup):
    """The sessions of the thesis tables. GRB061 waits for better data."""

    definition = """
    -> Session
    """
    default_contents = [
        {"subject_name": "GRB006", "session_name": "20240821_121447"},
        {"subject_name": "GRB064", "session_name": "20261008_143359"},
    ]
