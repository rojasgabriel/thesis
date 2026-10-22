"""Keep database imports read-only until a command explicitly applies changes."""

from contextlib import contextmanager


def initialize_tables(*tables):
    """Declare missing tables, in dependency order, for an explicit apply call."""
    for table in tables:
        instance = table()
        if not instance.is_declared:
            instance.declare(table.declaration_context)
            instance.connection.dependencies.clear()


@contextmanager
def read_only_database_imports():
    """Bind LabData tables without creating schemas, tables, or lookup rows.

    DataJoint's creation flags do not disable automatic Lookup.contents inserts.
    Suppress those defaults only while each table is bound, then restore them.
    This context is for the single-threaded import stage, before workers start.
    """
    import datajoint as dj

    factory = dj.schema
    decorate = dj.Schema._decorate_table

    def schema(*args, **kwargs):
        kwargs.update(create_schema=False, create_tables=False)
        return factory(*args, **kwargs)

    def bind(self, table_class, *args, **kwargs):
        own_contents = table_class.__dict__.get("contents")
        has_own_contents = "contents" in table_class.__dict__
        if issubclass(table_class, dj.Lookup):
            table_class.contents = ()
        try:
            return decorate(self, table_class, *args, **kwargs)
        finally:
            if issubclass(table_class, dj.Lookup):
                if has_own_contents:
                    table_class.contents = own_contents
                else:
                    del table_class.contents

    dj.schema = schema
    dj.Schema._decorate_table = bind
    try:
        yield
    finally:
        dj.Schema._decorate_table = decorate
        dj.schema = factory
