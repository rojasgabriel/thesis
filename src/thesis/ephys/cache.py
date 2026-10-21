"""Keep database tables in local files, to work with no database connection.

Set the environment variable `THESIS_CACHE_DIR` to a directory. A function
with `@cached` then saves its table there on the first call and reads the file
on the next calls with the same arguments. With the variable not set, the
function reads the database each time. Delete a file to read its table from
the database again.
"""

from __future__ import annotations

import functools
import inspect
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def atomic_output(path: Path):
    """Publish a complete binary file without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
        # Linking a sibling file publishes it atomically and refuses to clobber
        # a file another process finished while this one was still writing.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cached(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        directory = os.environ.get("THESIS_CACHE_DIR")
        if not directory:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = "_".join(f"{name}-{value}" for name, value in bound.arguments.items())
        path = Path(directory) / f"{function.__name__}_{key}.pkl"
        if path.exists():
            return pd.read_pickle(path)
        table = function(*args, **kwargs)
        try:
            with atomic_output(path) as handle:
                table.to_pickle(handle)
        except FileExistsError:
            return pd.read_pickle(path)
        return table

    return wrapper
