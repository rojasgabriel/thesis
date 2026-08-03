from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


CONFIG_TABLE = "#psychophysical_kernel_fit_config"
KERNEL_TABLE = "__psychophysical_kernel"
CONFIG_COLUMNS = (
    (
        "kernel_method",
        "enum('legacy_variable','fixed_window') NOT NULL DEFAULT 'legacy_variable'",
        "regularization_c",
    ),
    ("bin_width_s", "float DEFAULT NULL", "kernel_method"),
    (
        "evidence_encoding",
        "enum('max_rate','trial_rate') DEFAULT NULL",
        "bin_width_s",
    ),
    ("min_trials_per_bin", "int DEFAULT NULL", "evidence_encoding"),
    (
        "observation_window",
        "enum('center_exit','response') DEFAULT NULL",
        "min_trials_per_bin",
    ),
)
KERNEL_COLUMNS = (
    (
        "timing_source",
        "enum('nidaq','bpod','mixed') NOT NULL DEFAULT 'bpod'",
        "fit_message",
    ),
    ("n_bins_fit", "int DEFAULT NULL", "n_trials_fit"),
    ("n_observed_per_bin", "longblob DEFAULT NULL", "n_bins_fit"),
    ("bin_centers_s", "longblob DEFAULT NULL", "n_observed_per_bin"),
    ("majority_accuracy", "float DEFAULT NULL", "score_mean"),
    ("score_above_majority", "float DEFAULT NULL", "majority_accuracy"),
    ("interpretation", "varchar(32) DEFAULT NULL", "bias_mean"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Alter the live kernel tables and insert fixed-window configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import datajoint as dj
    import labdata.schema as labdata_schema

    connection = dj.conn()
    database = f"{labdata_schema.dbase_name}_user"
    table_columns = {
        CONFIG_TABLE: CONFIG_COLUMNS,
        KERNEL_TABLE: KERNEL_COLUMNS,
    }
    existing = {
        table: _existing_columns(connection, database, table) for table in table_columns
    }
    statements = [
        statement
        for table, columns in table_columns.items()
        for statement in _missing_column_statements(
            database, table, columns, existing[table]
        )
    ]

    if not args.apply:
        for statement in statements:
            print(statement)
        if not statements:
            print("Kernel timing-source columns are already present.")
        print("Dry run only. Re-run with --apply after exact live-write approval.")
        return

    for statement in statements:
        connection.query(statement)

    from labdata_plugin.analysisschema import PsychophysicalKernelFitConfig

    PsychophysicalKernelFitConfig.insert(
        PsychophysicalKernelFitConfig.contents,
        skip_duplicates=True,
    )
    _validate_migration(connection, database)
    print("Kernel timing-source schema migration complete.")


def _existing_columns(connection, database: str, table: str) -> set[str]:
    return {
        row[0]
        for row in connection.query(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema={database!r} AND table_name={table!r}"
        ).fetchall()
    }


def _missing_column_statements(
    database: str,
    table: str,
    columns: tuple[tuple[str, str, str], ...],
    existing: set[str],
) -> list[str]:
    return [
        f"ALTER TABLE `{database}`.`{table}` ADD COLUMN `{name}` "
        f"{definition} AFTER `{after}`"
        for name, definition, after in columns
        if name not in existing
    ]


def _validate_migration(connection, database: str) -> None:
    expected = {
        CONFIG_TABLE: {name for name, _, _ in CONFIG_COLUMNS},
        KERNEL_TABLE: {name for name, _, _ in KERNEL_COLUMNS},
    }
    missing = {
        table: sorted(columns - _existing_columns(connection, database, table))
        for table, columns in expected.items()
    }
    missing = {table: columns for table, columns in missing.items() if columns}
    if missing:
        raise RuntimeError(f"Kernel timing-source migration incomplete: {missing}")

    invalid_sources = connection.query(
        f"SELECT COUNT(*) FROM `{database}`.`{KERNEL_TABLE}` "
        "WHERE timing_source NOT IN ('nidaq','bpod','mixed')"
    ).fetchone()[0]
    if invalid_sources:
        raise RuntimeError(
            f"Kernel timing-source migration found {invalid_sources} invalid rows"
        )


if __name__ == "__main__":
    main()
