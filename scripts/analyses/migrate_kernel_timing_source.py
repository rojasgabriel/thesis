from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


CONFIG_TABLE = "#psychophysical_kernel_fit_config"
KERNEL_TABLE = "__psychophysical_kernel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Recreate the disposable kernel tables with the canonical schema.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import datajoint as dj
    import labdata.schema as labdata_schema

    connection = dj.conn()
    database = f"{labdata_schema.dbase_name}_user"
    statements = _reset_statements(database)

    if not args.apply:
        for table in (KERNEL_TABLE, CONFIG_TABLE):
            count = connection.query(
                f"SELECT COUNT(*) FROM `{database}`.`{table}`"
            ).fetchone()[0]
            print(f"{table}: {count} rows will be deleted")
        for statement in statements:
            print(statement)
        print("Dry run only. Re-run with --apply after exact live-write approval.")
        return

    for statement in statements:
        connection.query(statement)

    from labdata_plugin.analysisschema import PsychophysicalKernelFitConfig

    PsychophysicalKernelFitConfig.insert(
        PsychophysicalKernelFitConfig.contents,
        skip_duplicates=True,
    )
    _validate_schema(connection, database)
    print("Kernel schema reset complete.")


def _reset_statements(database: str) -> list[str]:
    return [
        f"DROP TABLE IF EXISTS `{database}`.`{KERNEL_TABLE}`",
        f"DROP TABLE IF EXISTS `{database}`.`{CONFIG_TABLE}`",
    ]


def _existing_columns(connection, database: str, table: str) -> set[str]:
    return {
        row[0]
        for row in connection.query(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema={database!r} AND table_name={table!r}"
        ).fetchall()
    }


def _validate_schema(connection, database: str) -> None:
    config_columns = _existing_columns(connection, database, CONFIG_TABLE)
    required_config_columns = {
        "kernel_fit_config_id",
        "timebins",
        "binning_method",
        "bin_width_s",
        "observation_window",
        "evidence_model",
        "min_trials_per_bin",
        "cv_splits",
        "random_state",
        "regularization_c",
    }
    if config_columns != required_config_columns:
        raise RuntimeError(
            "Kernel config schema mismatch: "
            f"expected={sorted(required_config_columns)}, "
            f"found={sorted(config_columns)}"
        )

    kernel_columns = _existing_columns(connection, database, KERNEL_TABLE)
    required_kernel_columns = {
        "timing_source",
        "n_bins_fit",
        "n_observed_per_bin",
        "bin_centers_s",
        "majority_accuracy",
        "score_above_majority",
        "interpretation",
    }
    missing_kernel_columns = required_kernel_columns - kernel_columns
    if missing_kernel_columns:
        raise RuntimeError(
            f"Kernel result schema is missing: {sorted(missing_kernel_columns)}"
        )

    rows = connection.query(
        "SELECT `kernel_fit_config_id`, `binning_method`, "
        "`observation_window`, `evidence_model` "
        f"FROM `{database}`.`{CONFIG_TABLE}` ORDER BY `kernel_fit_config_id`"
    ).fetchall()
    expected = [
        (0, "fixed_width", "center_exit", "trial_rate_residual"),
        (1, "fixed_width", "response", "trial_rate_residual"),
    ]
    if list(rows) != expected:
        raise RuntimeError(
            f"Kernel config rows mismatch: expected={expected}, found={rows}"
        )


if __name__ == "__main__":
    main()
