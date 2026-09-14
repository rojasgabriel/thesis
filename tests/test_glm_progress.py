import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from thesis.ephys.analyses.glm import (
    _artifact_lock,
    _common_design_is_current,
    code_version,
    run_over_units,
    task_temporal_bases,
)
from thesis.ephys.analyses.glm_unique import _scored_folds


def test_unit_progress_distinguishes_cache_and_failure() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cached = root / "unit_1.json"
        cached.write_text(f'{{"record_version": {code_version("folds")}}}')
        jobs = [
            {"unit_id": 1, "output": str(cached)},
            {"unit_id": 2, "output": str(root / "unit_2.json")},
        ]

        def task(job: dict) -> dict:
            if job["unit_id"] == 2:
                raise ValueError("test failure")
            return {"unit_id": job["unit_id"]}

        output = StringIO()
        with (
            patch("thesis.ephys.analyses.glm._init_worker"),
            redirect_stdout(output),
        ):
            results = list(
                run_over_units(
                    task, jobs, root / "windows.npz", root / "design.npy", 1, "folds"
                )
            )

    assert results == [({"unit_id": 1}, True), (None, False)]
    assert "Failed unit 2: ValueError: test failure" in output.getvalue()
    assert "Skipped" not in output.getvalue()


def test_artifact_lock_rejects_a_second_command() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        with _artifact_lock(root):
            try:
                with _artifact_lock(root):
                    raise AssertionError("second lock unexpectedly succeeded")
            except RuntimeError as error:
                assert "Another GLM command is already running" in str(error)


def test_unique_pairs_alphas_after_an_empty_fold() -> None:
    partitions = [
        {"test": np.array([True, False, False])},
        {"test": np.array([False, True, False])},
        {"test": np.array([False, False, True])},
    ]
    pairs = _scored_folds(np.array([1, 0, 1]), partitions, [0.1, 0.2])

    assert pairs == [(partitions[0], 0.1), (partitions[2], 0.2)]


def test_old_common_design_is_not_reused() -> None:
    with TemporaryDirectory() as directory:
        design = Path(directory) / "common.npy"
        design.touch()
        bases = task_temporal_bases()
        metadata = {
            "video_columns_per_component": 1,
            "task_manifest": [
                {"name": name, "columns": basis.basis.shape[1]}
                for name, basis in bases.items()
            ],
        }
        design.with_suffix(".json").write_text(json.dumps(metadata))
        assert _common_design_is_current(design)
        metadata["video_columns_per_component"] = 3
        design.with_suffix(".json").write_text(json.dumps(metadata))
        assert not _common_design_is_current(design)


if __name__ == "__main__":
    test_unit_progress_distinguishes_cache_and_failure()
    test_artifact_lock_rejects_a_second_command()
    test_unique_pairs_alphas_after_an_empty_fold()
    test_old_common_design_is_not_reused()
