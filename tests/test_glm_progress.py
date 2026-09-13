from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from thesis.ephys.analyses.glm import _artifact_lock, run_over_units


def test_unit_progress_distinguishes_cache_and_failure() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cached = root / "unit_1.json"
        cached.write_text('{"record_version": 2}')
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


if __name__ == "__main__":
    test_unit_progress_distinguishes_cache_and_failure()
    test_artifact_lock_rejects_a_second_command()
