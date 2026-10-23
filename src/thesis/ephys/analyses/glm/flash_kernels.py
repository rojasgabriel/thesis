"""Save the flash-kernel response of every unit, for the grouping of units.

The sessions of the two-animal figures (`SESSIONS`) are also defined here.

No new fit; the kernels are from the final refit of each unit (`fit.py`).

- Kernel of a condition (`results.flash_kernel_at`): the flash kernel, plus
  the short-gap modifier for a gap up to 50 ms, plus the after-exit modifier
  for a flash after the center exit. Conditions (`CONDITIONS`): in port and
  after exit, at the 40 ms and the 80+ ms gap.
- Response of a unit in a condition: the mean of its kernel from
  `RESPONSE_MS` after the flash (Δ log rate).
- Fixation response: the mean response over the two in-port conditions.
  `flash_by_gap.py` uses it to sort units into the units that fire more and
  less after a flash (`KIND_LIMIT`).

Units (`--units`) are those of `fit.py`. The script writes `flash_kernels.npz`
in each session directory.

    uv run python -m thesis.ephys.analyses.glm.flash_kernels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.results import flash_kernel_at

# The sessions of the two-animal figures. Each test is within one animal.
SESSIONS = ("GRB006_20240821_121447", "GRB064_20260916_132939")

# name: (gap in ms that selects the gap class, after the center exit)
CONDITIONS = {
    "in port, 40 ms gap": (40, False),
    "in port, 80+ ms gap": (100, False),
    "after exit, 40 ms gap": (40, True),
    "after exit, 80+ ms gap": (100, True),
}
FIXATION = ("in port, 40 ms gap", "in port, 80+ ms gap")
RESPONSE_MS = (40, 100)
KIND_LIMIT = 0.2


def session_kernels(root: Path, unit_set: str) -> Path:
    """Write the kernel response of each unit of one session to `flash_kernels.npz`."""
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names = metadata["column_names"]
    scale = np.asarray(metadata["training_scale"])
    basis = dmat.task_temporal_bases()[dmat.FLASH_AFTER_EXIT]
    lags = 1000 * basis.basis_time
    window = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    columns = {
        term: [
            i for i, column in enumerate(names) if column.startswith(f"{term}_basis")
        ]
        for term in (dmat.FLASH_COMMON, dmat.FLASH_JUST_BEFORE, dmat.FLASH_AFTER_EXIT)
    }

    def response(coefficients) -> np.ndarray:
        """Return the mean kernel of each condition in the response window."""
        weights = np.asarray(coefficients, dtype=float)
        terms = {
            term: (lags, (weights[index] / scale[index]) @ basis.basis.T)
            for term, index in columns.items()
        }
        return np.stack(
            [flash_kernel_at(terms, gap, after) for gap, after in CONDITIONS.values()]
        )[:, window].mean(-1)

    final = load_unit_records(
        model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
    )
    responses = np.asarray([response(item["coefficients"]) for item in final])
    fixation = [list(CONDITIONS).index(name) for name in FIXATION]
    path = root / "flash_kernels.npz"
    np.savez(
        path,
        unit_ids=np.asarray([int(item["unit_id"]) for item in final]),
        conditions=np.asarray(list(CONDITIONS)),
        response=responses,
        fixation_response=responses[:, fixation].mean(1),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    for session in SESSIONS:
        print(session_kernels(args.out / session, args.units))


if __name__ == "__main__":
    main()
