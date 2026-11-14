"""Save the flash-kernel response of every unit, for the grouping of units.

The sessions of the two-animal figures (`SESSIONS`) are imported here from
`behavior.trials`.

No new fit; the kernels are from the final refit of each unit (`fit.py`).

- Kernel of a condition (`results.flash_kernel_at`): the flash kernel, plus
  the short-gap modifier for a gap up to 50 ms, plus the after-exit modifier
  for a flash after the center exit. Conditions (`CONDITIONS`): in port and
  after exit, at the 40 ms and the 80+ ms gap.
- Response of a unit in a condition: the mean of its kernel from
  `RESPONSE_MS` after the flash (Δ log rate).
- Peak of a condition (`peak`): the mean kernel within `peaks.PEAK_HALF_MS`
  of its largest value from 30 to 120 ms (`peaks.window_peak`, 1-ms lags),
  the peak measure of the observed and model rates.
- Dip of a condition (`dip`): the same window around its most negative value,
  for the flash-suppressed units.
- Spread of a response or peak (`response_sd`, `peak_sd`): its SD over the
  cross-validation fold fits.
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

from thesis.behavior.trials import SESSIONS
from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.peaks import window_peak
from thesis.ephys.analyses.glm.results import flash_kernel_at

# The sessions of the two-animal figures (`behavior.trials.RECORDED`). Each test
# is within one animal. GRB064: 20261008 since 2026-11-01 (9 rates, the decoder
# session; until then 20260916, which has only 4 extreme rates; old results in
# results/glm/_before_grb064_switch_2026_11_01/).

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
        for term in (
            dmat.FLASH_COMMON,
            dmat.FLASH_JUST_BEFORE,
            dmat.FLASH_AFTER_EXIT,
            dmat.FLASH_RECOVERY,
        )
    }
    columns = {term: index for term, index in columns.items() if index}
    center = metadata.get("flash_recovery_center")

    def response(coefficients) -> np.ndarray:
        """Return the mean kernel, its window peak and its window dip, (3, condition)."""
        weights = np.asarray(coefficients, dtype=float)
        terms = {
            term: (lags, (weights[index] / scale[index]) @ basis.basis.T)
            for term, index in columns.items()
        }
        curves = np.stack(
            [
                flash_kernel_at(terms, gap, after, center)
                for gap, after in CONDITIONS.values()
            ]
        )
        return np.stack(
            [
                curves[:, window].mean(-1),
                window_peak(curves, lags),
                -window_peak(-curves, lags),
            ]
        )

    final = load_unit_records(
        model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
    )
    responses = np.asarray([response(item["coefficients"]) for item in final])
    folds = {
        int(item["unit_id"]): item["folds"]
        for item in load_unit_records(
            model.fit_dir(unit_set) / "checkpoints", "unit_*_folds.json"
        )
    }
    spread = np.asarray(
        [
            np.std(
                [response(f["coefficients"]) for f in folds[int(item["unit_id"])]], 0
            )
            for item in final
        ]
    )
    fixation = [list(CONDITIONS).index(name) for name in FIXATION]
    path = root / "flash_kernels.npz"
    np.savez(
        path,
        unit_ids=np.asarray([int(item["unit_id"]) for item in final]),
        conditions=np.asarray(list(CONDITIONS)),
        response=responses[:, 0],
        response_sd=spread[:, 0],
        peak=responses[:, 1],
        peak_sd=spread[:, 1],
        dip=responses[:, 2],
        dip_sd=spread[:, 2],
        fixation_response=responses[:, 0, fixation].mean(1),
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
