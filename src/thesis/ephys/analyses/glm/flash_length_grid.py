"""Test the length of the flash kernels of the V1 GLM, with kernels by gap class.

Question: the observed response to a flash is about 200 ms long (a peak near
70 ms, then a dip to about 190 ms; `flash_by_gap.py`), and the flash kernels
are 150 ms long. The first test of the length was on a design with one flash
kernel for all flashes, where a longer kernel did not help: more than half of
the flashes are followed by another one after 40 ms, so the late part of an
average kernel is near zero. With one kernel for each gap class, a longer
kernel can hold the dip after a flash at a long gap. Does it predict better?

Comparison: the final design with the length of every flash kernel changed, on
the fixed 20-unit sample, with the validation D² of each unit. The raised
cosines keep the same spacing (about 19 ms), so a longer kernel has more of
them. The designs are in `DESIGNS` (length in ms: number of raised cosines);
`IN_USE` is the design of the model.

The comparison is paired over units. Where two options do not differ reliably,
the model uses the simpler one. Units, trials, alignment, baseline, and
exclusions are those of `fit.py`.

    uv run python -m thesis.ephys.analyses.glm.flash_length_grid

A design that has saved scores in `flash_length_grid/flash_length_grid.json`
is not fit again. `--samples N` scores each design on N sets of 20 units
(no unit is in two sets); the comparison is then paired over all those units. Each design takes about two minutes.
"""

from __future__ import annotations

from thesis.ephys.analyses.glm import grid

DESIGNS = {
    "150": {"FLASH_WINDOW_S": 0.15, "FLASH_BASES": 8},
    "200": {"FLASH_WINDOW_S": 0.20, "FLASH_BASES": 11},
    "250": {"FLASH_WINDOW_S": 0.25, "FLASH_BASES": 13},
    "300": {"FLASH_WINDOW_S": 0.30, "FLASH_BASES": 16},
}
IN_USE = "150"


if __name__ == "__main__":
    grid.main(DESIGNS, "flash_length_grid", __doc__.splitlines()[0])
