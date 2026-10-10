# Supplementary figures: how the V1 GLM is constructed

Explainer figures for the design of the V1 spike-prediction GLM. Each script is
self-contained, builds its figure from the real basis definitions in
`src/thesis/ephys/analyses/glm/dmat.py` (no fit needed), and writes its SVG next to
itself. Run any of them from the repo root:

```bash
uv run python figures/supp/regressor_overview.py
uv run python figures/supp/flash_interaction_design.py
uv run python figures/supp/flash_basis_columns.py
```

- `regressor_overview` — every regressor group (flash, task events, motion-energy
  PC, spike history) with its temporal basis, window, coding, and alignment.
- `flash_interaction_design` — three of the four flash regressors for one example
  trial (it does not show the flash × motion-energy regressor),
  showing they are the same 6-column basis placed at the same flash times, each
  regressor scaled by its pre/post-withdrawal contrast or centered adaptation resource.
  The one-trial example centers the resource over its own flashes; the fitted
  model centers over all modeled flashes in the session.
- `flash_basis_columns` — one flash expands into 6 basis columns; the diagonal
  "gradient" is the basis tiling time, not model weight.

## Interactive version

An interactive companion to these figures lets you move and scale the flash basis
set and watch the three regressors rebuild live (basis count, window, bump
width, linear/log spacing, flash count/ISI, withdrawal timing):

https://claude.ai/artifact/6xozqk84PStgC7UvL4q7uA

It is a standalone page (no fit or data needed). Colour shows the design value, a
fixed input — not a fitted weight.
