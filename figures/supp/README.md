# Supplementary figures: how the V1 GLM is constructed

Explainer figures for the design of the V1 spike-prediction GLM. Each script is
self-contained, builds its figure from the real basis definitions in
`src/thesis/ephys/analyses/glm/dmat.py` (no fit needed), and writes its SVG next to
itself. Run any of them from the repo root:

```bash
uv run python figures/supp/regressor_overview.py
uv run python figures/supp/flash_basis_columns.py
```

- `regressor_overview` — every regressor group (flash, task events, motion-energy
  PC, spike history) with its temporal basis, window, coding, and alignment.
- `flash_basis_columns` — one flash expands into 6 basis columns; the diagonal
  "gradient" is the basis tiling time, not model weight.
