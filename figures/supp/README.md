# Supplementary figures: how the V1 GLM is constructed

Explainer figures for the design of the V1 spike-prediction GLM. The script is
self-contained, builds its figure from the real basis definitions in
`src/thesis/ephys/analyses/glm/dmat.py` (no fit needed), and writes its SVG next to
itself. Run it from the repo root:

```bash
uv run python figures/supp/regressor_overview.py
```

- `regressor_overview` — every regressor group (flash, task events, motion-energy
  PC, spike history) with its temporal basis, window, coding, and alignment.
