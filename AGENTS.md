# Project instructions

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

- Inspect the current branch, working tree, code, and notebooks first.
- Treat repository code and docs as technical truth. Use Notion for
  current priorities and decisions.
- Keep domain code under `src/thesis/<domain>/`; do not keep empty domain
  folders.
- Keep complete ephys workflows under `src/thesis/ephys/analyses/` and
  interactive entry points under `src/thesis/ephys/tools/`. Run them as
  modules with `python -m`. Keep notebooks exploratory; do not import from them.
- Do not add machine-local paths or sibling-checkout dependencies.

Keep analysis code explicit and reproducible. State assumptions and the
scientific comparison. Report what ran, and separate results from
interpretation. Make the smallest complete change. Prefer simple fixes over complex ones. Do not overengineer things. This repo is managed by one person and their thesis project, not a software engineering team with an app in production.

When changing an analysis:

- Preserve the scientific question. Remove options or panels that do not help
  answer it.
- State the comparison, unit-selection rules, baseline, event alignment,
  exclusions, and sampling unit in the module docstring.
- Keep discovery datasets separate from follow-up control datasets unless the
  analysis explicitly requires pooling them.
- Preserve paired observations when the experimental design is paired.
- Show individual observations with the appropriate summary. If both plotted
  variables are estimates, show uncertainty in both directions.
- A display limit can hide outliers, but it must not silently remove them from
  statistics. Document the limit.
- Run the real analysis and inspect the rendered figure after plotting changes.
  Static checks alone are not sufficient.

For publication figures, label series directly with matching text colors.
Use a conventional legend only when direct labels would be unclear.
Keep Matplotlib spines at their default edge positions. Use `separate_axes`
only when intentionally trimming the visible axes; do not move spines to the
data origin by default.
Organize panels in scientific order: observation, characterization, then
experimental control or prediction. Use color for one consistent concept
throughout a figure, and use meaningful biological or experimental labels
instead of unit IDs. Do not use an overall figure title. Put bold lowercase
panel letters above and to the left of each panel, outside the plotting area.
Use short, stable output names instead of encoding every analysis option in the
filename.

Run the narrowest relevant checks, then:

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
```
