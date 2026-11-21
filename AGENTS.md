# Project instructions

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

Before writing any code:

- Inspect the current branch, working tree, code, and notebooks first.
- Keep this repository for code, a minimal README, and analysis timelines
  (see below). Put operational procedures, research narratives, and full
  results in Notion.
- Treat repository code as technical truth. Use Notion for
  current priorities and decisions. Notion might not have the most up to date
  information so use conversation context as well to understand goals.
- Keep domain code under `src/thesis/<domain>/`; do not keep empty domain
  folders.
- Keep complete ephys workflows under `src/thesis/ephys/analyses/` and
  interactive entry points under `src/thesis/ephys/tools/`. Run them with
  `uv run <script>` when `pyproject.toml` defines a script, otherwise with
  `uv run python -m <module>`.
- Keep notebooks exploratory.
- Do not add machine-local paths or sibling-checkout dependencies.

When changing an analysis:

- Preserve the scientific question. Remove options or panels that do not help
  answer it.
- State the comparison, unit-selection rules, baseline, event alignment,
  exclusions, and sampling unit in the module docstring.
- Preserve paired observations when the experimental design is paired.
- Show individual observations with the appropriate summary. If both plotted
  variables are estimates, show uncertainty in both directions.
- A display limit can hide outliers, but it must not silently remove them from
  statistics. Inform the researcher about the limit and let them decide.
- Run the real analysis and inspect the rendered figure after plotting changes.
  Static checks alone are not sufficient.

Analysis timelines:

- Every analysis has a timeline file next to its code: `<module>.timeline.md`
  for a module, `timeline.md` in the folder for a package. Create it when you
  create the analysis. It lets a human see when each decision was made and when
  each finding started to show.
- Add an entry in the same commit as the change. Add entries only at the end.
  Do not rewrite old entries. If an entry is wrong, add a `correction` entry
  that names the old date.
- Use this format, newest last:

  ```markdown
  # <analysis name>

  Question: <the scientific question, one sentence>

  ## 2026-11-21 · decision · Gabriel
  What: Use a ±20 ms window around the peak.
  Why: The single-bin maximum is noisy.
  Evidence: <commit, figure path, or number>
  ```

- Entry types: `question` (the question starts or changes), `decision` (a
  method choice, exclusion, or parameter), `finding` (a result with its
  numbers and N), `correction` (an earlier entry was wrong), `dead end` (a
  thing we tried and stopped, and why).
- Name who made the call: the researcher, or the AI with the model name.
  When the AI proposes and the researcher approves, name the researcher.
- Write the first result that suggests a finding, also if it is weak. Give
  the numbers, N units, N mice, and the figure or result path.
- Keep entries short. Link to Notion for the full narrative. Do not copy code.

When making figures:

- Label series directly with matching text colors. Make sure data labels are
  legible and not overlapping with the data.
- Keep Matplotlib spines at their default edge positions. Use `separate_axes`
  only when the origin point of the graph would not represent (0,0).
- Organize panels in scientific order: observation, characterization, then
  experimental control or prediction. Use color for one consistent concept
  throughout a figure, and use meaningful biological or experimental labels
  instead of unit IDs.
- Do not use an overall figure title. Put bold lowercase
  panel letters above and to the left of each panel, outside the plotting area.
- Use short, stable output names instead of encoding every analysis option in the
  filename. Prefer .svg as the output format.
