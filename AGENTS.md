# Project instructions

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

- Start from the current branch, working tree, scripts, notebooks, and tests.
- Read `README.md` and `docs/MIGRATION.md` before changing data access or the
  package surface.
- Keep reusable code under `src/thesis/behavior/` or `src/thesis/ephys/`.
- Keep scripts grouped as analyses, diagnostics, or tools.
- Keep notebooks exploratory. Do not import code from notebooks.
- Do not add machine-local paths or sibling-checkout dependencies.
- Use Notion for current priorities and decisions, not as a copy of repository
  state.

The live plugin schema is intentionally small. `EventMapping` and
`LocomotionPeaks` are protected. Do not change their definitions or import old
behavior schema modules. Treat a decorated DataJoint module import as a possible
database write.

Prefer readable, explicit, and reproducible code. Keep assumptions visible,
name the scientific comparison, report what ran, and separate results from
interpretation. Make the smallest change that fully answers the question.

Run the narrowest relevant checks, then:

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
uv run python -m unittest discover -s tests -v
```
