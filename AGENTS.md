# Project instructions

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

- Inspect the current branch, working tree, code, and notebooks first.
- Treat repository code and docs as technical truth. Use Notion for
  current priorities and decisions.
- Read `ANALYSES.md` for the maintained scientific surface.
- Use `docs/MIGRATION.md` only for merge history and recovery references.
- Keep reusable code under `src/thesis/behavior/` or `src/thesis/ephys/`.
- Keep scripts under `scripts/analyses`, `scripts/diagnostics`, or
  `scripts/tools`. Keep notebooks exploratory; do not import from them.
- Do not add machine-local paths or sibling-checkout dependencies.

Keep analysis code explicit and reproducible. State assumptions and the
scientific comparison. Report what ran, and separate results from
interpretation. Make the smallest complete change.

Run the narrowest relevant checks, then:

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
```
