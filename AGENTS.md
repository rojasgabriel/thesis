# Project instructions

Behavior and electrophysiology analysis code for Churchland Lab thesis work.

- Inspect the current branch, working tree, code, and notebooks first.
- Treat repository code and docs as technical truth. Use Notion for
  current priorities and decisions.
- Read `ANALYSES.md` for the maintained scientific surface.
- Keep domain code under `src/thesis/<domain>/`; do not keep empty domain
  folders.
- Keep complete ephys workflows under `src/thesis/ephys/analyses/` and
  interactive entry points under `src/thesis/ephys/tools/`. Run them as
  modules with `python -m`. Keep notebooks exploratory; do not import from them.
- Do not add machine-local paths or sibling-checkout dependencies.

Keep analysis code explicit and reproducible. State assumptions and the
scientific comparison. Report what ran, and separate results from
interpretation. Make the smallest complete change. Prefer simple fixes over complex ones. Do not overengineer things. This repo is managed by one person and their thesis project, not a software engineering team with an app in production.

Run the narrowest relevant checks, then:

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
```
