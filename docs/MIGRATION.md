# Repository migration

LAB-TASKS-522 combines the thesis behavior and electrophysiology code in one
repository. Ephys is the Git base because it owns the current environment, CI,
typing support, hardware metadata, and the two live plugin tables.

## Source revisions

- ephys `main`: `44b8d33af79d19ae1a9d887b2e022e70250999f5`
- behavior_analyses `dev`: `eead29629d9a1b5f40c995ba64423bbf88b654fe`
- recovery tags: `pre-thesis-merge-ephys-20260818` and
  `pre-thesis-merge-behavior-20260818`

Both histories are parents of the merge commit. Old files are available through
Git history and the recovery tags instead of a checked-in archive directory.

## Package changes

- `ephys.src.config.*` became `thesis.ephys.config.*`.
- `ephys.src.utils.*` became `thesis.ephys.utils.*`.
- `behavior_analyses.*` became `thesis.behavior.*`.
- `labdata_plugin.analysisschema` became `labdata_plugin.schema`.

No compatibility aliases were added. All maintained callers changed in the same
merge.

## Removed active surface

The merge does not carry forward the dropped behavior table definitions, their
migration and population scripts, or notebooks that query them. It also removes
duplicate psychometric helpers, stale schema diagrams, the unused SFN module,
and the discarded Stringer/RRR TODO. The pre-merge tags preserve these files.

## Database boundary

This migration is code-only. It does not run DataJoint schema decorators against
the live database. It does not create, alter, seed, populate, rename, or delete
tables. `EventMapping` and `LocomotionPeaks` keep their existing class names,
dependencies, fields, and physical tables.

Any redesigned behavior schema is separate future work.
