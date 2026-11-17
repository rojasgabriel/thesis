# Saved figures and numeric evidence

The SVGs accompany `docs/glm.md` and `docs/glm_checks.md`. They were copied from
existing outputs on 2026-11-16 without changing or refitting the analyses.
Selected numeric reports and the encoding-model caption were copied on the same
date into `reports/` so the main narrative can be checked without lab access.
JSON report trailing newlines are normalized by the repository hooks.

The source checkout was at commit
`7608250672ff9c529279ee8ef14152c1f4ba3417`. Generated outputs are not tracked
there, so this identifies the checkout, not a verified generation commit.
The [manifest](manifest.json) records SHA-256 digests for the saved figures,
reports, and caption. These are reference artifacts; CI does not reproduce fits.

## Cohorts and evidence

| Figure or result | Cohort | Numeric evidence |
| --- | --- | --- |
| Flash observation, matched predictions, isolated kernels | GRB006 / GRB064 / GRB061 | [flash_movement.json](reports/flash_movement.json) |
| Corrected fitted-term contribution | GRB006 / GRB064 / GRB061 | [leak_test.json](reports/leak_test.json), nested `main comparison` summaries |
| Gap/rank gain control | Two or three mice per group | [exit_flash_responses.json](reports/exit_flash_responses.json), `gain_test.*.mice` |
| Power | GRB006 / GRB064 / GRB061 | [power_test.json](reports/power_test.json) |
| First-flash response weights | GRB006 / GRB064 / GRB061 | [flash_weight_check.json](reports/flash_weight_check.json) |
| First-flash model miss | GRB006 / GRB064 / GRB061 | [first_flash_after_exit.json](reports/first_flash_after_exit.json) |
| In-port movement | GRB006 / GRB064 / GRB061 | [flash_movement_in_port.json](reports/flash_movement_in_port.json) |
| Encoding-model overview | GRB006 / GRB064 | [caption](reports/main_figure_caption.md), [inclusion tests](reports/main_figure_stats.json) |
| Raw-video, per-unit gain, recovery, decoding, side, depth, design controls | GRB006 / GRB064, except explicitly identified historical runs | Scope and historical reports are identified in [the supplement](../glm_checks.md) |

GRB061 is provisional because it fails the required-wait criterion. The
three-mouse pooled results include it; they do not establish a result restricted
to expert sessions. Most supporting controls have not been extended to it.
`leak_test.json` retains older two-mouse top-level kernel-validation entries;
its three-mouse nested `main comparison` summaries are the source for the primary
rate test. Do not infer one common cohort from the file's top-level keys.

The paired-peak means and tests were recalculated from all three mice's saved
per-unit peak arrays. Both correction scales, exclusions, and within-mouse tests
for GRB006 and GRB064 were also reproduced from their saved simulation arrays.
The GRB061 `leak_test.npz` simulation archive was absent
locally during reconciliation: its corrected results and the three-mouse pooled
intervals are preserved from the saved JSON report, not independently
recalculated from its simulations. Recover that archive before claiming complete
reproduction of the third mouse's correction.

The 2026-11-16 reconciliation updates cohort counts, panel descriptions, pooled
intervals, and interpretations. The main rate plots remain uncorrected observed
and predicted peaks; the corrected log-ratio results are in the accompanying
text and tables. Nonsignificant tests are not evidence of equivalence.

## Updating these snapshots

Run the relevant analysis with the inputs described in the
[GLM methods](../glm_checks.md#run-and-inspect), inspect the rendered output, and
copy both the figure and its report. Update manifest digests, provenance, cohort
scope, and narrative together. Do not infer the cohort from current code alone:
a saved figure may predate a session-list change.

Raw recordings, model fits, caches, and bulk outputs stay outside this directory.
A fresh query after parser, sync, or response-classification updates can differ
from these saved results. The reports may contain `NaN` for undefined per-unit
quantities; these are missing estimates, not zero effects.
