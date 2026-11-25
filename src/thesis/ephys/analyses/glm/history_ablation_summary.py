"""Summarize paired fixed-weight history removal on original flash pairs.

Use the saved matched trials, peak windows, response censoring and excitatory
unit selection from history_ablation.py. Bootstrap neurons within mouse and,
for corrected log ratios, resample the corresponding null estimates.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thesis.ephys.analyses.glm import history_sensitivity as hs
from thesis.ephys.analyses.glm import leak_test
from thesis.ephys.analyses.glm.short_history_summary import draw, paired
from thesis.plotting import FIGURE_STYLE, save_figure


def summarize(source, out, corrected):
    rng = np.random.default_rng(12)
    main = json.loads((source / "flash_movement.json").read_text())["sessions"]
    excited = json.loads((source / "flash_excited_units.json").read_text())
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
        for ax, s, letter in zip(axs, hs.SESSIONS, "ab", strict=True):
            mouse = s.split("_")[0]
            ref = main[mouse]
            ids = excited[s]
            a = []
            b = []
            mc = []
            null_max_rates = []
            with np.load(source / s / "power_test_held_out.npz") as z:
                old = dict(
                    zip(
                        z["unit_ids"].tolist(),
                        z["fit_peaks"][:, z["alphas"].tolist().index(0.0)],
                        strict=True,
                    )
                )
            for u in ids:
                folder = out / s / f"unit_{u}"
                r = json.loads((folder / "real.json").read_text())
                idx = ref["unit_ids"].index(u)
                scale = ref["normalization_rate_spikes_s"][idx]
                p = np.array([r["current"], r["ablated"]])
                np.testing.assert_allclose(
                    p[0], np.array(ref["rate_peaks"])[idx, :, 1:], rtol=2e-5, atol=2e-4
                )
                log_ratio = leak_test.log_part(p)[:, 0]
                n = leak_test.comparison_part(p)[:, 0] / scale
                draws = np.broadcast_to(log_ratio, (2000, 2)).copy()
                if corrected:
                    trial = json.loads((folder / "complete.json").read_text())
                    assert len(trial["null_peaks"]) == 20
                    null_max_rates.append(trial["max_expected_hz"])
                    for i, (peaks, norm) in enumerate(
                        ((old[u], 1), (np.array(trial["null_peaks"]), scale))
                    ):
                        vals = leak_test.log_part(peaks)[:, 0]
                        finite = vals[np.isfinite(vals)]
                        log_ratio[i] -= (
                            np.median(finite)
                            if len(finite) >= len(vals) / 2
                            else np.nan
                        )
                        n[i] -= np.median(leak_test.comparison_part(peaks)[:, 0]) / norm
                        draws[:, i] = (
                            log_ratio[i]
                            + np.median(finite)
                            - np.median(rng.choice(finite, (2000, len(finite))), axis=1)
                            if np.isfinite(log_ratio[i])
                            else np.nan
                        )
                a.append(log_ratio)
                b.append(n)
                mc.append(draws)
            a, b = np.array(a), np.array(b)
            logs = paired(a, rng, np.transpose(mc, (1, 0, 2)) if corrected else None)
            report[mouse] = {
                "log": logs,
                "normalized": paired(b, rng),
                "unit_ids": ids,
                "max_null_expected_hz": max(null_max_rates, default=None),
                "log_values": np.where(np.isfinite(a), a, None).tolist(),
                "normalized_values": b.tolist(),
            }
            draw(ax, a, logs, rng)
            ax.set_xticklabels(["Observed history", "History off"])
            ax.set_ylabel(
                ("Corrected" if corrected else "Uncorrected")
                + " flash × exit log ratio"
            )
            ax.axhline(0, color="black", lw=0.6)
            ax.text(
                0.025,
                1.02,
                f"{mouse} · {logs['n']} pairs",
                transform=ax.transAxes,
                va="bottom",
            )
            ax.text(-0.14, 1.04, letter, transform=ax.transAxes, weight="bold")
        name = "corrected" if corrected else "direct"
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        save_figure(fig, out / f"{name}.svg")
        (out / f"{name}_summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        for s, r in report.items():
            print(
                s,
                {
                    k: {x: y for x, y in r[k].items() if x != "keep"}
                    for k in ("log", "normalized")
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corrected", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.out, args.corrected)
