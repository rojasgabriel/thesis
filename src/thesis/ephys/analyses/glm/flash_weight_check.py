"""Do the flash weights of the GLM agree with the StimulusResponse table?

Question: the StimulusResponse table calls a unit flash-excited from its raw
response to the first flash of each trial. The GLM gives each unit a flash
kernel. If the model reads the flash response correctly, the units the table
calls excited must have positive flash kernels, and the size of the two
responses must go together.

Comparison, for each unit, both in Δ log rate:

- Table: log(rate 30–120 ms / rate −40–0 ms) around the first flash of each
  trial, the windows of StimulusResponse (param 0), from the spikes. The rate
  difference of these windows must equal the stored `rate_delta`. Uncertainty:
  bootstrap SE over trials. Class: the stored response type (excited,
  suppressed, or not responsive).
- Model (a), the flash weights: the flash kernel at the first flash of a trial
  (the kernel of the long-gap class with recovery r = 1, in the port), mean
  over 30–120 ms.
- Model (b), all terms locked to the first flash: (a) plus the fixation kernel
  (it starts at the first flash) over 30–120 ms, minus the pre-stimulus step
  (on from the center entry to the first flash), so the same before/after
  contrast as the table. Spike history is not included.

Model values are from the final refit; uncertainty is the jackknife SE over the
10 cross-validation folds (each fold leaves 10% of the trials out). A unit is
excited by the model when its value minus 1.96 SE is above 0 (suppressed: plus
1.96 SE below 0).

Units: the stable units of the GLM (`fit.py`: unit criteria 1, stability 0,
exclusions). The sampling unit is the unit; each statistic is within one mouse
(AUC of the model value for the table's excited units, with a bootstrap 95%
interval over units; Spearman ρ; cross-table of classes). The figure pools both
mice (marker shape).

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.flash_weight_check
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import fitted_kernels, flash_kernel_at
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    BOOTSTRAPS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
PARAM_ID = 0
BASELINE_S = (-0.04, 0.0)
RESPONSE_S = (0.03, 0.12)
FLOOR_HZ = 0.5  # added to both rates before the log ratio
CLASSES = {"excited": "C0", "suppressed": "C5", "not responsive": REFERENCE_COLOR}
MARKERS = {"GRB006": "o", "GRB064": "^"}
# Display limit of both axes. All points are inside; a few error bars go past it.
LIMITS = (-3.0, 3.0)


def table_values(subject: str, session: str, unit_ids: list[int]) -> dict:
    """Return the table class, the stored rate delta and the log ratio with its SE."""
    from labdata_plugin.schema import StimulusResponse, StimulusResponseParam
    from thesis.ephys.events import fetch_session_events

    param = (StimulusResponseParam & {"stim_response_param_id": PARAM_ID}).fetch1()
    assert (param["baseline_start"], param["baseline_end"]) == BASELINE_S
    assert (param["response_start"], param["response_end"]) == RESPONSE_S
    key = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": 1,
        "stim_response_param_id": PARAM_ID,
    }
    stored = {
        int(r["unit_id"]): r for r in (StimulusResponse.Unit & key).fetch(as_dict=True)
    }
    _, pulses = fetch_session_events(subject, session)
    onsets = pulses.loc[pulses["first_in_train"], "timestamp"].to_numpy(float)
    units = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(onsets), (BOOTSTRAPS, len(onsets)))
    out = {"class": [], "stored_delta": [], "delta": [], "log_ratio": [], "se": []}
    for unit in unit_ids:
        spikes = np.sort(np.asarray(units.loc[unit, "spike_times_s"], dtype=float))

        def rates(window):
            lo = np.searchsorted(spikes, onsets + window[0])
            hi = np.searchsorted(spikes, onsets + window[1])
            return (hi - lo) / (window[1] - window[0])

        base, resp = rates(BASELINE_S), rates(RESPONSE_S)
        ratio = np.log(
            (resp[draws].mean(1) + FLOOR_HZ) / (base[draws].mean(1) + FLOOR_HZ)
        )
        row = stored[unit]
        response_type = row["response_type"] or "not responsive"
        out["class"].append(
            response_type if response_type in CLASSES else "not responsive"
        )
        out["stored_delta"].append(float(row["rate_delta"]))
        out["delta"].append(float(resp.mean() - base.mean()))
        out["log_ratio"].append(
            float(np.log((resp.mean() + FLOOR_HZ) / (base.mean() + FLOOR_HZ)))
        )
        out["se"].append(float(ratio.std(ddof=1)))
    return out


def model_values(root: Path) -> dict:
    """Return the flash-weight and first-flash values of each unit, with jackknife SE."""
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    center = metadata["flash_recovery_center"]
    ones = np.ones(dmat.HISTORY_COLUMNS)  # history kernels are not used here
    checkpoints = model.fit_dir("all") / "checkpoints"
    finals = {
        r["unit_id"]: r for r in load_unit_records(checkpoints, "unit_*_final.json")
    }
    folds = {
        r["unit_id"]: r for r in load_unit_records(checkpoints, "unit_*_folds.json")
    }

    def measures(record) -> tuple[float, float]:
        kernels = fitted_kernels(record, metadata, ones)
        lags, flash = (
            kernels[dmat.FLASH_COMMON][0],
            flash_kernel_at(kernels, np.inf, False, center),
        )
        window = (lags >= 1000 * RESPONSE_S[0]) & (lags < 1000 * RESPONSE_S[1])
        weights = float(flash[window].mean())
        hold_lags, hold = kernels[dmat.HOLD]
        hold_window = (hold_lags >= 1000 * RESPONSE_S[0]) & (
            hold_lags < 1000 * RESPONSE_S[1]
        )
        pre_stimulus = float(kernels["fixation"][1][0])
        return weights, weights + float(hold[hold_window].mean()) - pre_stimulus

    out = {"unit_ids": [], "weights": [], "weights_se": [], "all": [], "all_se": []}
    for unit, final in sorted(finals.items()):
        fold_values = np.asarray(
            [
                measures({**fold, "components": final["components"]})
                for fold in folds[unit]["folds"]
            ]
        )
        k = len(fold_values)
        se = np.sqrt((k - 1) / k * ((fold_values - fold_values.mean(0)) ** 2).sum(0))
        point = measures(final)
        out["unit_ids"].append(int(unit))
        out["weights"].append(point[0])
        out["all"].append(point[1])
        out["weights_se"].append(float(se[0]))
        out["all_se"].append(float(se[1]))
    return out


def auc(values: np.ndarray, positive: np.ndarray) -> float:
    """Return P(value of a positive unit > value of a negative unit), ties half."""
    pos, neg = values[positive], values[~positive]
    greater = (pos[:, None] > neg[None, :]).mean()
    return float(greater + 0.5 * (pos[:, None] == neg[None, :]).mean())


def statistics(table: dict, model: dict) -> dict:
    excited = np.asarray(table["class"]) == "excited"
    x = np.asarray(table["log_ratio"])
    stats = {
        "units": len(x),
        "table excited": int(excited.sum()),
        "table suppressed": int(np.sum(np.asarray(table["class"]) == "suppressed")),
        "max |recomputed − stored rate delta| (spikes/s)": float(
            np.max(np.abs(np.asarray(table["delta"]) - table["stored_delta"]))
        ),
    }
    rng = np.random.default_rng(0)
    for name in ("weights", "all"):
        y = np.asarray(model[name])
        se = np.asarray(model[f"{name}_se"])
        boot = []
        for _ in range(BOOTSTRAPS):
            i = rng.integers(0, len(y), len(y))
            if 0 < excited[i].sum() < len(i):
                boot.append(auc(y[i], excited[i]))
        model_class = np.where(
            y - 1.96 * se > 0,
            "excited",
            np.where(y + 1.96 * se < 0, "suppressed", "not responsive"),
        )
        cross = {
            f"table {t} / model {m}": int(
                np.sum((np.asarray(table["class"]) == t) & (model_class == m))
            )
            for t in CLASSES
            for m in CLASSES
        }
        rho = spearmanr(x, y)
        stats[name] = {
            "auc_table_excited": auc(y, excited),
            "auc_95": [float(q) for q in np.percentile(boot, [2.5, 97.5])],
            "spearman_rho": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
            "cross_table": cross,
            "table excited with model value <= 0": [
                int(u)
                for u, e, v in zip(model["unit_ids"], excited, y, strict=True)
                if e and v <= 0
            ],
        }
    return stats


def plot(sessions: dict) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        titles = ("flash weights", "all first-flash terms")
        for axis, name, title in zip(axes, ("weights", "all"), titles, strict=True):
            for subject, (table, model) in sessions.items():
                classes = np.asarray(table["class"])
                for kind, color in CLASSES.items():
                    keep = classes == kind
                    axis.errorbar(
                        np.asarray(table["log_ratio"])[keep],
                        np.asarray(model[name])[keep],
                        xerr=np.asarray(table["se"])[keep],
                        yerr=np.asarray(model[f"{name}_se"])[keep],
                        fmt=MARKERS[subject],
                        ms=3.5,
                        mfc="none" if kind == "not responsive" else color,
                        mec=color,
                        ecolor=color,
                        elinewidth=0.4,
                        alpha=0.8,
                        zorder=3 if kind != "not responsive" else 2,
                    )
            lo, hi = LIMITS
            axis.plot(
                [lo, hi], [lo, hi], color=REFERENCE_COLOR, ls="--", lw=0.6, zorder=1
            )
            axis.set(xlim=LIMITS, ylim=LIMITS, aspect="equal")
            axis.axhline(0, color=REFERENCE_COLOR, lw=0.5, zorder=1)
            axis.axvline(0, color=REFERENCE_COLOR, lw=0.5, zorder=1)
            axis.set_xlabel("first-flash response, data (Δ log rate)")
            axis.set_ylabel(f"model, {title} (Δ log rate)")
        for row, (kind, color) in enumerate(CLASSES.items()):
            axes[0].text(
                0.03,
                0.97 - 0.07 * row,
                f"table: {kind}",
                color=color,
                transform=axes[0].transAxes,
                va="top",
            )
        for row, (subject, marker) in enumerate(MARKERS.items()):
            axes[1].plot(
                0.05,
                0.95 - 0.07 * row,
                marker,
                color="0.3",
                mfc="none",
                ms=4,
                transform=axes[1].transAxes,
            )
            axes[1].text(
                0.09,
                0.95 - 0.07 * row,
                subject,
                color="0.3",
                transform=axes[1].transAxes,
                va="center",
            )
        label_panels(axes)
        figure.tight_layout()
        return save_figure(figure, OUT / "flash_weight_check")


def main() -> None:
    sessions, report = (
        {},
        {"windows_s": {"baseline": BASELINE_S, "response": RESPONSE_S}},
    )
    for session in SESSIONS:
        subject, stamp = session.split("_", 1)
        model = model_values(OUT / session)
        table = table_values(subject, stamp, model["unit_ids"])
        sessions[subject] = (table, model)
        report[session] = statistics(table, model)
    (OUT / "flash_weight_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(plot(sessions))


if __name__ == "__main__":
    main()
