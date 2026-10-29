"""Do the decoders and the GLM use the same neurons for the same signals?

Question: after the center exit, the population tells the stimulus category
and the choice (`category_without_terms.py`, `decoding_time_course.py`). The
GLM gives each neuron a flash kernel and a turn-direction kernel. If the
category readout is visual and the choice readout is the turn, the neurons
with large category weights have large flash weights, and the neurons with
large choice weights have large turn weights. The crossed pairs (category
against turn, choice against flash) are the control. The turn term was added
to the GLM because the rates showed a side preference after the exit
(`side_preference.py`), so the agreement of the choice and turn weights is
partly a check of the fit, not an independent finding.

Comparison, for each unit:

- Decoder weights: the logistic weight of the unit (z-scored rates, so the
  weights are comparable between units) in a category decoder (equal numbers
  of the four category × choice groups) and in a choice decoder (the same
  groups), on the rate `WINDOW_S` after the center exit. Each of
  `N_BOOTSTRAPS` resamples draws each group with replacement and fits on all
  of its trials; the weight is the mean and its SE the SD over resamples.
  Positive: more spikes for fast trains, or for right choices.
- GLM weights: the flash weight of `flash_weight_check.model_values` (the
  in-port first-flash kernel, mean Δ log rate 30–120 ms, final refit, with
  the jackknife SE over the folds), and the turn-direction kernel (+1 right,
  −1 left) as mean Δ log rate from 0 to `WINDOW_S` after the exit, final
  refit, with the jackknife SE over the folds. Positive: more spikes for
  flashes, or for right turns.

- Sessions, units and trials: the GLM sessions (`flash_kernels.SESSIONS`),
  the GLM units with a final fit (172 and 79; not the decoder default of
  single units and multi-units), and the selected GLM trials that are not at
  the category boundary.
- Statistics: Spearman ρ over the units of each mouse, with a p value from
  `N_PERMUTATIONS` shuffles of the unit labels. Pooled: ρ of the within-mouse
  percentile ranks of all units, with a hierarchical bootstrap 95% interval
  (mice, then units). The sampling unit is the unit.
- Figure: rows are the decoders, columns the GLM weights; each unit with its
  SE in both directions. All units are drawn; there is no display limit.

The script writes `figures/glm/decoder_weights.svg` and
`results/glm/decoder_weights.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.decoder_weights
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses._decoding import decoder, window_rates
from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.flash_weight_check import model_values
from thesis.ephys.analyses.glm.results import fitted_kernels
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    BOOTSTRAPS,
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
NAME = "decoder_weights"
WINDOW_S = 0.3
N_BOOTSTRAPS = 100
N_PERMUTATIONS = 2000
SEED = 0
DECODERS = ("category", "choice")
GLM_WEIGHTS = ("flash", "turn")
COLORS = {"flash": CONCEPT_COLORS["visual"], "turn": CONCEPT_COLORS["task"]}
# The marker of each mouse as text, so the statistics name the marker.
GLYPHS = {"o": "○", "^": "△"}
LABELS = {
    "category": "category weight (fast +)",
    "choice": "choice weight (right +)",
    "flash": "flash weight (Δ log rate)",
    "turn": "turn weight, right + (Δ log rate)",
}


def turn_weights(root: Path) -> dict[int, tuple[float, float]]:
    """Return the turn-direction weight of each unit and its jackknife SE."""
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    ones = np.ones(dmat.HISTORY_COLUMNS)  # history kernels are not used here
    checkpoints = model.fit_dir("all") / "checkpoints"
    finals = {
        r["unit_id"]: r for r in load_unit_records(checkpoints, "unit_*_final.json")
    }
    folds = {
        r["unit_id"]: r for r in load_unit_records(checkpoints, "unit_*_folds.json")
    }

    def measure(record) -> float:
        lags, kernel = fitted_kernels(record, metadata, ones)[dmat.TURN_DIRECTION]
        return float(kernel[(lags >= 0) & (lags < 1000 * WINDOW_S)].mean())

    out = {}
    for unit, final in finals.items():
        values = np.asarray(
            [
                measure({**fold, "components": final["components"]})
                for fold in folds[unit]["folds"]
            ]
        )
        k = len(values)
        se = np.sqrt((k - 1) / k * ((values - values.mean()) ** 2).sum())
        out[int(unit)] = (measure(final), float(se))
    return out


def decoder_weights(rates, category, choice, rng) -> dict[str, np.ndarray]:
    """Return the mean and SD over resamples of each unit's decoder weights."""
    groups = [
        np.flatnonzero((category == c) & (choice == r)) for c in (0, 1) for r in (-1, 1)
    ]
    n = min(len(g) for g in groups)
    weights = {name: [] for name in DECODERS}
    for _ in range(N_BOOTSTRAPS):
        index = np.concatenate([rng.choice(g, n, replace=True) for g in groups])
        for name, labels in (("category", category), ("choice", choice == 1)):
            model = decoder().fit(rates[index], labels[index])
            weights[name].append(model[-1].coef_[0])
    return {name: (np.mean(w, 0), np.std(w, 0, ddof=1)) for name, w in weights.items()}


def session_values(session: str) -> dict:
    root = OUT / session
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    trials = trials[trials["stim_category"].ne("boundary")]
    subject, stamp = session.split("_", 1)
    flash = model_values(root)
    turn = turn_weights(root)
    ids = flash["unit_ids"]
    if sorted(turn) != ids:
        raise ValueError(f"The flash and turn weights have other units in {session}.")
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False)
    units = table.set_index("unit_id").loc[ids].reset_index()
    exit_s = trials["center_exit_s"].to_numpy(float)
    rates = window_rates(units, exit_s, exit_s + WINDOW_S)
    category = trials["stim_category"].eq("high_rate").to_numpy(int)
    choice = trials["response"].to_numpy(int)
    weights = decoder_weights(rates, category, choice, np.random.default_rng(SEED))
    return {
        "subject": subject,
        "unit_ids": ids,
        "category": weights["category"],
        "choice": weights["choice"],
        "flash": (np.asarray(flash["weights"]), np.asarray(flash["weights_se"])),
        "turn": (
            np.asarray([turn[u][0] for u in ids]),
            np.asarray([turn[u][1] for u in ids]),
        ),
    }


def correlation(sessions: list[dict], a: str, b: str) -> dict:
    """Spearman ρ of each mouse with a unit-label shuffle, and pooled."""
    rng = np.random.default_rng(SEED)
    out = {}
    for data in sessions:
        x, y = data[a][0], data[b][0]
        rho = spearmanr(x, y).statistic
        null = np.asarray(
            [spearmanr(rng.permutation(x), y).statistic for _ in range(N_PERMUTATIONS)]
        )
        p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1)
        out[data["subject"]] = {"units": len(x), "rho": float(rho), "p_shuffle": p}
    # Pooled: the percentile ranks within each mouse, then the mice and units.
    ranks = [
        (rankdata(d[a][0]) / len(d[a][0]), rankdata(d[b][0]) / len(d[b][0]))
        for d in sessions
    ]
    pooled = spearmanr(
        np.concatenate([r[0] for r in ranks]), np.concatenate([r[1] for r in ranks])
    ).statistic
    boots = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        picked = [ranks[g] for g in rng.integers(0, len(ranks), len(ranks))]
        draws = [rng.integers(0, len(r[0]), len(r[0])) for r in picked]
        boots[i] = spearmanr(
            np.concatenate([r[0][d] for r, d in zip(picked, draws, strict=True)]),
            np.concatenate([r[1][d] for r, d in zip(picked, draws, strict=True)]),
        ).statistic
    out["pooled"] = {
        "rho": float(pooled),
        "interval_95": np.percentile(boots, [2.5, 97.5]).tolist(),
    }
    return out


def plot(sessions: list[dict], stats: dict) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(DECODERS), len(GLM_WEIGHTS), figsize=(6.6, 6.0), sharex="col"
        )
        for row, a in enumerate(DECODERS):
            for col, b in enumerate(GLM_WEIGHTS):
                axis = axes[row, col]
                for index, data in enumerate(sessions):
                    (x, x_se), (y, y_se) = data[b], data[a]
                    axis.errorbar(
                        x,
                        y,
                        xerr=x_se,
                        yerr=y_se,
                        fmt="none",
                        ecolor=COLORS[b],
                        alpha=0.25,
                        linewidth=0.5,
                    )
                    axis.scatter(
                        x,
                        y,
                        s=10,
                        marker=mouse_marker(data["subject"], index),
                        facecolors="none",
                        edgecolors=COLORS[b],
                        linewidths=0.6,
                    )
                axis.axhline(0, color="0.7", linewidth=0.6)
                axis.axvline(0, color="0.7", linewidth=0.6)
                lines = [
                    f"{GLYPHS.get(mouse_marker(s), '')} {s}: ρ = {v['rho']:+.2f}, "
                    + (
                        "p < 0.001"
                        if v["p_shuffle"] < 0.001
                        else f"p = {v['p_shuffle']:.3f}"
                    )
                    for s, v in stats[f"{a} vs {b}"].items()
                    if s != "pooled"
                ]
                axis.text(
                    0.02,
                    0.98,
                    "\n".join(lines),
                    transform=axis.transAxes,
                    va="top",
                    fontsize=6.5,
                    color=OBSERVED_COLOR,
                )
                if row == len(DECODERS) - 1:
                    axis.set_xlabel(LABELS[b], color=COLORS[b])
                if col == 0:
                    axis.set_ylabel(LABELS[a])
        label_panels(axes.ravel(), x=-0.2, y=1.04)
        figure.tight_layout()
        return save_figure(figure, OUT / NAME)


def main() -> None:
    sessions = [session_values(s) for s in SESSIONS]
    stats = {
        f"{a} vs {b}": correlation(sessions, a, b)
        for a in DECODERS
        for b in GLM_WEIGHTS
    }
    report = {
        "units": {d["subject"]: len(d["unit_ids"]) for d in sessions},
        "correlations": stats,
    }
    (OUT / f"{NAME}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(plot(sessions, stats))


if __name__ == "__main__":
    main()
