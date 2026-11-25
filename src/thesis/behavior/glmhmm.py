"""Does the mouse change its choice strategy during a session (engagement states)?

Question: the psychometric curve (`psychometric.py`) gives one strategy for all
trials. Does a mouse change between a strategy that follows the flash rate
(engaged) and strategies that follow a side or the trial before (disengaged)?
The model is the GLM-HMM of Ashwood et al. 2022 (Nat Neurosci 25:201,
doi:10.1038/s41593-021-01007-z), with the settings of its flash-rate data set
(Odoemene et al. 2018, the same task). No model of the neurons: the states
come from the choices only.

Model: a hidden Markov model with `K` states. In each state the choice
(right = 1) is a Bernoulli GLM (logistic) of four inputs (paper, 4.4.1):
- stimulus: flash rate minus the boundary, z-scored over the choice trials of
  the mouse;
- bias: 1;
- previous choice: ±1, the last trial with a choice in the session;
- win-stay: previous choice × (+1 rewarded, −1 not rewarded) of that trial.
One transition matrix for all sessions; each session is one sequence, with no
link between sessions. Priors: Gaussian on the weights (SD `SIGMA`), Dirichlet
on each row of the transition matrix (`ALPHA`), flat on the first state.

Sessions (the rule of `behavior.trials`, but with no need of a recording):
discrimination sessions with flashes (visual or visual + audio) and at least
`MIN_RATES` flash rates among the choice trials. Each mouse is fit alone.
Trials: every trial of a session stays in the sequence. A trial gives a choice
to the model if it has a left or right choice, no early withdrawal, and
flashes. Every other trial (early withdrawal, no choice, auditory only) is a
missing observation: its likelihood is 1, so the state only moves on by the
transition matrix (paper, 4.4, "violation" trials). The sampling unit is the
trial, nested in sessions.

Fit: EM of the `dynamax` package (`LogisticRegressionHMM`): forward-backward
E-step; M-step in closed form for the transitions and 50 Adam steps for the
weights. K = 1 is one logistic regression (scikit-learn) with the same prior. `ITERATIONS` EM iterations, with no stop rule; the report has the gain
of the log posterior in the last 10 iterations. `INITS` starts for each fit,
from the 1-state weights plus Gaussian noise (SD 0.2) and a transition matrix
near 0.95 on the diagonal; the start with the highest log posterior is kept.
`dynamax` has no missing observations and no sessions of different lengths.
Thus the sessions are padded to the longest one, and `MaskedEmissions` sets
the log-likelihood to 0 on a trial with no choice and on the padding. The bias
of `dynamax` stays at 0; the bias is an input, so that it has the same prior
as the other weights.

Model comparison: 5-fold cross-validation over whole sessions, K = 1 to
`MAX_STATES`. Score: log-likelihood of the held-out sessions in bits per choice
trial above a Bernoulli model with the fraction of right choices of the
training sessions (paper, eq. 22). K = 1 is the baseline (one GLM).

States: sorted by the stimulus weight, largest first. State 1 is named
"engaged" only as a label of that order; read the weights before you use the
name. A trial belongs to its most probable state (`glmhmm_trials.csv` has the
posterior of each state for every trial, so that a threshold can be used).

Differences from the paper: (1) no fit of all mice together before the fit of
each mouse (there are 2 or 3 mice, with different sessions); states are matched
between mice by the order above. (2) The first trials of a session, before the
first choice, have 0 for the two history inputs. (3) The 1-state start is fit
with the prior. (4) The EM is from `dynamax`, not the `ssm` package of the
paper: the weights get Adam steps, not BFGS to convergence. `--check`
simulates a 3-state model and tests that the fit recovers it.

Figure: (a) held-out score against K; (b) weights of each state; (c) fraction
of right choices against stimulus strength for the trials of each state, with
the curve of the state (history inputs at 0); (d) posterior of each state over
the trials of the recorded session (`trials.RECORDED`). Error bars in (c): 95%
binomial (Wilson) interval over trials; it treats trials as independent.
Colors are the states; the mouse is the row and the marker.

Writes `figures/behavior/glmhmm.svg`, `results/behavior/glmhmm.json` and
`results/behavior/glmhmm_trials.csv`.

    THESIS_CACHE_DIR=figures/cache uv run python -m thesis.behavior.glmhmm
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import multiprocessing
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from dynamax.hidden_markov_model import LogisticRegressionHMM
from dynamax.hidden_markov_model.models.logreg_hmm import (
    LogisticRegressionHMMEmissions,
)
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.behavior.trials import (
    MIN_RATES,
    MIN_RATES_BY_SUBJECT,
    MODALITIES,
    RECORDED,
    SUBJECTS,
    _bpod_table,
)
from thesis.ephys.trials import CATEGORY_BOUNDARY_HZ, stimulus_strength
from thesis.plotting import (
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

jax.config.update("jax_enable_x64", True)

OUT = Path("results/behavior")
INPUTS = ("stimulus", "bias", "previous choice", "win-stay")
# The priors that the paper selected for the flash-rate task (4.1.1).
SIGMA = 0.75
ALPHA = 2.0
MAX_STATES = 5
# 3 states: see `--states`. The held-out score decides if more states help.
STATES = 3
FOLDS = 5
INITS = 20
SEED = 20261120
ITERATIONS = 300
# State 1 (the largest stimulus weight) first. Not the colors of the model
# terms of the V1 GLM (`plotting.CONCEPT_COLORS`).
STATE_COLORS = ("#008B8B", "#8C510A", "#B2182B", "#808000", "0.4")


def state_name(state: int) -> str:
    return "state 1 (engaged)" if state == 0 else f"state {state + 1}"


# --- data ------------------------------------------------------------------


def all_trials(subject: str) -> pd.DataFrame:
    """Return every trial of the used sessions, with the model inputs.

    `observed` marks the trials that give a choice to the model.
    """
    table = _bpod_table(subject).sort_values(["session_name", "trial_num"])
    table = table[
        (table["setting_task_mode"].astype(str) == "discrimination")
        & table["setting_modalities"].astype(str).isin(MODALITIES)
    ].copy()
    choice = table["response"].isin((-1, 1))
    rate = table["stim_rate_vision"].astype(float)
    table["observed"] = (
        choice
        & (table["early_withdrawal"] == 0)
        & table["rewarded_modality"].astype(str).isin(MODALITIES)
        & np.isfinite(rate)
    )
    rates = table[table["observed"]].groupby("session_name")["stim_rate_vision"]
    enough = rates.nunique() >= MIN_RATES_BY_SUBJECT.get(subject, MIN_RATES)
    table = table[table["session_name"].isin(enough.index[enough])].copy()
    if not np.allclose(
        table.loc[table["observed"], "category_boundary"], CATEGORY_BOUNDARY_HZ
    ):
        raise ValueError(f"{subject}: a boundary is not {CATEGORY_BOUNDARY_HZ} Hz.")
    # The last trial with a choice (of any modality) before this one.
    choice = table["response"].isin((-1, 1))
    side = table["response"].where(choice).astype(float)
    outcome = pd.Series(
        np.where(table["rewarded"] == 1, 1.0, -1.0), index=table.index
    ).where(choice)
    by_session = table["session_name"]
    previous = side.groupby(by_session).transform(lambda s: s.ffill().shift(1))
    previous_outcome = outcome.groupby(by_session).transform(
        lambda s: s.ffill().shift(1)
    )
    evidence = table["stim_rate_vision"].astype(float) - CATEGORY_BOUNDARY_HZ
    table["rate_hz"] = table["stim_rate_vision"].astype(float)
    table["right"] = (table["response"] == 1).astype(float)
    # Trials with no flash rate are not observed; their stimulus value is not used.
    table["stimulus"] = (evidence / evidence[table["observed"]].std()).fillna(0.0)
    table["bias"] = 1.0
    table["previous choice"] = previous.fillna(0.0)
    table["win-stay"] = (previous * previous_outcome).fillna(0.0)
    return table.reset_index(drop=True)


def padded(trials: pd.DataFrame) -> dict:
    """Return the sessions as arrays (sessions, longest session, ...)."""
    groups = [g for _, g in trials.groupby("session_name", sort=True)]
    longest = max(map(len, groups))
    x = np.zeros((len(groups), longest, len(INPUTS)))
    y = np.zeros((len(groups), longest))
    observed = np.zeros((len(groups), longest), dtype=bool)
    real = np.zeros((len(groups), longest), dtype=bool)
    for s, g in enumerate(groups):
        x[s, : len(g)] = g[list(INPUTS)].to_numpy(float)
        y[s, : len(g)] = g["right"].to_numpy()
        observed[s, : len(g)] = g["observed"].to_numpy()
        real[s, : len(g)] = True
    return {"x": x, "y": y, "observed": observed, "real": real}


def subset(data: dict, sessions) -> dict:
    return {name: value[sessions] for name, value in data.items()}


# --- model -----------------------------------------------------------------


class MaskedEmissions(LogisticRegressionHMMEmissions):
    """Logistic emissions; the last input is 1 on a trial with a choice.

    The log-likelihood is 0 on every other trial (and on the padding), so the
    E-step and the M-step both treat it as a missing observation.
    """

    def _compute_conditional_logliks(self, params, emissions, inputs=None):
        logliks = super()._compute_conditional_logliks(
            params, emissions, inputs[:, :-1]
        )
        return inputs[:, -1:] * logliks


def hmm(n_states: int) -> LogisticRegressionHMM:
    model = LogisticRegressionHMM(
        n_states,
        len(INPUTS) + 1,  # + the mask of the choice trials
        initial_probs_concentration=1.0,
        transition_matrix_concentration=ALPHA,
    )
    model.emission_component = MaskedEmissions(
        n_states, len(INPUTS), emission_matrices_scale=SIGMA
    )
    return model


def parameters(model: LogisticRegressionHMM, first, transitions, weights):
    params, props = model.initialize(
        initial_probs=jnp.asarray(first),
        transition_matrix=jnp.asarray(transitions),
        emission_weights=jnp.asarray(weights),
        emission_biases=jnp.zeros(model.num_states),
    )
    props.emissions.biases.trainable = False  # the bias is an input
    return params, props


def arrays(data: dict):
    """Return the choices and the inputs (with the mask) for `dynamax`."""
    inputs = np.concatenate([data["x"], data["observed"][..., None]], axis=-1)
    return jnp.asarray(data["y"]), jnp.asarray(inputs)


def em(data: dict, weights, transitions) -> dict:
    """Run EM from each start (starts, states, ...); return the best fit."""
    n_states = weights.shape[1]
    model = hmm(n_states)
    choices, inputs = arrays(data)

    def run(weights, transitions):
        first = jnp.full(n_states, 1 / n_states)
        params, props = parameters(model, first, transitions, weights)
        return model.fit_em(
            params, props, choices, inputs, num_iters=ITERATIONS, verbose=False
        )

    params, log_posterior = jax.vmap(run)(
        jnp.asarray(weights), jnp.asarray(transitions)
    )
    log_posterior = np.asarray(log_posterior)
    best = int(np.nanargmax(log_posterior[:, -1]))
    return {
        "weights": np.asarray(params.emissions.weights[best]),
        "transitions": np.asarray(params.transitions.transition_matrix[best]),
        "first": np.asarray(params.initial.probs[best]),
        "log_posterior": float(log_posterior[best, -1]),
        "last_gain": float(log_posterior[best, -1] - log_posterior[best, -11]),
    }


def fit(data: dict, n_states: int, seed: int, inits: int = INITS) -> dict:
    """Return the best of `inits` EM fits, with the states sorted by stimulus weight."""
    # 1 state is one GLM; `dynamax` needs 2 states or more. C = SIGMA² gives
    # the same Gaussian prior (the bias is an input, so it has the prior too).
    glm = LogisticRegression(C=SIGMA**2, fit_intercept=False, tol=1e-8, max_iter=1000)
    glm.fit(data["x"][data["observed"]], data["y"][data["observed"]])
    one = {
        "weights": glm.coef_,
        "transitions": np.ones((1, 1)),
        "first": np.ones(1),
        "last_gain": 0.0,
    }
    if n_states == 1:
        return one
    rng = np.random.default_rng(seed)
    weights = one["weights"] + rng.normal(0, 0.2, (inits, n_states, len(INPUTS)))
    transitions = np.abs(
        0.95 * np.eye(n_states) + rng.normal(0, 0.05, (inits, n_states, n_states))
    )
    transitions /= transitions.sum(2, keepdims=True)
    best = em(data, weights, transitions)
    order = np.argsort(-best["weights"][:, 0])
    return {
        **best,
        "weights": best["weights"][order],
        "transitions": best["transitions"][np.ix_(order, order)],
        "first": best["first"][order],
    }


def posteriors(model: dict, data: dict) -> tuple[np.ndarray, float]:
    """Return the state posterior of each trial and log p(choices)."""
    if len(model["weights"]) == 1:
        z = data["x"] @ model["weights"][0]
        loglik = (data["y"] * z - np.logaddexp(0, z))[data["observed"]].sum()
        return np.ones((*data["y"].shape, 1)), float(loglik)
    machine = hmm(len(model["weights"]))
    params, _ = parameters(
        machine, model["first"], model["transitions"], model["weights"]
    )
    posterior = jax.vmap(partial(machine.smoother, params))(*arrays(data))
    return np.asarray(posterior.smoothed_probs), float(posterior.marginal_loglik.sum())


# --- cross-validation --------------------------------------------------------


def _fold_score(job) -> float:
    """Return the held-out score of one (K, fold): bits per choice trial."""
    data, n_states, fold, folds = job
    train, test = subset(data, folds != fold), subset(data, folds == fold)
    model = fit(train, n_states, SEED + 100 * n_states + fold)
    _, loglik = posteriors(model, test)
    p_right = train["y"][train["observed"]].mean()
    y = test["y"][test["observed"]]
    baseline = np.sum(y * np.log(p_right) + (1 - y) * np.log(1 - p_right))
    return float((loglik - baseline) / (len(y) * np.log(2)))


def cross_validate(data: dict, pool) -> np.ndarray:
    """Return the held-out score, (K from 1 to `MAX_STATES`, folds)."""
    n_sessions = len(data["x"])
    folds = np.empty(n_sessions, dtype=int)
    order = np.random.default_rng(SEED).permutation(n_sessions)
    for fold, block in enumerate(np.array_split(order, FOLDS)):
        folds[block] = fold
    jobs = [
        (data, n_states, fold, folds)
        for n_states in range(1, MAX_STATES + 1)
        for fold in range(FOLDS)
    ]
    return np.array(list(pool.map(_fold_score, jobs))).reshape(MAX_STATES, FOLDS)


# --- analysis ----------------------------------------------------------------


def mouse_states(subject: str, n_states: int, pool) -> tuple[dict, pd.DataFrame]:
    trials = all_trials(subject)
    data = padded(trials)
    scores = cross_validate(data, pool)
    model = fit(data, n_states, SEED)
    gamma, loglik = posteriors(model, data)
    names = [f"p_state_{k + 1}" for k in range(n_states)]
    trials[names] = gamma[data["real"]]
    trials["state"] = gamma[data["real"]].argmax(1) + 1
    used = trials[trials["observed"]]
    correct = used[used["rate_hz"] != CATEGORY_BOUNDARY_HZ]
    recorded = used[used["session_name"] == RECORDED[subject]]
    report = {
        "subject": subject,
        "sessions": int(trials["session_name"].nunique()),
        "trials": int(len(trials)),
        "choice_trials": int(len(used)),
        "states": n_states,
        "inputs": list(INPUTS),
        "held_out_bits_per_trial": scores.tolist(),
        "weights": model["weights"].tolist(),
        "transitions": model["transitions"].tolist(),
        "expected_dwell_trials": (1 / (1 - np.diag(model["transitions"]))).tolist(),
        "first_state": model["first"].tolist(),
        "log_likelihood": loglik,
        "em_iterations": ITERATIONS,
        "log_posterior_gain_last_10_iterations": model["last_gain"],
        "occupancy": [float(np.mean(used["state"] == k + 1)) for k in range(n_states)],
        "fraction_correct": [
            float(correct.loc[correct["state"] == k + 1, "rewarded"].mean())
            for k in range(n_states)
        ],
        "posterior_at_least_0_8": float(np.mean(used[names].max(axis=1) >= 0.8)),
        "recorded_session": RECORDED[subject],
        "recorded_choice_trials_by_state": [
            int(np.sum(recorded["state"] == k + 1)) for k in range(n_states)
        ],
    }
    columns = ["subject_name", "session_name", "trial_num", "observed", "rate_hz"]
    return report, trials[[*columns, "right", "rewarded", *names, "state"]]


def plot(reports: list[dict], table: pd.DataFrame) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        rows = len(reports)
        figure, axes = plt.subplots(
            rows,
            4,
            figsize=(11, 2.5 * rows),
            layout="constrained",
            width_ratios=(0.8, 1.1, 1, 1.6),
            squeeze=False,
        )
        for row, (report, panels) in enumerate(zip(reports, axes, strict=True)):
            subject, n_states = report["subject"], report["states"]
            marker = mouse_marker(subject, row)
            comparison, weights, curves, session = panels
            scores = np.asarray(report["held_out_bits_per_trial"])
            k = np.arange(1, len(scores) + 1)
            comparison.plot(k, scores, marker, color="0.75", ms=2.5, ls="none")
            comparison.plot(k, scores.mean(1), f"{marker}-", color="black", ms=4, lw=1)
            comparison.axvline(n_states, color=REFERENCE_COLOR, ls=":", lw=0.8)
            comparison.set(
                xticks=k,
                xlabel="states",
                ylabel="held-out log-likelihood\n(bits per trial above chance)",
            )
            comparison.text(0, 1.03, subject, transform=comparison.transAxes)
            weights.axhline(0, color=REFERENCE_COLOR, lw=0.6)
            used = table[(table["subject_name"] == subject) & table["observed"]]
            strength = np.linspace(-1, 1, 101)
            sd = (used["rate_hz"] - CATEGORY_BOUNDARY_HZ).std()
            for state in range(n_states):
                color = STATE_COLORS[state]
                w = np.asarray(report["weights"][state])
                weights.plot(range(len(INPUTS)), w, f"{marker}-", color=color, ms=4)
                weights.text(
                    1.0,
                    0.98 - 0.09 * state,
                    f"{state_name(state)}, {100 * report['occupancy'][state]:.0f}%",
                    transform=weights.transAxes,
                    color=color,
                    fontsize=6.5,
                    ha="right",
                    va="top",
                )
                mine = used[used["state"] == state + 1]
                by_rate = mine.groupby("rate_hz")["right"].agg(["sum", "count"])
                low, high = np.array(
                    [
                        binomtest(int(k), int(n)).proportion_ci(method="wilson")
                        for k, n in by_rate.to_numpy()
                    ]
                ).T
                fraction = by_rate["sum"] / by_rate["count"]
                curves.errorbar(
                    stimulus_strength(by_rate.index.to_numpy()),
                    fraction,
                    yerr=[fraction - low, high - fraction],
                    fmt=marker,
                    color=color,
                    ms=3,
                    lw=0.8,
                    capsize=1.5,
                )
                curves.plot(
                    strength,
                    expit(w[0] * 8 * strength / sd + w[1]),
                    color=color,
                    lw=1,
                )
                recorded = table[
                    (table["subject_name"] == subject)
                    & (table["session_name"] == report["recorded_session"])
                ]
                session.plot(
                    recorded["trial_num"],
                    recorded[f"p_state_{state + 1}"],
                    color=color,
                    lw=1,
                )
            weights.set_xticks(
                range(len(INPUTS)), [name.replace(" ", "\n") for name in INPUTS]
            )
            weights.set(xlim=(-0.4, len(INPUTS) - 0.6), ylabel="weight (log-odds)")
            curves.axhline(0.5, color=REFERENCE_COLOR, lw=0.6)
            curves.axvline(0, color=REFERENCE_COLOR, lw=0.6)
            curves.set(
                xlabel="stimulus strength",
                ylabel="fraction of right choices",
                ylim=(0, 1),
            )
            session.set(
                xlabel=f"trial of the recorded session ({report['recorded_session'][:8]})",
                ylabel="posterior of the state",
                ylim=(-0.02, 1.02),
            )
        label_panels(axes[0], x=-0.2, y=1.06)
        return save_figure(figure, OUT / "glmhmm")


def check() -> None:
    """Simulate a 3-state model and test that the fit recovers it."""
    rng = np.random.default_rng(0)
    weights = np.array([[2.5, 0, 0, 0], [1.0, -1.5, 0, 0.3], [0.2, 1.8, 0, 0]])
    transitions = np.full((3, 3), 0.02) + 0.94 * np.eye(3)
    n_sessions, n_trials = 40, 400
    x = np.zeros((n_sessions, n_trials, 4))
    y = np.zeros((n_sessions, n_trials))
    states = np.zeros((n_sessions, n_trials), dtype=int)
    for s in range(n_sessions):
        state, side, reward = rng.integers(3), 0.0, 0.0
        for t in range(n_trials):
            if t:
                state = rng.choice(3, p=transitions[state])
            stimulus = rng.choice(np.linspace(-1.5, 1.5, 9))
            x[s, t] = stimulus, 1, side, side * reward
            y[s, t] = rng.random() < expit(x[s, t] @ weights[state])
            side = 2 * y[s, t] - 1
            reward = 1.0 if (stimulus > 0) == (side > 0) else -1.0
            states[s, t] = state
    observed = rng.random((n_sessions, n_trials)) > 0.3  # 30% missing choices
    data = {"x": x, "y": y, "observed": observed, "real": np.ones_like(observed)}
    model = fit(data, 3, seed=1, inits=5)
    gamma, loglik = posteriors(model, data)
    # The simulated states are in the order of the sort (stimulus weight).
    assert np.abs(model["weights"] - weights).max() < 0.35, model["weights"]
    assert np.abs(model["transitions"] - transitions).max() < 0.03
    assert np.mean(gamma.argmax(2) == states) > 0.85
    assert loglik > posteriors(fit(data, 1, seed=1), data)[1] + 100
    print("check passed:", np.round(model["weights"], 2).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=int, default=STATES, help="K of the figure.")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="Recovery test only.")
    parser.add_argument("--replot", action="store_true", help="Saved fits; no new fit.")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.replot:
        table = pd.read_csv(OUT / "glmhmm_trials.csv", dtype={"session_name": str})
        print(plot(json.loads((OUT / "glmhmm.json").read_text()), table))
        return
    reports, tables = [], []
    # spawn: a fork of a process with JAX threads can stop.
    spawn = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(args.workers, mp_context=spawn) as pool:
        for subject in args.subjects:
            report, table = mouse_states(subject, args.states, pool)
            reports.append(report)
            tables.append(table)
            print(json.dumps(report))
    table = pd.concat(tables, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "glmhmm.json").write_text(json.dumps(reports, indent=1))
    table.to_csv(OUT / "glmhmm_trials.csv", index=False)
    print(plot(reports, table))


if __name__ == "__main__":
    main()
