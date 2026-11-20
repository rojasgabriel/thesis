"""Behavior of the thesis sessions: psychometric fit."""

from ._base import dj, np, rojasbowe_schema
from .sessions import ThesisSession  # noqa: F401


@rojasbowe_schema
class Psychometric(dj.Computed):
    """Choices of the recorded session (thesis.behavior.psychometric, --recorded).

    Trials: completed, left or right choice, full 1-s train, visual and
    multisensory pooled. Cumulative-Gaussian fit (fit_psychometric) of the
    choice on x = rate − 12 Hz:
    P(right) = γ + (1 − γ − λ) · ½ erfc(−β (x − α) / √2).
    Bounds: β 0.001–0.5, γ and λ 0–0.5; a value at a bound is not constrained
    by the data (GRB064 β = 0.5).
    Accuracy: rewarded fraction of the non-boundary trials.
    """

    definition = """
    -> ThesisSession
    ---
    choice_trials                          : smallint
    alpha                                  : float  # bias (Hz from the boundary)
    beta                                   : float  # sensitivity (1/Hz)
    gamma                                  : float  # left lapse (P(right) floor)
    lambda                                 : float  # right offset (1 − P(right) ceiling)
    accuracy_non_boundary                  : float
    """

    class Rate(dj.Part):
        definition = """
        -> master
        rate_hz                            : float
        ---
        trials                             : smallint
        right_choices                      : smallint
        """

    def make(self, key, **_kwargs):
        from fit_psychometric import fit_psychometric

        from thesis.behavior.trials import RECORDED, task_trials
        from thesis.ephys.trials import CATEGORY_BOUNDARY_HZ

        if RECORDED.get(key["subject_name"]) != key["session_name"]:
            raise ValueError(f"{key} is not the recorded session of the mouse.")
        trials = task_trials(key["subject_name"], recorded=True)
        rates = trials["rate_hz"].to_numpy()
        right = trials["right"].to_numpy()
        fit = fit_psychometric(rates - CATEGORY_BOUNDARY_HZ, right)
        if fit["fit_params"] is None or not fit["fit"].mle_retvals["converged"]:
            raise ValueError(f"{key}: the psychometric fit did not converge.")
        alpha, beta, gamma, lapse = (float(p) for p in fit["fit_params"])
        self.insert1(
            {
                **key,
                "choice_trials": len(trials),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda": lapse,
                "accuracy_non_boundary": trials.loc[
                    rates != CATEGORY_BOUNDARY_HZ, "rewarded"
                ].mean(),
            }
        )
        self.Rate.insert(
            {
                **key,
                "rate_hz": rate,
                "trials": int((rates == rate).sum()),
                "right_choices": int(right[rates == rate].sum()),
            }
            for rate in np.unique(rates)
        )
