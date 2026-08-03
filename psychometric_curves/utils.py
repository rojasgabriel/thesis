"""Labdata-backed psychometric plotting helpers.

Archived djchurchland notebooks live under ``archive/djchurchland/``.
Maintained plotting entry points are ``scripts/analyses/plot_psychometrics.py``
and the helpers in this module.
"""

from __future__ import annotations

import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from behavior_analyses.io import get_chipmunk_table


def _as_session_name_strings(sessions_list: list[Any] | None) -> list[str] | None:
    if sessions_list is None:
        return None
    out = []
    for session in sessions_list:
        if isinstance(session, datetime.datetime):
            out.append(session.strftime("%Y%m%d_%H%M%S"))
        else:
            out.append(str(session))
    return out


def _fetch_choice_and_stim(
    mouse_id: str, query: str | None, sessions_list: list[Any] | None
):
    Chipmunk = get_chipmunk_table()
    relation = Chipmunk.Trial() * Chipmunk.TrialParameters() & {
        "subject_name": mouse_id
    }
    if sessions_list is not None:
        session_strings = _as_session_name_strings(sessions_list)
        assert session_strings is not None
        relation = relation & [{"session_name": name} for name in session_strings]
    if query:
        relation = relation & query
    relation = relation & "response != 0"
    response, modality, audio_rate, visual_rate, boundary = relation.fetch(
        "response",
        "rewarded_modality",
        "stim_rate_audio",
        "stim_rate_vision",
        "category_boundary",
    )
    modality = np.asarray(modality)
    stim_rate = np.where(
        np.isin(modality, ["visual", "visual+audio"]),
        np.asarray(visual_rate, dtype=float),
        np.asarray(audio_rate, dtype=float),
    )
    return response, stim_rate - np.asarray(boundary, dtype=float)


def plot_single_mouse_psychometric_fit(
    mouse_id,
    query=None,
    ax=None,
    plot_mode="both",
    sessions_list=None,
    session_fits=None,
):
    """Plot individual and/or average psychometric fits for one subject.

    ``session_fits`` may be a list of dict rows with ``stims``, ``p_side`` /
    ``p_right``, and ``fit_params`` (for example from
    ``PsychometricSessionFit``). When omitted, only the pooled average fit from
    Chipmunk trials is drawn.
    """
    from behavior_analyses.psychometrics import (
        cumulative_gaussian,
        fit_psychometric_labdata,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    color = "black"
    fits = list(session_fits or [])

    if fits and plot_mode in ["individual", "both"]:
        for fit in fits:
            stims = np.asarray(fit["stims"], dtype=float)
            params = np.asarray(fit["fit_params"], dtype=float)
            p_side = np.asarray(fit.get("p_right", fit.get("p_side")), dtype=float)
            nx = np.linspace(np.min(stims), np.max(stims), 100)
            ax.plot(
                nx,
                cumulative_gaussian(*params, nx),
                linewidth=1,
                alpha=0.1,
                color=color,
            )
            ax.plot(stims, p_side, "o", markersize=4, alpha=0.1, color=color)

    if plot_mode in ["average", "both"]:
        response_values, stim_values = _fetch_choice_and_stim(
            mouse_id, query, sessions_list
        )
        if len(response_values) == 0:
            print(f"No trial data available for mouse {mouse_id}.")
        else:
            fit = fit_psychometric_labdata(
                np.asarray(stim_values, dtype=float),
                np.asarray(response_values, dtype=float),
                min_choices=20,
            )
            if fit is None:
                print(f"Failed to fit average curve for mouse {mouse_id}")
            else:
                nx = np.linspace(np.min(fit["stims"]), np.max(fit["stims"]), 100)
                ax.plot(
                    nx,
                    cumulative_gaussian(*fit["fit_params"], nx),
                    linewidth=2,
                    label=f"{mouse_id} Average",
                    color=color,
                )
                for stim, p_side, ci in zip(
                    fit["stims"], fit["p_right"], fit["p_right_ci"]
                ):
                    ax.plot([stim, stim], ci, "-_", color=color)
                ax.plot(
                    fit["stims"],
                    fit["p_right"],
                    "o",
                    markerfacecolor="lightgray",
                    markersize=6,
                    color=color,
                )

    ax.set_ylabel("P(right choice)", fontsize=14)
    ax.set_xlabel("Stimulus rate relative to boundary (Hz)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_ylim([0, 1])
    return ax


def plot_multi_mouse_psychometric_fit(
    mouse_sessions_dict, mice_list=None, query=None, ax=None
):
    """Plot average psychometric fits for multiple mice."""
    from behavior_analyses.psychometrics import (
        cumulative_gaussian,
        fit_psychometric_labdata,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if not mice_list:
        mice_list = list(mouse_sessions_dict.keys())

    colors = sns.color_palette("Set1")
    for mouse_id, color in zip(mice_list, colors):
        sessions_list = mouse_sessions_dict[mouse_id]
        response_values, stim_values = _fetch_choice_and_stim(
            mouse_id, query, sessions_list
        )
        if len(response_values) == 0:
            print(f"No trial data available for mouse {mouse_id}.")
            continue
        fit = fit_psychometric_labdata(
            np.asarray(stim_values, dtype=float),
            np.asarray(response_values, dtype=float),
            min_choices=20,
        )
        if fit is None:
            print(f"Failed to fit average curve for mouse {mouse_id}")
            continue
        nx = np.linspace(np.min(fit["stims"]), np.max(fit["stims"]), 100)
        ax.plot(
            nx,
            cumulative_gaussian(*fit["fit_params"], nx),
            linewidth=2,
            label=f"{mouse_id}",
            color=color,
        )

    ax.set_ylabel("P(right choice)", fontsize=14)
    ax.set_xlabel("Stimulus rate relative to boundary (Hz)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_ylim([0, 1])
    return ax
