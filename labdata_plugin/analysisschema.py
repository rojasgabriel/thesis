import datajoint as dj
import numpy as np
from labdata.schema import (
    DatasetEvents,  # noqa: F401
    EphysRecording,
    Session,
    SpikeSorting,
    UnitCount,
    get_user_schema,
)

rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class EventMapping(dj.Manual):  # TODO: remove renaming of Digital keys
    definition = """
    -> Session
    event_name                           : varchar(54)   # shared logical event role
    ---
    -> DatasetEvents.Digital.proj(source_dataset_name='dataset_name', source_stream_name='stream_name', source_event_name='event_name')
    """


@rojasbowe_schema
class LocomotionPeaks(dj.Computed):
    definition = """
    -> UnitCount.Unit
    ---
    stat_peak       : float  # peak amplitude of stat event (sp/s)
    stat_latency    : float  # latency of stat event (s)
    move_peak       : float  # peak amplitude of move event (sp/s)
    move_latency    : float  # latency of move event (s)
    """

    key_source = UnitCount.Unit & "unit_criteria_id = 1" & "passes = 1"
    _session_cache = {}

    def make(self, key):
        from ephys.src.config.locomotion import (
            BASELINE_WINDOW,
            PETH_KWARGS,
            RESP_WINDOW,
        )
        from ephys.src.utils.analysis_conditioned_stim import (
            build_trial_stim_classification,
            extract_conditioned_stim_anchors,
        )
        from ephys.src.utils.analysis_peth import compute_population_peth
        from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata
        from ephys.src.utils.io_digital_events import fetch_session_events
        from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table

        subject = key["subject_name"]
        session = key["session_name"]
        unit_id = key["unit_id"]

        cache_key = (subject, session)
        if cache_key not in self._session_cache:
            aligned_events = fetch_session_events(subject, session)
            trial_table = fetch_trial_metadata(subject, session, aligned_events)
            if trial_table is None:
                raise RuntimeError(
                    f"Could not load trial metadata for {subject} {session}."
                )
            trial_table = enrich_chipmunk_trial_table(trial_table)
            trial_classification = build_trial_stim_classification(
                aligned_events, trial_table
            )
            anchors = extract_conditioned_stim_anchors(trial_classification)
            stationary_event_times = anchors["paired_last_stationary"]
            movement_event_times = anchors["paired_first_movement"]
            if stationary_event_times.size == 0 or movement_event_times.size == 0:
                raise RuntimeError(
                    f"No paired locomotion trials for {subject} {session}."
                )
            session_query = (
                SpikeSorting()
                & f'subject_name = "{subject}"'
                & f'session_name = "{session}"'
            ).proj()
            sampling_rate_hz = float(
                (EphysRecording.ProbeSetting() & session_query).fetch1("sampling_rate")
            )
            self._session_cache[cache_key] = (
                stationary_event_times,
                movement_event_times,
                sampling_rate_hz,
                session_query,
            )
        else:
            (
                stationary_event_times,
                movement_event_times,
                sampling_rate_hz,
                session_query,
            ) = self._session_cache[cache_key]

        if stationary_event_times.size == 0 or movement_event_times.size == 0:
            raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")

        unit_query = SpikeSorting.Unit & session_query & f"unit_id = {unit_id}"
        spike_times_samples = unit_query.fetch1("spike_times")
        spike_times = np.asarray(spike_times_samples, dtype=float) / sampling_rate_hz

        stationary_peth, _, bin_centers_s = compute_population_peth(
            [spike_times],
            stationary_event_times,
            **PETH_KWARGS,
        )
        movement_peth, _, _ = compute_population_peth(
            [spike_times],
            movement_event_times,
            **PETH_KWARGS,
        )

        stationary_mean_rate = stationary_peth.mean(axis=1)[0]
        movement_mean_rate = movement_peth.mean(axis=1)[0]
        baseline_mask = (bin_centers_s >= BASELINE_WINDOW[0]) & (
            bin_centers_s < BASELINE_WINDOW[1]
        )
        stationary_baseline_rate = stationary_mean_rate[baseline_mask].mean()

        response_mask = (bin_centers_s >= RESP_WINDOW[0]) & (
            bin_centers_s < RESP_WINDOW[1]
        )
        response_bin_centers_s = bin_centers_s[response_mask]
        stationary_response = (
            stationary_mean_rate[response_mask] - stationary_baseline_rate
        )
        movement_response = movement_mean_rate[response_mask] - stationary_baseline_rate

        stationary_peak_idx = int(np.argmax(stationary_response))
        movement_peak_idx = int(np.argmax(movement_response))

        self.insert1(
            {
                **key,
                "stat_peak": float(stationary_response[stationary_peak_idx]),
                "stat_latency": float(response_bin_centers_s[stationary_peak_idx]),
                "move_peak": float(movement_response[movement_peak_idx]),
                "move_latency": float(response_bin_centers_s[movement_peak_idx]),
            }
        )

    def plot(
        self,
        subject_sessions=None,
        *,
        output_dir="/Users/gabriel/lib/ephys/figures/locomotion",
        filename="condition_peak_from_locomotion_peaks",
        extension="pdf",
        dpi=300,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from ephys.src.utils.analysis_stats import mean_and_t_ci
        from matplotlib.widgets import Button, RadioButtons, TextBox

        if subject_sessions is None:
            subject_sessions = [
                ("GRB006", "20240821_121447"),
                ("GRB058", "20260312_134952"),
            ]

        rows_by_subject = []
        for subject, session in subject_sessions:
            restriction = {
                "subject_name": subject,
                "session_name": session,
                "unit_criteria_id": 1,
            }
            relation = self * self.key_source & restriction & "passes = 1"
            rows = relation.fetch("stat_peak", "move_peak", as_dict=True)
            if not rows:
                raise RuntimeError(
                    f"No LocomotionPeaks rows found for {subject} {session}."
                )
            stat_peak = np.asarray([row["stat_peak"] for row in rows], dtype=float)
            move_peak = np.asarray([row["move_peak"] for row in rows], dtype=float)
            rows_by_subject.append(
                {
                    "subject": subject,
                    "stat_peak": stat_peak,
                    "move_peak": move_peak,
                }
            )

        fig, ax = plt.subplots(figsize=(7.2, 7.0))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.34, top=0.96)
        subject_colors = sns.color_palette("Set1")
        plotted_values = []
        for subject_index, subject_rows in enumerate(rows_by_subject):
            color = subject_colors[subject_index]
            stat_peak = subject_rows["stat_peak"]
            move_peak = subject_rows["move_peak"]
            plotted_stat = np.maximum(stat_peak, 0) + 0.1
            plotted_move = np.maximum(move_peak, 0) + 0.1
            plotted_values.extend([plotted_stat, plotted_move])
            ax.scatter(
                plotted_stat,
                plotted_move,
                s=18,
                alpha=0.2,
                color=color,
                linewidths=0,
                label="_nolegend_",
                zorder=2,
            )
            mean_x, lower_x, upper_x = mean_and_t_ci(
                plotted_stat,
                log_scale=True,
                ci_level=0.95,
                drop_nonfinite=False,
            )
            mean_y, lower_y, upper_y = mean_and_t_ci(
                plotted_move,
                log_scale=True,
                ci_level=0.95,
                drop_nonfinite=False,
            )
            ax.errorbar(
                mean_x,
                mean_y,
                xerr=np.array([[mean_x - lower_x], [upper_x - mean_x]]),
                yerr=np.array([[mean_y - lower_y], [upper_y - mean_y]]),
                fmt="o",
                ms=9,
                color=color,
                mfc=color,
                mec="white",
                mew=0.8,
                elinewidth=1.2,
                ecolor=color,
                capsize=2.5,
                alpha=0.95,
                zorder=5,
                label=subject_rows["subject"],
            )

        all_peak_values = np.concatenate(plotted_values)
        lower_limit = 0.1
        upper_limit = max(1.0, float(np.percentile(all_peak_values, 99) * 1.05))
        ax.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            "k--",
            alpha=0.4,
            lw=0.8,
        )
        ax.set_xlim(lower_limit, upper_limit)
        ax.set_ylim(lower_limit, upper_limit)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal")
        ax.set_xlabel("Stationary peak (baseline-corrected sp/s)")
        ax.set_ylabel("Movement peak (baseline-corrected sp/s)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        dir_box = TextBox(fig.add_axes((0.12, 0.23, 0.52, 0.04)), "Dir", output_dir)
        file_box = TextBox(fig.add_axes((0.12, 0.17, 0.52, 0.04)), "File", filename)
        dpi_box = TextBox(fig.add_axes((0.12, 0.11, 0.18, 0.04)), "DPI", str(dpi))
        ext_buttons = RadioButtons(
            fig.add_axes((0.72, 0.11, 0.12, 0.16)),
            ("pdf", "png", "svg"),
            active=("pdf", "png", "svg").index(extension),
        )
        save_button = Button(fig.add_axes((0.72, 0.04, 0.16, 0.05)), "Save")
        status_text = fig.text(0.12, 0.035, "", fontsize=8)

        def save(_event):
            from pathlib import Path

            target_dir = Path(dir_box.text).expanduser()
            target_dir.mkdir(parents=True, exist_ok=True)
            selected_extension = ext_buttons.value_selected
            target_name = Path(file_box.text).stem + f".{selected_extension}"
            try:
                target_dpi = int(dpi_box.text)
            except ValueError:
                status_text.set_text("DPI must be an integer.")
                fig.canvas.draw_idle()
                return
            target_path = target_dir / target_name
            fig.savefig(target_path, bbox_inches="tight", dpi=target_dpi)
            status_text.set_text(f"Saved {target_path}")
            fig.canvas.draw_idle()

        save_button.on_clicked(save)
        plt.show()
        return fig


@rojasbowe_schema
class PsychophysicalKernelParam(dj.Lookup):
    """Analysis params for ``PsychophysicalKernel`` (Odoemene-style logistic kernel)."""

    definition = """
    kernel_param_id                      : varchar(48)
    ---
    timebins                             : int
    bin_width_s                          : float
    cv_splits                            : int
    random_state                         : int
    max_rate_hz                          : float
    evidence_encoding                    : varchar(32)  # max_rate | trial_rate
    min_trials_per_bin                   : int
    regularization_c                     : float
    observation_window                   : varchar(32)  # center_exit | response
    analysis_version                     : varchar(32)
    """
    contents = [  # noqa: RUF012
        # Fixation-only: flashes until center-port exit.
        (
            "v1_100ms_10bin_center",
            10,
            0.1,
            10,
            0,
            20.0,
            "max_rate",
            50,
            1.0,
            "center_exit",
            "v1",
        ),
        # Through response: flashes until chosen response-port poke.
        (
            "v1_100ms_10bin_response",
            10,
            0.1,
            10,
            0,
            20.0,
            "max_rate",
            50,
            1.0,
            "response",
            "v1",
        ),
        # Locked primary: fixation-only, centered on each trial's generative rate.
        (
            "v2_100ms_10bin_center_rate",
            10,
            0.1,
            10,
            0,
            20.0,
            "trial_rate",
            50,
            1.0,
            "center_exit",
            "v2",
        ),
        # Window sensitivity: includes movement-period flashes through response.
        (
            "v2_100ms_10bin_response_rate",
            10,
            0.1,
            10,
            0,
            20.0,
            "trial_rate",
            50,
            1.0,
            "response",
            "v2",
        ),
    ]


@rojasbowe_schema
class SessionPsychophysicalKernel(dj.Computed):
    """Session psychophysical kernel (logistic reverse correlation).

    See ``ephys.src.utils.psychophysical_kernel`` and Odoemene et al. 2018
    doi:10.1523/JNEUROSCI.3478-17.2018. Requires ``EventMapping`` + Chipmunk;
    missing NIDAQ mappings raise (no Bpod-timestamp fallback).
    """

    definition = """
    -> Session
    -> PsychophysicalKernelParam
    ---
    n_trials                             : int
    n_trials_fit                         : int
    n_bins_fit                           : int
    weights                              : longblob  # cv x timebins; kernel folds
    weights_mean                         : longblob  # psychophysical kernel w(t)
    weights_error                        : longblob  # approx. SE on w(t)
    n_observed_per_bin                   : longblob  # trials with finite evidence
    bin_centers_s                        : longblob  # time from first flash
    scores                               : longblob  # CV holdout accuracy
    score_mean                           : float
    majority_accuracy = NULL             : float
    score_above_majority = NULL          : float
    bias                                 : longblob  # CV intercepts β0
    bias_mean                            : float
    interpretation                       : varchar(32)  # early/late/flat/failed
    wait_time_mean                       : float
    wait_time_std                        : float
    response_time_mean                   : float
    response_time_std                    : float
    fit_converged                        : tinyint
    """

    @property
    def key_source(self):
        return (Session * PsychophysicalKernelParam) & EventMapping.proj()

    def make(self, key):
        from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata
        from ephys.src.utils.io_digital_events import fetch_session_events
        from ephys.src.utils.psychophysical_kernel import (
            compute_session_psychophysical_kernel,
        )
        from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table

        params = (PsychophysicalKernelParam() & key).fetch1()
        subject = key["subject_name"]
        session = key["session_name"]

        align_ev = fetch_session_events(subject, session)
        trial_df = fetch_trial_metadata(subject, session, align_ev)
        if trial_df is None:
            raise RuntimeError(
                f"Could not load Chipmunk trial metadata for {subject} {session}."
            )
        trial_df = enrich_chipmunk_trial_table(trial_df)

        result = compute_session_psychophysical_kernel(
            align_ev,
            trial_df,
            timebins=int(params["timebins"]),
            bin_width_s=float(params["bin_width_s"]),
            max_rate_hz=float(params["max_rate_hz"]),
            cv_splits=int(params["cv_splits"]),
            random_state=int(params["random_state"]),
            min_trials_per_bin=int(params["min_trials_per_bin"]),
            regularization_C=float(params["regularization_c"]),
            observation_window=str(params["observation_window"]),
            evidence_encoding=str(params["evidence_encoding"]),
        )
        if not result["fit_converged"]:
            raise RuntimeError(
                f"Psychophysical kernel fit failed for {subject} {session} "
                f"(n_trials={result['n_trials']}, "
                f"n_observed_per_bin={result['n_observed_per_bin'].tolist()})."
            )

        self.insert1(
            {
                **key,
                "n_trials": int(result["n_trials"]),
                "n_trials_fit": int(result["n_trials_fit"]),
                "n_bins_fit": int(result["n_bins_fit"]),
                "weights": result["weights"],
                "weights_mean": result["weights_mean"],
                "weights_error": result["weights_error"],
                "n_observed_per_bin": result["n_observed_per_bin"],
                "bin_centers_s": result["bin_centers_s"],
                "scores": result["scores"],
                "score_mean": float(result["score_mean"]),
                "majority_accuracy": float(result["majority_accuracy"]),
                "score_above_majority": float(result["score_above_majority"]),
                "bias": result["bias"],
                "bias_mean": float(result["bias_mean"]),
                "interpretation": result["interpretation"],
                "wait_time_mean": float(result["wait_time_mean"]),
                "wait_time_std": float(result["wait_time_std"]),
                "response_time_mean": float(result["response_time_mean"]),
                "response_time_std": float(result["response_time_std"]),
                "fit_converged": 1,
            }
        )
