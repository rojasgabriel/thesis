import datajoint as dj
import numpy as np
from labdata.schema import (
    EphysRecording,
    SpikeSorting,
    UnitCount,
    get_user_schema,
)

rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class EventMapping(dj.Lookup):
    definition = """
    stream_name                          : varchar(54)
    event_name                           : varchar(54)
    ---
    event_role                           : varchar(54)
    """
    contents = [
        ("nidq", "ai0", "visual_stim"),
        ("nidq", "0", "trig"),
        ("nidq", "1", "frames"),
        ("nidq", "2", "trial_start"),
        ("nidq", "3", "left_port"),
        ("nidq", "4", "center_port"),
        ("nidq", "5", "right_port"),
        ("obx", "io0", "visual_stim"),
        ("obx", "io2", "trial_start"),
        ("obx", "io3", "frames"),
        ("obx", "io4", "left_port"),
        ("obx", "io5", "center_port"),
        ("obx", "io6", "right_port"),
    ]


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
        from thesis.ephys.config.locomotion import (
            BASELINE_WINDOW,
            PETH_KWARGS,
            RESP_WINDOW,
        )
        from thesis.ephys.utils.analysis_conditioned_stim import (
            build_trial_stim_classification,
            extract_conditioned_stim_anchors,
        )
        from thesis.ephys.utils.analysis_peth import compute_population_peth
        from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
        from thesis.ephys.utils.io_digital_events import fetch_session_events
        from thesis.ephys.utils.trial_alignment import enrich_chipmunk_trial_table

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
