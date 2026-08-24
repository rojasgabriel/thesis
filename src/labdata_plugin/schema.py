import warnings

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

import datajoint as dj  # noqa: E402
from labdata.schema import (  # noqa: E402
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
        {"stream_name": "nidq", "event_name": "ai0", "event_role": "visual_stim"},
        {"stream_name": "nidq", "event_name": "0", "event_role": "trig"},
        {"stream_name": "nidq", "event_name": "1", "event_role": "frames"},
        {"stream_name": "nidq", "event_name": "2", "event_role": "trial_start"},
        {"stream_name": "nidq", "event_name": "3", "event_role": "left_port"},
        {"stream_name": "nidq", "event_name": "4", "event_role": "center_port"},
        {"stream_name": "nidq", "event_name": "5", "event_role": "right_port"},
        {"stream_name": "obx", "event_name": "io0", "event_role": "visual_stim"},
        {"stream_name": "obx", "event_name": "io2", "event_role": "trial_start"},
        {"stream_name": "obx", "event_name": "io3", "event_role": "frames"},
        {"stream_name": "obx", "event_name": "io4", "event_role": "left_port"},
        {"stream_name": "obx", "event_name": "io5", "event_role": "center_port"},
        {"stream_name": "obx", "event_name": "io6", "event_role": "right_port"},
    ]


@rojasbowe_schema
class StimulusResponsivenessParams(dj.Lookup):
    definition = """
    responsiveness_param_id               : tinyint
    ---
    baseline_start                         : float  # seconds from first stimulus
    baseline_end                           : float
    response_start                         : float
    response_end                           : float
    peth_pre_seconds                       : float
    peth_post_seconds                      : float
    binwidth_ms                            : float
    alpha                                  : float
    peak_search_start                      : float
    peak_search_end                        : float
    min_prominence_fraction                : float
    min_prominence_spikes_per_second       : float
    min_peak_distance_ms                   : float
    """

    contents = [
        {
            "responsiveness_param_id": 0,
            "baseline_start": -0.04,
            "baseline_end": 0.0,
            "response_start": 0.03,
            "response_end": 0.12,
            "peth_pre_seconds": 0.1,
            "peth_post_seconds": 0.15,
            "binwidth_ms": 10.0,
            "alpha": 0.05,
            "peak_search_start": 0.0,
            "peak_search_end": 0.12,
            "min_prominence_fraction": 0.25,
            "min_prominence_spikes_per_second": 1.0,
            "min_peak_distance_ms": 20.0,
        }
    ]


@rojasbowe_schema
class UnitStabilityParams(dj.Lookup):
    definition = """
    unit_stability_param_id                : tinyint
    ---
    n_time_windows                         : tinyint
    max_amplitude_drift                    : float
    dip_alpha                              : float
    max_dip_samples                        : int
    """

    contents = [
        {
            "unit_stability_param_id": 0,
            "n_time_windows": 5,
            "max_amplitude_drift": 0.10,
            "dip_alpha": 0.05,
            "max_dip_samples": 72_000,
        }
    ]


@rojasbowe_schema
class UnitStability(dj.Computed):
    definition = """
    -> UnitCount
    -> UnitStabilityParams
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        amplitude_drift                      : float
        dip_statistic                        : float
        dip_p_value                          : float
        dip_q_value                          : float
        dip_sample_size                      : int
        passes_amplitude_stability            : tinyint
        passes_unimodality                    : tinyint
        passes                                : tinyint
        """

    def make(self, key, **_kwargs):
        from thesis.ephys.unit_stability import compute_unit_stability

        params = (UnitStabilityParams & key).fetch1()
        unit_keys = (UnitCount.Unit & key & "passes = 1").fetch("KEY")
        units = sorted(
            (SpikeSorting.Unit & unit_keys).fetch(
                "unit_id", "spike_times", "spike_amplitudes", as_dict=True
            ),
            key=lambda row: row["unit_id"],
        )
        if not units:
            raise ValueError(f"No passing units for {key}")

        recording_duration, sampling_rate = (
            EphysRecording * EphysRecording.ProbeSetting & key
        ).fetch1("recording_duration", "sampling_rate")
        results = compute_unit_stability(
            spike_times=[row["spike_times"] for row in units],
            spike_amplitudes=[row["spike_amplitudes"] for row in units],
            unit_ids=[row["unit_id"] for row in units],
            recording_end=float(recording_duration) * float(sampling_rate),
            n_time_windows=params["n_time_windows"],
            max_amplitude_drift=params["max_amplitude_drift"],
            dip_alpha=params["dip_alpha"],
            max_dip_samples=params["max_dip_samples"],
        )

        self.insert1(key)
        self.Unit.insert([{**key, **row} for row in results.to_dict("records")])


@rojasbowe_schema
class StimulusResponsiveness(dj.Computed):
    definition = """
    -> UnitCount
    -> StimulusResponsivenessParams
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        mean_baseline_rate                 : float  # spikes per second
        mean_response_rate                 : float  # spikes per second
        response_delta                     : float  # response minus baseline
        response_p_value                   : float
        response_q_value                   : float
        response_type                      : enum('none', 'excited', 'suppressed')
        n_response_components              : tinyint
        response_component_times           : blob   # seconds from first stimulus
        response_component_rates           : blob   # spikes per second
        """

    def make(self, key, **_kwargs):
        import numpy as np
        from spks.event_aligned import population_peth

        from thesis.ephys.events import fetch_session_events
        from thesis.ephys.peaks import classify_peak_count
        from thesis.ephys.responsiveness import compute_unit_responsiveness

        params = (StimulusResponsivenessParams & key).fetch1()
        unit_keys = (UnitCount.Unit & key & "passes = 1").fetch("KEY")
        units = (SpikeSorting.Unit & unit_keys).get_spike_times()
        if not units:
            raise ValueError(f"No passing units for {key}")

        unit_ids = [row["unit_id"] for row in units]
        _, stimulus_pulses = fetch_session_events(
            key["subject_name"], key["session_name"]
        )
        first_stimulus = stimulus_pulses.loc[
            stimulus_pulses["first_in_train"], "timestamp"
        ].to_numpy(dtype=float)
        if first_stimulus.size == 0:
            raise ValueError(f"No first-stimulus events for {key}")

        peth, bin_edges, _ = population_peth(
            all_spike_times=[row["spike_times"] for row in units],
            alignment_times=first_stimulus,
            pre_seconds=params["peth_pre_seconds"],
            post_seconds=params["peth_post_seconds"],
            binwidth_ms=params["binwidth_ms"],
        )
        peth = peth / (params["binwidth_ms"] / 1000)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        response_stats, masks = compute_unit_responsiveness(
            peth,
            bin_edges,
            unit_ids,
            base_window=(params["baseline_start"], params["baseline_end"]),
            resp_window=(params["response_start"], params["response_end"]),
            alpha=params["alpha"],
        )

        components = {}
        for response_type, mode in (("excited", "peaks"), ("suppressed", "dips")):
            indices = np.flatnonzero(masks[response_type])
            rows = classify_peak_count(
                peth[indices],
                bin_centers,
                [unit_ids[index] for index in indices],
                search_window=(
                    params["peak_search_start"],
                    params["peak_search_end"],
                ),
                baseline_window=(params["baseline_start"], params["baseline_end"]),
                min_prominence_frac=params["min_prominence_fraction"],
                min_prominence_abs=params["min_prominence_spikes_per_second"],
                min_distance_ms=params["min_peak_distance_ms"],
                binwidth_ms=params["binwidth_ms"],
                mode=mode,
            )
            components.update(
                {
                    row["unit"]: (
                        response_type,
                        row["n_peaks"],
                        row["peak_times"],
                        row["peak_heights"],
                    )
                    for row in rows.to_dict("records")
                }
            )

        part_rows = []
        for row in response_stats.to_dict("records"):
            response_type, count, times, rates = components.get(
                row["unit"], ("none", 0, [], [])
            )
            part_rows.append(
                {
                    **key,
                    "unit_id": row["unit"],
                    "mean_baseline_rate": row["mean_base"],
                    "mean_response_rate": row["mean_resp"],
                    "response_delta": row["delta"],
                    "response_p_value": row["p"],
                    "response_q_value": row["q"],
                    "response_type": response_type,
                    "n_response_components": count,
                    "response_component_times": np.asarray(times, dtype=float),
                    "response_component_rates": np.asarray(rates, dtype=float),
                }
            )

        self.insert1(key)
        self.Unit.insert(part_rows)
