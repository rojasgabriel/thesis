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

    @staticmethod
    def compute(
        spike_times,
        spike_amplitudes,
        unit_ids,
        recording_end,
        *,
        n_time_windows=5,
        max_amplitude_drift=0.10,
        dip_alpha=0.05,
        max_dip_samples=72_000,
    ):
        """Return amplitude-stability and unimodality results for each unit."""
        import numpy as np
        import pandas as pd
        from diptest import diptest
        from statsmodels.stats.multitest import multipletests

        if not (len(spike_times) == len(spike_amplitudes) == len(unit_ids)):
            raise ValueError(
                "Spike times, amplitudes, and unit IDs must have equal length."
            )
        if not unit_ids:
            raise ValueError("At least one unit is required.")

        stability_rows = []
        time_edges = np.linspace(0, recording_end, n_time_windows + 1)[1:-1]
        for unit_id, unit_spike_times, unit_amplitudes in zip(
            unit_ids, spike_times, spike_amplitudes, strict=True
        ):
            unit_spike_times = np.asarray(unit_spike_times)
            unit_amplitudes = np.asarray(unit_amplitudes)
            if len(unit_spike_times) != len(unit_amplitudes) or not len(
                unit_amplitudes
            ):
                raise ValueError(f"Unit {unit_id} has invalid spike amplitude data.")

            time_window_index = np.digitize(unit_spike_times, time_edges)
            window_means = np.array(
                [
                    unit_amplitudes[time_window_index == index].mean()
                    if np.any(time_window_index == index)
                    else np.nan
                    for index in range(n_time_windows)
                ]
            )
            mean_amplitude = unit_amplitudes.mean()
            amplitude_drift = (
                (np.nanmax(window_means) - np.nanmin(window_means)) / mean_amplitude
                if mean_amplitude
                else np.nan
            )

            dip_sample = unit_amplitudes
            if len(dip_sample) > max_dip_samples:
                dip_sample = np.random.default_rng(0).choice(
                    dip_sample, max_dip_samples, replace=False
                )
            dip_statistic, dip_p_value = diptest(dip_sample)
            stability_rows.append(
                {
                    "unit_id": unit_id,
                    "amplitude_drift": amplitude_drift,
                    "dip_statistic": dip_statistic,
                    "dip_p_value": dip_p_value,
                    "dip_sample_size": len(dip_sample),
                }
            )

        stability_results = pd.DataFrame(stability_rows)
        _, stability_results["dip_q_value"], _, _ = multipletests(
            stability_results["dip_p_value"], alpha=dip_alpha, method="fdr_bh"
        )
        stability_results["passes_amplitude_stability"] = np.isfinite(
            stability_results["amplitude_drift"]
        ) & (stability_results["amplitude_drift"] <= max_amplitude_drift)
        stability_results["passes_unimodality"] = (
            stability_results["dip_q_value"] >= dip_alpha
        )
        stability_results["passes"] = (
            stability_results["passes_amplitude_stability"]
            & stability_results["passes_unimodality"]
        )
        return stability_results

    def make(self, key, **_kwargs):
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
        results = self.compute(
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

    @staticmethod
    def compute(
        peth,
        bin_edges,
        unit_ids,
        baseline_window,
        response_window,
        alpha,
    ):
        """Compare each unit's paired baseline and stimulus-response rates."""
        import numpy as np
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        n_units = peth.shape[0]
        if len(unit_ids) != n_units:
            raise ValueError(f"len(unit_ids)={len(unit_ids)} != peth n_units={n_units}")

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        baseline_bins = (bin_centers >= baseline_window[0]) & (
            bin_centers < baseline_window[1]
        )
        response_bins = (bin_centers >= response_window[0]) & (
            bin_centers < response_window[1]
        )
        if not baseline_bins.any() or not response_bins.any():
            raise ValueError(
                "Baseline or response window does not overlap the available bins"
            )

        baseline_rates = peth[:, :, baseline_bins].mean(axis=2)
        response_rates = peth[:, :, response_bins].mean(axis=2)
        response_deltas = response_rates - baseline_rates
        p_values = np.ones(n_units, dtype=float)
        for unit_index, unit_deltas in enumerate(response_deltas):
            if np.allclose(unit_deltas, 0):
                continue
            try:
                _, p_values[unit_index] = stats.wilcoxon(
                    response_rates[unit_index],
                    baseline_rates[unit_index],
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError:
                pass

        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        mean_baseline_rates = baseline_rates.mean(axis=1)
        mean_response_rates = response_rates.mean(axis=1)
        mean_response_deltas = response_deltas.mean(axis=1)
        excited = (q_values < alpha) & (mean_response_deltas > 0)
        suppressed = (q_values < alpha) & (mean_response_deltas < 0)
        return pd.DataFrame(
            {
                "unit": unit_ids,
                "mean_base": mean_baseline_rates,
                "mean_resp": mean_response_rates,
                "delta": mean_response_deltas,
                "p": p_values,
                "q": q_values,
            }
        ), {"excited": excited, "suppressed": suppressed}

    @staticmethod
    def classify_components(
        peth,
        bin_centers,
        unit_ids,
        *,
        search_window,
        baseline_window,
        min_prominence_fraction,
        min_prominence_spikes_per_second,
        min_peak_distance_ms,
        binwidth_ms,
        mode="peaks",
    ):
        """Find peaks or dips in each unit's trial-averaged response."""
        import numpy as np
        import pandas as pd
        from scipy.signal import find_peaks, peak_prominences

        if mode not in ("peaks", "dips"):
            raise ValueError("mode must be 'peaks' or 'dips'")
        if len(unit_ids) != peth.shape[0]:
            raise ValueError(
                f"len(unit_ids)={len(unit_ids)} != peth n_units={peth.shape[0]}"
            )

        baseline_bins = (bin_centers >= baseline_window[0]) & (
            bin_centers < baseline_window[1]
        )
        search_bins = (bin_centers >= search_window[0]) & (
            bin_centers < search_window[1]
        )
        search_indices = np.flatnonzero(search_bins)
        if not search_indices.size:
            raise ValueError("search_window does not overlap available bins.")

        component_rows = []
        for unit_index, unit_id in enumerate(unit_ids):
            mean_peth = peth[unit_index].mean(axis=0)
            baseline = mean_peth[baseline_bins].mean() if baseline_bins.any() else 0.0
            response_delta = mean_peth - baseline
            response_signal = -response_delta if mode == "dips" else response_delta
            max_signal = float(response_signal[search_bins].max())
            prominence = max(
                min_prominence_fraction * max_signal,
                min_prominence_spikes_per_second,
            )
            component_indices, _ = find_peaks(
                response_signal,
                prominence=prominence,
                distance=max(1, round(min_peak_distance_ms / binwidth_ms)),
            )
            component_indices = component_indices[search_bins[component_indices]]
            component_indices = component_indices[
                response_delta[component_indices] < 0
                if mode == "dips"
                else response_delta[component_indices] > 0
            ]

            if not component_indices.size and max_signal > 0:
                best_index = int(
                    search_indices[np.argmax(response_signal[search_bins])]
                )
                if (
                    0 < best_index < len(response_signal) - 1
                    and response_signal[best_index] > response_signal[best_index - 1]
                    and response_signal[best_index] > response_signal[best_index + 1]
                    and peak_prominences(response_signal, [best_index])[0][0]
                    >= prominence
                ):
                    component_indices = np.array([best_index])

            component_rows.append(
                {
                    "unit": unit_id,
                    "n_peaks": len(component_indices),
                    "peak_times": bin_centers[component_indices].tolist(),
                    "peak_heights": mean_peth[component_indices].tolist(),
                }
            )

        return pd.DataFrame(component_rows)

    @classmethod
    def fetch_excited_unit_ids(
        cls,
        subject,
        session,
        unit_criteria_id=1,
        responsiveness_param_id=0,
        stability_param_id=None,
    ):
        """Return a set of stored unit IDs classified as stimulus excited."""
        responsiveness_query = {
            "subject_name": subject,
            "session_name": session,
            "unit_criteria_id": unit_criteria_id,
            "responsiveness_param_id": responsiveness_param_id,
        }
        if len(cls & responsiveness_query) == 0:
            raise ValueError(
                "StimulusResponsiveness has not been populated for "
                f"{responsiveness_query}"
            )
        excited_units = cls.Unit & responsiveness_query & {"response_type": "excited"}
        if stability_param_id is not None:
            stability_query = {
                "subject_name": subject,
                "session_name": session,
                "unit_criteria_id": unit_criteria_id,
                "unit_stability_param_id": stability_param_id,
            }
            if len(UnitStability & stability_query) == 0:
                raise ValueError(
                    f"UnitStability has not been populated for {stability_query}"
                )
            excited_units &= UnitStability.Unit & stability_query & {"passes": 1}

        unit_ids = [int(unit_id) for unit_id in excited_units.fetch("unit_id")]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Unit IDs are not unique across this session")
        return set(unit_ids)

    def make(self, key, **_kwargs):
        import numpy as np
        from spks.event_aligned import population_peth

        from thesis.ephys.events import fetch_session_events

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
        response_stats, masks = self.compute(
            peth,
            bin_edges,
            unit_ids,
            baseline_window=(params["baseline_start"], params["baseline_end"]),
            response_window=(params["response_start"], params["response_end"]),
            alpha=params["alpha"],
        )

        components = {}
        for response_type, mode in (("excited", "peaks"), ("suppressed", "dips")):
            indices = np.flatnonzero(masks[response_type])
            rows = self.classify_components(
                peth[indices],
                bin_centers,
                [unit_ids[index] for index in indices],
                search_window=(
                    params["peak_search_start"],
                    params["peak_search_end"],
                ),
                baseline_window=(params["baseline_start"], params["baseline_end"]),
                min_prominence_fraction=params["min_prominence_fraction"],
                min_prominence_spikes_per_second=params[
                    "min_prominence_spikes_per_second"
                ],
                min_peak_distance_ms=params["min_peak_distance_ms"],
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
