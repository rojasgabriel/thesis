"""First-flash response class of each unit (excited, suppressed, not responsive)."""

from ._base import SpikeSorting, UnitCount, dj, rojasbowe_schema  # noqa: F401


@rojasbowe_schema
class StimulusResponseParam(dj.Lookup):
    definition = """
    stim_response_param_id                 : tinyint
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

    default_contents = [
        {
            "stim_response_param_id": 0,
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
class StimulusResponse(dj.Computed):
    """Task-only first-pulse responses; see thesis.ephys.task_events for refresh requirements."""

    definition = """
    -> UnitCount
    -> StimulusResponseParam
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        baseline_rate                      : float  # mean baseline rate in spikes per second
        response_rate                      : float  # mean response rate in spikes per second
        rate_delta                         : float  # response rate minus baseline rate
        p_value                            : float  # paired baseline-response test p-value
        q_value                            : float  # FDR-adjusted p-value
        response_type                      : enum('none', 'excited', 'suppressed')
        n_components                       : tinyint # response peaks or dips
        component_latencies_s              : blob    # seconds from first stimulus
        component_rates                    : blob    # spikes per second at each component
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

        q_values = stats.false_discovery_control(p_values, method="bh")
        mean_baseline_rates = baseline_rates.mean(axis=1)
        mean_response_rates = response_rates.mean(axis=1)
        mean_response_deltas = response_deltas.mean(axis=1)
        excited = (q_values < alpha) & (mean_response_deltas > 0)
        suppressed = (q_values < alpha) & (mean_response_deltas < 0)
        return pd.DataFrame(
            {
                "unit_id": unit_ids,
                "baseline_rate": mean_baseline_rates,
                "response_rate": mean_response_rates,
                "rate_delta": mean_response_deltas,
                "p_value": p_values,
                "q_value": q_values,
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
                (
                    response_delta[component_indices] < 0
                    if mode == "dips"
                    else response_delta[component_indices] > 0
                )
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
                    "unit_id": unit_id,
                    "n_components": len(component_indices),
                    "component_latencies_s": bin_centers[component_indices].tolist(),
                    "component_rates": mean_peth[component_indices].tolist(),
                }
            )

        return pd.DataFrame(component_rows)

    def make(self, key, **_kwargs):
        import numpy as np
        from spks.event_aligned import population_peth

        from thesis.ephys.task_events import fetch_task_stimulus_pulses

        response_param = (StimulusResponseParam & key).fetch1()
        unit_keys = (UnitCount.Unit & key & "passes = 1").fetch("KEY")
        unit_records = (SpikeSorting.Unit & unit_keys).get_spike_times()
        if not unit_records:
            raise ValueError(f"No passing units for {key}")

        unit_ids = [unit_record["unit_id"] for unit_record in unit_records]
        stimulus_pulses = fetch_task_stimulus_pulses(
            key["subject_name"], key["session_name"]
        )
        first_stimulus = stimulus_pulses.loc[
            stimulus_pulses["first_in_train"], "timestamp"
        ].to_numpy(dtype=float)
        if first_stimulus.size == 0:
            raise ValueError(f"No first-stimulus events for {key}")

        peth, bin_edges, _ = population_peth(
            all_spike_times=[record["spike_times"] for record in unit_records],
            alignment_times=first_stimulus,
            pre_seconds=response_param["peth_pre_seconds"],
            post_seconds=response_param["peth_post_seconds"],
            binwidth_ms=response_param["binwidth_ms"],
        )
        peth = peth / (response_param["binwidth_ms"] / 1000)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        response_stats, masks = self.compute(
            peth,
            bin_edges,
            unit_ids,
            baseline_window=(
                response_param["baseline_start"],
                response_param["baseline_end"],
            ),
            response_window=(
                response_param["response_start"],
                response_param["response_end"],
            ),
            alpha=response_param["alpha"],
        )

        components = {}
        for response_type, mode in (("excited", "peaks"), ("suppressed", "dips")):
            indices = np.flatnonzero(masks[response_type])
            component_table = self.classify_components(
                peth[indices],
                bin_centers,
                [unit_ids[index] for index in indices],
                search_window=(
                    response_param["peak_search_start"],
                    response_param["peak_search_end"],
                ),
                baseline_window=(
                    response_param["baseline_start"],
                    response_param["baseline_end"],
                ),
                min_prominence_fraction=response_param["min_prominence_fraction"],
                min_prominence_spikes_per_second=response_param[
                    "min_prominence_spikes_per_second"
                ],
                min_peak_distance_ms=response_param["min_peak_distance_ms"],
                binwidth_ms=response_param["binwidth_ms"],
                mode=mode,
            )
            components.update(
                {
                    component_record["unit_id"]: (
                        response_type,
                        component_record["n_components"],
                        component_record["component_latencies_s"],
                        component_record["component_rates"],
                    )
                    for component_record in component_table.to_dict("records")
                }
            )

        unit_response_records = []
        for response_record in response_stats.to_dict("records"):
            response_type, count, times, rates = components.get(
                response_record["unit_id"], ("none", 0, [], [])
            )
            unit_response_records.append(
                {
                    **key,
                    "unit_id": response_record["unit_id"],
                    "baseline_rate": response_record["baseline_rate"],
                    "response_rate": response_record["response_rate"],
                    "rate_delta": response_record["rate_delta"],
                    "p_value": response_record["p_value"],
                    "q_value": response_record["q_value"],
                    "response_type": response_type,
                    "n_components": count,
                    "component_latencies_s": np.asarray(times, dtype=float),
                    "component_rates": np.asarray(rates, dtype=float),
                }
            )

        self.insert1(key)
        self.Unit.insert(unit_response_records)
