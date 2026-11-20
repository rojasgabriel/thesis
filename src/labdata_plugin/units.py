"""Unit quality: amplitude stability and movement artifacts."""

from ._base import (  # noqa: F401
    EphysRecording,
    SpikeSorting,
    UnitCount,
    dj,
    rojasbowe_schema,
)


@rojasbowe_schema
class UnitStabilityParam(dj.Lookup):
    definition = """
    stability_param_id                     : tinyint
    ---
    n_time_windows                         : tinyint  # equal recording windows used to measure amplitude drift
    max_amplitude_drift                    : float    # maximum fractional range of window-mean amplitudes
    dip_alpha                              : float    # FDR threshold for the amplitude dip test
    max_dip_samples                        : int      # maximum spike amplitudes sampled per unit
    """

    default_contents = [
        {
            "stability_param_id": 0,
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
    -> UnitStabilityParam
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        amplitude_drift                    : float    # range of window-mean amplitudes divided by the overall mean
        dip_statistic                      : float    # Hartigan dip statistic for spike amplitudes
        dip_p_value                        : float    # dip-test p-value
        dip_q_value                        : float    # FDR-adjusted dip-test p-value
        dip_sample_size                    : int      # spike amplitudes included in the dip test
        amplitude_stable                   : tinyint  # amplitude drift is within the configured limit
        amplitude_unimodal                 : tinyint  # dip q-value is at or above the configured alpha
        passes                             : tinyint  # amplitude is stable and unimodal
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
        """Return amplitude-stability and unimodality results for each unit.

        Adapted from code provided by Michael Ryan, Ph.D.
        """
        import numpy as np
        import pandas as pd
        from diptest import diptest
        from scipy.stats import false_discovery_control

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
                    (
                        unit_amplitudes[time_window_index == index].mean()
                        if np.any(time_window_index == index)
                        else np.nan
                    )
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
        stability_results["dip_q_value"] = false_discovery_control(
            stability_results["dip_p_value"], method="bh"
        )
        stability_results["amplitude_stable"] = np.isfinite(
            stability_results["amplitude_drift"]
        ) & (stability_results["amplitude_drift"] <= max_amplitude_drift)
        stability_results["amplitude_unimodal"] = (
            stability_results["dip_q_value"] >= dip_alpha
        )
        stability_results["passes"] = (
            stability_results["amplitude_stable"]
            & stability_results["amplitude_unimodal"]
        )
        return stability_results

    def make(self, key, **_kwargs):
        stability_param = (UnitStabilityParam & key).fetch1()
        unit_keys = (UnitCount.Unit & key & "passes = 1").fetch("KEY")
        unit_records = sorted(
            (SpikeSorting.Unit & unit_keys).fetch(
                "unit_id", "spike_times", "spike_amplitudes", as_dict=True
            ),
            key=lambda unit_record: unit_record["unit_id"],
        )
        if not unit_records:
            raise ValueError(f"No passing units for {key}")

        recording_duration, sampling_rate = (
            EphysRecording * EphysRecording.ProbeSetting & key
        ).fetch1("recording_duration", "sampling_rate")
        stability_table = self.compute(
            spike_times=[record["spike_times"] for record in unit_records],
            spike_amplitudes=[record["spike_amplitudes"] for record in unit_records],
            unit_ids=[record["unit_id"] for record in unit_records],
            recording_end=float(recording_duration) * float(sampling_rate),
            n_time_windows=stability_param["n_time_windows"],
            max_amplitude_drift=stability_param["max_amplitude_drift"],
            dip_alpha=stability_param["dip_alpha"],
            max_dip_samples=stability_param["max_dip_samples"],
        )

        self.insert1(key)
        self.Unit.insert(
            [
                {**key, **stability_record}
                for stability_record in stability_table.to_dict("records")
            ]
        )


@rojasbowe_schema
class MovementArtifactParam(dj.Lookup):
    definition = """
    movement_param_id                      : tinyint
    ---
    max_in_port_ratio                      : float  # in-port rate / recording rate below this: silent in the port
    min_fw3m_ms                            : float  # spike width at 3/4 of the trough above this: broad
    """
    default_contents = [
        {"movement_param_id": 0, "max_in_port_ratio": 0.1, "min_fw3m_ms": 0.25}
    ]


@rojasbowe_schema
class MovementArtifact(dj.Computed):
    """Stable units that look like movement artifacts (preprocessing.movement_artifacts).

    A unit is an artifact when it is silent in the center port AND has a broad
    spike. Units: the units of the UnitStability key that pass it.
    """

    definition = """
    -> UnitStability
    -> MovementArtifactParam
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        in_port_ratio                      : float    # in-port rate / recording rate
        fw3m_ms                            : float    # spike width at 3/4 of the trough
        artifact                           : tinyint  # silent in the port and broad
        """

    def make(self, key, **_kwargs):
        from thesis.ephys.preprocessing.movement_artifacts import (
            is_artifact,
            score_session,
        )

        param = (MovementArtifactParam & key).fetch1()
        units = score_session(
            key["subject_name"],
            key["session_name"],
            unit_criteria_id=key["unit_criteria_id"],
            stability_param_id=key["stability_param_id"],
        )
        artifact = is_artifact(
            units["in_port_ratio"],
            units["fw3m"],
            param["max_in_port_ratio"],
            param["min_fw3m_ms"],
        )
        self.insert1(key)
        self.Unit.insert(
            {
                **key,
                "unit_id": int(unit_id),
                "in_port_ratio": ratio,
                "fw3m_ms": fw3m,
                "artifact": int(flag),
            }
            for unit_id, ratio, fw3m, flag in zip(
                units["unit_id"],
                units["in_port_ratio"],
                units["fw3m"],
                artifact,
                strict=True,
            )
        )
