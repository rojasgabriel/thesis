"""Insert unit-quality criteria and populate unit counts."""

import warnings

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

from labdata.schema import (  # noqa: E402
    SpikeSorting,
    UnitCount,
    UnitCountCriteria,
    UnitMetrics,
)

CRITERIA = [
    {
        "unit_criteria_id": 0,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.1 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.6 & firing_rate > 1",
        "mua_criteria": None,
    },
    {
        "unit_criteria_id": 1,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.1 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.6 & firing_rate > 1 & depth_drift_start_to_end < 6",
        "mua_criteria": None,
    },
    {
        "unit_criteria_id": 2,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.01 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.9 & firing_rate > 2 & depth_drift_start_to_end < 6",
        "mua_criteria": None,
    },
]


def main() -> None:
    UnitCountCriteria().insert(CRITERIA, skip_duplicates=True)

    missing_metrics = SpikeSorting.Unit - UnitMetrics
    if len(missing_metrics):
        raise RuntimeError(
            f"UnitMetrics is incomplete for {len(missing_metrics)} units. "
            "Populate UnitMetrics before UnitCount."
        )

    print(UnitCount().populate())


if __name__ == "__main__":
    main()
