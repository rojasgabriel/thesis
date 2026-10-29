"""Insert unit-quality criteria and populate unit counts."""

import argparse

# Criteria as stored in the lab database (Joao Couto, 2026-10-24, labdata 0.1.25):
# depth drift is an absolute distance, and noise units get passes = -1.
MUA_CRITERIA = "span_um <= 350 & spike_duration >= 0.05"
CRITERIA = [
    {
        "unit_criteria_id": 0,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.1 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.6 & firing_rate > 1",
        "mua_criteria": MUA_CRITERIA,
    },
    {
        "unit_criteria_id": 1,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.1 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.6 & firing_rate > 1 & abs(depth_drift_start_to_end) < 6",
        "mua_criteria": MUA_CRITERIA,
    },
    {
        "unit_criteria_id": 2,
        "sua_criteria": "isi_contamination < 0.1 & amplitude_cutoff < 0.01 & spike_duration > 0.1 & spike_amplitude > 50 & presence_ratio > 0.9 & firing_rate > 2 & abs(depth_drift_start_to_end) < 6",
        "mua_criteria": MUA_CRITERIA,
    },
]


def main() -> None:
    # Parse first so --help or a stray argument exits before any database write.
    argparse.ArgumentParser(
        description="Insert unit-quality criteria and populate unit counts"
    ).parse_args()
    from labdata.schema import SpikeSorting, UnitCount, UnitCountCriteria, UnitMetrics

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
