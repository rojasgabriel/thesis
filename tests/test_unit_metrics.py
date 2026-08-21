import unittest
from unittest.mock import patch

import numpy as np

from thesis.ephys.utils import unit_metrics


class FetchSpikeDurationTests(unittest.TestCase):
    def test_tolerant_alias_requests_non_strict_waveform_durations(self):
        with patch.object(
            unit_metrics,
            "fetch_waveform_durations_ms",
            return_value=np.array([0.3, np.nan]),
        ) as fetch:
            durations = unit_metrics.fetch_spike_duration_ms(
                "GRB058", "session", [1, 2]
            )

        np.testing.assert_allclose(durations, [0.3, np.nan])
        fetch.assert_called_once_with(
            "GRB058", "session", [1, 2], strict=False, unit_criteria_id=1
        )


if __name__ == "__main__":
    unittest.main()
