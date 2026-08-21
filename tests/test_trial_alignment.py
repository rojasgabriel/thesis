import unittest

import numpy as np
import pandas as pd

from thesis.ephys.utils.trial_alignment import (
    enrich_chipmunk_trial_table,
    map_local_trial_rows_to_chipmunk_trials,
)


class TrialAlignmentTests(unittest.TestCase):
    def test_enrich_chipmunk_trial_table_adds_previous_trial_columns(self):
        trial_table = pd.DataFrame(
            {
                "response": [1, -1],
                "rewarded": [1, 0],
                "stim_rate_vision": [4, 8],
            }
        )

        enriched = enrich_chipmunk_trial_table(trial_table)

        self.assertTrue(np.isnan(enriched.loc[0, "prev_response"]))
        self.assertEqual(enriched.loc[1, "prev_response"], 1)
        self.assertEqual(enriched.loc[1, "prev_rewarded"], 1)
        self.assertEqual(enriched.loc[1, "prev_stim_rate"], 4)

    def test_map_local_trial_rows_falls_back_to_rate_and_outcome(self):
        local_trial_table = pd.DataFrame(
            {
                "trial_rate": [4],
                "trial_outcome": [1],
                "response_side": [-1],
            }
        )
        chipmunk_trial_table = pd.DataFrame(
            {
                "stim_rate_vision": [4],
                "response": [1],
                "rewarded": [1],
                "with_choice": [1],
            }
        )

        matched = map_local_trial_rows_to_chipmunk_trials(
            local_trial_table, chipmunk_trial_table
        )

        np.testing.assert_array_equal(matched, [0])


if __name__ == "__main__":
    unittest.main()
