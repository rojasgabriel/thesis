# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["UnitMatchPy==3.6.1"]
# ///
"""Run UnitMatch on the waveform files of several sessions of one probe.

Step 2 of `thesis.ephys.preprocessing.unit_match`. It runs in its own
environment because UnitMatchPy needs numpy < 2 and pandas < 3:

    uv run src/thesis/ephys/preprocessing/unit_match_run.py \
        results/unit_match/GRB064_*.npz --out results/unit_match/GRB064

Sessions are ordered by file name (the session date). UnitMatch default
parameters; drift between sessions is corrected per shank. A pair is a match
when the match probability is above 0.5 in both cross-validation directions
(half 1 of A against half 2 of B, and the reverse).

Writes `<out>_pairs.csv` (every pair within 100 µm or with a probability above
0.05, with its probability in both directions) and `<out>_ids.csv` (one row per unit, with the UnitMatch
liberal, intermediate and conservative track IDs).
"""

import argparse

import numpy as np
import pandas as pd
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.default_params as default_params
import UnitMatchPy.overlord as ov
import UnitMatchPy.utils as util


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sessions = [dict(np.load(f)) for f in sorted(args.files)]
    if len({s["subject"].item() for s in sessions}) != 1:
        raise ValueError("All files must come from one subject (one probe).")
    positions = [s["channel_pos"] for s in sessions]
    if any(not np.array_equal(p, positions[0]) for p in positions):
        raise ValueError("Channel maps differ; UnitMatch needs one map.")

    param = util.get_probe_geometry(positions[0], default_params.get_default_param())
    waveform = np.concatenate([s["waveforms"] for s in sessions]).astype(float)
    n_per_session = np.array([len(s["unit_ids"]) for s in sessions])
    param["n_units"], session_id, session_switch, param["n_sessions"] = (
        util.get_session_data(n_per_session)
    )
    param["n_channels"] = waveform.shape[2]
    within_session = util.get_within_session(session_id, param)
    # UnitMatch wants 3-D positions; the first axis is unused.
    channel_pos = [np.insert(p, 0, 1.0, axis=1) for p in positions]
    original_ids = np.concatenate([s["unit_ids"] for s in sessions])
    clus_info = {
        "session_switch": session_switch,
        "session_id": session_id,
        "original_ids": original_ids,
    }

    props = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    total_score, candidate_pairs, scores, predictors = ov.extract_metric_scores(
        props, session_switch, within_session, param, niter=2
    )
    prior_match = 1 - param["n_expected_matches"] / param["n_units"] ** 2
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    kernels = bf.get_parameter_kernels(scores, labels, cond, param, add_one=1)
    prob = bf.apply_naive_bayes(
        kernels, np.array((prior_match, 1 - prior_match)), predictors, param, cond
    )[:, 1].reshape(param["n_units"], param["n_units"])
    liberal, intermediate, conservative, _ = aid.assign_unique_id(
        prob, param, clus_info
    )

    names = np.array([s["session"].item() for s in sessions])[session_id]
    centroid = props["avg_centroid"].mean(axis=2)  # (3, n_units), mean of halves
    pd.DataFrame(
        {
            "session": names,
            "unit_id": original_ids,
            "x_um": centroid[1],
            "depth_um": centroid[2],
            "self_prob": np.diag(prob),
            "track_liberal": liberal,
            "track": intermediate,
            "track_conservative": conservative,
        }
    ).to_csv(f"{args.out}_ids.csv", index=False)

    dist = np.linalg.norm(centroid[1:, :, None] - centroid[1:, None, :], axis=0)
    # Centroids are before drift correction, so keep any pair with some probability.
    near = (dist < param["max_dist"]) | (prob > 0.05) | (prob.T > 0.05)
    a, b = np.nonzero(near & (np.triu(np.ones_like(dist), 1) > 0))
    pd.DataFrame(
        {
            "session_a": names[a],
            "unit_a": original_ids[a],
            "session_b": names[b],
            "unit_b": original_ids[b],
            "distance_um": dist[a, b],
            "prob_ab": prob[a, b],
            "prob_ba": prob[b, a],
            "match": (prob[a, b] > param["match_threshold"])
            & (prob[b, a] > param["match_threshold"]),
        }
    ).to_csv(f"{args.out}_pairs.csv", index=False)
    print(
        f"{args.out}: {param['n_units']} units, {len(np.unique(intermediate))} tracks"
    )


if __name__ == "__main__":
    main()
