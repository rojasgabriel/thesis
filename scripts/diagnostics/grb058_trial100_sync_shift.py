"""
Deep dive into what changes at trial 100.
"""

import numpy as np
from scipy.stats import linregress

from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
from thesis.ephys.utils.io_digital_events import fetch_session_events

# Load data
subject, session = "GRB058", "20260312_134952"
align_ev = fetch_session_events(subject, session)
trial_df = fetch_trial_metadata(subject, session, align_ev)
if trial_df is None:
    raise RuntimeError(f"Missing trial_df for {subject} {session}.")

n = min(len(trial_df), len(align_ev["trial_start"]))
bpod_sync = trial_df["t_sync"].iloc[:n].to_numpy(dtype=float)
obx_sync = align_ev["trial_start"][:n].astype(float)
valid = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
bpod_s = bpod_sync[valid]
obx_s = obx_sync[valid]
trial_indices = np.arange(n)[valid]

# Get residuals from early fit to extrapolate into late period
mask_early = trial_indices < 100
slope_early, intercept_early, _, _, _ = linregress(
    trial_indices[mask_early], obx_s[mask_early] - bpod_s[mask_early]
)

expected_error_at_100 = slope_early * 100 + intercept_early
actual_error_at_100 = (
    obx_s[trial_indices == 100][0] - bpod_s[trial_indices == 100][0]
    if 100 in trial_indices
    else None
)

print("=" * 70)
print("DEEP DIVE: What changes at trial 100?")
print("=" * 70)

print("\nAt trial 100:")
print(f"  Predicted error (from early fit): {expected_error_at_100 * 1000:+.3f} ms")
if actual_error_at_100 is not None:
    print(f"  Actual error: {actual_error_at_100 * 1000:+.3f} ms")
    print(f"  Jump: {(actual_error_at_100 - expected_error_at_100) * 1000:+.3f} ms")

# Window around trial 100
window_before = trial_indices[(trial_indices >= 90) & (trial_indices < 100)]
window_after = trial_indices[(trial_indices >= 100) & (trial_indices <= 110)]

print(f"\nTrials 90-100 (n={len(window_before)}):")
for idx in window_before[-5:]:  # Last 5
    error = obx_s[trial_indices == idx][0] - bpod_s[trial_indices == idx][0]
    predicted = slope_early * idx + intercept_early
    residual = error - predicted
    print(
        f"  Trial {idx}: error={error * 1000:+6.3f} ms, predicted={predicted * 1000:+6.3f} ms, residual={residual * 1000:+6.3f} ms"
    )

print(f"\nTrials 100-110 (n={len(window_after)}):")
for idx in window_after[:5]:  # First 5
    error = obx_s[trial_indices == idx][0] - bpod_s[trial_indices == idx][0]
    predicted = slope_early * idx + intercept_early
    residual = error - predicted
    print(
        f"  Trial {idx}: error={error * 1000:+6.3f} ms, predicted={predicted * 1000:+6.3f} ms, residual={residual * 1000:+6.3f} ms"
    )

# Check if error grows smoothly or jumps discontinuously
print("\n" + "=" * 70)
print("SMOOTHNESS CHECK: Is the transition gradual or abrupt?")
print("=" * 70)

errors = obx_s - bpod_s
error_diffs = np.diff(errors)
trial_diffs = trial_indices[:-1]

# Filter to windows
mask_early_diffs = trial_diffs < 50
mask_pre_100 = (trial_diffs >= 80) & (trial_diffs < 100)
mask_post_100 = trial_diffs >= 100

if np.sum(mask_early_diffs) > 0:
    print("\nError jumps (OBX - BPod per trial):")
    print(
        f"  Trials 0-50:    mean={np.mean(error_diffs[mask_early_diffs]) * 1000:+.3f} ms/trial, std={np.std(error_diffs[mask_early_diffs]) * 1000:.3f} ms"
    )
    print(
        f"  Trials 80-100:  mean={np.mean(error_diffs[mask_pre_100]) * 1000:+.3f} ms/trial, std={np.std(error_diffs[mask_pre_100]) * 1000:.3f} ms"
    )
    print(
        f"  Trials 100+:    mean={np.mean(error_diffs[mask_post_100]) * 1000:+.3f} ms/trial, std={np.std(error_diffs[mask_post_100]) * 1000:.3f} ms"
    )

# Look for a single large jump
large_jumps = np.where(np.abs(error_diffs) > np.std(error_diffs) * 3)[0]
print(
    f"\nLarge error jumps (>3σ = >{np.std(error_diffs) * 3 * 1000:.3f} ms): {len(large_jumps)}"
)
for idx in large_jumps:
    if trial_diffs[idx] >= 90 and trial_diffs[idx] <= 110:
        print(f"  Trial {trial_diffs[idx]}: {error_diffs[idx] * 1000:+.3f} ms")

# Check clock rates more carefully
print("\n" + "=" * 70)
print("CLOCK RATE ANALYSIS: Are the clocks running at different rates after trial 100?")
print("=" * 70)

# Compute rolling clock rate (trial-to-trial interval)
window_size = 10
for start_trial in [50, 95, 100, 105, 150]:
    mask = (trial_indices >= start_trial) & (trial_indices < start_trial + window_size)
    if np.sum(mask) >= window_size:
        obx_intervals = np.diff(obx_s[mask])
        bpod_intervals = np.diff(bpod_s[mask])
        rate_ratio = np.mean(obx_intervals) / np.mean(bpod_intervals)
        print(
            f"Trials {start_trial}-{start_trial + window_size}: OBX/BPod interval ratio = {rate_ratio:.8f}"
        )
