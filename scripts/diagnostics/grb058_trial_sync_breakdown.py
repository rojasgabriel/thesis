"""
Diagnose sync breakdown at trial 100 in GRB058 / 20260312_134952.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
from thesis.ephys.utils.io_digital_events import fetch_session_events

# Load data
subject, session = "GRB058", "20260312_134952"
print(f"\n{'=' * 70}")
print(f"Analyzing {subject} / {session}")
print(f"{'=' * 70}\n")

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

print(f"Total trials loaded: {n}")
print(f"Valid sync pairs: {len(bpod_s)}")
print(f"NaN rate: {(n - len(bpod_s)) / n * 100:.1f}%\n")

# =============================================================================
# 1. DRIFT CHANGE: Compare slopes before/after trial 100
# =============================================================================
print("=" * 70)
print("1. DRIFT CHANGE: Linear model slopes")
print("=" * 70)

# Split at trial 100
mask_early = trial_indices < 100
mask_late = trial_indices >= 100

if np.sum(mask_early) > 2 and np.sum(mask_late) > 2:
    bpod_early, obx_early = bpod_s[mask_early], obx_s[mask_early]
    bpod_late, obx_s_late = bpod_s[mask_late], obx_s[mask_late]

    slope_early, intercept_early, r_early, p_early, se_early = linregress(
        trial_indices[mask_early], obx_early - bpod_early
    )
    slope_late, intercept_late, r_late, p_late, se_late = linregress(
        trial_indices[mask_late], obx_s_late - bpod_late
    )

    print(f"\nTrials 0-99 ({np.sum(mask_early)} points):")
    print(f"  Slope:  {slope_early:+.6e} s/trial")
    print(f"  Intercept: {intercept_early:+.6f} s")
    print(f"  R²: {r_early**2:.4f}")
    print(f"  StdErr: {se_early:+.6e}")

    print(f"\nTrials 100-235 ({np.sum(mask_late)} points):")
    print(f"  Slope:  {slope_late:+.6e} s/trial")
    print(f"  Intercept: {intercept_late:+.6f} s")
    print(f"  R²: {r_late**2:.4f}")
    print(f"  StdErr: {se_late:+.6e}")

    slope_diff = slope_late - slope_early
    slope_ratio = slope_late / slope_early if slope_early != 0 else np.inf
    print(f"\nSlope difference: {slope_diff:+.6e} s/trial")
    print(f"Slope ratio (late/early): {slope_ratio:.2f}x")
    print(
        f"Interpretation: {'SIGNIFICANT change' if abs(slope_diff) > 1e-5 else 'no significant change'}"
    )
else:
    print("Not enough valid points in one or both periods.")

# =============================================================================
# 2. RESIDUAL PATTERN: Global fit + residuals by trial index
# =============================================================================
print("\n" + "=" * 70)
print("2. RESIDUAL PATTERN: Global linear fit")
print("=" * 70)

slope_global, intercept_global, r_global, p_global, se_global = linregress(
    trial_indices, obx_s - bpod_s
)
residuals_global = (obx_s - bpod_s) - (slope_global * trial_indices + intercept_global)

print(f"\nGlobal fit (all {len(trial_indices)} points):")
print(f"  Slope:  {slope_global:+.6e} s/trial")
print(f"  Intercept: {intercept_global:+.6f} s")
print(f"  R²: {r_global**2:.4f}")

print("\nResiduals:")
print(f"  Mean: {np.mean(residuals_global):+.6e} s")
print(f"  Std:  {np.std(residuals_global):+.6e} s")
print(f"  Min:  {np.min(residuals_global):+.6e} s")
print(f"  Max:  {np.max(residuals_global):+.6e} s")

# Residuals by period
res_early = residuals_global[mask_early]
res_late = residuals_global[mask_late]
print(
    f"\nResiduals (trials 0-99): mean={np.mean(res_early):+.6e}, std={np.std(res_early):+.6e}"
)
print(
    f"Residuals (trials 100-235): mean={np.mean(res_late):+.6e}, std={np.std(res_late):+.6e}"
)

if len(res_early) > 0 and len(res_late) > 0:
    std_ratio = np.std(res_late) / np.std(res_early)
    print(f"Residual std ratio (late/early): {std_ratio:.2f}x")

# =============================================================================
# 3. INTERPOLATION GAPS: NaN patterns
# =============================================================================
print("\n" + "=" * 70)
print("3. INTERPOLATION GAPS: NaN clustering")
print("=" * 70)

invalid_mask = ~valid
invalid_indices = np.arange(n)[invalid_mask]

print(f"\nTotal NaN sync points: {len(invalid_indices)}")

# Find clusters of consecutive NaNs
if len(invalid_indices) > 0:
    gaps = np.diff(invalid_indices)
    consecutive = np.where(gaps == 1)[0]

    print(
        f"Consecutive NaN clusters: {len(np.unique(np.concatenate(([0], consecutive + 1)))) if len(consecutive) > 0 else 0}"
    )

    # Look for clusters around trial 100
    near_100 = invalid_indices[(invalid_indices >= 80) & (invalid_indices < 120)]
    if len(near_100) > 0:
        print(f"NaN indices near trial 100 (80-120): {near_100.tolist()}")
    else:
        print("No NaN indices in trials 80-120.")

    # Check if there's a change in NaN rate before/after 100
    nan_rate_early = np.sum(~valid[:100]) / min(100, n)
    nan_rate_late = np.sum(~valid[100:]) / max(1, n - 100)
    print(f"\nNaN rate (trials 0-99): {nan_rate_early * 100:.1f}%")
    print(f"NaN rate (trials 100+): {nan_rate_late * 100:.1f}%")

# =============================================================================
# 4. CLOCK CONSISTENCY: Look for jumps/resets
# =============================================================================
print("\n" + "=" * 70)
print("4. CLOCK CONSISTENCY: Timestamp discontinuities")
print("=" * 70)

# Differences between consecutive valid points
obx_diffs = np.diff(obx_s)
bpod_diffs = np.diff(bpod_s)

print("\nOBX clock (trial_start) differences:")
print(f"  Mean interval: {np.mean(obx_diffs):.6f} s")
print(f"  Std interval:  {np.std(obx_diffs):.6f} s")
print(f"  Min interval:  {np.min(obx_diffs):.6f} s")
print(f"  Max interval:  {np.max(obx_diffs):.6f} s")

print("\nBpod clock (t_sync) differences:")
print(f"  Mean interval: {np.mean(bpod_diffs):.6f} s")
print(f"  Std interval:  {np.std(bpod_diffs):.6f} s")
print(f"  Min interval:  {np.min(bpod_diffs):.6f} s")
print(f"  Max interval:  {np.max(bpod_diffs):.6f} s")

# Look for large jumps (>5x std)
obx_mean_int = np.mean(obx_diffs)
obx_std_int = np.std(obx_diffs)
large_jumps_obx = np.where(np.abs(obx_diffs - obx_mean_int) > 5 * obx_std_int)[0]

bpod_mean_int = np.mean(bpod_diffs)
bpod_std_int = np.std(bpod_diffs)
large_jumps_bpod = np.where(np.abs(bpod_diffs - bpod_mean_int) > 5 * bpod_std_int)[0]

print(f"\nLarge OBX jumps (>5σ): {len(large_jumps_obx)}")
if len(large_jumps_obx) > 0:
    for idx in large_jumps_obx[:10]:  # Show first 10
        print(
            f"  Between trials {trial_indices[idx]}-{trial_indices[idx + 1]}: {obx_diffs[idx]:.6f} s"
        )

print(f"\nLarge Bpod jumps (>5σ): {len(large_jumps_bpod)}")
if len(large_jumps_bpod) > 0:
    for idx in large_jumps_bpod[:10]:  # Show first 10
        print(
            f"  Between trials {trial_indices[idx]}-{trial_indices[idx + 1]}: {bpod_diffs[idx]:.6f} s"
        )

# =============================================================================
# PLOT: Residuals and errors over trial index
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Residuals from global fit
ax = axes[0]
ax.scatter(trial_indices, residuals_global * 1000, s=10, alpha=0.6)
ax.axvline(100, color="r", linestyle="--", linewidth=2, label="Trial 100")
ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Trial index")
ax.set_ylabel("Residual (ms)")
ax.set_title("Residuals from global linear fit (OBX - BPod)")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Raw error (OBX - BPod) by trial
ax = axes[1]
ax.scatter(trial_indices, (obx_s - bpod_s) * 1000, s=10, alpha=0.6)
ax.axvline(100, color="r", linestyle="--", linewidth=2, label="Trial 100")
fit_line = slope_global * trial_indices + intercept_global
ax.plot(trial_indices, fit_line * 1000, "r-", linewidth=1.5, label="Linear fit")
ax.set_xlabel("Trial index")
ax.set_ylabel("OBX - BPod (ms)")
ax.set_title("Raw sync error over trials")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Cumulative NaN count
ax = axes[2]
nan_cumsum = np.cumsum(~valid[:n])
ax.plot(np.arange(n), nan_cumsum, linewidth=2)
ax.axvline(100, color="r", linestyle="--", linewidth=2, label="Trial 100")
ax.set_xlabel("Trial index")
ax.set_ylabel("Cumulative NaN count")
ax.set_title("Cumulative NaN sync points")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(
    "figures/sync_breakdown_GRB058_20260312.png",
    dpi=150,
    bbox_inches="tight",
)
print("\n" + "=" * 70)
print("Plot saved: figures/sync_breakdown_GRB058_20260312.png")
print("=" * 70)
