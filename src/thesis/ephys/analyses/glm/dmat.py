"""Build the shared design matrix for the V1 GLM.

This module owns the design side of the GLM, from the raw trial table to the
standardized numeric design. `write_stimulus_windows` selects trials, lays down
the equal first-flash bin grid, fixes the 60/20/20 whole-trial split, and maps
camera frames to trial windows. From that window file the rest of the module
builds every regressor column: the task and sensory regressors (flash kernel,
the short-interval flash term, the stimulus-during-hold kernel, withdrawal),
the contemporaneous motion-energy video PCs, and each unit's own strictly past spike-history block. `prepare_common_design`
writes the shared task and video columns to a memmapped `.npy`;
`build_unit_history` and `_design_with_history` add the per-unit history block
at fit time. `fit.py` consumes what is here and never rebuilds the design.

Comparison, unit-selection, baseline, event alignment, exclusions, and sampling
unit are stated in `fit.py`; this module is the deterministic, validated bridge
from trial table plus video to fit-ready inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from damn.alignment import construct_timebins
from damn.objects.basis_function_objects import BasisFunction, RaisedCosineBasis
from damn.objects.design_matrix_objects import DesignMatrix
from damn.objects.regressor_objects import EventRegressor

from thesis.ephys.trials import build_trial_table

# The window starts this long before the first flash. With 0.5 s it has about
# 0.45 s before the center entry (the entry is 26 to 106 ms before the first
# flash), the baseline of the model, and in 95% of the trials of GRB006 it
# starts after the exit from the side port of the trial before (2026-10-15;
# it was 0.1 s, which left about 50 ms before the entry). The window ends this
# long after the first flash, so that the windows of the closest trials (2.70 s
# apart) and their video frames do not overlap (at least 96 ms between
# windows); 4 of 294 trials of GRB006 and 6 of 169 of GRB064 reach the response
# entry later and lose their last bins (0.3% and 0.8% of the modeled time; it
# was 2.54 s).
PRE_S = 0.5
POST_S = 2.1
BINWIDTH_S = 0.001
SPLIT_SEED = 20260912
# One count. In the first design, 25 and 50 PCs were not reliably better than
# 10 (paired over units), and 100 and 200 were worse; the smallest is used.
VIDEO_COMPONENT_COUNTS = (10,)
# Flash kernel: this long after each flash, with this many bases.
FLASH_WINDOW_S = 0.15
FLASH_BASES = 8
# The flash kernel by the gap from the flash before: one kernel for each gap
# class, and no common kernel. A class is the flashes with a gap up to its
# limit; the last class has the longer gaps and the first flash of a trial,
# which has no flash before it. So each kernel is the response at that gap, and
# the classes do not overlap. With no limits (an empty tuple) the design has
# one common kernel for all flashes, as a comparison (`gap_class_grid.py`).
# Two classes (a flash within 50 ms of another, and every other flash): more
# classes do not predict better on 80 units or on all units. An earlier design
# had one common kernel with a fast-adaptation and a flash-recovery term,
# which were correlated (-0.86) and could not be read one by one.
# From 2026-10-17 the design has the two classes in the form of a modifier
# (`FLASH_JUST_BEFORE_TERM`): no classes here.
FLASH_GAP_EDGES_S = ()
FLASH_GAP = "flash_gap"
FLASH_COMMON = "visual_flash"
# The gap from the flash before, as a modifier (the design from 2026-10-17,
# decided with Gabriel for a reading in words): one common kernel on every
# flash, and a second kernel on the flashes with a flash up to
# `FLASH_JUST_BEFORE_S` before them. So the common kernel is the response to a
# flash after a long gap, and the second kernel is how a flash just before
# changes it. The columns span the same space as two gap classes
# (`FLASH_GAP_EDGES_S = (0.05,)`, the design to 2026-10-16), which predict the
# same (`gap_class_grid.py`); only the L2 penalty differs: it pulls the change
# to zero, not each class kernel.
FLASH_JUST_BEFORE_TERM = True
FLASH_JUST_BEFORE = "flash_just_before"
FLASH_JUST_BEFORE_S = 0.05
# The flashes come in slots of this length.
FLASH_SLOT_S = 0.04
# Causal withdrawal kernel: this long after center exit, with this many bases.
# It is short so that it holds few flashes, and it has no part before the event.
WITHDRAWAL_PRE_S = 0.0
WITHDRAWAL_WINDOW_S = 0.1
WITHDRAWAL_BASES = 4
# `withdrawal_grid.py` turns this off to test the model without the term. Off
# from 2026-10-15: the response period covers the time after the exit, with
# bases as close together just after the exit (first canonical correlation of
# the two was 0.91).
WITHDRAWAL_KERNEL = False
# Stimulus during hold: one kernel on the first flash, set to zero from the
# center exit on. Units fire less from about 80 ms after the first flash until
# the animal leaves the center port; the flashes go on (the train is 1 s long
# and stops early only at the response entry).
# The bases have even spacing in log(t + stretch). They are close together
# early, where the rate changes fast and every trial gives data (about 40 ms
# apart near the first-flash response), and far apart late, where the rate is
# almost flat and only the longer holds give data. No step goes with the
# kernel. In a trial with a hold longer than the window, the kernel keeps its
# last value up to the center exit. A stretch of 0 gives even spacing in time
# (`hold_grid.py`). The window is 0.7 s, the 95th percentile of the fixation in
# both mice (2026-10-15): with 0.95 s, the part after 0.7 s rested on 1 to 3%
# of the trials and was not stable between halves of the trials.
HOLD_KERNEL = True
HOLD_WINDOW_S = 0.7
# 10 bases with a stretch of 50 ms (2026-10-16): on the design of 2026-10-15
# this predicts better than 12 bases with 25 ms (+0.12%, 50 of 80 units,
# p = 0.0047, `hold_grid.py`), and the kernels are recovered as well
# (`parameter_recovery --grid hold_grid`).
HOLD_BASES = 10
HOLD_LOG_STRETCH_S = 0.05
# A factor on the columns of the fixation kernel after the z-score. With 1 the
# fixation kernel has the same L2 penalty as the other terms: the model in use,
# and the common practice. With a factor above 1 the penalty on its weights is
# that factor squared times weaker. With the same penalty, the fit makes the
# fixation kernel too small and gives a part of the level to the flash × move
# kernels (`leak_test.py`); a factor of 10 removes that leak with the same
# prediction, and is the sensitivity analysis (`_weaker_fixation_penalty`). The
# saved scale of the columns has the factor in it, so the kernels are read in
# the same way. A factor above 10 stops the convergence.
HOLD_PENALTY_SCALE = 1.0
# Pre-stimulus: one column that is 1 from the center entry to the first flash,
# where no other term is on. Its weight has no common sign over units; it sets
# the rate of each unit in the port before the stimulus.
# Names: the figures and the docs call this column "pre-stimulus" (it was
# "initiation" to 2026-10-15) and the hold kernel "fixation". The keys ("fixation", "hold") stay, because
# the saved fits use them.
FIXATION_COLUMN = True
FIXATION = "fixation"
# Earlier parameterizations of that term, for `hold_grid.py`: one column that is 1 from
# center entry (or from the first flash) to center exit.
IN_PORT_COLUMN = False
STATE_FROM_FIRST_FLASH = False
IN_PORT = "in_port"
# One column for the side of the response after the center exit (the figures
# and the docs call it "choice"; the key stays). In the model from 2026-10-12
# to 2026-10-15 (`candidate_terms.py`); turn direction, the same side with a
# time course, replaces it (first canonical correlation 0.9 between the two).
SIDE_COLUMN = False
SIDE = "side_after_exit"
# One column for the side of the last response port that the animal went to
# before this trial (`trials.last_side`), from the start of the window to the
# center exit: the side from which the animal comes to the center port. On raw
# rates, about half of the units differ by it. In the model from 2026-10-12
# (`candidate_terms.py`).
# `unique_dev.py` shuffles it over trials.
PREVIOUS_SIDE_COLUMN = True
PREVIOUS_SIDE = "previous_side"
# A slow change of the rate over the session: raised cosines over the time of
# the trial in the session, constant inside a trial (`session_drift_columns`).
# In the model from 2026-10-12. `unique_dev.py` shuffles it over trials, not
# inside a trial. `DRIFT_SPLINE` gives the first parameterization that was tested, a cubic spline
# with `DRIFT_KNOTS` knots, for `drift_grid.py`.
SESSION_DRIFT = True
DRIFT = "session_drift"
DRIFT_BASES = 4
DRIFT_SPLINE = False
DRIFT_KNOTS = 6
# Camera views of cam0 that the motion energy can come from: the back view of
# GRB006, and the bottom view from GRB064 on. The view differs between these
# animals, so their motion-energy PCs are not the same measure.
VIDEO_VIEWS = ("BackStereoView", "BottomView")
# A second flash kernel for a flash after the center exit against a flash in
# the fixation. In the model from 2026-10-12 (`candidate_terms.py`).
FLASH_AFTER_EXIT_TERM = True
# Names: the figures and the docs call a flash in the fixation "still" and a
# flash after the center exit "move", and this term "flash × move". These are
# task periods; the model has no measure of locomotion in them.
FLASH_AFTER_EXIT = "flash_after_exit"
# The coding of the design to 2026-10-16, for a comparison.
FLASH_AFTER_EXIT_CENTERED = False
# A flash kernel for the product of the period and the gap class: with it, the
# change from the fixation to after the exit can differ between the gap classes.
# In the model from 2026-10-13 to 2026-10-15 (`candidate_terms.py`: about a
# quarter of the units need it in the two sessions). Out: it was the least
# stable kernel between halves of the trials (agreement 0.61 and 0.56), and the
# flash × move effect is not reliable after the leak correction.
FLASH_MOVE_GAP_TERM = False
FLASH_MOVE_GAP = "flash_move_gap"
# Candidate terms. `candidate_terms.py` tested them with the inclusion rule
# (decided with Gabriel, 2026-10-15). Time in trial and turn direction passed
# in both mice; accumulated evidence and the previous outcome failed in both.
# Time in trial is not in the model: over the fixation it was the same time
# course as the fixation kernel (first canonical correlation 1.00), and the
# fixation kernel is the term that reads the fixation effect. The response
# period below takes its place after the exit. Turn direction is in the model
# from 2026-10-15, in the form below.
# - Time in trial: `TRIAL_TIME_BASES` raised cosines over the time from the
#   first flash, 0 to `TRIAL_TIME_WINDOW_S`; 0 outside. From the residual
#   checks of 2026-10-13.
# - Accumulated evidence: the number of flashes of the trial up to the bin,
#   minus its expected value for the rate of the trial. A trial with rate N has
#   N flashes in 25 slots of `FLASH_SLOT_S`; the first slot always has one, and
#   the other N - 1 go to random slots of the other 24. So the expected count
#   when k of those 24 slots have started is 1 + (N - 1) k / 24. The term is 0
#   before the first flash and varies only by the random placement, so it is
#   separable from the rate, the category, and the choice. It tests whether
#   V1 carries the accumulated flashes on top of the 150-ms flash kernels.
# - Turn direction: the side of the response (+1 right, -1 left) times
#   the `exit_basis` (`TURN_DIRECTION_BASES` raised cosines, even in
#   log(t + `EXIT_LOG_STRETCH_S`), from the center exit to `TURN_DIRECTION_S`
#   after it, the 95th percentile of the time from the exit to the response
#   entry in both mice); after that the last cosine keeps its value up to the
#   response entry, where the bins of the trial end. 0 before the exit. A direction-specific movement
#   signal, which motion energy cannot carry; it adds a time course to the
#   choice column after the exit. Nothing before the exit, where the side is
#   confounded with the stimulus.
# - Previous outcome: +1 if the last trial with a response was rewarded, -1 if
#   not, over the whole window of the trial (0 before the first response of
#   the session). `unique_dev.py` shuffles it over trials.
TIME_IN_TRIAL_TERM = False
TIME_IN_TRIAL = "time_in_trial"
TRIAL_TIME_WINDOW_S = 1.5
TRIAL_TIME_BASES = 6
ACCUMULATED_EVIDENCE_TERM = False
ACCUMULATED_EVIDENCE = "accumulated_evidence"
STIMULUS_SLOTS = 25
TURN_DIRECTION_TERM = True
TURN_DIRECTION = "turn_direction"
TURN_DIRECTION_S = 1.3
TURN_DIRECTION_BASES = 8
# The log stretch of the exit basis. It has the value of the fixation kernel,
# but its own name, so that a grid of the fixation kernel (`hold_grid.py`)
# does not change the exit basis also.
EXIT_LOG_STRETCH_S = 0.025
PREVIOUS_OUTCOME_COLUMN = False
PREVIOUS_OUTCOME = "previous_outcome"
# Response period (decided with Gabriel, 2026-10-15): the `exit_basis` with no
# sign, from the center exit to the response entry; it takes the place of the
# withdrawal kernel. With turn direction it splits the time after the exit into
# a part common to both sides and a part of opposite sign for left and right.
# The two share a direction where the sides differ in number or in the time to
# the response (first canonical correlation 0.39 in GRB006, 0.80 in GRB064,
# where right turns take 734 against 638 ms). It also gives the rate step at
# the exit a term of its own.
RESPONSE_PERIOD_TERM = True
RESPONSE_PERIOD = "response_period"
# `THESIS_GLM_TERMS` switches terms on for a whole run, also in the worker
# processes, which import this module again: a comma-separated list of the
# switch names above, for example "TIME_IN_TRIAL_TERM,TURN_DIRECTION_TERM".
for _switch in filter(None, os.environ.get("THESIS_GLM_TERMS", "").split(",")):
    if globals().get(_switch) is not False:
        raise ValueError(f"THESIS_GLM_TERMS: {_switch} is not a switch of the design.")
    globals()[_switch] = True
HOLD = "hold"
HISTORY_COLUMNS = 10


def exit_basis() -> "StretchedCosineBasis":
    """Return the basis of the response period and the turn direction."""
    return StretchedCosineBasis(
        TURN_DIRECTION_BASES,
        TURN_DIRECTION_S + BINWIDTH_S,
        BINWIDTH_S,
        EXIT_LOG_STRETCH_S,
    )


class StretchedCosineBasis(BasisFunction):
    """Causal raised cosines with even spacing in log(t + stretch_s).

    The bases are close together just after the event and farther apart later.
    A larger `stretch_s` gives a more even spacing; DAMN's log basis is the case
    of a stretch of one bin, which puts most bases in the first 30 ms.
    """

    def __init__(
        self, n_funcs: int, post_s: float, binwidth_s: float, stretch_s: float
    ):
        self.n_funcs = n_funcs
        self.stretch_s = stretch_s
        super().__init__(0, post_s, binwidth_s)

    def construct_basis(self):
        times, _, _ = construct_timebins(0, self.post_s, self.binwidth_s)
        warped = np.log(times + self.stretch_s)
        centers = np.linspace(warped[0], warped[-1], self.n_funcs)
        phase = (warped[:, None] - centers[None, :]) * np.pi / (centers[1] - centers[0])
        return 0.5 * (1 + np.cos(np.clip(phase, -np.pi, np.pi))), times


def training_zscore(
    values: np.ndarray, train_rows: np.ndarray | slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each column with training rows and reject constants."""
    mean = values[train_rows].mean(axis=0)
    scale = values[train_rows].std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Scaling statistics must be finite.")
    if np.any(scale == 0):
        raise ValueError("Every column must vary in the training rows.")
    scaled = (values - mean) / scale
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled values must be finite.")
    return scaled, mean, scale


def validate_frame_times(times: np.ndarray, n_frames: int) -> np.ndarray:
    """Reject missing, duplicate, unordered, or mismatched frame timestamps."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) != n_frames or n_frames < 2:
        raise ValueError("Video frame count and timestamp count must match.")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing.")
    return times


def trial_bins(alignments: np.ndarray) -> dict:
    """Use DAMN's native equal grid and a random whole-trial 60/20/20 split."""
    alignments = np.asarray(alignments, dtype=float)
    if alignments.ndim != 1 or len(alignments) < 5:
        raise ValueError("At least five trial alignment times are required.")
    if not np.isfinite(alignments).all():
        raise ValueError("Trial alignment times must be finite.")
    centers, edges, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    starts, stops = alignments + edges[0], alignments + edges[-1]
    if np.any(starts[1:] < stops[:-1]):
        raise ValueError("Windows must be ordered and non-overlapping.")
    n_bins = len(centers)
    n_trials = len(starts)
    order = np.random.default_rng(SPLIT_SEED).permutation(n_trials)
    split = np.full(n_trials, 2, dtype=np.int8)
    split[order[: int(0.8 * n_trials)]] = 1
    split[order[: int(0.6 * n_trials)]] = 0
    rows = np.repeat(np.arange(len(starts)), n_bins)
    return dict(
        bin_center_s=(alignments[:, None] + centers).ravel(),
        relative_bin_centers_s=centers,
        relative_bin_edges_s=edges,
        window_start_s=starts,
        window_stop_s=stops,
        bin_split=split[rows],
        trial_split=split,
    )


def write_stimulus_windows(
    subject: str, session: str, output: Path, frame_times: Path
) -> None:
    """Write first-flash windows, the random split, and the camera-frame map."""
    if output.exists():
        raise FileExistsError(output)
    trials = build_trial_table(subject, session, include_frames=False)
    first = trials["stim_pulse_times_s"].str[0].to_numpy(dtype=float)
    t = trials[["center_entry_s", "center_exit_s", "response_port_entry_s"]].to_numpy(
        dtype=float
    )
    completed = (
        trials["response"].isin((-1, 1)).to_numpy()
        & trials["early_withdrawal"].fillna(1).eq(0).to_numpy()
    )
    eligible = completed & np.isfinite(first) & np.isfinite(t).all(axis=1)
    ordered = (t[:, 0] <= first) & (first < t[:, 1]) & (t[:, 1] < t[:, 2])
    if np.any(eligible & ~ordered):
        raise ValueError(
            f"Resolve event order before selecting trials: {trials.loc[eligible & ~ordered, 'trial_num'].tolist()}"
        )
    bins = trial_bins(first[eligible])
    starts, stops = bins["window_start_s"], bins["window_stop_s"]
    summary = dict(
        subject_name=subject,
        session_name=session,
        alignment_event="first measured flash",
        pre_s=PRE_S,
        post_s=POST_S,
        binwidth_s=BINWIDTH_S,
        eligible_trials=int(eligible.sum()),
        completed_trials_missing_events=int((completed & ~eligible).sum()),
        bins_per_trial=len(bins["relative_bin_centers_s"]),
        grid_source="damn.alignment.construct_timebins",
        actual_window_edges_s=bins["relative_bin_edges_s"][[0, -1]].tolist(),
        split="random whole-trial 60/20/20",
        split_seed=SPLIT_SEED,
        split_labels=["train", "validation", "test"],
        split_trial_counts=np.bincount(bins["trial_split"], minlength=3).tolist(),
        video_aligned=False,
    )
    arrays = dict(
        trial_num=trials["trial_num"].to_numpy(),
        eligible_trials=eligible,
        selected_trial_rows=np.flatnonzero(eligible),
        first_stim_s=first,
        response_entry_s=t[:, 2],
        **bins,
    )
    from labdata.schema import DatasetVideo, File

    key = dict(subject_name=subject, session_name=session, video_name="cam0")
    video = (DatasetVideo & key).fetch1()
    paths, missing = (File & (DatasetVideo.File & key)).check_if_files_local()
    if (
        missing
        or len(paths) != 1
        or not any(view in str(paths[0]) for view in VIDEO_VIEWS)
    ):
        raise ValueError(
            f"Expected one local cam0 video ({' or '.join(VIDEO_VIEWS)}); "
            f"missing: {missing}."
        )
    frames = validate_frame_times(
        np.load(frame_times, allow_pickle=False), int(video["n_frames"])
    )
    rows = np.full(len(frames), -1, dtype=int)
    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        lo = np.searchsorted(frames, start, side="right") - 1
        hi = np.searchsorted(frames, stop)
        if lo < 0 or hi >= len(frames):
            raise ValueError(f"Video does not cover window {i}.")
        if np.any(rows[lo : hi + 1] != -1):
            raise ValueError("Video support overlaps between trial windows.")
        rows[lo : hi + 1] = i
    split = np.full(len(frames), -1, dtype=np.int8)
    keep = rows >= 0
    split[keep] = bins["trial_split"][rows[keep]]
    arrays.update(frame_times_s=frames, frame_trial_row=rows, frame_split=split)
    import subprocess

    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height",
                "-of",
                "json",
                str(paths[0]),
            ],
            text=True,
        )
    )["streams"][0]
    if int(info["nb_frames"]) != len(frames):
        raise ValueError("Video header and frame timestamp counts differ.")
    summary.update(
        video_aligned=True,
        video_path=str(paths[0]),
        n_frames=len(frames),
        width=info["width"],
        height=info["height"],
        frame_times_source=str(frame_times.resolve()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(
            handle, allow_pickle=False, metadata_json=json.dumps(summary), **arrays
        )
    print(json.dumps(summary, indent=2))


def task_temporal_bases() -> dict[str, BasisFunction]:
    """Return the temporal basis used for each sensory or task regressor."""
    # Most flashes are 40 ms apart. Linear spacing keeps resolution across the
    # short response window instead of spending log-spaced bases below 10 ms.
    flash = RaisedCosineBasis(FLASH_BASES, 0, FLASH_WINDOW_S + BINWIDTH_S, BINWIDTH_S)
    withdrawal = RaisedCosineBasis(
        WITHDRAWAL_BASES,
        WITHDRAWAL_PRE_S + BINWIDTH_S * (WITHDRAWAL_PRE_S > 0),
        WITHDRAWAL_WINDOW_S + BINWIDTH_S,
        BINWIDTH_S,
    )
    bases = {
        FLASH_COMMON: flash,
        **{name: flash for name in flash_gap_names()},
        FLASH_AFTER_EXIT: flash,
        FLASH_MOVE_GAP: flash,
        FLASH_JUST_BEFORE: flash,
        "hold": StretchedCosineBasis(
            HOLD_BASES, HOLD_WINDOW_S + BINWIDTH_S, BINWIDTH_S, HOLD_LOG_STRETCH_S
        )
        if HOLD_LOG_STRETCH_S
        else RaisedCosineBasis(HOLD_BASES, 0, HOLD_WINDOW_S + BINWIDTH_S, BINWIDTH_S),
        "withdrawal": withdrawal,
    }
    if not HOLD_KERNEL:
        del bases["hold"]
    if FLASH_GAP_EDGES_S:
        del bases[FLASH_COMMON]
    if not FLASH_AFTER_EXIT_TERM:
        del bases[FLASH_AFTER_EXIT]
    if not FLASH_MOVE_GAP_TERM:
        del bases[FLASH_MOVE_GAP]
    if not FLASH_JUST_BEFORE_TERM:
        del bases[FLASH_JUST_BEFORE]
    if not WITHDRAWAL_KERNEL:
        del bases["withdrawal"]
    return bases


def flash_gap_names() -> list[str]:
    """Return the name of the flash kernel of each gap class, shortest gap first."""
    return [
        f"{FLASH_GAP}_{index + 1}" for index in range(len(FLASH_GAP_EDGES_S) + 1)
    ] * bool(FLASH_GAP_EDGES_S)


def gap_flash_terms() -> list[str]:
    """Return the flash terms that code the gap: the classes, or common + modifier."""
    if FLASH_GAP_EDGES_S:
        return flash_gap_names()
    return [FLASH_COMMON, *[FLASH_JUST_BEFORE] * FLASH_JUST_BEFORE_TERM]


def short_gap_s() -> float:
    """Return the limit of a short gap: the first class limit, or the modifier's."""
    return FLASH_GAP_EDGES_S[0] if FLASH_GAP_EDGES_S else FLASH_JUST_BEFORE_S


def flash_gap_class(trials) -> np.ndarray:
    """Return the gap class of each modeled flash, in the order of `flash_events`.

    The class is the number of limits of `FLASH_GAP_EDGES_S` that are below the
    gap from the flash before. The first flash of a trial is in the last class.
    """
    return np.searchsorted(FLASH_GAP_EDGES_S, flash_gaps(trials), side="left")


def flash_gaps(trials) -> np.ndarray:
    """Return the gap from the flash before of each modeled flash (s).

    In the order of `flash_events`; the first flash of a trial has an infinite
    gap.
    """
    gaps_by_trial = []
    for flashes, center_entry, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        gaps_by_trial.append(np.diff(modeled, prepend=-np.inf))
    return np.concatenate(gaps_by_trial)


def flash_events(trials) -> np.ndarray:
    """Return the times of the modeled flashes: center entry to response entry."""
    times = []
    for flashes, center_entry, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        if not len(modeled):
            raise ValueError("Every modeled trial must contain at least one flash.")
        times.append(modeled)
    return np.concatenate(times)


def flash_after_exit(trials) -> np.ndarray:
    """Return the period of each modeled flash, in the order of `flash_events`.

    1 for a flash at or after the center exit and 0 for a flash in the
    fixation, so the common flash kernel is the response in the port and the
    flash × move kernel is how the exit changes it (to 2026-10-16 the values
    were centered, which made the common kernel a mean of the two periods).
    With `FLASH_AFTER_EXIT_CENTERED` they are centered again. The first flash
    of a trial gets 0: it is always in the fixation. The period is that of the
    flash time; the kernel of a flash just before the exit goes on after it.
    """
    after, first = [], []
    for flashes, center_entry, center_exit, response_entry in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["center_exit_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        modeled = flashes[(flashes >= center_entry) & (flashes <= response_entry)]
        after.append(modeled >= center_exit)
        first.append(np.arange(len(modeled)) == 0)
    after = np.concatenate(after).astype(float)
    first = np.concatenate(first)
    if FLASH_AFTER_EXIT_CENTERED:
        return np.where(first, 0.0, after - after[~first].mean())
    return np.where(first, 0.0, after)


def flash_move_gap(trials) -> np.ndarray:
    """Return the period-by-gap product of each modeled flash, in the order of `flash_events`.

    The centered period of `flash_after_exit` times the centered indicator of
    the class of the shortest gaps. The first flash of a trial gets 0.
    """
    period = flash_after_exit(trials)
    used = period != 0  # the first flash of a trial is exactly 0
    short = (flash_gap_class(trials) == 0).astype(float)
    return np.where(used, period * (short - short[used].mean()), 0.0)


def _bumps(position: np.ndarray, count: int) -> np.ndarray:
    """Return `count` raised cosines with even spacing over a position of 0 to 1.

    Each one is two spacings wide, so they add up to 1 from 0 to 1; outside
    that range they are 0.
    """
    centers = np.linspace(0, 1, count)
    distance = (np.asarray(position)[..., None] - centers) / (centers[1] - centers[0])
    return np.where(np.abs(distance) < 1, 0.5 * (1 + np.cos(np.pi * distance)), 0.0)


def accumulated_evidence(trials, times: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Return the flashes up to each bin minus their expected number (trial x bin).

    `times` are the bin times (trial x bin) and `first` the first flash of each
    trial. See `ACCUMULATED_EVIDENCE` for the expected number.
    """
    out = np.zeros(times.shape)
    for row, (flashes, rate) in enumerate(
        zip(trials["stim_pulse_times_s"], trials["visual_stim_rate_hz"], strict=True)
    ):
        flashes = np.sort(np.asarray(flashes, dtype=float))
        flashes = flashes[flashes >= first[row] - 1e-6]
        since = times[row] - first[row]
        count = np.searchsorted(flashes - first[row], since, side="right")
        started = np.clip(np.floor(since / FLASH_SLOT_S), 0, STIMULUS_SLOTS - 1)
        expected = 1 + (float(rate) - 1) * started / (STIMULUS_SLOTS - 1)
        out[row] = np.where(since >= 0, count - expected, 0.0)
    return out


def session_drift_columns(trial_times: np.ndarray) -> np.ndarray:
    """Return the drift columns of each trial: smooth bumps over the session.

    `DRIFT_BASES` raised cosines with even spacing from the first to the last
    trial, each two spacings wide, so they add up to 1 at every time. The last
    one is left out, because the model has an intercept. So there are
    `DRIFT_BASES` - 1 columns.
    """
    if DRIFT_SPLINE:
        from sklearn.preprocessing import SplineTransformer

        return SplineTransformer(
            n_knots=DRIFT_KNOTS, degree=3, include_bias=False
        ).fit_transform(trial_times[:, None])
    position = (trial_times - trial_times.min()) / np.ptp(trial_times)
    return _bumps(position, DRIFT_BASES)[:, :-1]


def build_task_design(alignments: np.ndarray, trials) -> tuple[np.ndarray, list[dict]]:
    """Build the sensory and task columns: DAMN event kernels and the port state."""
    bases = task_temporal_bases()
    holds = trials["center_exit_s"].to_numpy(dtype=float) - np.asarray(alignments)
    flashes = flash_events(trials)
    gap_class = flash_gap_class(trials)
    just_before = flash_gaps(trials) <= FLASH_JUST_BEFORE_S
    limits_ms = [f"{1000 * limit:.0f}" for limit in FLASH_GAP_EDGES_S]
    gap_coding = [
        *(f"flashes with a gap up to {limit} ms" for limit in limits_ms),
        f"flashes with a gap above {limits_ms[-1]} ms, and the first flash of a trial"
        if limits_ms
        else "",
    ]
    specifications = [
        *(
            (name, flashes[gap_class == index], None, "sensory", gap_coding[index])
            for index, name in enumerate(flash_gap_names())
        ),
        (
            FLASH_COMMON,
            flashes,
            None,
            "sensory",
            "each measured flash from initiation through response entry",
        ),
        (
            FLASH_JUST_BEFORE,
            flashes[just_before],
            None,
            "sensory",
            f"flashes with a flash up to {1000 * FLASH_JUST_BEFORE_S:.0f} ms before",
        ),
        (
            FLASH_AFTER_EXIT,
            flashes,
            flash_after_exit(trials),
            "sensory",
            "centered; 1 for a flash after the center exit, 0 in the fixation; "
            "the first flash is 0",
        ),
        (
            FLASH_MOVE_GAP,
            flashes,
            flash_move_gap(trials),
            "sensory",
            "the centered period (flash × move) times the centered gap class "
            "(1 for the shortest gaps, else 0); the first flash is 0",
        ),
        (
            "hold",
            np.asarray(alignments, dtype=float),
            None,
            "task",
            "one kernel on the first flash, zero from the center exit on",
        ),
        (
            "withdrawal",
            trials["center_exit_s"].to_numpy(dtype=float),
            None,
            "task",
            "one withdrawal per completed trial",
        ),
    ]
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    manifest = []
    for name, times, values, group, coding in specifications:
        if name not in bases:
            continue
        basis = bases[name]
        design.add_regressor(
            EventRegressor(
                name,
                times,
                BINWIDTH_S,
                event_values=values,
                basis_objects=[basis],
                tags=group.split(),
            )
        )
        manifest.append(
            {
                "name": name,
                "group": group,
                "events": len(times),
                "columns": basis.basis.shape[1],
                "kernel_range_s": [
                    float(basis.basis_time[0]),
                    float(basis.basis_time[-1]),
                ],
                "coding": coding,
            }
        )
    design.build_matrix()
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    times = np.asarray(alignments, dtype=float)[:, None] + centers
    before_exit = times < trials["center_exit_s"].to_numpy(dtype=float)[:, None]
    values = np.array(design.X, dtype=float)
    cursor = 0
    for item in manifest:  # the hold kernel ends at the center exit
        if item["name"] == "hold":
            columns = slice(cursor, cursor + item["columns"])
            # No step goes with the kernel. In a trial with a hold longer than
            # the window, the kernel keeps its last value up to the center exit.
            beyond = (times - np.asarray(alignments)[:, None]) > HOLD_WINDOW_S
            values[beyond.ravel(), columns] = bases["hold"].basis[-1]
            values[:, columns] *= before_exit.ravel()[:, None]
            item["trials_longer_than_window"] = int(np.sum(holds > HOLD_WINDOW_S))
        cursor += item["columns"]
    if FIXATION_COLUMN:
        entry = trials["center_entry_s"].to_numpy(dtype=float)[:, None]
        first = np.asarray(alignments, dtype=float)[:, None]
        values = np.column_stack((values, ((times >= entry) & (times < first)).ravel()))
        manifest.append(
            {
                "name": FIXATION,
                "group": "task",
                "events": len(alignments),
                "columns": 1,
                "coding": "1 from center entry to the first flash, else 0",
            }
        )
    if SIDE_COLUMN:
        # +1 for a response to the right port and -1 for the left port, from the
        # center exit on; 0 before the exit.
        sign = np.where(trials["response"].to_numpy() == 1, 1.0, -1.0)[:, None]
        values = np.column_stack((values, (sign * ~before_exit).ravel()))
        manifest.append(
            {
                "name": SIDE,
                "group": "task",
                "events": len(alignments),
                "columns": 1,
                "coding": "+1 (right) or -1 (left) from the center exit on, else 0",
            }
        )
    if PREVIOUS_SIDE_COLUMN:
        # +1 (right) or -1 (left) up to the center exit; 0 from the exit on, and
        # in the trials before the first response of the session.
        last = np.nan_to_num(trials["last_side"].to_numpy(dtype=float))[:, None]
        values = np.column_stack((values, (last * before_exit).ravel()))
        manifest.append(
            {
                "name": PREVIOUS_SIDE,
                "group": "task",
                "events": int(np.count_nonzero(last)),
                "columns": 1,
                "coding": "+1 (right) or -1 (left) for the side of the last "
                "response, up to the center exit, else 0",
            }
        )
    if PREVIOUS_OUTCOME_COLUMN:
        outcome = np.nan_to_num(trials["last_outcome"].to_numpy(dtype=float))[:, None]
        values = np.column_stack(
            (values, np.repeat(outcome, times.shape[1], axis=1).ravel())
        )
        manifest.append(
            {
                "name": PREVIOUS_OUTCOME,
                "group": "task",
                "events": int(np.count_nonzero(outcome)),
                "columns": 1,
                "coding": "+1 (rewarded) or -1 (not) for the last trial with a "
                "response, over the whole window, else 0",
            }
        )
    if SESSION_DRIFT:
        drift = session_drift_columns(np.asarray(alignments, dtype=float))
        values = np.column_stack((values, np.repeat(drift, times.shape[1], axis=0)))
        manifest.append(
            {
                "name": DRIFT,
                "group": "task",
                "events": len(alignments),
                "columns": drift.shape[1],
                "coding": "raised cosines (or a cubic spline) of the time of the "
                "trial in the session",
            }
        )
    since_first = times - np.asarray(alignments, dtype=float)[:, None]
    in_window = (since_first >= 0) & (since_first <= TRIAL_TIME_WINDOW_S)
    over_trial = (
        _bumps(since_first / TRIAL_TIME_WINDOW_S, TRIAL_TIME_BASES)
        * in_window[..., None]
    )
    since_exit = times - trials["center_exit_s"].to_numpy(dtype=float)[:, None]
    after_exit = since_exit >= 0
    # After `TURN_DIRECTION_S`, the last cosine keeps its value (1).
    basis = exit_basis().basis
    index = np.clip(np.rint(since_exit / BINWIDTH_S).astype(int), 0, len(basis) - 1)
    turn = basis[index] * after_exit[..., None]
    side = np.where(trials["response"].to_numpy() == 1, 1.0, -1.0)[:, None, None]
    for name, used, block, coding in (
        (
            TIME_IN_TRIAL,
            TIME_IN_TRIAL_TERM,
            over_trial,
            f"{TRIAL_TIME_BASES} raised cosines over 0 to {TRIAL_TIME_WINDOW_S} s "
            "from the first flash",
        ),
        (
            ACCUMULATED_EVIDENCE,
            ACCUMULATED_EVIDENCE_TERM,
            accumulated_evidence(trials, times, np.asarray(alignments, dtype=float))[
                ..., None
            ]
            if ACCUMULATED_EVIDENCE_TERM
            else None,
            "flashes up to the bin minus 1 + (N - 1) k / 24 (N flashes, k of the 24 "
            "random slots started); 0 before the first flash",
        ),
        (
            RESPONSE_PERIOD,
            RESPONSE_PERIOD_TERM,
            turn,
            f"{TURN_DIRECTION_BASES} raised cosines from the center exit to "
            f"{TURN_DIRECTION_S} s after it, the last one held to the response entry",
        ),
        (
            TURN_DIRECTION,
            TURN_DIRECTION_TERM,
            side * turn,
            f"+1 (right) or -1 (left) times {TURN_DIRECTION_BASES} raised cosines "
            f"from the center exit to {TURN_DIRECTION_S} s after it, the last one "
            "held to the response entry",
        ),
    ):
        if used:
            values = np.column_stack((values, block.reshape(-1, block.shape[-1])))
            manifest.append(
                {
                    "name": name,
                    "group": "task",
                    "events": len(alignments),
                    "columns": block.shape[-1],
                    "coding": coding,
                }
            )
    if not IN_PORT_COLUMN:
        return values, manifest
    # Earlier parameterization of the hold term: one state column with no temporal basis.
    start = (
        np.asarray(alignments, dtype=float)
        if STATE_FROM_FIRST_FLASH
        else trials["center_entry_s"].to_numpy(dtype=float)
    )
    in_port = (times >= start[:, None]) & (
        times < trials["center_exit_s"].to_numpy(dtype=float)[:, None]
    )
    manifest.append(
        {
            "name": IN_PORT,
            "group": "task",
            "events": len(alignments),
            "columns": 1,
            "coding": f"1 from {'first flash' if STATE_FROM_FIRST_FLASH else 'center entry'}"
            " to center exit, else 0",
        }
    )
    return np.column_stack((values, in_port.ravel())), manifest


def spike_history_basis() -> RaisedCosineBasis:
    """Return 10 log-spaced functions at strictly past lags 1 through 100 ms."""
    return RaisedCosineBasis(HISTORY_COLUMNS, 0, 0.1, BINWIDTH_S, log_scale=True)


def build_unit_history(alignments: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    """Return one unit's 10 strictly past self-history columns.

    Shifting the spike times by one bin keeps the filter causal, so the count in
    a bin is never a predictor of itself.
    """
    design = DesignMatrix(alignments, PRE_S, POST_S, BINWIDTH_S)
    design.add_regressor(
        EventRegressor(
            "self_history",
            np.asarray(spike_times, dtype=float) + BINWIDTH_S,
            BINWIDTH_S,
            basis_objects=[spike_history_basis()],
            tags="history",
        )
    )
    design.build_matrix()
    history = np.asarray(design.X)
    if history.shape[1] != HISTORY_COLUMNS:
        raise ValueError("Expected 10 strictly past self-history columns.")
    return history


def _design_with_history(
    common: np.ndarray,
    history: np.ndarray,
    common_columns: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Append a unit's standardized history block to the shared design.

    Pass `rows` to build only the rows a fit needs. Materializing all 775,866
    and then slicing costs the full array plus every slice; roughly half those
    rows are masked out after response entry and never scored.
    """
    index = slice(None) if rows is None else rows
    height = common.shape[0] if rows is None else int(np.count_nonzero(rows))
    design = np.empty((height, common_columns + history.shape[1]), dtype=np.float32)
    design[:, :common_columns] = common[index, :common_columns]
    design[:, common_columns:] = history[index]
    return design


def _load_windows(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as windows:
        selected_rows = windows["selected_trial_rows"]
        eligible = windows["eligible_trials"]
        return {
            "metadata": json.loads(str(windows["metadata_json"])),
            "selected_rows": selected_rows.copy(),
            "selected_trial_numbers": windows["trial_num"][selected_rows].copy(),
            "alignments": windows["first_stim_s"][eligible].copy(),
            "split": windows["bin_split"].copy(),
            "bin_times": windows["bin_center_s"].copy(),
            "relative_centers": windows["relative_bin_centers_s"].copy(),
            "response_times": windows["response_entry_s"][eligible].copy(),
        }


def _load_selected_trials(prepared: dict):
    metadata = prepared["metadata"]
    trials = build_trial_table(
        metadata["subject_name"], metadata["session_name"], include_frames=False
    )
    # The outcome of the last trial with a response before this one: +1 rewarded,
    # -1 not; from the full table, so that early withdrawals are skipped.
    responded = trials["response"].isin((-1, 1))
    trials = trials.assign(
        last_outcome=np.where(trials["rewarded"] == 1, 1.0, -1.0)
    ).assign(last_outcome=lambda t: t["last_outcome"].where(responded).ffill().shift(1))
    trials = trials.iloc[prepared["selected_rows"]].reset_index(drop=True)
    if not np.array_equal(trials["trial_num"], prepared["selected_trial_numbers"]):
        raise ValueError("Prepared and live trial selections differ.")
    return trials


def _valid_bin_mask(prepared: dict) -> np.ndarray:
    """Keep bins at or before that trial's response entry."""
    limits = prepared["response_times"] - prepared["alignments"]
    return (prepared["relative_centers"][None, :] <= limits[:, None]).ravel()


def _split_masks(
    split: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test rows after the response-entry mask."""
    split = np.asarray(split)
    if not np.array_equal(np.unique(split), [0, 1, 2]):
        raise ValueError("Rows must contain train, validation, and test labels.")
    return tuple((split == value) & valid for value in (0, 1, 2))


def prepare_common_design(windows: Path, video: Path, output: Path) -> None:
    """Build and save the shared task and motion-energy design columns."""
    prepared = _load_windows(windows)
    trials = _load_selected_trials(prepared)
    with np.load(video, allow_pickle=False) as data:
        frame_times = data["frame_times_s"].copy()
        # The video file has more PCs than the model can use; keep the first ones.
        scores = data["scores"][:, : max(VIDEO_COMPONENT_COUNTS)].copy()
    if scores.ndim != 2 or scores.shape[1] != max(VIDEO_COMPONENT_COUNTS):
        raise ValueError("The video file has too few motion-energy PCs.")
    if len(frame_times) != len(scores):
        raise ValueError("Video times and score rows must match.")
    if (
        not np.isfinite(frame_times).all()
        or not np.isfinite(scores).all()
        or np.any(np.diff(frame_times) <= 0)
    ):
        raise ValueError("Video times and scores must be finite and time ordered.")

    task_values, task_manifest = build_task_design(prepared["alignments"], trials)
    task_columns = task_values.shape[1]
    valid = _valid_bin_mask(prepared)
    train_rows, _, _ = _split_masks(prepared["split"], valid)
    scaled_base, base_mean, base_scale = training_zscore(task_values, train_rows)
    cursor = 0
    for item in task_manifest:  # a weaker penalty on the fixation kernel
        if item["name"] == HOLD:
            columns = slice(cursor, cursor + item["columns"])
            scaled_base[:, columns] *= HOLD_PENALTY_SCALE
            base_scale[columns] /= HOLD_PENALTY_SCALE
        cursor += item["columns"]
    base_names = [
        f"{item['name']}_basis_{index + 1:02d}"
        for item in task_manifest
        for index in range(item["columns"])
    ]
    base_columns = len(base_names)
    total_columns = base_columns + scores.shape[1]

    output.parent.mkdir(parents=True, exist_ok=True)
    matrix_partial = output.with_name(f"{output.name}.partial")
    metadata_path = output.with_suffix(".json")
    metadata_partial = metadata_path.with_name(f"{metadata_path.name}.partial")
    for path in (matrix_partial, metadata_partial):
        if path.exists():
            raise FileExistsError(path)
    matrix = np.lib.format.open_memmap(
        matrix_partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(prepared["split"]), total_columns),
    )
    matrix[:, :base_columns] = scaled_base.astype(np.float32)
    del scaled_base, task_values

    # Sample each PC at the bin centers, on the grid rebuilt from the constants
    # like the spike counts and task design, not the reusable windows file.
    centers, _, _ = construct_timebins(PRE_S, POST_S, BINWIDTH_S)
    sample_times = (prepared["alignments"][:, None] + centers).ravel()
    video_mean = []
    video_scale = []
    video_names = []
    for component in range(scores.shape[1]):
        block = np.interp(
            sample_times, frame_times, scores[:, component], left=0, right=0
        )
        scaled, mean, scale = training_zscore(block[:, None], train_rows)
        matrix[:, base_columns + component] = scaled[:, 0].astype(np.float32)
        video_mean.extend(mean.tolist())
        video_scale.extend(scale.tolist())
        video_names.append(f"video_pc_{component + 1:03d}")
        if (component + 1) % 10 == 0:
            matrix.flush()
            print(
                f"Built {component + 1} of {scores.shape[1]} video components.",
                flush=True,
            )

    metadata = {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "dtype": str(matrix.dtype),
        "task_columns": task_columns,
        "base_columns": base_columns,
        "video_components": scores.shape[1],
        "video_columns_per_component": 1,
        "video_component_candidates": list(VIDEO_COMPONENT_COUNTS),
        "column_names": base_names + video_names,
        "task_manifest": task_manifest,
        "training_mean": base_mean.tolist() + video_mean,
        "training_scale": base_scale.tolist() + video_scale,
        "split_row_counts": np.bincount(prepared["split"], minlength=3).tolist(),
        "hold_penalty_scale": HOLD_PENALTY_SCALE,
    }
    matrix.flush()
    del matrix
    with metadata_partial.open("x") as handle:
        json.dump(metadata, handle, indent=2)
    matrix_partial.replace(output)
    metadata_partial.replace(metadata_path)
    omitted = {"column_names", "task_manifest", "training_mean", "training_scale"}
    print(
        json.dumps(
            {key: metadata[key] for key in metadata if key not in omitted}, indent=2
        )
    )


def _common_design_is_current(output: Path) -> bool:
    """Return whether a saved design has the current task and video columns."""
    metadata_path = output.with_suffix(".json")
    if not output.exists() or not metadata_path.exists():
        return False
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    bases = task_temporal_bases()
    manifest = metadata.get("task_manifest", [])
    # The columns with no temporal basis, in the order of `build_task_design`.
    drift_columns = 7 if DRIFT_SPLINE else DRIFT_BASES - 1
    state = [
        (name, columns)
        for name, columns, used in (
            (FIXATION, 1, FIXATION_COLUMN),
            (SIDE, 1, SIDE_COLUMN),
            (PREVIOUS_SIDE, 1, PREVIOUS_SIDE_COLUMN),
            (PREVIOUS_OUTCOME, 1, PREVIOUS_OUTCOME_COLUMN),
            (DRIFT, drift_columns, SESSION_DRIFT),
            (TIME_IN_TRIAL, TRIAL_TIME_BASES, TIME_IN_TRIAL_TERM),
            (ACCUMULATED_EVIDENCE, 1, ACCUMULATED_EVIDENCE_TERM),
            (RESPONSE_PERIOD, TURN_DIRECTION_BASES, RESPONSE_PERIOD_TERM),
            (TURN_DIRECTION, TURN_DIRECTION_BASES, TURN_DIRECTION_TERM),
            (IN_PORT, 1, IN_PORT_COLUMN),
        )
        if used
    ]
    return (
        metadata.get("video_columns_per_component") == 1
        and [item.get("name") for item in manifest]
        == [*bases, *(name for name, _ in state)]
        and [item.get("columns") for item in manifest]
        == [
            *(basis.basis.shape[1] for basis in bases.values()),
            *(columns for _, columns in state),
        ]
        and metadata.get("hold_penalty_scale", 1.0) == HOLD_PENALTY_SCALE
    )
