"""Category, choice, and rate decoding from GRB006 V1 population activity.

This script reproduces the category-decoding pipeline in
``notebooks/categorydecoding.ipynb`` and extends it to ask whether the category
signal in V1 is separable from a choice signal, whether category and choice
interact at the single-unit level, and whether stimulus rate is decodable. It
restricts to GRB006 ``20240821_121447`` only.

Every classifier uses **balanced classes**: the majority class is randomly
undersampled to the minority count and the result is averaged over many
resamples, so chance is exactly 50% (binary) or 1/k (multiclass) and accuracy
is directly interpretable.

It answers the following questions and writes a small figure set plus an
HTML report under ``reports/category_choice_decoding.html``:

1. Is stimulus category decodable from V1, and how does that relate to
   single-unit category tuning?
2. Is choice decodable from V1, and how does that relate to choice tuning?
3. If choice variance is regressed out of the population (fold-safe), can
   category still be decoded, and vice versa?
4. Are units that are category-tuned also choice-tuned in a consistent way
   (an interaction)?
5. Is stimulus rate decodable (graded regression + balanced multiclass), and
   how does that relate to rate tuning?

Run requires DataJoint / labdata DB access (VPN), like the source notebook.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from html import escape
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import matplotlib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

import statsmodels.api as sm  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
)
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata  # noqa: E402
from ephys.src.utils.io_digital_events import fetch_session_events  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical parameters (single source; do not redefine inline downstream).
# ---------------------------------------------------------------------------
SUBJECT = "GRB006"
SESSION = "20240821_121447"
UNIT_CRITERIA_ID = 1

POST_STIM_WINDOW_S = 1.0  # integration window after first stimulus (notebook)
N_TIMEPOINTS = 10  # bins for the time-course decoder (notebook)
N_TIMEPOINTS_200MS = 5  # smoother 200 ms bins across the same 1 s window
CV_FOLDS = 10
N_BALANCE_RESAMPLES = 20  # balanced-undersample draws per accuracy estimate
N_SHUFFLES = 100  # permutation null reps for headline accuracies
TIMECOURSE_N_SHUFFLES = 10  # null reps for the per-bin time-course
RANDOM_STATE = 0
FDR_ALPHA = 0.05

FIGURE_ROOT = Path(os.environ.get("EPHYS_FIGURE_ROOT", str(REPO_ROOT / "figures")))
FIGURE_DIR = FIGURE_ROOT / "category_choice_decoding"
REPORT_PATH = REPO_ROOT / "reports" / "category_choice_decoding.html"


def repository_provenance() -> dict:
    def git(*args):
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    script_path = Path(__file__).resolve()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "software_versions": software_versions(),
    }


def software_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "datajoint",
        "labdata",
    ]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed as a named distribution"
    return versions


def audit_motion_cache(n_trials: int) -> dict:
    """Describe the recoverable June motion-energy cache without overclaiming it."""
    motion_path = REPO_ROOT / ".cache/categorydecoding/motion_early_values.npz"
    video_path = (
        REPO_ROOT / ".cache/labdata_video/GRB006/20240821_121447/chipmunk/"
        "GRB006_20240821_121447_chipmunk_DemonstratorAudiTask_"
        "BackStereoView_00000000.avi"
    )
    source_ref = "stringer-subspaces:notebooks/categorydecoding.ipynb"
    source_check = subprocess.run(
        ["git", "cat-file", "-e", source_ref],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    out: dict[str, object] = {
        "cache_path": str(motion_path.relative_to(REPO_ROOT)),
        "cache_exists": motion_path.exists(),
        "video_path": str(video_path.relative_to(REPO_ROOT)),
        "video_exists": video_path.exists(),
        "video_frame_rate_hz": 60.0,
        "video_frame_size_px": [640, 512],
        "alignment": (
            "DatasetVideo frame_times when available, otherwise aligned_events['frames']; "
            "motion timestamps use frame_times[1:]"
        ),
        "spatial_processing": "full grayscale frame; no crop, mask, PCA, or smoothing",
        "frame_check": (
            "recovered source requires video, database n_frames, and frame_times "
            "counts to agree within one; the cached run does not store the exact difference"
        ),
        "label_free_feature": True,
        "source_ref": source_ref,
        "source_ref_available": source_check.returncode == 0,
        "n_analysis_trials": int(n_trials),
        "metric": "mean absolute grayscale frame-to-frame difference",
        "window": "0 to 0.5 s after first stimulus onset",
        "directional": False,
    }
    if not motion_path.exists():
        return out
    with np.load(motion_path, allow_pickle=False) as cache:
        values = np.asarray(cache["motion_early"], dtype=float)
        threshold = float(cache["threshold"]) if "threshold" in cache else None
    finite = np.isfinite(values)
    out.update(
        {
            "n_cached_trials": int(len(values)),
            "n_finite": int(finite.sum()),
            "n_missing": int((~finite).sum()),
            "coverage_fraction": float(finite.sum() / n_trials),
            "median_split_threshold": threshold,
        }
    )
    return out


def artifact_inventory() -> dict:
    paths = {
        "active_script": REPO_ROOT / "scripts/analyses/category_choice_decoding.py",
        "active_notebook": REPO_ROOT / "notebooks/categorydecoding.ipynb",
        "html_report": REPORT_PATH,
        "pi_update_deck": REPO_ROOT / "reports/category_choice_decoding_pi_update.pptx",
        "powerpoint_lock_file": REPO_ROOT
        / "reports/~$category_choice_decoding_pi_update.pptx",
    }
    out = {}
    for label, path in paths.items():
        out[label] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.exists(),
            "modified_at_utc": (
                datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                if path.exists()
                else None
            ),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib windows after building figures.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the analysis without writing figures or the report.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce resample/shuffle counts for a fast smoke run (not for real results).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Decoders (one definition each; data is balanced before fitting).
# ---------------------------------------------------------------------------
def make_decoder():
    """Binary L2 logistic decoder pipeline (no class_weight; data is balanced)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            cv=CV_FOLDS,
            l1_ratios=(0,),  # type: ignore[arg-type]
            solver="lbfgs",
            scoring="accuracy",
            max_iter=1000,
            fit_intercept=True,
            use_legacy_attributes=False,
        ),
    )


def make_fast_decoder():
    """Fixed-regularization logistic decoder for the per-bin time-course.

    The time-course asks *when* a signal appears (a relative comparison across
    bins), so the nested-CV regularization search in ``make_decoder`` is
    unnecessary and ~100x too slow when repeated over every bin x resample x
    shuffle. A single L2 logistic with a fixed C gives the same trend far
    faster; headline accuracies still use the cross-validated decoder.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def make_multiclass():
    """One-vs-rest multiclass logistic decoder (for graded stimulus rate)."""
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    )


def inner_cv():
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_rate_regressor(_X_train=None, args=None):
    alphas = getattr(args, "ridge_alphas", np.logspace(-4, 6, 21))
    return GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": alphas},
        cv=inner_cv(),
        scoring="r2",
    )


def make_fixed_rate_regressor(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


def make_mlp_rate_regressor(_X_train=None, args=None):
    alphas = getattr(args, "mlp_alphas", np.logspace(-4, 2, 7))
    hidden = getattr(args, "mlp_hidden_layer_sizes", [(4,), (8,), (16,)])
    return GridSearchCV(
        make_pipeline(
            StandardScaler(),
            TransformedTargetRegressor(
                regressor=MLPRegressor(
                    activation="tanh",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    max_iter=getattr(args, "mlp_max_iter", 2500),
                    max_fun=getattr(args, "mlp_max_fun", 20000),
                ),
                transformer=StandardScaler(),
            ),
        ),
        {
            "transformedtargetregressor__regressor__hidden_layer_sizes": hidden,
            "transformedtargetregressor__regressor__alpha": alphas,
        },
        cv=inner_cv(),
        scoring="r2",
    )


def logreg_coefs(decoder) -> np.ndarray:
    return decoder.named_steps["logisticregressioncv"].coef_[0]


# ---------------------------------------------------------------------------
# Balanced-decoding helpers (undersample the majority class to equal size).
# ---------------------------------------------------------------------------
def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices for a class-balanced subset (majority undersampled to minority)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_per_class = int(counts.min())
    picks = [
        rng.choice(np.flatnonzero(y == c), size=n_per_class, replace=False)
        for c in classes
    ]
    return np.sort(np.concatenate(picks))


def balanced_joint_indices(
    y: np.ndarray, nuisance: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Indices balanced for every target x nuisance cell."""
    groups = np.array([f"{a}|{b}" for a, b in zip(y, nuisance)])
    cells, counts = np.unique(groups, return_counts=True)
    n_per_cell = int(counts.min())
    picks = [
        rng.choice(np.flatnonzero(groups == cell), size=n_per_cell, replace=False)
        for cell in cells
    ]
    return np.sort(np.concatenate(picks))


def _kfold_for(y: np.ndarray) -> StratifiedKFold:
    """Stratified K-fold with fold count capped by the smallest class."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(CV_FOLDS, max(2, counts.min())))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def balanced_decode(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
    decoder_factory=make_decoder,
) -> np.ndarray:
    """Per-resample mean CV accuracy on class-balanced subsets."""
    means = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        idx = balanced_indices(y, rng)
        Xb, yb = X[idx], y[idx]
        means[i] = cross_val_score(
            decoder_factory(), Xb, yb, cv=_kfold_for(yb), scoring="accuracy"
        ).mean()
    return means


def balanced_joint_decode(
    X: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
    decoder_factory=make_decoder,
) -> np.ndarray:
    """Decode target after balancing each target x nuisance cell."""
    means = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        idx = balanced_joint_indices(y, nuisance, rng)
        Xb, yb = X[idx], y[idx]
        means[i] = cross_val_score(
            decoder_factory(), Xb, yb, cv=_kfold_for(yb), scoring="accuracy"
        ).mean()
    return means


def balanced_joint_shuffle(
    X: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    rng: np.random.Generator,
    n_shuffles: int,
    decoder_factory=make_decoder,
) -> np.ndarray:
    """Null distribution for joint-balanced decoding."""
    out = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        ys = rng.permutation(y)
        idx = balanced_joint_indices(ys, nuisance, rng)
        Xb, yb = X[idx], ys[idx]
        out[i] = cross_val_score(
            decoder_factory(), Xb, yb, cv=_kfold_for(yb), scoring="accuracy"
        ).mean()
    return out


def category_to_boundary_choice_decode(
    X_train: np.ndarray,
    y_cat: np.ndarray,
    y_choice: np.ndarray,
    X_boundary: np.ndarray,
    boundary_choice: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train category readout, then test whether it predicts boundary choices."""
    accs = np.full(n_resamples, np.nan)
    high_frac = np.full(n_resamples, np.nan)
    preds = np.full((n_resamples, len(boundary_choice)), np.nan)
    for i in range(n_resamples):
        train_idx = balanced_joint_indices(y_cat, y_choice, rng)
        clf = make_fast_decoder().fit(X_train[train_idx], y_cat[train_idx])
        pred = clf.predict(X_boundary)
        accs[i] = balanced_accuracy_score(boundary_choice, pred)
        high_frac[i] = float(np.mean(pred == 1))
        preds[i] = pred
    return accs, high_frac, preds


def category_to_boundary_choice_shuffle(
    real_predictions: np.ndarray,
    boundary_choice: np.ndarray,
    rng: np.random.Generator,
    n_shuffles: int,
) -> np.ndarray:
    """Null for boundary choice alignment: shuffle boundary choices."""
    out = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        pred = real_predictions[i % len(real_predictions)]
        out[i] = balanced_accuracy_score(rng.permutation(boundary_choice), pred)
    return out


def balanced_shuffle(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_shuffles: int,
    decoder_factory=make_decoder,
) -> np.ndarray:
    """Null distribution: permute labels, then balance and decode (one draw each)."""
    out = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        ys = rng.permutation(y)
        idx = balanced_indices(ys, rng)
        Xb, yb = X[idx], ys[idx]
        out[i] = cross_val_score(
            decoder_factory(), Xb, yb, cv=_kfold_for(yb), scoring="accuracy"
        ).mean()
    return out


def balanced_residual_decode(
    X: np.ndarray,
    y: np.ndarray,
    regressor: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
    decoder_factory=make_decoder,
) -> np.ndarray:
    """Balanced decode of ``y`` after fold-safe linear removal of ``regressor``.

    The nuisance regression is fit on the training split only and applied to
    both train and test, so no information leaks across the held-out fold.
    """
    regressor = np.asarray(regressor, dtype=float).reshape(len(X), -1)
    means = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        idx = balanced_indices(y, rng)
        Xb, yb, rb = X[idx], y[idx], regressor[idx]
        fold_scores = []
        for train_idx, test_idx in _kfold_for(yb).split(Xb, yb):
            nuisance = LinearRegression().fit(rb[train_idx], Xb[train_idx])
            x_train = Xb[train_idx] - nuisance.predict(rb[train_idx])
            x_test = Xb[test_idx] - nuisance.predict(rb[test_idx])
            clf = decoder_factory().fit(x_train, yb[train_idx])
            fold_scores.append(clf.score(x_test, yb[test_idx]))
        means[i] = float(np.mean(fold_scores))
    return means


def balanced_residual_shuffle(
    X: np.ndarray,
    y: np.ndarray,
    regressor: np.ndarray,
    rng: np.random.Generator,
    n_shuffles: int,
    decoder_factory=make_decoder,
) -> np.ndarray:
    out = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        ys = rng.permutation(y)
        out[i] = balanced_residual_decode(
            X, ys, regressor, rng, 1, decoder_factory=decoder_factory
        )[0]
    return out


def perm_p(real_mean: float, null: np.ndarray) -> float:
    return float((np.sum(null >= real_mean) + 1) / (len(null) + 1))


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------
def load_session():
    """Load good units, sampling rate, and aligned trial metadata."""
    from labdata.schema import Dataset, EphysRecording, SpikeSorting, UnitCount

    dset = Dataset & f'subject_name = "{SUBJECT}"' & f'session_name = "{SESSION}"'
    units = pd.DataFrame(
        (
            (
                SpikeSorting().Unit()
                & dset.fetch("subject_name", "session_name", as_dict=True)
            )
            * (
                UnitCount.Unit()
                & f"unit_criteria_id = {UNIT_CRITERIA_ID}"
                & "passes = 1"
            )
        ).fetch(
            "subject_name",
            "session_name",
            "dataset_name",
            "unit_id",
            "spike_times",
            as_dict=True,
        )
    )
    srate = float(
        (
            EphysRecording.ProbeSetting()
            & (
                SpikeSorting()
                & f'subject_name = "{SUBJECT}"'
                & f'session_name = "{SESSION}"'
            ).proj()
        ).fetch1("sampling_rate")
    )
    aligned_events = fetch_session_events(SUBJECT, SESSION)
    trials = fetch_trial_metadata(SUBJECT, SESSION, aligned_events)
    return units, srate, trials


def build_design_matrix(
    units: pd.DataFrame,
    srate: float,
    valid_trials: pd.DataFrame,
    window_s: float = POST_STIM_WINDOW_S,
):
    """Trials x units firing rate (sp/s) in the post-first-stim window."""
    n_trials = len(valid_trials)
    n_units = len(units)
    X = np.full((n_trials, n_units), np.nan)
    first_stim = valid_trials["first_stim_ts"].to_numpy(dtype=float)
    for unit_idx, spike_samples in enumerate(units.spike_times):
        st = np.asarray(spike_samples, dtype=float) / srate
        for trial_idx in range(n_trials):
            start = first_stim[trial_idx]
            stop = start + window_s
            X[trial_idx, unit_idx] = ((st >= start) & (st < stop)).sum() / (
                stop - start
            )
    assert not np.isnan(X).any()
    return X


def build_timecourse(
    units: pd.DataFrame,
    srate: float,
    valid_trials: pd.DataFrame,
    n_timepoints: int = N_TIMEPOINTS,
):
    """Trials x units x timepoints rate tensor across the integration window."""
    n_trials = len(valid_trials)
    n_units = len(units)
    bin_edges = np.linspace(0, POST_STIM_WINDOW_S, n_timepoints + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    first_stim = valid_trials["first_stim_ts"].to_numpy(dtype=float)
    X_time = np.full((n_trials, n_units, n_timepoints), np.nan)
    for unit_idx, spike_samples in enumerate(units.spike_times):
        st = np.asarray(spike_samples, dtype=float) / srate
        for trial_idx in range(n_trials):
            bins = first_stim[trial_idx] + bin_edges
            durations = np.diff(bins)
            counts = np.histogram(st, bins=bins)[0]
            X_time[trial_idx, unit_idx] = counts / durations
    assert not np.isnan(X_time).any()
    return X_time, bin_centers


# ---------------------------------------------------------------------------
# Plotting helpers.
# ---------------------------------------------------------------------------
def _set_style(plt) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.dpi"] = 110


def _box(ax, datasets, labels, colors=None):
    positions = np.arange(1, len(datasets) + 1)
    bp = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="k"),
        boxprops=dict(color="k"),
        whiskerprops=dict(color="k"),
        capprops=dict(color="k"),
    )
    face = colors if colors is not None else ["0.8"] * len(datasets)
    for patch, color in zip(bp["boxes"], face):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def _style_decoder_axis(
    ax, ylabel: str = "CV accuracy (balanced trials)", ylim=(0.4, 1.0)
) -> None:
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)


def _cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)
    n_pos = len(x_pos)
    n_neg = len(x_neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    pooled = np.sqrt(
        ((n_pos - 1) * np.var(x_pos, ddof=1) + (n_neg - 1) * np.var(x_neg, ddof=1))
        / (n_pos + n_neg - 2)
    )
    if pooled == 0:
        return 0.0
    return float((np.mean(x_pos) - np.mean(x_neg)) / pooled)


def _binary_effect_d(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([_cohens_d(X[y == 1, u], X[y == 0, u]) for u in range(X.shape[1])])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(
        pd.Series(a).rank(method="average").to_numpy(),
        pd.Series(b).rank(method="average").to_numpy(),
    )


def save_fig(fig, name: str, no_save: bool) -> None:
    if no_save:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"Saved -> {FIGURE_DIR / name}.(pdf|png)")


# ---------------------------------------------------------------------------
# Q1: category decoding + category tuning.
# ---------------------------------------------------------------------------
def analyze_category(plt, X, y_cat, rng, args) -> dict:
    print("\n[Q1] Category decoding ...")
    real = balanced_decode(X, y_cat, rng, args.n_balance_resamples)
    shuffle = balanced_shuffle(X, y_cat, rng, args.n_shuffles)

    weight_idx = balanced_indices(y_cat, np.random.default_rng(RANDOM_STATE))
    decoder = make_decoder().fit(X[weight_idx], y_cat[weight_idx])
    coefs = logreg_coefs(decoder)
    tuning = _binary_effect_d(X, y_cat)
    r_tuning_weight = _corr(tuning, coefs)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    _box(axs[0], [real, shuffle], ["real", "shuffle"], ["tab:blue", "0.7"])
    _style_decoder_axis(axs[0])
    axs[0].set_xlabel("train: non-boundary category\ntest: held-out trials")

    axs[1].scatter(tuning, coefs, color="k", alpha=0.3, s=18)
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].axvline(0, color="k", lw=0.8)
    axs[1].text(
        0.05, 0.95, f"r = {r_tuning_weight:.2f}", transform=axs[1].transAxes, va="top"
    )
    axs[1].set_xlabel("category discriminability: high vs low (Cohen's d)")
    axs[1].set_ylabel("decoder weight")
    fig.tight_layout()
    save_fig(fig, "fig1_category_decoding", args.no_save)

    return {
        "real_mean": float(real.mean()),
        "real_std": float(real.std(ddof=1)),
        "shuffle_mean": float(shuffle.mean()),
        "shuffle_max": float(shuffle.max()),
        "p_perm": perm_p(real.mean(), shuffle),
        "r_tuning_weight": r_tuning_weight,
    }


# ---------------------------------------------------------------------------
# Q2: choice decoding + choice tuning.
# ---------------------------------------------------------------------------
def analyze_choice(plt, X, y_choice, rng, args) -> dict:
    print("\n[Q2] Choice decoding ...")
    real = balanced_decode(X, y_choice, rng, args.n_balance_resamples)
    shuffle = balanced_shuffle(X, y_choice, rng, args.n_shuffles)

    weight_idx = balanced_indices(y_choice, np.random.default_rng(RANDOM_STATE + 1))
    decoder = make_decoder().fit(X[weight_idx], y_choice[weight_idx])
    coefs = logreg_coefs(decoder)
    tuning = _binary_effect_d(X, y_choice)
    r_tuning_weight = _corr(tuning, coefs)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    _box(axs[0], [real, shuffle], ["real", "shuffle"], ["tab:green", "0.7"])
    _style_decoder_axis(axs[0])
    axs[0].set_xlabel("train: non-boundary choice\ntest: held-out trials")

    axs[1].scatter(tuning, coefs, color="k", alpha=0.3, s=18)
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].axvline(0, color="k", lw=0.8)
    axs[1].text(
        0.05, 0.95, f"r = {r_tuning_weight:.2f}", transform=axs[1].transAxes, va="top"
    )
    axs[1].set_xlabel("choice discriminability: right vs left (Cohen's d)")
    axs[1].set_ylabel("decoder weight")
    fig.tight_layout()
    save_fig(fig, "fig2_choice_decoding", args.no_save)

    return {
        "real_mean": float(real.mean()),
        "real_std": float(real.std(ddof=1)),
        "shuffle_mean": float(shuffle.mean()),
        "shuffle_max": float(shuffle.max()),
        "p_perm": perm_p(real.mean(), shuffle),
        "r_tuning_weight": r_tuning_weight,
    }


# ---------------------------------------------------------------------------
# Q3: fold-safe, balanced residualized decoding (both directions).
# ---------------------------------------------------------------------------
def analyze_residual(plt, X, y_cat, y_choice, rng, args) -> dict:
    print("\n[Q3] Residualized decoding ...")
    cat_raw = balanced_decode(X, y_cat, rng, args.n_balance_resamples)
    cat_resid = balanced_residual_decode(
        X, y_cat, y_choice, rng, args.n_balance_resamples
    )
    cat_resid_shuffle = balanced_residual_shuffle(
        X, y_cat, y_choice, rng, args.n_shuffles
    )

    cho_raw = balanced_decode(X, y_choice, rng, args.n_balance_resamples)
    cho_resid = balanced_residual_decode(
        X, y_choice, y_cat, rng, args.n_balance_resamples
    )
    cho_resid_shuffle = balanced_residual_shuffle(
        X, y_choice, y_cat, rng, args.n_shuffles
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _box(
        ax,
        [cat_resid, cat_resid_shuffle, cho_resid, cho_resid_shuffle],
        [
            "category\nminus choice",
            "shuffle",
            "choice\nminus category",
            "shuffle",
        ],
        ["tab:cyan", "0.7", "tab:olive", "0.7"],
    )
    _style_decoder_axis(ax)
    ax.axvline(2.5, color="0.8", lw=1)
    ax.set_title("train: residualized decoder\ntest: held-out residual trials")
    fig.tight_layout()
    save_fig(fig, "fig3_residual_decoding", args.no_save)

    return {
        "cat_raw_mean": float(cat_raw.mean()),
        "cat_resid_mean": float(cat_resid.mean()),
        "cat_resid_shuffle_mean": float(cat_resid_shuffle.mean()),
        "cat_resid_p": perm_p(cat_resid.mean(), cat_resid_shuffle),
        "cho_raw_mean": float(cho_raw.mean()),
        "cho_resid_mean": float(cho_resid.mean()),
        "cho_resid_shuffle_mean": float(cho_resid_shuffle.mean()),
        "cho_resid_p": perm_p(cho_resid.mean(), cho_resid_shuffle),
    }


# ---------------------------------------------------------------------------
# Residualization sanity checks (self-residual negative controls).
# ---------------------------------------------------------------------------
def analyze_residual_sanity(plt, X, y_cat, y_choice, rng, args) -> dict:
    """Sanity checks for fold-safe linear residualization.

    If removal works, decoding a variable from activity after regressing *that
    same variable* out should fall to chance (~50%). Cross-residual decoding
    (remove the other variable) is shown alongside for contrast: category should
    stay readable after choice removal only if it carries unique information;
    it should *not* stay readable after category self-removal.

    Residual signal above chance after self-removal means the linear nuisance
    model did not fully subtract what the decoder reads — interpret cross-
    residual survival cautiously and prefer stronger controls in future work.
    """
    print("\n[Sanity] Residualization negative controls ...")
    cat_self = balanced_residual_decode(
        X,
        y_cat,
        y_cat,
        rng,
        args.n_balance_resamples,
        decoder_factory=make_fast_decoder,
    )
    cat_self_shuffle = balanced_residual_shuffle(
        X, y_cat, y_cat, rng, args.n_shuffles, decoder_factory=make_fast_decoder
    )

    cho_self = balanced_residual_decode(
        X,
        y_choice,
        y_choice,
        rng,
        args.n_balance_resamples,
        decoder_factory=make_fast_decoder,
    )
    cho_self_shuffle = balanced_residual_shuffle(
        X,
        y_choice,
        y_choice,
        rng,
        args.n_shuffles,
        decoder_factory=make_fast_decoder,
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _box(
        ax,
        [cat_self, cat_self_shuffle, cho_self, cho_self_shuffle],
        [
            "category\nminus category",
            "shuffle",
            "choice\nminus choice",
            "shuffle",
        ],
        ["tab:orange", "0.7", "tab:orange", "0.7"],
    )
    _style_decoder_axis(ax)
    ax.axvline(2.5, color="0.8", lw=1)
    ax.set_title("self-removal negative control")
    fig.tight_layout()
    save_fig(fig, "fig7_residual_sanity", args.no_save)

    cat_self_p = perm_p(cat_self.mean(), cat_self_shuffle)
    cho_self_p = perm_p(cho_self.mean(), cho_self_shuffle)

    return {
        "cat_self_mean": float(cat_self.mean()),
        "cat_self_shuffle_mean": float(cat_self_shuffle.mean()),
        "cat_self_p": cat_self_p,
        "cat_self_at_chance": bool(cat_self.mean() <= 0.55),
        "cho_self_mean": float(cho_self.mean()),
        "cho_self_shuffle_mean": float(cho_self_shuffle.mean()),
        "cho_self_p": cho_self_p,
        "cho_self_at_chance": bool(cho_self.mean() <= 0.55),
    }


# ---------------------------------------------------------------------------
# Q4: category x choice interaction (single-unit structure).
# ---------------------------------------------------------------------------
def _scatter_equal(ax, x, y, *, xlabel: str, ylabel: str, color="k") -> float:
    r = _corr(x, y)
    lim = np.nanpercentile(np.abs(np.concatenate([x, y])), 99)
    lim = float(lim) if np.isfinite(lim) and lim > 0 else 1.0
    ax.scatter(x, y, color=color, alpha=0.3, s=18)
    ax.axline((0, 0), slope=1, color="k", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return r


def _plot_interaction_psths(
    plt,
    units: pd.DataFrame,
    srate: float,
    valid_trials: pd.DataFrame,
    selected_units: np.ndarray,
    y_cat: np.ndarray,
    y_choice: np.ndarray,
    inter_coef: np.ndarray,
    model_r2: np.ndarray,
    inter_q: np.ndarray,
    no_save: bool,
) -> None:
    if selected_units.size == 0:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No FDR-significant category x choice interaction units",
            ha="center",
            va="center",
        )
        save_fig(fig, "fig4_interaction_psths", no_save)
        return

    edges = np.arange(-0.2, 1.0 + 0.05, 0.05)
    centers = edges[:-1] + np.diff(edges) / 2
    first_stim = valid_trials["first_stim_ts"].to_numpy(dtype=float)
    conds = [
        ("low / left", (y_cat == 0) & (y_choice == 0), "tab:blue", "-"),
        ("low / right", (y_cat == 0) & (y_choice == 1), "tab:blue", "--"),
        ("high / left", (y_cat == 1) & (y_choice == 0), "tab:red", "-"),
        ("high / right", (y_cat == 1) & (y_choice == 1), "tab:red", "--"),
    ]
    n_cols = min(3, selected_units.size)
    n_rows = int(np.ceil(selected_units.size / n_cols))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.1 * n_cols, 2.8 * n_rows),
        sharex=True,
        squeeze=False,
    )
    for ax, unit_idx in zip(axs.ravel(), selected_units):
        st = np.asarray(units.iloc[unit_idx].spike_times, dtype=float) / srate
        for label, mask, color, linestyle in conds:
            trial_rates = []
            for start in first_stim[mask]:
                counts = np.histogram(st - start, bins=edges)[0]
                trial_rates.append(counts / np.diff(edges))
            if trial_rates:
                ax.plot(
                    centers,
                    np.mean(trial_rates, axis=0),
                    color=color,
                    ls=linestyle,
                    lw=1.5,
                    label=label,
                )
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(
            f"unit {units.iloc[unit_idx].unit_id}\n"
            f"coef={inter_coef[unit_idx]:.2f}, R2={model_r2[unit_idx]:.2f}, q={inter_q[unit_idx]:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("sp/s")
    for ax in axs.ravel()[selected_units.size :]:
        ax.axis("off")
    axs[-1, 0].set_xlabel("time from first stimulus (s)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig4_interaction_psths", no_save)


def analyze_interaction(
    plt, units, srate, valid_trials, X, y_cat, y_choice, args
) -> dict:
    print("\n[Q4] Category x choice interaction ...")
    cat = y_cat.astype(float)
    cho = y_choice.astype(float)
    design = np.column_stack([np.ones_like(cat), cat, cho, cat * cho])
    n_units = X.shape[1]
    inter_coef = np.full(n_units, np.nan)
    inter_p = np.full(n_units, np.nan)
    model_r2 = np.full(n_units, np.nan)
    for u in range(n_units):
        fit = sm.OLS(X[:, u], design).fit()
        inter_coef[u] = fit.params[3]
        inter_p[u] = fit.pvalues[3]
        model_r2[u] = fit.rsquared
    reject, inter_q, _, _ = multipletests(inter_p, alpha=FDR_ALPHA, method="fdr_bh")
    n_sig_inter = int(reject.sum())

    high, low = y_cat == 1, y_cat == 0
    left, right = y_choice == 0, y_choice == 1
    delta_left = X[high & left].mean(axis=0) - X[low & left].mean(axis=0)
    delta_right = X[high & right].mean(axis=0) - X[low & right].mean(axis=0)
    choice_low = X[right & low].mean(axis=0) - X[left & low].mean(axis=0)
    choice_high = X[right & high].mean(axis=0) - X[left & high].mean(axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    r_left_right = _scatter_equal(
        axs[0],
        delta_left,
        delta_right,
        xlabel="category effect | left choice (sp/s)",
        ylabel="category effect | right choice (sp/s)",
    )
    r_low_high = _scatter_equal(
        axs[1],
        choice_low,
        choice_high,
        xlabel="choice effect | low category (sp/s)",
        ylabel="choice effect | high category (sp/s)",
    )
    axs[2].hist(inter_coef, bins=30, color="0.4")
    axs[2].axvline(0, color="k", lw=1)
    axs[2].text(
        0.05,
        0.95,
        f"{n_sig_inter}/{n_units} FDR-sig",
        transform=axs[2].transAxes,
        va="top",
    )
    axs[2].set_xlabel("category x choice coefficient (sp/s)")
    axs[2].set_ylabel("units")
    fig.tight_layout()
    save_fig(fig, "fig4_interaction", args.no_save)

    sig_units = np.flatnonzero(reject)
    if sig_units.size:
        order = np.argsort(np.abs(inter_coef[sig_units]))[::-1]
        sig_units = sig_units[order][:6]
    _plot_interaction_psths(
        plt,
        units,
        srate,
        valid_trials,
        sig_units,
        y_cat,
        y_choice,
        inter_coef,
        model_r2,
        inter_q,
        args.no_save,
    )

    sig_rows = [
        {
            "unit_id": str(units.iloc[int(u)].unit_id),
            "interaction_coef": float(inter_coef[u]),
            "interaction_q": float(inter_q[u]),
            "model_r2": float(model_r2[u]),
        }
        for u in sig_units
    ]

    return {
        "n_units": n_units,
        "n_sig_interaction": n_sig_inter,
        "r_category_effect_left_right": r_left_right,
        "r_choice_effect_low_high": r_low_high,
        "interaction_coef_mean": float(np.nanmean(inter_coef)),
        "interaction_coef_abs95": float(np.nanpercentile(np.abs(inter_coef), 95)),
        "significant_interaction_units": sig_rows,
    }


# ---------------------------------------------------------------------------
# Q5: stimulus rate decoding (graded ridge + balanced multiclass) + tuning.
# ---------------------------------------------------------------------------
def balanced_multiclass_rate(X, rate, rng, n_resamples):
    """Balanced multiclass one-vs-rest decoding of rate; returns acc + confusion."""
    labels = np.unique(rate)
    accs = np.full(n_resamples, np.nan)
    cms = np.full((n_resamples, len(labels), len(labels)), np.nan)
    for i in range(n_resamples):
        idx = balanced_indices(rate, rng)
        Xb, rb = X[idx], rate[idx]
        pred = cross_val_predict(make_multiclass(), Xb, rb, cv=_kfold_for(rb))
        accs[i] = balanced_accuracy_score(rb, pred)
        cms[i] = confusion_matrix(rb, pred, labels=labels, normalize="true")
    return accs, np.nanmean(cms, axis=0), labels


def jsonable_param_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [jsonable_param_value(v) for v in value]
    if isinstance(value, list):
        return [jsonable_param_value(v) for v in value]
    return value


def format_param_value(value) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return f"({format_param_value(value[0])},)"
        return "(" + ", ".join(format_param_value(v) for v in value) + ")"
    if isinstance(value, float | np.floating):
        return f"{float(value):.4g}"
    if isinstance(value, int | np.integer):
        return str(int(value))
    return str(value)


def param_equal(a, b) -> bool:
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return tuple(a) == tuple(b)
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return a == b


def search_info(model):
    if not hasattr(model, "best_params_"):
        return {}
    best = {key: jsonable_param_value(val) for key, val in model.best_params_.items()}
    edges = {}
    best_indices = {}
    grid_sizes = {}
    for key, vals in model.param_grid.items():
        values = list(vals)
        raw_val = model.best_params_[key]
        idx = next(
            (
                i
                for i, candidate in enumerate(values)
                if param_equal(candidate, raw_val)
            ),
            0,
        )
        best_indices[key] = int(idx)
        grid_sizes[key] = int(len(values))
        edges[key] = {
            "min": jsonable_param_value(values[0]),
            "max": jsonable_param_value(values[-1]),
            "hit_min": bool(idx == 0),
            "hit_max": bool(idx == len(values) - 1),
        }
    return {
        "best_params": best,
        "best_indices": best_indices,
        "grid_sizes": grid_sizes,
        "edge_hits": edges,
    }


def summarize_search_infos(fold_infos, final_model):
    counts = {}
    edge_hits = {}
    for info in fold_infos:
        for key, val in info.get("best_params", {}).items():
            counts.setdefault(key, {})
            if key.endswith("__gamma"):
                idx = info["best_indices"][key] + 1
                size = info["grid_sizes"][key]
                label = f"grid {idx}/{size}"
            else:
                label = format_param_value(val)
            counts[key][label] = counts[key].get(label, 0) + 1
        for key, edges in info.get("edge_hits", {}).items():
            edge_hits.setdefault(key, {"hit_min": False, "hit_max": False})
            edge_hits[key]["hit_min"] |= edges["hit_min"]
            edge_hits[key]["hit_max"] |= edges["hit_max"]
    return {
        "outer_fold_best_params": [info.get("best_params", {}) for info in fold_infos],
        "selected_param_counts": counts,
        "edge_hits": edge_hits,
        "final_refit": search_info(final_model).get("best_params", {}),
    }


def rate_cv_readout(decoder_factory, X, rate, cv_reg, return_search=False):
    pred = np.full(len(rate), np.nan)
    scores = []
    infos = []
    for train_idx, test_idx in cv_reg.split(X, rate):
        model = decoder_factory(X[train_idx]).fit(X[train_idx], rate[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
        scores.append(model.score(X[test_idx], rate[test_idx]))
        infos.append(search_info(model))
    if return_search:
        return pred, np.asarray(scores), infos
    return pred, np.asarray(scores)


def analyze_time_aware_cv(plt, X, y_cat, y_choice, rate, args) -> dict:
    """Compare shuffled trial folds with contiguous held-out trial blocks."""
    print("\n[control] Random vs contiguous-block cross-validation ...")
    random_binary = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    blocked = KFold(n_splits=CV_FOLDS, shuffle=False)

    def binary_scores(y, splitter):
        scores = []
        folds = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ).fit(X[train_idx], y[train_idx])
            scores.append(
                balanced_accuracy_score(y[test_idx], clf.predict(X[test_idx]))
            )
            folds.append(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_index_min": int(test_idx.min()),
                    "test_index_max": int(test_idx.max()),
                    "test_class_counts": {
                        str(int(label)): int(count)
                        for label, count in zip(
                            *np.unique(y[test_idx], return_counts=True)
                        )
                    },
                }
            )
        return np.asarray(scores), folds

    cat_random, cat_random_folds = binary_scores(y_cat, random_binary)
    cat_blocked, cat_folds = binary_scores(y_cat, blocked)
    choice_random, choice_random_folds = binary_scores(y_choice, random_binary)
    choice_blocked, choice_folds = binary_scores(y_choice, blocked)

    def ridge_factory(X_train=None):
        return make_rate_regressor(X_train, args)

    random_rate_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    _, rate_random, rate_random_search = rate_cv_readout(
        ridge_factory, X, rate, random_rate_cv, return_search=True
    )
    _, rate_blocked, rate_blocked_search = rate_cv_readout(
        ridge_factory, X, rate, blocked, return_search=True
    )
    rate_blocked_folds = []
    for fold, (train_idx, test_idx) in enumerate(blocked.split(X, rate)):
        labels, counts = np.unique(rate[test_idx], return_counts=True)
        rate_blocked_folds.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_index_min": int(test_idx.min()),
                "test_index_max": int(test_idx.max()),
                "test_rate_counts": {
                    f"{float(label):g}": int(count)
                    for label, count in zip(labels, counts)
                },
            }
        )

    comparisons = [
        ("category balanced accuracy", cat_random, cat_blocked, 0.5),
        ("choice balanced accuracy", choice_random, choice_blocked, 0.5),
        ("rate CV R2", rate_random, rate_blocked, 0.0),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    for ax, (title, random_scores, blocked_scores, chance) in zip(axs, comparisons):
        _box(
            ax,
            [random_scores, blocked_scores],
            ["random\nfolds", "contiguous\nblocks"],
            ["0.65", "tab:orange"],
        )
        ax.axhline(chance, color="k", ls="--", lw=1)
        ax.set_ylabel(title)
        ax.set_title("train: remaining trials\ntest: held-out fold")
    save_fig(fig, "fig9_time_aware_cv", args.no_save)

    def summary(random_scores, blocked_scores):
        return {
            "random_mean": float(np.nanmean(random_scores)),
            "random_folds": [round(float(v), 4) for v in random_scores],
            "blocked_mean": float(np.nanmean(blocked_scores)),
            "blocked_folds": [round(float(v), 4) for v in blocked_scores],
            "blocked_minus_random": float(
                np.nanmean(blocked_scores) - np.nanmean(random_scores)
            ),
        }

    return {
        "design": (
            "Ten contiguous trial blocks preserve acquisition order; each block is "
            "held out while the model trains on all remaining trials. Binary models "
            "use fixed-C class-balanced logistic regression; the rate model retains "
            "fold-internal ridge alpha selection."
        ),
        "category": summary(cat_random, cat_blocked),
        "choice": summary(choice_random, choice_blocked),
        "rate": summary(rate_random, rate_blocked),
        "category_random_folds": cat_random_folds,
        "blocked_folds": cat_folds,
        "choice_random_folds": choice_random_folds,
        "choice_blocked_folds": choice_folds,
        "rate_blocked_folds": rate_blocked_folds,
        "rate_random_search": rate_random_search,
        "rate_blocked_search": rate_blocked_search,
    }


def residualized_rate_readout(
    decoder_factory, X, rate, nuisance, cv_reg, return_search=False
):
    """Fold-safe rate readout after linear removal of a nuisance variable."""
    nuisance = np.asarray(nuisance, dtype=float).reshape(len(X), -1)
    pred = np.full(len(rate), np.nan)
    scores = []
    infos = []
    for train_idx, test_idx in cv_reg.split(X, rate):
        nuisance_model = LinearRegression().fit(nuisance[train_idx], X[train_idx])
        x_train = X[train_idx] - nuisance_model.predict(nuisance[train_idx])
        x_test = X[test_idx] - nuisance_model.predict(nuisance[test_idx])
        model = decoder_factory(x_train).fit(x_train, rate[train_idx])
        pred[test_idx] = model.predict(x_test)
        scores.append(model.score(x_test, rate[test_idx]))
        infos.append(search_info(model))
    if return_search:
        return pred, np.asarray(scores), infos
    return pred, np.asarray(scores)


def choice_balanced_ridge_rate_control(
    X, rate, y_choice, boundary_rate, n_resamples, decoder_factory
):
    rate_labels = np.unique(rate)
    rng = np.random.default_rng(RANDOM_STATE + 17)
    high = np.full((n_resamples, len(rate_labels)), np.nan)
    error = np.full_like(high, np.nan)
    r2 = []
    n_trials = []
    n_per_rate = {}

    for j, r in enumerate(rate_labels):
        n_left = int(np.sum((rate == r) & (y_choice == 0)))
        n_right = int(np.sum((rate == r) & (y_choice == 1)))
        n_per_rate[float(r)] = 2 * min(n_left, n_right)

    for i in range(n_resamples):
        picks = []
        for r in rate_labels:
            left = np.flatnonzero((rate == r) & (y_choice == 0))
            right = np.flatnonzero((rate == r) & (y_choice == 1))
            n = min(len(left), len(right))
            if n == 0:
                continue
            picks.extend(rng.choice(left, size=n, replace=False))
            picks.extend(rng.choice(right, size=n, replace=False))
        idx = np.sort(np.asarray(picks, dtype=int))
        rb = rate[idx]
        cv = KFold(
            n_splits=min(CV_FOLDS, len(idx)),
            shuffle=True,
            random_state=RANDOM_STATE + i,
        )
        pred, fold_r2 = rate_cv_readout(decoder_factory, X[idx], rb, cv)
        r2.extend(fold_r2)
        n_trials.append(len(idx))
        for j, r in enumerate(rate_labels):
            mask = rb == r
            high[i, j] = np.mean(pred[mask] > boundary_rate)
            error[i, j] = high[i, j] if r < boundary_rate else 1 - high[i, j]

    def sem(vals):
        valid = np.isfinite(vals)
        if valid.sum() < 2:
            return 0.0
        return float(np.nanstd(vals, ddof=1) / np.sqrt(valid.sum()))

    rows = []
    for j, r in enumerate(rate_labels):
        rows.append(
            {
                "rate": float(r),
                "choice_balanced_ridge_high": float(np.nanmean(high[:, j])),
                "choice_balanced_ridge_high_sem": sem(high[:, j]),
                "choice_balanced_ridge_error": float(np.nanmean(error[:, j])),
                "choice_balanced_ridge_error_sem": sem(error[:, j]),
                "n_trials_per_resample": int(n_per_rate[float(r)]),
            }
        )
    return {
        "by_rate": rows,
        "cv_r2_mean": float(np.mean(r2)),
        "category_collapsed_acc": float(1 - np.nanmean(error)),
        "n_trials_per_resample": int(np.mean(n_trials)),
    }


def continuous_rate_metrics(name, rate, pred, r2_folds, boundary_rate):
    residual = pred - rate
    calibration_slope, calibration_intercept = np.polyfit(rate, pred, 1)
    residual_slope, residual_intercept = np.polyfit(rate, residual, 1)
    rate_labels = np.unique(rate)
    trial_low_side = rate < boundary_rate
    trial_high_side = rate > boundary_rate
    summary = (
        pd.DataFrame({"rate": rate, "prediction": pred})
        .groupby("rate", sort=True)["prediction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"])
    summary["residual"] = summary["mean"] - summary["rate"]
    high_by_rate = (
        pd.DataFrame({"rate": rate, "high": (pred > boundary_rate).astype(float)})
        .groupby("rate", sort=True)["high"]
        .agg(["mean", "count"])
        .reset_index()
    )
    low_side = high_by_rate["rate"] < boundary_rate
    high_by_rate["sem"] = np.sqrt(
        high_by_rate["mean"] * (1 - high_by_rate["mean"]) / high_by_rate["count"]
    )
    high_by_rate["error"] = np.where(
        low_side, high_by_rate["mean"], 1 - high_by_rate["mean"]
    )
    return {
        "name": name,
        "pred": pred,
        "r2_folds": r2_folds,
        "summary": summary,
        "high_by_rate": high_by_rate,
        "metrics": {
            "cv_r2_mean": float(np.mean(r2_folds)),
            "cv_r2_folds": [round(float(v), 4) for v in r2_folds],
            "calibration_slope": float(calibration_slope),
            "calibration_intercept": float(calibration_intercept),
            "residual_slope": float(residual_slope),
            "residual_intercept": float(residual_intercept),
            "spearman_rho": _spearman(rate, pred),
            "mae_hz": float(np.mean(np.abs(residual))),
            "rmse_hz": float(np.sqrt(np.mean(residual**2))),
            "low_side_mean_residual": float(np.mean(residual[trial_low_side])),
            "high_side_mean_residual": float(np.mean(residual[trial_high_side])),
            "lowest_rate_residual": float(
                summary.loc[summary["rate"] == rate_labels[0], "residual"].iloc[0]
            ),
            "highest_rate_residual": float(
                summary.loc[summary["rate"] == rate_labels[-1], "residual"].iloc[0]
            ),
            "category_collapsed_acc": float(1 - high_by_rate["error"].mean()),
        },
    }


def evidence_target_summary(rate, target, pred, r2_folds, y_choice):
    residual = pred - target
    summary = (
        pd.DataFrame({"rate": rate, "target": target, "prediction": pred})
        .groupby("rate", sort=True)
        .agg(
            target=("target", "mean"),
            mean=("prediction", "mean"),
            std=("prediction", "std"),
            count=("prediction", "size"),
        )
        .reset_index()
    )
    summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"])
    summary["residual"] = summary["mean"] - summary["target"]
    pred_high = pred > 0
    true_high = target > 0
    return {
        "cv_r2_mean": float(np.mean(r2_folds)),
        "cv_r2_folds": [round(float(v), 4) for v in r2_folds],
        "spearman_rho": _spearman(rate, pred),
        "mae_evidence": float(np.mean(np.abs(residual))),
        "rmse_evidence": float(np.sqrt(np.mean(residual**2))),
        "category_collapsed_acc": float(np.mean(pred_high == true_high)),
        "same_trial_choice_alignment": float(np.mean(pred_high == y_choice)),
        "right_minus_left_d": mean_readout_choice_effect(pred, y_choice, rate),
        "summary": summary,
    }


def analyze_rate_target_family(
    plt, X, rate, y_choice, cv_reg, args, boundary_rate, endpoint_distance
):
    specs = [
        ("exact Hz", "exact_hz", None, "k", "o", "-"),
        ("linear", "linear", np.inf, "tab:blue", "s", "-"),
        ("tanh scale 8", "tanh_scale_8", 8.0, "tab:green", "^", "-"),
        ("tanh scale 4", "tanh_scale_4", 4.0, "tab:orange", "D", "-"),
        ("tanh scale 2", "tanh_scale_2", 2.0, "tab:red", "P", "-"),
    ]
    signed = rate - boundary_rate
    out = []
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for label, key, scale, color, marker, ls in specs:
        if scale is None:
            target = rate
            plot_target = signed / endpoint_distance
        elif np.isinf(scale):
            target = signed / endpoint_distance
            plot_target = target
        else:
            target = np.tanh(signed / scale)
            plot_target = target

        pred, r2, infos = rate_cv_readout(
            lambda X_train: make_rate_regressor(X_train, args),
            X,
            target,
            cv_reg,
            return_search=True,
        )
        plot_pred = (
            (pred - boundary_rate) / endpoint_distance if scale is None else pred
        )
        readout = evidence_target_summary(rate, plot_target, plot_pred, r2, y_choice)
        summary = readout.pop("summary")
        refit = make_rate_regressor(X, args).fit(X, target)
        out.append(
            {
                "label": label,
                "key": key,
                "scale_hz": None if scale is None or np.isinf(scale) else float(scale),
                "target": "exact rate" if scale is None else "boundary evidence",
                "metrics": readout,
                "hyperparameter_tuning": summarize_search_infos(infos, refit),
                "by_rate": summary.round(4).to_dict(orient="records"),
            }
        )

        axs[0].plot(
            summary["rate"],
            summary["target"],
            color=color,
            marker=marker,
            ls=ls,
            label=label,
        )
        axs[1].errorbar(
            summary["rate"],
            summary["mean"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            ls=ls,
            lw=2,
            capsize=3,
            label=label,
        )
        axs[2].errorbar(
            summary["rate"],
            summary["residual"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            ls=ls,
            lw=2,
            capsize=3,
            label=label,
        )

    for ax in axs:
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.axvline(boundary_rate, color="0.35", ls="--", lw=1)
        ax.set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("target evidence")
    axs[0].set_title("train target functions\ntest: boundary-centered evidence")
    axs[1].set_ylabel("predicted evidence")
    axs[1].set_title("train: target family\ntest: held-out trials")
    axs[2].set_ylabel("evidence residual")
    axs[2].set_title("train: target family\ntest: predicted - target")
    axs[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "fig5_rate_target_family", args.no_save)
    return out


def choice_residualized_rate_diagnostic(
    plt, X, rate, y_choice, cv_reg, args, boundary_rate, ridge_factory, raw_readout
):
    choice_pred, choice_r2 = residualized_rate_readout(
        ridge_factory, X, rate, y_choice, cv_reg
    )
    self_pred, self_r2 = residualized_rate_readout(ridge_factory, X, rate, rate, cv_reg)
    readouts = {
        "raw_ridge": raw_readout,
        "choice_residualized": continuous_rate_metrics(
            "choice residualized", rate, choice_pred, choice_r2, boundary_rate
        ),
        "rate_self_residualized": continuous_rate_metrics(
            "rate self residualized", rate, self_pred, self_r2, boundary_rate
        ),
    }
    styles = {
        "raw_ridge": ("k", "o", "-", "raw ridge"),
        "choice_residualized": ("tab:purple", "s", "-", "choice-resid ridge"),
        "rate_self_residualized": ("0.55", "D", "--", "rate-self resid"),
    }

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    rate_min, rate_max = float(rate.min()), float(rate.max())
    for key, readout in readouts.items():
        color, marker, ls, label = styles[key]
        summary = readout["summary"]
        axs[0].errorbar(
            summary["rate"],
            summary["mean"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            ls=ls,
            lw=2,
            capsize=3,
            label=label,
        )
        axs[1].errorbar(
            summary["rate"],
            summary["residual"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            ls=ls,
            lw=2,
            capsize=3,
            label=label,
        )
    axs[0].plot([rate_min, rate_max], [rate_min, rate_max], color="k", ls="--", lw=1)
    axs[0].set_xlim(rate_min, rate_max)
    axs[0].set_ylim(rate_min, rate_max)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("predicted rate (Hz)")
    axs[0].set_title("train: stimulus rate\ntest: held-out trials")
    axs[0].legend(frameon=False, fontsize=8)

    axs[1].axhline(0, color="k", ls="--", lw=1)
    axs[1].set_xlabel("true stimulus rate (Hz)")
    axs[1].set_ylabel("prediction residual (Hz)")
    axs[1].set_title("train: stimulus rate\ntest: predicted - true")

    metric_keys = [
        ("cv_r2_mean", "CV R2"),
        ("calibration_slope", "calib slope"),
        ("residual_slope", "resid slope"),
        ("category_collapsed_acc", "cat acc"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.25
    for i, key in enumerate(readouts):
        color, _, _, label = styles[key]
        vals = [readouts[key]["metrics"][metric] for metric, _ in metric_keys]
        axs[2].bar(x + (i - 1) * width, vals, width=width, color=color, label=label)
    axs[2].axhline(0, color="0.5", ls="--", lw=1)
    axs[2].axhline(1, color="0.8", ls=":", lw=1)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([label for _, label in metric_keys], rotation=35, ha="right")
    axs[2].set_title("train: stimulus rate\ntest: metric summary")
    axs[2].legend(frameon=False, fontsize=8)
    save_fig(fig, "fig5_choice_residualized_rate", args.no_save)

    out = {}
    for key in ["choice_residualized", "rate_self_residualized"]:
        out[key] = {
            "metrics": readouts[key]["metrics"],
            "by_rate": readouts[key]["summary"].round(4).to_dict(orient="records"),
        }
    return out


def mlp_rate_diagnostic(plt, rate, readouts, args, boundary_rate):
    styles = {
        "ridge": ("k", "o", "-", "ridge"),
        "mlp": ("tab:orange", "s", "-", "MLP"),
    }
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    rate_min, rate_max = float(rate.min()), float(rate.max())
    for key in ["ridge", "mlp"]:
        color, marker, ls, label = styles[key]
        summary = readouts[key]["summary"]
        axs[0].errorbar(
            summary["rate"],
            summary["mean"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            ls=ls,
            lw=2,
            capsize=3,
            label=label,
        )
        axs[1].errorbar(
            summary["rate"],
            summary["residual"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            ls=ls,
            lw=2,
            capsize=3,
            label=label,
        )

    axs[0].plot([rate_min, rate_max], [rate_min, rate_max], color="k", ls="--", lw=1)
    axs[0].set_xlim(rate_min, rate_max)
    axs[0].set_ylim(rate_min, rate_max)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("predicted rate (Hz)")
    axs[0].set_title("train: stimulus rate\ntest: held-out trials")
    axs[0].legend(frameon=False, fontsize=8)

    axs[1].axhline(0, color="k", ls="--", lw=1)
    axs[1].axvline(boundary_rate, color="0.35", ls="--", lw=1)
    axs[1].set_xlabel("true stimulus rate (Hz)")
    axs[1].set_ylabel("prediction residual (Hz)")
    axs[1].set_title("train: stimulus rate\ntest: predicted - true")

    metric_keys = [
        ("cv_r2_mean", "CV R2"),
        ("calibration_slope", "calib slope"),
        ("residual_slope", "resid slope"),
        ("category_collapsed_acc", "cat acc"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.35
    for i, key in enumerate(["ridge", "mlp"]):
        color, _, _, label = styles[key]
        vals = [readouts[key]["metrics"][metric] for metric, _ in metric_keys]
        axs[2].bar(x + (i - 0.5) * width, vals, width=width, color=color, label=label)
    axs[2].axhline(0, color="0.5", ls="--", lw=1)
    axs[2].axhline(1, color="0.8", ls=":", lw=1)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([label for _, label in metric_keys], rotation=35, ha="right")
    axs[2].set_title("train: stimulus rate\ntest: metric summary")
    axs[2].legend(frameon=False, fontsize=8)
    save_fig(fig, "fig5_mlp_rate_diagnostic", args.no_save)


def ridge_alpha_path_diagnostic(
    plt, X, rate, cv_reg, args, boundary_rate, selected_alpha
):
    rows = []
    alphas = np.asarray(args.ridge_alphas, dtype=float)
    ols_pred, ols_r2 = rate_cv_readout(
        lambda _X_train: make_pipeline(StandardScaler(), LinearRegression()),
        X,
        rate,
        cv_reg,
    )
    rows.append(
        {
            "model": "OLS",
            "alpha": None,
            "is_selected_main_alpha": False,
            **continuous_rate_metrics("OLS", rate, ols_pred, ols_r2, boundary_rate)[
                "metrics"
            ],
        }
    )
    for alpha in alphas:
        pred, r2 = rate_cv_readout(
            lambda _X_train, alpha=alpha: make_fixed_rate_regressor(alpha),
            X,
            rate,
            cv_reg,
        )
        metrics = continuous_rate_metrics(
            f"alpha {alpha:.4g}", rate, pred, r2, boundary_rate
        )["metrics"]
        rows.append(
            {
                "model": "ridge",
                "alpha": float(alpha),
                "is_selected_main_alpha": bool(
                    selected_alpha is not None and np.isclose(alpha, selected_alpha)
                ),
                **metrics,
            }
        )

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ols_x = alphas.min() / 10
    panels = [
        (axs[0, 0], "cv_r2_mean", "CV $R^2$", 0),
        (axs[0, 1], "calibration_slope", "calibration slope", 1),
        (axs[1, 0], "residual_slope", "residual slope (Hz/Hz)", 0),
        (axs[1, 1], "category_collapsed_acc", "category accuracy", 0.5),
    ]
    for ax, key, ylabel, ref in panels:
        ax.plot(alphas, [row[key] for row in rows[1:]], color="k", marker="o", lw=2)
        ax.scatter(
            [ols_x],
            [rows[0][key]],
            color="tab:blue",
            marker="D",
            s=45,
            label="OLS",
            zorder=4,
        )
        ax.axhline(ref, color="0.5", ls="--", lw=1)
        if selected_alpha is not None:
            ax.axvline(selected_alpha, color="tab:red", ls=":", lw=1.5)
        ax.set_xscale("log")
        ax.text(ols_x, rows[0][key], " OLS", color="tab:blue", fontsize=8, va="bottom")
        ax.set_xlabel("ridge alpha (OLS at left)")
        ax.set_ylabel(ylabel)
    axs[0, 0].set_title("train: fixed-alpha ridge\ntest: held-out rate")
    axs[0, 1].set_title("train: fixed-alpha ridge\ntest: prediction slope")
    axs[1, 0].set_title("train: fixed-alpha ridge\ntest: residual compression")
    axs[1, 1].set_title("train: fixed-alpha ridge\ntest: category side")
    if selected_alpha is not None:
        axs[0, 0].text(
            0.04,
            0.08,
            f"red line = nested-CV alpha {selected_alpha:.3g}",
            transform=axs[0, 0].transAxes,
            color="tab:red",
            fontsize=9,
        )
    save_fig(fig, "fig5_ridge_alpha_path", args.no_save)
    return rows


def unit_rate_slopes(X, rate):
    return np.array([np.polyfit(rate, X[:, u], 1)[0] for u in range(X.shape[1])])


def rate_slope_bins(slopes, n_bins=5):
    return np.array_split(np.argsort(np.abs(slopes)), n_bins)


def slope_subset_rate_readout(X, rate, cv_reg, args, bin_idx):
    pred = np.full(len(rate), np.nan)
    scores = []
    n_units_used = []
    for train_idx, test_idx in cv_reg.split(X, rate):
        if bin_idx is None:
            unit_idx = np.arange(X.shape[1])
        else:
            slopes = unit_rate_slopes(X[train_idx], rate[train_idx])
            unit_idx = rate_slope_bins(slopes, n_bins=5)[bin_idx]
        model = make_rate_regressor(X[train_idx][:, unit_idx], args).fit(
            X[train_idx][:, unit_idx], rate[train_idx]
        )
        pred[test_idx] = model.predict(X[test_idx][:, unit_idx])
        scores.append(model.score(X[test_idx][:, unit_idx], rate[test_idx]))
        n_units_used.append(len(unit_idx))
    return pred, np.asarray(scores), int(np.mean(n_units_used))


def slope_subset_rate_diagnostic(plt, X, rate, cv_reg, args, boundary_rate):
    specs = [
        ("all units", None, "k", "o", "--", 3.0),
        ("bin 1/5\nflattest", 0, "0.55", "o", "-", 2.0),
        ("bin 2/5", 1, "tab:purple", "s", "-", 2.0),
        ("bin 3/5", 2, "tab:blue", "D", "-", 2.0),
        ("bin 4/5", 3, "tab:green", "^", "-", 2.0),
        ("bin 5/5\nsteepest", 4, "tab:red", "P", "-", 2.0),
    ]
    rows = []
    readouts = []
    for label, bin_idx, color, marker, ls, lw in specs:
        pred, r2, n_units_used = slope_subset_rate_readout(
            X, rate, cv_reg, args, bin_idx
        )
        readout = continuous_rate_metrics(label, rate, pred, r2, boundary_rate)
        rows.append(
            {
                "label": label,
                "bin": None if bin_idx is None else int(bin_idx + 1),
                "n_units": n_units_used,
                **readout["metrics"],
            }
        )
        readouts.append((readout, color, marker, ls, lw))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for readout, color, marker, ls, lw in [*readouts[1:], readouts[0]]:
        summary = readout["summary"]
        label = readout["name"]
        axs[0].errorbar(
            summary["rate"],
            summary["residual"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            lw=lw,
            ls=ls,
            capsize=3,
            label=label,
            zorder=3 if label == "all units" else 2,
        )
    axs[0].axhline(0, color="k", ls="--", lw=1)
    axs[0].set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("prediction residual (Hz)")
    axs[0].set_title("train: slope-selected units\ntest: residual compression")
    axs[0].legend(frameon=False, fontsize=8, ncol=2)

    x = np.arange(len(rows))
    colors = [spec[2] for spec in specs]
    labels = [row["label"] for row in rows]
    axs[1].bar(x, [row["calibration_slope"] for row in rows], color=colors)
    axs[1].axhline(1, color="k", ls="--", lw=1)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axs[1].set_ylabel("calibration slope")
    axs[1].set_title("train: stimulus rate\ntest: slope of prediction")

    axs[2].bar(x, [row["cv_r2_mean"] for row in rows], color=colors)
    axs[2].axhline(0, color="k", ls="--", lw=1)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axs[2].set_ylabel("CV $R^2$")
    axs[2].set_title("train: stimulus rate\ntest: held-out trials")
    for ax in axs[1:]:
        for i, row in enumerate(rows):
            ax.text(
                i, 0.02, f"n={row['n_units']}", ha="center", va="bottom", fontsize=8
            )
    save_fig(fig, "fig5_rate_slope_subset_diagnostic", args.no_save)
    return rows


def single_unit_slope_example_diagnostic(plt, X, rate, cv_reg, args, boundary_rate):
    slopes = unit_rate_slopes(X, rate)
    bins = rate_slope_bins(slopes, n_bins=5)
    rng = np.random.default_rng(RANDOM_STATE + 97)
    specs = [
        ("bin 1/5\nflattest", 0, "0.55", "o"),
        ("bin 2/5", 1, "tab:purple", "s"),
        ("bin 3/5", 2, "tab:blue", "D"),
        ("bin 4/5", 3, "tab:green", "^"),
        ("bin 5/5\nsteepest", 4, "tab:red", "P"),
    ]
    rows = []
    readouts = []
    for label, bin_idx, color, marker in specs:
        unit_idx = int(rng.choice(bins[bin_idx]))
        pred, r2 = rate_cv_readout(
            lambda X_train: make_rate_regressor(X_train, args),
            X[:, [unit_idx]],
            rate,
            cv_reg,
        )
        readout = continuous_rate_metrics(label, rate, pred, r2, boundary_rate)
        rows.append(
            {
                "label": label,
                "bin": int(bin_idx + 1),
                "unit_index": unit_idx,
                "rate_slope_sp_s_hz": float(slopes[unit_idx]),
                "abs_rate_slope_sp_s_hz": float(abs(slopes[unit_idx])),
                **readout["metrics"],
            }
        )
        readouts.append((readout, color, marker))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for readout, color, marker in readouts:
        summary = readout["summary"]
        axs[0].errorbar(
            summary["rate"],
            summary["residual"],
            yerr=summary["sem"],
            color=color,
            marker=marker,
            lw=2,
            capsize=3,
            label=readout["name"],
        )
    axs[0].axhline(0, color="k", ls="--", lw=1)
    axs[0].set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("prediction residual (Hz)")
    axs[0].set_title("train: one random unit per bin\ntest: residual compression")
    axs[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(rows))
    axs[1].bar(x, [row["residual_slope"] for row in rows], color=[s[2] for s in specs])
    axs[1].axhline(0, color="k", ls="--", lw=1)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(
        [row["label"] for row in rows], rotation=35, ha="right", fontsize=8
    )
    axs[1].set_ylabel("residual slope (Hz/Hz)")
    axs[1].set_title("train: one random unit per bin\ntest: residual slope")
    save_fig(fig, "fig5_single_unit_slope_examples", args.no_save)
    return rows


def steepest_unit_ols_diagnostic(plt, X, rate, cv_reg, args, boundary_rate):
    slopes = unit_rate_slopes(X, rate)
    unit_idx = int(np.argmax(np.abs(slopes)))
    firing = X[:, unit_idx]
    encoding = LinearRegression().fit(rate[:, None], firing)
    pred, r2 = rate_cv_readout(
        lambda _X_train: make_pipeline(StandardScaler(), LinearRegression()),
        X[:, [unit_idx]],
        rate,
        cv_reg,
    )
    readout = continuous_rate_metrics(
        "steepest unit OLS", rate, pred, r2, boundary_rate
    )
    tuning = (
        pd.DataFrame({"rate": rate, "firing": firing})
        .groupby("rate", sort=True)["firing"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    tuning["sem"] = tuning["std"].fillna(0) / np.sqrt(tuning["count"])
    tuning["linear_fit"] = encoding.predict(tuning["rate"].to_numpy()[:, None])
    tuning["linear_residual"] = tuning["mean"] - tuning["linear_fit"]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axs[0].scatter(rate, firing, color="0.2", alpha=0.25, s=18, label="trials")
    axs[0].errorbar(
        tuning["rate"],
        tuning["mean"],
        yerr=tuning["sem"],
        color="tab:red",
        marker="o",
        lw=2,
        capsize=3,
        label="mean +/- SEM",
    )
    line_x = np.linspace(rate.min(), rate.max(), 100)
    axs[0].plot(
        line_x,
        encoding.predict(line_x[:, None]),
        color="k",
        ls="--",
        lw=1.5,
        label="OLS line",
    )
    axs[0].set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("unit firing rate (sp/s)")
    axs[0].set_title("steepest rate-tuned unit\nraw tuning")
    axs[0].legend(frameon=False, fontsize=8)

    axs[1].errorbar(
        tuning["rate"],
        tuning["linear_residual"],
        yerr=tuning["sem"],
        color="tab:red",
        marker="o",
        lw=2,
        capsize=3,
    )
    axs[1].axhline(0, color="k", ls="--", lw=1)
    axs[1].set_xlabel("true stimulus rate (Hz)")
    axs[1].set_ylabel("firing residual (sp/s)")
    axs[1].set_title("raw tuning residual\nmean - linear fit")

    summary = readout["summary"]
    axs[2].errorbar(
        summary["rate"],
        summary["residual"],
        yerr=summary["sem"],
        color="tab:blue",
        marker="o",
        lw=2,
        capsize=3,
    )
    axs[2].axhline(0, color="k", ls="--", lw=1)
    axs[2].set_xlabel("true stimulus rate (Hz)")
    axs[2].set_ylabel("prediction residual (Hz)")
    axs[2].set_title("train: single-unit OLS\ntest: held-out rate")
    save_fig(fig, "fig5_steepest_unit_ols", args.no_save)

    return {
        "unit_index": unit_idx,
        "rate_slope_sp_s_hz": float(slopes[unit_idx]),
        "abs_rate_slope_sp_s_hz": float(abs(slopes[unit_idx])),
        "encoding_intercept_sp_s": float(encoding.intercept_),
        "encoding_r2": float(encoding.score(rate[:, None], firing)),
        "decoder_metrics": readout["metrics"],
        "tuning_by_rate": tuning.round(4).to_dict(orient="records"),
        "decoder_by_rate": summary.round(4).to_dict(orient="records"),
    }


def _choice_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def nested_rate_choice_cv(X, rate, y_choice, args, state_X=None):
    """Evaluate rate-readout-to-choice models with fully nested stacking.

    Each outer choice fold receives rate-readout predictions from a ridge model
    that has not seen the outer test trials. The choice model is trained on
    cross-fitted ridge predictions within the outer training fold, so neither
    neural-feature fitting nor ridge hyperparameter selection can leak from the
    choice-test fold into the stacked predictor.
    """
    rate = np.asarray(rate, dtype=float)
    y_choice = np.asarray(y_choice, dtype=int)
    rate_labels = np.unique(rate)
    outer_cv = _kfold_for(y_choice)
    model_names = [
        "bias only",
        "stimulus rate",
        "nested ridge readout",
        "rate + nested ridge",
        "rate fixed effects",
        "rate FE + nested ridge",
    ]
    if state_X is not None:
        model_names.extend(["rate FE + state", "rate FE + state + nested ridge"])

    probs = {name: np.full(len(y_choice), np.nan) for name in model_names}
    losses = {name: [] for name in model_names}
    nested_readout = np.full(len(y_choice), np.nan)
    fold_metadata = []
    upstream_search = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(y_choice)), y_choice)
    ):
        inner_oof = np.full(len(train_idx), np.nan)
        inner_splitter = KFold(
            n_splits=min(5, len(train_idx)),
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )
        for inner_train, inner_valid in inner_splitter.split(train_idx):
            fit_idx = train_idx[inner_train]
            valid_idx = train_idx[inner_valid]
            rate_model = make_rate_regressor(X[fit_idx], args).fit(
                X[fit_idx], rate[fit_idx]
            )
            inner_oof[inner_valid] = rate_model.predict(X[valid_idx])
        if np.isnan(inner_oof).any():
            raise RuntimeError("Nested rate readout left training predictions missing")

        outer_rate_model = make_rate_regressor(X[train_idx], args).fit(
            X[train_idx], rate[train_idx]
        )
        test_readout = outer_rate_model.predict(X[test_idx])
        nested_readout[test_idx] = test_readout
        upstream_search.append(search_info(outer_rate_model))

        train_rate_fe = rate_feature_matrix(rate[train_idx], rate_labels)
        test_rate_fe = rate_feature_matrix(rate[test_idx], rate_labels)
        train_predictors = {
            "stimulus rate": rate[train_idx, None],
            "nested ridge readout": inner_oof[:, None],
            "rate + nested ridge": np.column_stack([rate[train_idx], inner_oof]),
            "rate fixed effects": train_rate_fe,
            "rate FE + nested ridge": np.column_stack([train_rate_fe, inner_oof]),
        }
        test_predictors = {
            "stimulus rate": rate[test_idx, None],
            "nested ridge readout": test_readout[:, None],
            "rate + nested ridge": np.column_stack([rate[test_idx], test_readout]),
            "rate fixed effects": test_rate_fe,
            "rate FE + nested ridge": np.column_stack([test_rate_fe, test_readout]),
        }
        if state_X is not None:
            train_predictors["rate FE + state"] = np.column_stack(
                [train_rate_fe, state_X[train_idx]]
            )
            test_predictors["rate FE + state"] = np.column_stack(
                [test_rate_fe, state_X[test_idx]]
            )
            train_predictors["rate FE + state + nested ridge"] = np.column_stack(
                [train_rate_fe, state_X[train_idx], inner_oof]
            )
            test_predictors["rate FE + state + nested ridge"] = np.column_stack(
                [test_rate_fe, state_X[test_idx], test_readout]
            )

        for name in model_names:
            if name == "bias only":
                p_right = np.clip(y_choice[train_idx].mean(), 1e-6, 1 - 1e-6)
                fold_prob = np.full(len(test_idx), p_right)
            else:
                clf = _choice_pipeline().fit(
                    train_predictors[name], y_choice[train_idx]
                )
                fold_prob = clf.predict_proba(test_predictors[name])[:, 1]
            probs[name][test_idx] = fold_prob
            losses[name].append(log_loss(y_choice[test_idx], fold_prob, labels=[0, 1]))
        fold_metadata.append(
            {
                "fold": outer_fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_index_min": int(test_idx.min()),
                "test_index_max": int(test_idx.max()),
            }
        )

    rows = []
    for name in model_names:
        row = {
            "model": name,
            "log_loss": float(log_loss(y_choice, probs[name], labels=[0, 1])),
            "fold_log_loss": [round(float(v), 4) for v in losses[name]],
            "balanced_accuracy": float(
                balanced_accuracy_score(y_choice, probs[name] >= 0.5)
            ),
        }
        rows.append(row)

    by_name = {row["model"]: row for row in rows}
    bias_loss = by_name["bias only"]["log_loss"]
    rate_fe_loss = by_name["rate fixed effects"]["log_loss"]
    for row in rows:
        row["delta_log_loss_vs_bias"] = float(bias_loss - row["log_loss"])
        row["pseudo_r2_vs_bias"] = float(1 - row["log_loss"] / bias_loss)
        row["delta_log_loss_vs_rate_fe"] = float(rate_fe_loss - row["log_loss"])
        row["pseudo_r2_vs_rate_fe"] = float(1 - row["log_loss"] / rate_fe_loss)

    return {
        "models": rows,
        "fold_losses": {
            name: [float(v) for v in values] for name, values in losses.items()
        },
        "nested_rate_readout": nested_readout,
        "folds": fold_metadata,
        "upstream_rate_search": upstream_search,
    }


def heldout_choice_model_comparison(plt, nested_choice, args):
    names = [
        "bias only",
        "stimulus rate",
        "nested ridge readout",
        "rate + nested ridge",
    ]
    out = [
        next(row for row in nested_choice["models"] if row["model"] == name)
        for name in names
    ]
    fold_losses = [np.asarray(nested_choice["fold_losses"][name]) for name in names]

    labels = [
        "bias\nonly",
        "stimulus\nrate",
        "nested ridge\nreadout",
        "rate + nested\nridge",
    ]
    colors = ["0.7", "k", "tab:blue", "tab:blue"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _box(ax, fold_losses, labels, colors)
    ax.set_ylabel("held-out choice log loss")
    ax.set_title("train: fully nested stacked model\ntest: outer-fold choices")
    ax.text(0.02, 0.95, "lower = better", transform=ax.transAxes, va="top")
    fig.tight_layout()
    save_fig(fig, "fig5_choice_model_comparison", args.no_save)
    return out


def shuffle_within_rate(y, rate, rng):
    shuffled = np.asarray(y).copy()
    for r in np.unique(rate):
        idx = np.flatnonzero(rate == r)
        shuffled[idx] = rng.permutation(shuffled[idx])
    return shuffled


def hard_alignment_by_rate(pred_high, y_choice, rate):
    rows = []
    for r in np.unique(rate):
        mask = rate == r
        mouse_high = float(np.mean(y_choice[mask]))
        model_high = float(np.mean(pred_high[mask]))
        expected = mouse_high * model_high + (1 - mouse_high) * (1 - model_high)
        observed = float(np.mean(pred_high[mask] == y_choice[mask]))
        rows.append(
            {
                "rate": float(r),
                "alignment": observed,
                "expected_alignment": expected,
                "excess_alignment": observed - expected,
                "mouse_high": mouse_high,
                "model_high": model_high,
                "n_trials": int(mask.sum()),
            }
        )
    return rows


def mean_hard_alignment(pred_high, y_choice, rate):
    return float(
        np.mean(
            [
                row["alignment"]
                for row in hard_alignment_by_rate(pred_high, y_choice, rate)
            ]
        )
    )


def mean_hard_excess(pred_high, y_choice, rate):
    return float(
        np.mean(
            [
                row["excess_alignment"]
                for row in hard_alignment_by_rate(pred_high, y_choice, rate)
            ]
        )
    )


def readout_choice_effect_by_rate(readout, y_choice, rate):
    rows = []
    for r in np.unique(rate):
        mask = rate == r
        right = readout[mask & (y_choice == 1)]
        left = readout[mask & (y_choice == 0)]
        rows.append(
            {
                "rate": float(r),
                "right_minus_left_d": _cohens_d(right, left),
                "n_left": int(len(left)),
                "n_right": int(len(right)),
            }
        )
    return rows


def mean_readout_choice_effect(readout, y_choice, rate):
    vals = [
        row["right_minus_left_d"]
        for row in readout_choice_effect_by_rate(readout, y_choice, rate)
    ]
    return float(np.nanmean(vals))


def rate_feature_matrix(rate, rate_labels=None):
    if rate_labels is None:
        rate_labels = np.unique(rate)
    rate_labels = np.asarray(rate_labels)
    return (rate[:, None] == rate_labels[None, :]).astype(float)


def rate_fixed_choice_tests(
    plt, rate, y_choice, nested_readout, nested_choice, boundary_rate, rng, args
):
    readouts = {
        "nested ridge": nested_readout,
    }
    hard_summary = {}
    hard_rows = []
    continuous_summary = {}
    continuous_rows = []

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axs = axs.ravel()
    for label, readout in readouts.items():
        pred_high = readout > boundary_rate
        rows = hard_alignment_by_rate(pred_high, y_choice, rate)
        hard_rows.extend({"model": label, **row} for row in rows)
        observed = mean_hard_alignment(pred_high, y_choice, rate)
        excess = mean_hard_excess(pred_high, y_choice, rate)
        null = np.array(
            [
                mean_hard_alignment(
                    pred_high, shuffle_within_rate(y_choice, rate, rng), rate
                )
                for _ in range(args.n_shuffles)
            ]
        )
        hard_summary[label] = {
            "rate_balanced_alignment": observed,
            "mean_expected_alignment": float(
                np.mean([row["expected_alignment"] for row in rows])
            ),
            "mean_excess_alignment": excess,
            "shuffle_mean": float(np.mean(null)),
            "shuffle_p": perm_p(observed, null),
        }

        effect_rows = readout_choice_effect_by_rate(readout, y_choice, rate)
        continuous_rows.extend({"model": label, **row} for row in effect_rows)
        effect = mean_readout_choice_effect(readout, y_choice, rate)
        effect_null = np.array(
            [
                mean_readout_choice_effect(
                    readout, shuffle_within_rate(y_choice, rate, rng), rate
                )
                for _ in range(args.n_shuffles)
            ]
        )
        continuous_summary[label] = {
            "mean_right_minus_left_d": effect,
            "shuffle_mean": float(np.nanmean(effect_null)),
            "shuffle_p": perm_p(effect, effect_null),
        }

        color = "tab:blue"
        marker = "s"
        axs[0].plot(
            [row["rate"] for row in rows],
            [row["alignment"] for row in rows],
            color=color,
            marker=marker,
            lw=2,
            label=label,
        )
        axs[1].plot(
            [row["rate"] for row in rows],
            [row["excess_alignment"] for row in rows],
            color=color,
            marker=marker,
            lw=2,
            label=label,
        )
        axs[2].plot(
            [row["rate"] for row in effect_rows],
            [row["right_minus_left_d"] for row in effect_rows],
            color=color,
            marker=marker,
            lw=2,
            label=label,
        )

    fixed_names = ["rate fixed effects", "rate FE + nested ridge"]
    fixed_effect_models = [
        next(row for row in nested_choice["models"] if row["model"] == name)
        for name in fixed_names
    ]
    fixed_effect_losses = [
        np.asarray(nested_choice["fold_losses"][name]) for name in fixed_names
    ]

    axs[0].axhline(
        hard_summary["nested ridge"]["shuffle_mean"],
        color="tab:blue",
        ls=":",
        lw=1,
    )
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("true stimulus rate (Hz)")
    axs[0].set_ylabel("choice alignment")
    axs[0].set_title("train: hard readout threshold\ntest: same-trial choice")
    axs[0].text(
        0.03,
        0.05,
        "dotted = within-rate shuffle",
        transform=axs[0].transAxes,
        va="bottom",
    )
    axs[0].legend(frameon=False)

    axs[1].axhline(0, color="0.5", ls="--", lw=1)
    axs[1].set_xlabel("true stimulus rate (Hz)")
    axs[1].set_ylabel("excess alignment")
    axs[1].set_title(
        "train: hard readout threshold\ntest: observed - rate-bin expected"
    )
    axs[1].legend(frameon=False)

    axs[2].axhline(0, color="0.5", ls="--", lw=1)
    axs[2].set_xlabel("true stimulus rate (Hz)")
    axs[2].set_ylabel("right-minus-left readout (Cohen's d)")
    axs[2].set_title("train: continuous readout\ntest: within-rate choice effect")
    axs[2].legend(frameon=False)

    _box(
        axs[3],
        fixed_effect_losses,
        ["rate\nFE", "rate FE +\nnested ridge"],
        ["0.7", "tab:blue"],
    )
    axs[3].set_ylabel("held-out choice log loss")
    axs[3].set_title(
        "train: fully nested stacked model\ntest: outer-fold choices | rate FE"
    )
    axs[3].text(0.05, 0.95, "lower = better", transform=axs[3].transAxes, va="top")
    save_fig(fig, "fig5_within_rate_choice_tests", args.no_save)

    return {
        "hard_alignment_summary": hard_summary,
        "hard_alignment_by_rate": hard_rows,
        "continuous_readout_summary": continuous_summary,
        "continuous_readout_by_rate": continuous_rows,
        "rate_fixed_choice_models": fixed_effect_models,
    }


def time_resolved_rate_fixed_choice(
    plt, X_time, bin_centers, valid_trials, rate, y_choice, args
) -> dict:
    rows = []
    baseline_loss = None
    for t, bin_center in enumerate(bin_centers):
        nested = nested_rate_choice_cv(X_time[:, :, t], rate, y_choice, args)
        by_name = {row["model"]: row for row in nested["models"]}
        base_row = by_name["rate fixed effects"]
        model_row = by_name["rate FE + nested ridge"]
        baseline_loss = base_row["log_loss"]
        rate_readout = nested["nested_rate_readout"]
        ridge_r2 = float(
            1 - np.sum((rate - rate_readout) ** 2) / np.sum((rate - np.mean(rate)) ** 2)
        )
        rows.append(
            {
                "bin_center_s": float(bin_center),
                "bin_center_ms": float(bin_center * 1000),
                "n_trials": int(len(y_choice)),
                "ridge_rate_cv_r2": ridge_r2,
                "log_loss": model_row["log_loss"],
                "delta_log_loss_vs_rate_fe": model_row["delta_log_loss_vs_rate_fe"],
                "pseudo_r2_vs_rate_fe": model_row["pseudo_r2_vs_rate_fe"],
                "balanced_accuracy": model_row["balanced_accuracy"],
            }
        )

    exit_latency = valid_trials["center_port_exit_ts"].to_numpy(
        dtype=float
    ) - valid_trials["first_stim_ts"].to_numpy(dtype=float)
    finite_exit = np.isfinite(exit_latency)
    exit_latency = exit_latency[finite_exit]
    if exit_latency.size:
        p25_s = float(np.percentile(exit_latency, 25))
        median_s = float(np.median(exit_latency))
        p75_s = float(np.percentile(exit_latency, 75))
        exit_summary = {
            "n_trials": int(exit_latency.size),
            "p25_s": p25_s,
            "median_s": median_s,
            "p75_s": p75_s,
        }
    else:
        p25_s = None
        median_s = None
        p75_s = None
        exit_summary = {"n_trials": 0, "p25_s": None, "median_s": None, "p75_s": None}

    pre_exit_rows = []
    bin_width = float(np.median(np.diff(bin_centers))) if len(bin_centers) > 1 else 0.1
    bin_ends = np.asarray(bin_centers) + bin_width / 2
    all_exit_latency = valid_trials["center_port_exit_ts"].to_numpy(
        dtype=float
    ) - valid_trials["first_stim_ts"].to_numpy(dtype=float)
    min_pre_exit_trials = 50  # ponytail: coarse stability floor for this diagnostic.
    for t, (bin_center, bin_end) in enumerate(zip(bin_centers, bin_ends)):
        mask = np.isfinite(all_exit_latency) & (all_exit_latency >= bin_end)
        y_sub = y_choice[mask]
        rate_sub = rate[mask]
        row = {
            "bin_center_s": float(bin_center),
            "bin_center_ms": float(bin_center * 1000),
            "bin_end_ms": float(bin_end * 1000),
            "n_trials": int(mask.sum()),
            "ridge_rate_cv_r2": None,
            "log_loss": None,
            "delta_log_loss_vs_rate_fe": None,
            "pseudo_r2_vs_rate_fe": None,
            "balanced_accuracy": None,
        }
        _, choice_counts = np.unique(y_sub, return_counts=True)
        enough_choice = len(choice_counts) == 2 and int(choice_counts.min()) >= 2
        if (
            mask.sum() >= min_pre_exit_trials
            and enough_choice
            and len(np.unique(rate_sub)) >= 2
        ):
            nested = nested_rate_choice_cv(X_time[mask, :, t], rate_sub, y_sub, args)
            by_name = {item["model"]: item for item in nested["models"]}
            model_row = by_name["rate FE + nested ridge"]
            readout = nested["nested_rate_readout"]
            ridge_r2 = float(
                1
                - np.sum((rate_sub - readout) ** 2)
                / np.sum((rate_sub - np.mean(rate_sub)) ** 2)
            )
            row.update(
                {
                    "ridge_rate_cv_r2": ridge_r2,
                    "log_loss": model_row["log_loss"],
                    "delta_log_loss_vs_rate_fe": model_row["delta_log_loss_vs_rate_fe"],
                    "pseudo_r2_vs_rate_fe": model_row["pseudo_r2_vs_rate_fe"],
                    "balanced_accuracy": model_row["balanced_accuracy"],
                }
            )
        pre_exit_rows.append(row)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    x = [row["bin_center_ms"] for row in rows]
    pre_y = [
        np.nan
        if row["delta_log_loss_vs_rate_fe"] is None
        else row["delta_log_loss_vs_rate_fe"]
        for row in pre_exit_rows
    ]
    if median_s is not None and p25_s is not None and p75_s is not None:
        p25 = p25_s * 1000
        p75 = p75_s * 1000
        median = median_s * 1000
        for ax in axs:
            ax.axvspan(p25, p75, color="0.9", zorder=0)
            ax.axvline(median, color="0.4", ls=":", lw=1.5)
    axs[0].plot(
        x,
        [row["delta_log_loss_vs_rate_fe"] for row in rows],
        color="tab:blue",
        marker="o",
        lw=2,
        label="all trials",
    )
    axs[0].plot(
        x,
        pre_y,
        color="tab:green",
        marker="s",
        lw=2,
        ls="--",
        label="pre-exit bins",
    )
    axs[0].axhline(0, color="0.5", ls="--", lw=1)
    axs[0].set_xlabel("time after first stimulus onset (ms)")
    axs[0].set_ylabel("choice log-loss gain vs rate FE")
    axs[0].set_title("train: fully nested rate-fixed model\ntest: outer-fold choices")
    axs[0].legend(frameon=False)

    axs[1].plot(
        x,
        [row["ridge_rate_cv_r2"] for row in rows],
        color="k",
        marker="o",
        lw=2,
        label="all trials",
    )
    axs[1].axhline(0, color="0.5", ls="--", lw=1)
    axs[1].set_xlabel("time after first stimulus onset (ms)")
    axs[1].set_ylabel("ridge rate CV $R^2$")
    axs[1].set_title("train: stimulus rate\ntest: held-out same-bin trials")
    axs[1].legend(frameon=False)
    save_fig(fig, "fig5_time_resolved_choice_readout", args.no_save)

    if baseline_loss is None:
        raise ValueError("time-resolved decoder requires at least one time bin")

    return {
        "baseline_rate_fe_log_loss": float(baseline_loss),
        "center_exit_latency": exit_summary,
        "min_pre_exit_trials": int(min_pre_exit_trials),
        "bins": rows,
        "pre_exit_bins": pre_exit_rows,
    }


def clean_numeric_feature(values) -> np.ndarray | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    if float(np.std(arr[finite])) < 1e-12:
        return None
    return arr


def behavior_state_features(
    valid_trials: pd.DataFrame,
) -> tuple[np.ndarray | None, list[str]]:
    cols = []
    names = []
    for col, label in [
        ("prev_response", "previous choice"),
        ("prev_rewarded", "previous reward"),
        ("prev_stim_rate", "previous rate"),
        ("early_withdrawal", "early withdrawal"),
    ]:
        if col in valid_trials:
            arr = clean_numeric_feature(valid_trials[col])
            if arr is not None:
                cols.append(arr)
                names.append(label)

    for label, later, earlier in [
        ("response latency", "response_ts", "first_stim_ts"),
        ("center-exit latency", "center_port_exit_ts", "first_stim_ts"),
        ("response after center exit", "response_ts", "center_port_exit_ts"),
    ]:
        if later in valid_trials and earlier in valid_trials:
            arr = clean_numeric_feature(valid_trials[later] - valid_trials[earlier])
            if arr is not None:
                cols.append(arr)
                names.append(label)

    if not cols:
        return None, []
    return np.column_stack(cols), names


def behavior_state_choice_controls(plt, nested_choice, state_names, args) -> dict:
    if not state_names:
        return {
            "state_feature_names": [],
            "models": [],
            "ridge_gain_after_rate_state": None,
        }

    names = [
        "rate fixed effects",
        "rate FE + nested ridge",
        "rate FE + state",
        "rate FE + state + nested ridge",
    ]
    rows = [
        next(row for row in nested_choice["models"] if row["model"] == name)
        for name in names
    ]
    losses = [np.asarray(nested_choice["fold_losses"][name]) for name in names]

    state_row = next(row for row in rows if row["model"] == "rate FE + state")
    full_row = next(
        row for row in rows if row["model"] == "rate FE + state + nested ridge"
    )
    ridge_gain_after_state = state_row["log_loss"] - full_row["log_loss"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _box(
        ax,
        losses,
        [
            "rate\nFE",
            "rate FE +\nnested ridge",
            "rate FE\n+ state",
            "rate FE + state\n+ nested ridge",
        ],
        ["0.7", "tab:blue", "tab:purple", "tab:green"],
    )
    ax.set_ylabel("held-out choice log loss")
    ax.set_title("train: fully nested stacked model\ntest: outer-fold choices")
    ax.text(0.02, 0.95, "lower = better", transform=ax.transAxes, va="top")
    fig.tight_layout()
    save_fig(fig, "fig5_behavior_state_choice_controls", args.no_save)

    return {
        "state_feature_names": state_names,
        "models": rows,
        "ridge_gain_after_rate_state": float(ridge_gain_after_state),
    }


def count_events_between(events, start, stop) -> float:
    if not (np.isfinite(start) and np.isfinite(stop)) or stop < start:
        return np.nan
    try:
        arr = np.asarray(events, dtype=float)
    except (TypeError, ValueError):
        return np.nan
    arr = arr[np.isfinite(arr)]
    return float(((arr >= start) & (arr < stop)).sum())


def summarize_flash_count_behavior(df, count_col, min_n):
    rows = []
    for count, group in df.groupby(count_col, sort=True):
        if len(group) < min_n:
            continue
        row: dict[str, int | float] = {
            "flash_count": int(count),
            "n": int(len(group)),
        }
        for col in ["mouse_high", "ridge_high"]:
            p = float(group[col].mean())
            row[f"{col}_fraction"] = p
            row[f"{col}_sem"] = float(np.sqrt(p * (1 - p) / len(group)))
        rows.append(row)
    return rows


def realized_flash_behavior(
    plt, valid_trials, y_choice, ridge_pred, boundary_rate, args
) -> dict:
    min_n = 5
    rows = []
    for i, trial in valid_trials.reset_index(drop=True).iterrows():
        first_stim = float(trial["first_stim_ts"])
        center_exit = float(trial["center_port_exit_ts"])
        response = float(trial["response_ts"])
        pre_exit = count_events_between(trial["stim_ts"], first_stim, center_exit)
        movement = count_events_between(trial["stim_ts"], center_exit, response)
        pre_response = count_events_between(trial["stim_ts"], first_stim, response)
        if not np.isfinite(pre_exit + movement + pre_response):
            continue
        rows.append(
            {
                "pre_exit_flashes": int(pre_exit),
                "movement_flashes": int(movement),
                "pre_response_flashes": int(pre_response),
                "mouse_high": int(y_choice[i]),
                "ridge_high": int(ridge_pred[i] > boundary_rate),
            }
        )
    flash_cols = [
        "pre_exit_flashes",
        "movement_flashes",
        "pre_response_flashes",
        "mouse_high",
        "ridge_high",
    ]
    df = pd.DataFrame(rows, columns=pd.Index(flash_cols))
    summaries = {
        "pre_exit": summarize_flash_count_behavior(df, "pre_exit_flashes", min_n),
        "movement": summarize_flash_count_behavior(df, "movement_flashes", min_n),
        "pre_response": summarize_flash_count_behavior(
            df, "pre_response_flashes", min_n
        ),
    }

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    plot_specs = [
        ("mouse_high", "k", "o", "-", "mouse: right/high choice"),
        ("ridge_high", "tab:blue", "s", "--", "ridge: predicted high side"),
    ]
    for ax, key, xlabel in zip(
        axs,
        ["pre_exit", "pre_response"],
        ["flashes before center exit", "flashes before response"],
        strict=True,
    ):
        summary = summaries[key]
        x = np.array([row["flash_count"] for row in summary])
        for col, color, marker, ls, label in plot_specs:
            y = np.array([row[f"{col}_fraction"] for row in summary])
            yerr = np.array([row[f"{col}_sem"] for row in summary])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=color,
                marker=marker,
                lw=2,
                ls=ls,
                capsize=3,
                label=label,
            )
        ax.axhline(0.5, color="0.5", ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.03, 1.03)
        ax.text(
            0.03,
            0.05,
            f"shown counts: n >= {min_n}",
            transform=ax.transAxes,
            color="0.35",
            fontsize=9,
        )
    axs[0].set_ylabel("fraction high / right")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_fig(fig, "fig5_realized_flash_behavior", args.no_save)

    return {
        "min_n_per_flash_count": min_n,
        "n_trials": int(len(df)),
        "summaries": summaries,
    }


def pca_dimensionality_grid(X, y_choice):
    n_trials, n_units = X.shape
    outer_train = n_trials - int(np.ceil(n_trials / CV_FOLDS))
    inner_train = outer_train - int(np.ceil(outer_train / inner_cv().get_n_splits()))
    _, choice_counts = np.unique(y_choice, return_counts=True)
    balanced_n = int(2 * choice_counts.min())
    choice_splits = int(min(CV_FOLDS, choice_counts.min()))
    choice_train = balanced_n - int(np.ceil(balanced_n / choice_splits))
    max_pc = int(max(1, min(n_units, inner_train, choice_train)))
    grid = [n for n in [1, 2, 3, 5, 10, 20, 50, 100, max_pc] if n <= max_pc]
    return sorted(set(grid)), max_pc


def make_pca_rate_regressor(n_components, args):
    alphas = getattr(args, "ridge_alphas", np.logspace(-4, 6, 21))
    return GridSearchCV(
        make_pipeline(StandardScaler(), PCA(n_components=n_components), Ridge()),
        {"ridge__alpha": alphas},
        cv=inner_cv(),
        scoring="r2",
    )


def make_pca_choice_decoder(n_components):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def analyze_encoding_dimensionality(plt, X, rate, y_choice, cv_reg, args) -> dict:
    pc_grid, max_pc = pca_dimensionality_grid(X, y_choice)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=max_pc).fit(Xs)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dim_rng = np.random.default_rng(RANDOM_STATE + 41)

    rows = []
    for n_pc in pc_grid:
        _, rate_scores = rate_cv_readout(
            lambda _X_train, n=n_pc: make_pca_rate_regressor(n, args),
            X,
            rate,
            cv_reg,
        )
        choice_scores = balanced_decode(
            X,
            y_choice,
            dim_rng,
            args.n_balance_resamples,
            decoder_factory=lambda n=n_pc: make_pca_choice_decoder(n),
        )
        rows.append(
            {
                "n_pcs": int(n_pc),
                "cumulative_variance": float(cumulative[n_pc - 1]),
                "rate_r2_mean": float(np.mean(rate_scores)),
                "rate_r2_sem": float(
                    np.std(rate_scores, ddof=1) / np.sqrt(len(rate_scores))
                ),
                "choice_acc_mean": float(np.mean(choice_scores)),
                "choice_acc_sem": float(
                    np.std(choice_scores, ddof=1) / np.sqrt(len(choice_scores))
                ),
            }
        )

    x = np.asarray(pc_grid)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    axs[0].plot(np.arange(1, max_pc + 1), cumulative, color="k", lw=2)
    axs[0].scatter(x, [row["cumulative_variance"] for row in rows], color="k", s=24)
    axs[0].set_ylabel("cumulative variance explained")
    axs[0].set_ylim(0, 1.02)
    axs[0].set_title("PCA of population activity")

    axs[1].errorbar(
        x,
        [row["rate_r2_mean"] for row in rows],
        yerr=[row["rate_r2_sem"] for row in rows],
        color="tab:blue",
        marker="o",
        lw=2,
        capsize=3,
    )
    axs[1].axhline(0, color="0.5", ls="--", lw=1)
    axs[1].set_ylabel("rate CV $R^2$")
    axs[1].set_title("train: stimulus rate\ntest: held-out trials")

    axs[2].errorbar(
        x,
        [row["choice_acc_mean"] for row in rows],
        yerr=[row["choice_acc_sem"] for row in rows],
        color="tab:red",
        marker="o",
        lw=2,
        capsize=3,
    )
    axs[2].axhline(0.5, color="0.5", ls="--", lw=1)
    axs[2].set_ylabel("choice CV accuracy")
    axs[2].set_ylim(0.45, 1.02)
    axs[2].set_title("train: non-boundary choice\ntest: held-out trials")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("number of PCs")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(int(v)) for v in x], rotation=55, ha="right", fontsize=8
        )
    save_fig(fig, "fig5_encoding_dimensionality", args.no_save)

    return {"pc_grid": [int(v) for v in pc_grid], "max_pc": max_pc, "curves": rows}


def analyze_rate(
    plt, X, X_time, bin_centers, valid_trials, rate, y_choice, cv_reg, rng, args
) -> dict:
    print("\n[Q5] Rate decoding ...")
    boundary_rate = 12.0

    def ridge_factory(X_train=None):
        return make_rate_regressor(X_train, args)

    ridge_pred, real_r2, ridge_search_infos = rate_cv_readout(
        ridge_factory, X, rate, cv_reg, return_search=True
    )
    readouts = {
        "ridge": continuous_rate_metrics(
            "ridge", rate, ridge_pred, real_r2, boundary_rate
        ),
    }
    ridge_metrics = readouts["ridge"]["metrics"]

    shuffle_r2 = np.full(args.n_shuffles, np.nan)
    for i in range(args.n_shuffles):
        _, fold_r2 = rate_cv_readout(ridge_factory, X, rng.permutation(rate), cv_reg)
        shuffle_r2[i] = fold_r2.mean()

    ridge_refit = ridge_factory(X).fit(X, rate)
    tuning = {
        "ridge": summarize_search_infos(ridge_search_infos, ridge_refit),
    }
    weights = ridge_refit.best_estimator_.named_steps["ridge"].coef_
    slopes = np.array([np.polyfit(rate, X[:, u], 1)[0] for u in range(X.shape[1])])
    r_slope_weight = _corr(slopes, weights)

    # balanced multiclass one-vs-rest decoder (interpretable confusion matrix)
    mc_acc, cm, rate_labels = balanced_multiclass_rate(
        X, rate, rng, args.n_balance_resamples
    )
    _, per_class_counts = np.unique(rate, return_counts=True)
    mc_chance = 1.0 / len(rate_labels)
    mc_high_by_rate = np.asarray(
        [cm[i, rate_labels > boundary_rate].sum() for i in range(len(rate_labels))]
    )
    mc_error_by_rate = np.where(
        rate_labels < boundary_rate, mc_high_by_rate, 1 - mc_high_by_rate
    )
    mc_category_acc = float(1 - mc_error_by_rate.mean())
    choice_balanced_ridge = choice_balanced_ridge_rate_control(
        X, rate, y_choice, boundary_rate, args.n_balance_resamples, ridge_factory
    )

    rate_min, rate_max = float(rate.min()), float(rate.max())
    readout_styles = {
        "ridge": ("k", "o", "ridge"),
    }

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    for key, readout in readouts.items():
        color, marker, label = readout_styles[key]
        summary = readout["summary"]
        axs[0, 0].errorbar(
            summary["rate"],
            summary["mean"],
            yerr=summary["sem"],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=3,
            ms=5,
            label=label,
        )
    axs[0, 0].plot([rate_min, rate_max], [rate_min, rate_max], color="k", ls="--", lw=1)
    axs[0, 0].text(
        0.05,
        0.95,
        "dot = mean held-out prediction per rate\n"
        "error bars = SEM across trials\n"
        f"calibration slope: {ridge_metrics['calibration_slope']:.2f}",
        transform=axs[0, 0].transAxes,
        va="top",
    )
    axs[0, 0].legend(frameon=False, loc="lower right")
    axs[0, 0].set_xlim(rate_min, rate_max)
    axs[0, 0].set_ylim(rate_min, rate_max)
    axs[0, 0].set_aspect("equal", adjustable="box")
    axs[0, 0].set_xlabel("true stimulus rate (Hz)")
    axs[0, 0].set_ylabel("predicted rate (Hz)")

    for key, readout in readouts.items():
        color, marker, label = readout_styles[key]
        summary = readout["summary"]
        axs[0, 1].errorbar(
            summary["rate"],
            summary["residual"],
            yerr=summary["sem"],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=3,
            ms=5,
            label=label,
        )
    axs[0, 1].axhline(0, color="k", ls="--", lw=1)
    axs[0, 1].text(
        0.05,
        0.08,
        f"residual slope: {ridge_metrics['residual_slope']:.2f}\n"
        f"MAE: {ridge_metrics['mae_hz']:.2f} Hz",
        transform=axs[0, 1].transAxes,
        va="bottom",
    )
    axs[0, 1].legend(frameon=False, loc="upper right")
    axs[0, 1].set_xlabel("true stimulus rate (Hz)")
    axs[0, 1].set_ylabel("prediction residual (Hz)")

    _box(
        axs[1, 0],
        [real_r2, shuffle_r2],
        ["ridge", "ridge\nshuffle"],
        ["k", "0.7"],
    )
    axs[1, 0].axhline(0, color="k", ls="--", lw=1)
    axs[1, 0].set_ylabel("CV $R^2$")

    axs[1, 1].scatter(slopes, weights, color="k", alpha=0.3, s=18)
    axs[1, 1].axhline(0, color="k", lw=0.8)
    axs[1, 1].axvline(0, color="k", lw=0.8)
    axs[1, 1].text(
        0.05, 0.95, f"r = {r_slope_weight:.2f}", transform=axs[1, 1].transAxes, va="top"
    )
    axs[1, 1].set_xlabel("per-unit rate slope (sp/s/Hz)")
    axs[1, 1].set_ylabel("linear ridge weight")
    fig.tight_layout()
    save_fig(fig, "fig5_rate_decoding", args.no_save)

    def mlp_factory(X_train=None):
        return make_mlp_rate_regressor(X_train, args)

    mlp_pred, mlp_r2, mlp_search_infos = rate_cv_readout(
        mlp_factory, X, rate, cv_reg, return_search=True
    )
    readouts["mlp"] = continuous_rate_metrics(
        "MLP", rate, mlp_pred, mlp_r2, boundary_rate
    )
    mlp_refit = mlp_factory(X).fit(X, rate)
    tuning["mlp"] = summarize_search_infos(mlp_search_infos, mlp_refit)
    mlp_rate_diagnostic(plt, rate, readouts, args, boundary_rate)

    residualized_rate = choice_residualized_rate_diagnostic(
        plt,
        X,
        rate,
        y_choice,
        cv_reg,
        args,
        boundary_rate,
        ridge_factory,
        readouts["ridge"],
    )
    slope_subset_diagnostic = slope_subset_rate_diagnostic(
        plt, X, rate, cv_reg, args, boundary_rate
    )
    ridge_alpha_path = ridge_alpha_path_diagnostic(
        plt,
        X,
        rate,
        cv_reg,
        args,
        boundary_rate,
        tuning["ridge"]["final_refit"].get("ridge__alpha"),
    )
    single_unit_slope_examples = single_unit_slope_example_diagnostic(
        plt, X, rate, cv_reg, args, boundary_rate
    )
    steepest_unit_ols = steepest_unit_ols_diagnostic(
        plt, X, rate, cv_reg, args, boundary_rate
    )

    encoding_dimensionality = analyze_encoding_dimensionality(
        plt, X, rate, y_choice, cv_reg, args
    )

    fig_cm, ax_cm = plt.subplots(figsize=(5.8, 4.8))
    im = ax_cm.imshow(cm, vmin=0, vmax=0.5, cmap="magma", aspect="auto")
    tick = np.arange(len(rate_labels))
    ax_cm.set_xticks(tick)
    ax_cm.set_xticklabels([f"{int(r)}" for r in rate_labels], fontsize=8)
    ax_cm.set_yticks(tick)
    ax_cm.set_yticklabels([f"{int(r)}" for r in rate_labels], fontsize=8)
    ax_cm.set_xlabel(
        f"predicted rate (Hz)\nbalanced acc = {mc_acc.mean():.2f} (chance {mc_chance:.2f})"
    )
    ax_cm.set_ylabel("true rate (Hz)")
    cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of trials (color max 0.5)")
    fig_cm.tight_layout()
    save_fig(fig_cm, "fig5_rate_confusion", args.no_save)

    rate_target_family = analyze_rate_target_family(
        plt,
        X,
        rate,
        y_choice,
        cv_reg,
        args,
        boundary_rate,
        endpoint_distance=8.0,
    )

    psych = pd.DataFrame(
        {
            "rate": rate,
            "mouse_high": y_choice,
            "ridge_high": (ridge_pred > boundary_rate).astype(float),
        }
    )
    psych_summary = (
        psych.groupby("rate", sort=True)
        .agg(
            mouse_high=("mouse_high", "mean"),
            ridge_high=("ridge_high", "mean"),
            n=("mouse_high", "size"),
        )
        .reset_index()
    )
    psych_summary["mouse_sem"] = np.sqrt(
        psych_summary["mouse_high"]
        * (1 - psych_summary["mouse_high"])
        / psych_summary["n"]
    )
    for col in ["ridge_high"]:
        psych_summary[f"{col}_sem"] = np.sqrt(
            psych_summary[col] * (1 - psych_summary[col]) / psych_summary["n"]
        )
    psych_summary["multiclass_high"] = mc_high_by_rate
    low_side = psych_summary["rate"] < 12.0
    psych_summary["mouse_error"] = np.where(
        low_side, psych_summary["mouse_high"], 1 - psych_summary["mouse_high"]
    )
    for col in ["ridge_high", "multiclass_high"]:
        psych_summary[f"{col}_error"] = np.where(
            low_side, psych_summary[col], 1 - psych_summary[col]
        )

    fig_psy, axs_psy = plt.subplots(
        2,
        1,
        figsize=(7.5, 7),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    axs_psy[0].errorbar(
        psych_summary["rate"],
        psych_summary["mouse_high"],
        yerr=psych_summary["mouse_sem"],
        color="k",
        marker="o",
        lw=2,
        capsize=3,
        label="mouse: right/high choice",
    )
    for col, color, marker, label in [
        ("ridge_high", "tab:blue", "s", "ridge: predicted high side"),
    ]:
        axs_psy[0].errorbar(
            psych_summary["rate"],
            psych_summary[col],
            yerr=psych_summary[f"{col}_sem"],
            color=color,
            marker=marker,
            lw=2,
            ls="--",
            capsize=3,
            label=label,
        )
    axs_psy[0].axhline(0.5, color="0.5", ls=":", lw=1)
    axs_psy[0].axvline(12.0, color="0.35", ls="--", lw=1)
    axs_psy[0].set_ylim(-0.03, 1.03)
    axs_psy[0].set_ylabel("fraction high / right")
    axs_psy[0].legend(frameon=False, loc="upper left")

    axs_psy[1].plot(
        psych_summary["rate"],
        psych_summary["mouse_error"],
        color="k",
        marker="o",
        lw=2,
        label="mouse category error",
    )
    for col, color, marker, ls, label in [
        ("ridge_high_error", "tab:blue", "s", "--", "ridge category error"),
    ]:
        axs_psy[1].errorbar(
            psych_summary["rate"],
            psych_summary[col],
            yerr=psych_summary.get(f"{col}_sem"),
            color=color,
            marker=marker,
            lw=2,
            ls=ls,
            capsize=3,
            label=label,
        )
    axs_psy[1].axvline(12.0, color="0.35", ls="--", lw=1)
    axs_psy[1].set_ylim(-0.03, 1.03)
    axs_psy[1].set_ylabel("category error fraction")
    axs_psy[1].set_xlabel("true stimulus rate (Hz)")
    axs_psy[1].set_xticks(rate_labels)
    axs_psy[1].legend(frameon=False, loc="upper left")
    save_fig(fig_psy, "fig5_rate_behavior", args.no_save)

    flash_behavior = realized_flash_behavior(
        plt, valid_trials, y_choice, ridge_pred, boundary_rate, args
    )
    state_X, state_names = behavior_state_features(valid_trials)
    nested_choice = nested_rate_choice_cv(X, rate, y_choice, args, state_X=state_X)
    nested_readout = nested_choice["nested_rate_readout"]
    choice_model_comparison = heldout_choice_model_comparison(plt, nested_choice, args)
    within_rate_choice_tests = rate_fixed_choice_tests(
        plt,
        rate,
        y_choice,
        nested_readout,
        nested_choice,
        boundary_rate,
        rng,
        args,
    )
    time_resolved_choice = time_resolved_rate_fixed_choice(
        plt, X_time, bin_centers, valid_trials, rate, y_choice, args
    )
    behavior_state_choice = behavior_state_choice_controls(
        plt, nested_choice, state_names, args
    )

    model_comparison = {
        "ridge": ridge_metrics,
        "mlp": readouts["mlp"]["metrics"],
        "multiclass_collapse": {
            "exact_rate_acc": float(mc_acc.mean()),
            "exact_rate_chance": float(mc_chance),
            "category_collapsed_acc": mc_category_acc,
        },
    }

    return {
        "real_r2_mean": ridge_metrics["cv_r2_mean"],
        "real_r2_folds": [round(float(v), 4) for v in real_r2],
        "shuffle_r2_mean": float(shuffle_r2.mean()),
        "r2_p": perm_p(real_r2.mean(), shuffle_r2),
        "calibration_slope": ridge_metrics["calibration_slope"],
        "calibration_intercept": ridge_metrics["calibration_intercept"],
        "residual_slope": ridge_metrics["residual_slope"],
        "residual_intercept": ridge_metrics["residual_intercept"],
        "spearman_rho": ridge_metrics["spearman_rho"],
        "mae_hz": ridge_metrics["mae_hz"],
        "rmse_hz": ridge_metrics["rmse_hz"],
        "category_collapsed_acc": ridge_metrics["category_collapsed_acc"],
        "r_slope_weight": r_slope_weight,
        "multiclass_acc": float(mc_acc.mean()),
        "multiclass_chance": float(mc_chance),
        "multiclass_category_collapsed_acc": mc_category_acc,
        "n_rate_classes": int(len(rate_labels)),
        "min_per_rate": int(per_class_counts.min()),
        "model_comparison": model_comparison,
        "hyperparameter_tuning": tuning,
        "choice_residualized_rate": residualized_rate["choice_residualized"],
        "rate_self_residualized_control": residualized_rate["rate_self_residualized"],
        "ridge_alpha_path": ridge_alpha_path,
        "slope_subset_diagnostic": slope_subset_diagnostic,
        "single_unit_slope_examples": single_unit_slope_examples,
        "steepest_unit_ols": steepest_unit_ols,
        "encoding_dimensionality": encoding_dimensionality,
        "rate_target_family": rate_target_family,
        "choice_balanced_ridge": choice_balanced_ridge,
        "rate_prediction": readouts["ridge"]["summary"]
        .round(4)
        .to_dict(orient="records"),
        "rate_predictions": {
            key: readout["summary"].round(4).to_dict(orient="records")
            for key, readout in readouts.items()
        },
        "psychometric": psych_summary.round(4).to_dict(orient="records"),
        "realized_flash_behavior": flash_behavior,
        "choice_model_comparison": choice_model_comparison,
        "nested_choice_design": {
            "folds": nested_choice["folds"],
            "upstream_rate_search": nested_choice["upstream_rate_search"],
        },
        "within_rate_choice_tests": within_rate_choice_tests,
        "time_resolved_choice": time_resolved_choice,
        "behavior_state_choice_controls": behavior_state_choice,
    }


# ---------------------------------------------------------------------------
# Time-course decoding for category, choice, and interaction.
# ---------------------------------------------------------------------------
def timecourse_balanced(X_time, y, rng, n_resamples, n_shuffles):
    n_t = X_time.shape[2]
    real = np.full((n_t, n_resamples), np.nan)
    for t in range(n_t):
        real[t] = balanced_decode(
            X_time[:, :, t], y, rng, n_resamples, decoder_factory=make_fast_decoder
        )
    shuffle = np.full((n_shuffles, n_t), np.nan)
    for s in range(n_shuffles):
        ys = rng.permutation(y)
        for t in range(n_t):
            shuffle[s, t] = balanced_decode(
                X_time[:, :, t], ys, rng, 1, decoder_factory=make_fast_decoder
            )[0]
    return real, shuffle


def timecourse_joint_balanced(
    X_time, y, nuisance, rng, n_resamples, n_shuffles
) -> tuple[np.ndarray, np.ndarray]:
    n_t = X_time.shape[2]
    real = np.full((n_t, n_resamples), np.nan)
    for t in range(n_t):
        real[t] = balanced_joint_decode(
            X_time[:, :, t],
            y,
            nuisance,
            rng,
            n_resamples,
            decoder_factory=make_fast_decoder,
        )
    shuffle = np.full((n_shuffles, n_t), np.nan)
    for s in range(n_shuffles):
        ys = rng.permutation(y)
        for t in range(n_t):
            shuffle[s, t] = balanced_joint_decode(
                X_time[:, :, t],
                ys,
                nuisance,
                rng,
                1,
                decoder_factory=make_fast_decoder,
            )[0]
    return real, shuffle


def _plot_timecourse_decoder(ax, bin_centers, real, shuffle, *, color, label):
    rmean = real.mean(axis=1)
    lo = np.nanpercentile(real, 2.5, axis=1)
    hi = np.nanpercentile(real, 97.5, axis=1)
    smean = shuffle.mean(axis=0)
    ax.plot(bin_centers, rmean, color=color, label=label)
    ax.fill_between(bin_centers, lo, hi, color=color, alpha=0.18)
    ax.plot(bin_centers, smean, color=color, lw=0.8, ls=":", alpha=0.6)
    return rmean, smean


def analyze_timecourse(
    plt,
    X_time_100,
    bin_centers_100,
    X_time_200,
    bin_centers_200,
    y_cat,
    y_choice,
    rng,
    args,
) -> dict:
    print("\n[TC] Time-course decoding (category and choice) ...")
    colors = {"category": "tab:blue", "choice": "tab:green"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    out = {}
    for ax, label, X_time, bin_centers in [
        (axs[0], "100ms", X_time_100, bin_centers_100),
        (axs[1], "200ms", X_time_200, bin_centers_200),
    ]:
        out[label] = {"bin_centers": [round(float(b), 3) for b in bin_centers]}
        for name, y in {"category": y_cat, "choice": y_choice}.items():
            real, shuffle = timecourse_balanced(
                X_time, y, rng, args.tc_resamples, args.timecourse_n_shuffles
            )
            rmean, smean = _plot_timecourse_decoder(
                ax, bin_centers, real, shuffle, color=colors[name], label=name
            )
            out[label][f"peak_{name}"] = float(rmean.max())
            out[label][f"curve_{name}"] = [round(float(v), 4) for v in rmean]
            out[label][f"shuffle_{name}"] = [round(float(v), 4) for v in smean]
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_ylim(0.4, 1)
        ax.set_xlabel("time after first stimulus (s)")
        ax.set_title(f"{label} bins\ntrain: category or choice; test: held-out bin")
    axs[0].set_ylabel("CV accuracy (balanced trials)")
    axs[0].legend(frameon=False, title="solid = real, dotted = shuffle")
    fig.tight_layout()
    save_fig(fig, "fig6_timecourse", args.no_save)
    out["peak_category"] = out["100ms"]["peak_category"]
    out["peak_choice"] = out["100ms"]["peak_choice"]
    return out


# ---------------------------------------------------------------------------
# June 29 controls: stimulus-strength timing and 12 Hz boundary transfer.
# ---------------------------------------------------------------------------
def analyze_june29_controls(
    plt,
    X,
    X_strength_time,
    strength_bin_centers,
    X_boundary,
    valid_trials,
    boundary_trials,
    y_cat,
    y_choice,
    rng,
    args,
) -> dict:
    print("\n[Controls] Stimulus-strength time-course and 12 Hz boundary transfer ...")
    boundary_rate = float(boundary_trials["stim_rate_vision"].median())
    rates = valid_trials["stim_rate_vision"].to_numpy(dtype=float)
    signed_strength = rates - boundary_rate
    abs_strength = np.abs(signed_strength)
    strengths = sorted(float(s) for s in np.unique(abs_strength) if s > 0)
    strength_curves = {}
    strength_counts = {}

    boundary_choice = (boundary_trials["response"] == 1).astype(int).to_numpy()
    cat_boundary_align, cat_boundary_high_frac, cat_boundary_pred = (
        category_to_boundary_choice_decode(
            X,
            y_cat,
            y_choice,
            X_boundary,
            boundary_choice,
            rng,
            args.n_balance_resamples,
        )
    )
    cat_boundary_shuffle = category_to_boundary_choice_shuffle(
        cat_boundary_pred, boundary_choice, rng, args.n_shuffles
    )

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, strength in enumerate(strengths):
        mask = np.isclose(abs_strength, strength)
        counts = pd.crosstab(
            pd.Series(y_cat[mask], name="category"),
            pd.Series(y_choice[mask], name="choice"),
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        strength_counts[f"{strength:g} Hz"] = counts.to_string()
        if (counts.to_numpy() == 0).any():
            continue
        real, shuffle = timecourse_joint_balanced(
            X_strength_time[mask],
            y_cat[mask],
            y_choice[mask],
            rng,
            args.tc_resamples,
            args.timecourse_n_shuffles,
        )
        mean = real.mean(axis=1)
        shuffle_mean = shuffle.mean(axis=0)
        color = cmap(0.15 + 0.75 * i / max(1, len(strengths) - 1))
        axs[0].plot(
            strength_bin_centers,
            mean,
            color=color,
            marker="o",
            label=f"|rate - 12| = {strength:g} Hz",
        )
        strength_curves[f"{strength:g}"] = {
            "n_trials": int(mask.sum()),
            "curve": [round(float(v), 4) for v in mean],
            "shuffle": [round(float(v), 4) for v in shuffle_mean],
            "peak": float(np.nanmax(mean)),
        }
    axs[0].axhline(0.5, color="k", ls="--", lw=1)
    axs[0].set_ylim(0.35, 1.0)
    axs[0].set_ylabel("CV accuracy (balanced trials)")
    axs[0].set_title("train: category within strength\ntest: held-out same-bin trials")
    axs[0].set_xlabel("time from first stimulus (s)")
    axs[0].legend(frameon=False, fontsize=8)

    _box(
        axs[1],
        [cat_boundary_align, cat_boundary_shuffle],
        ["real", "shuffle"],
        ["tab:purple", "0.7"],
    )
    axs[1].axhline(0.5, color="k", ls="--", lw=1)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("choice alignment on 12 Hz trials")
    axs[1].set_title("train: non-boundary category\ntest: 12 Hz choice alignment")
    fig.tight_layout()
    save_fig(fig, "fig8_june29_controls", args.no_save)

    projection_scores = []
    projection_scores_boundary = []
    for _ in range(args.n_balance_resamples):
        train_idx = balanced_joint_indices(y_cat, y_choice, rng)
        clf = make_fast_decoder().fit(X[train_idx], y_cat[train_idx])
        projection_scores.append(clf.decision_function(X))
        projection_scores_boundary.append(clf.decision_function(X_boundary))
    projection_scores = np.mean(projection_scores, axis=0)
    projection_scores_boundary = np.mean(projection_scores_boundary, axis=0)
    groups = [
        ("low\nnon-boundary", projection_scores[y_cat == 0], "tab:blue"),
        (
            "12 Hz\nleft choice",
            projection_scores_boundary[boundary_choice == 0],
            "tab:purple",
        ),
        (
            "12 Hz\nright choice",
            projection_scores_boundary[boundary_choice == 1],
            "tab:purple",
        ),
        ("high\nnon-boundary", projection_scores[y_cat == 1], "tab:red"),
    ]
    fig_proj, ax_proj = plt.subplots(figsize=(7.5, 4.5))
    for pos, (label, values, color) in enumerate(groups, start=1):
        jitter = rng.normal(0, 0.04, size=len(values))
        ax_proj.scatter(
            np.full(len(values), pos) + jitter,
            values,
            color=color,
            alpha=0.35,
            s=18,
        )
        if len(values):
            med = float(np.median(values))
            ax_proj.plot([pos - 0.22, pos + 0.22], [med, med], color="k", lw=2)
    ax_proj.axhline(0, color="k", ls="--", lw=1)
    ax_proj.set_xticks(np.arange(1, len(groups) + 1))
    ax_proj.set_xticklabels([g[0] for g in groups])
    ax_proj.set_ylabel("category decoder score")
    ax_proj.set_title("train: non-boundary category\ntest: category-axis projection")
    fig_proj.tight_layout()
    save_fig(fig_proj, "fig8_category_axis_projection", args.no_save)

    return {
        "strength_bin_centers": [round(float(b), 3) for b in strength_bin_centers],
        "strength_curves": strength_curves,
        "strength_cell_counts": strength_counts,
        "boundary_rate_hz": boundary_rate,
        "decoder": "fixed-C logistic",
        "cat_boundary_decoder_window_s": float(POST_STIM_WINDOW_S),
        "boundary_n_trials": int(len(boundary_trials)),
        "cat_boundary_choice_align_mean": float(cat_boundary_align.mean()),
        "cat_boundary_choice_align_shuffle_mean": float(cat_boundary_shuffle.mean()),
        "cat_boundary_choice_align_p": perm_p(
            cat_boundary_align.mean(), cat_boundary_shuffle
        ),
        "cat_boundary_high_prediction_fraction": float(cat_boundary_high_frac.mean()),
        "projection_medians": {
            label.replace("\n", " "): float(np.median(values)) if len(values) else None
            for label, values, _ in groups
        },
    }


# ---------------------------------------------------------------------------
# Report (reader-facing artifact).
# ---------------------------------------------------------------------------
def _decodable(p: float, mean: float, chance: float) -> str:
    if p < 0.05 and mean > chance + 0.02:
        return "yes"
    return "not clearly"


# ---------------------------------------------------------------------------
# HTML report (reader-facing artifact).
# ---------------------------------------------------------------------------
def write_report_html(results: dict, no_save: bool) -> None:
    if no_save:
        print("\n[report] --no-save set; skipping report.")
        return
    meta = results["meta"]
    q1, q2, q3 = results["q1"], results["q2"], results["q3"]
    q4, q5, tc = results["q4"], results["q5"], results["tc"]
    controls = results["june29_controls"]
    sanity = results["sanity"]
    common_population = (
        f"Population: {meta['n_trials']} non-boundary choice trials x "
        f"{meta['n_units']} units; neural feature is firing rate in the 0-1 s "
        "window after first stimulus."
    )
    figure_notes = {
        "fig1_category_decoding": common_population
        + " Left: balanced ten-fold accuracy over 20 undersampling draws versus 100 label shuffles; 0.5 is chance. Right: each point is a unit; x is high-minus-low Cohen's d, y is its standardized logistic weight, and r is Pearson correlation.",
        "fig2_choice_decoding": common_population
        + " Left: left-versus-right balanced ten-fold accuracy over 20 draws versus 100 shuffles. Right: each point is a unit; x is right-minus-left Cohen's d and y is its standardized logistic weight. Random folds mix acquisition times.",
        "fig3_residual_decoding": "In every balanced outer fold, a linear nuisance model is fit only on training trials and removed from training and test neural matrices before classification. Boxes contain 20 balanced draws or 100 shuffles; the unit is a resample, not an animal.",
        "fig7_residual_sanity": "Negative control using the same fold-safe subtraction but removing the target itself. Boxes contain 20 draws or 100 shuffles; category-minus-category and choice-minus-choice should approach 0.5.",
        "fig4_interaction": f"Each point is one of {meta['n_units']} units. Firing rate is the 0-1 s mean. Left/middle compare category or choice effects across the other variable; dashed lines are identity. Interaction OLS p values are Benjamini-Hochberg corrected across units.",
        "fig4_interaction_psths": "Only q < 0.05 interaction units are shown. Curves are trial-mean firing rates in non-overlapping 50 ms bins from -0.2 to 1.0 s around first stimulus; color is category and line style is report. No uncertainty band is plotted.",
        "fig9_time_aware_cv": "Ten random or contiguous outer-fold scores. Binary models use fixed-C class-balanced logistic regression and balanced accuracy; rate uses fold-tuned ridge and R2. Boxes contain folds; 0.5 and 0 are the binary and R2 reference levels.",
        "fig6_timecourse": "First-stimulus-aligned, non-overlapping 100 or 200 ms bins across 0-1 s. A separate fixed-C decoder is fit at each bin. Solid lines are means over five balanced draws, bands are their 2.5th-97.5th percentiles, dotted lines are ten-shuffle means, and 0.5 is chance. No time-point multiplicity test is claimed.",
        "fig8_june29_controls": "Left: non-overlapping 200 ms bins stratified by distance from 12 Hz and balanced across category x choice cells; curves average five draws. Right: a non-boundary category decoder is scored for report alignment on 53 boundary trials; boxes are 20 draws and 100 shuffled-choice nulls.",
        "fig8_category_axis_projection": "A category axis is fit on balanced non-boundary training trials. Held-out non-boundary and 53 boundary-trial projections use the same standardized decoder coordinates; boundary groups are defined only by left/right report.",
        "fig5_rate_decoding": common_population
        + " Outer ten-fold ridge predictions are held out and alpha is selected by inner five-fold CV. Points are mean +/- trial SEM by rate; diagonal is perfect calibration; residual = prediction - truth; R2 boxes contain ten outer folds or 100 shuffled-target fits.",
        "fig5_mlp_rate_diagnostic": "Ridge and bounded tanh-MLP use identical outer folds and target scaling. Points are mean +/- trial SEM by rate. MLP uses LBFGS without early stopping; hidden sizes 4/8/16 and alpha 1e-4 to 1e2 are selected inside each fold.",
        "fig5_rate_confusion": "Balanced one-vs-rest classification of eight nominal rates. Rows are true rate, columns predicted rate, and each row sums to one; color is the mean fraction over 20 balanced resamples. Exact-rate chance is 1/8.",
        "fig5_choice_residualized_rate": "The choice nuisance model is fit inside each outer rate-training fold and subtracted before ridge fitting. Raw, choice-residualized, and rate-self-residualized controls use the same folds; points are mean held-out residual +/- trial SEM by nominal rate.",
        "fig5_rate_slope_subset_diagnostic": "Units are ranked by absolute full-session rate slope, then nested subsets use the same ridge pipeline. Curves are mean held-out residual +/- trial SEM by rate. Full-session subset selection makes this descriptive and potentially double-dipped.",
        "fig5_ridge_alpha_path": "Fixed alpha values, including OLS, use the same outer folds. Curves are mean held-out residual +/- trial SEM by rate; summaries show mean outer-fold R2 and residual slope. This isolates shrinkage, not squared-error loss behavior.",
        "fig5_steepest_unit_ols": "The unit with largest absolute full-session rate slope is selected descriptively. Panels show raw trials, per-rate mean +/- SEM, the in-sample encoding line, and outer-fold single-unit OLS residuals. Selection is not nested.",
        "fig5_single_unit_slope_examples": "One fixed-seed unit is sampled from each full-session slope bin. Curves are mean held-out single-unit ridge residual +/- trial SEM; bars are residual slopes. Examples are descriptive, not population inference.",
        "fig5_rate_target_family": "Identical folds compare exact Hz, boundary-centered linear evidence, and tanh evidence with scales 2/4/8. Each target is scored in its own units; category-side accuracy and same-trial alignment are separate, so R2 values are not a common biological scale.",
        "fig5_encoding_dimensionality": "PCA is refit inside every outer training fold. X is retained PC count; points are mean ten-fold category accuracy, rate R2, or choice accuracy and bars are fold SEM. This is a variance-dimensionality diagnostic.",
        "fig5_rate_behavior": "At each nominal rate, points are mouse-right and outer-fold ridge-above-12-Hz fractions; binomial SEM uses the trial count in that rate bin. The lower panel converts both to category-side error.",
        "fig5_realized_flash_behavior": "Trials are binned by counted stimulus events before center exit or response. Points are mouse-right and ridge-high fractions with binomial SEM; bins with fewer than five trials are omitted. Post-exit counts are report-period descriptors.",
        "fig5_choice_model_comparison": "Boxes contain ten outer-choice-fold log losses. In each fold the upstream ridge and alpha selection exclude test trials; meta-training readouts use five-fold cross-fitting. Lower is better; fold variation is not animal-level uncertainty.",
        "fig5_within_rate_choice_tests": "Hard alignment and right-minus-left Cohen's d use outer-fold readouts within nominal-rate bins; nulls shuffle choice within rate 100 times. The fourth panel compares ten outer-fold log losses for rate fixed effects with or without the nested readout.",
        "fig5_time_resolved_choice_readout": "Separate fully nested stacks use each non-overlapping 100 ms bin. Blue uses all trials; green retains trials whose center exit is after the bin end. Gray is exit-time IQR and dotted line the median. No time-point multiplicity test is claimed.",
        "fig5_behavior_state_choice_controls": "Ten outer-fold log-loss distributions compare rate fixed effects, nested neural readout, and history/port-timing covariates. Missing numeric state values are mean-imputed inside each training fold; these proxies are not a video movement model.",
    }
    figure_counter = 0

    def img(name: str, alt: str) -> str:
        nonlocal figure_counter
        figure_counter += 1
        path = FIGURE_DIR / f"{name}.png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return (
            f'<figure id="{escape(name)}"><img '
            f'src="data:image/png;base64,{encoded}" alt="{escape(alt)}">'
            f"<figcaption><strong>Figure {figure_counter}.</strong> "
            f"{escape(alt)}. {escape(figure_notes.get(name, ''))}</figcaption></figure>"
        )

    strength_items = "".join(
        "<li>"
        f"|rate - 12| = {escape(str(strength))} Hz: "
        f"peak CV accuracy {vals['peak']:.2f}, n = {vals['n_trials']} trials"
        "</li>"
        for strength, vals in controls["strength_curves"].items()
    )
    strength_counts = "\n\n".join(
        f"|rate - 12| = {strength}\n{table}"
        for strength, table in controls["strength_cell_counts"].items()
    )
    sig_units = q4["significant_interaction_units"]
    if sig_units:
        sig_rows = "".join(
            "<tr>"
            f"<td>{escape(row['unit_id'])}</td>"
            f"<td>{row['interaction_coef']:.2f}</td>"
            f"<td>{row['model_r2']:.2f}</td>"
            f"<td>{row['interaction_q']:.3f}</td>"
            "</tr>"
            for row in sig_units
        )
        sig_table = (
            "<table><caption><strong>Table 9.</strong> FDR-significant single-unit interactions.</caption><thead><tr><th>unit</th><th>interaction coef (sp/s)</th>"
            "<th>model R2</th><th>FDR q</th></tr></thead>"
            f"<tbody>{sig_rows}</tbody></table>"
        )
    else:
        sig_table = "<p>No units passed the FDR threshold for the interaction term.</p>"

    def metric_cell(value) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.2f}"

    mlp_metrics = q5["model_comparison"].get("mlp")
    if mlp_metrics:
        ridge_metrics = q5["model_comparison"]["ridge"]
        mlp_delta_r2 = mlp_metrics["cv_r2_mean"] - ridge_metrics["cv_r2_mean"]
        mlp_delta_resid = (
            mlp_metrics["residual_slope"] - ridge_metrics["residual_slope"]
        )
        if mlp_delta_r2 > 0.02 and mlp_delta_resid > 0.05:
            mlp_takeaway = (
                "This would suggest that the linear ridge readout was missing "
                "nonlinear decodable rate information."
            )
        elif mlp_delta_r2 < -0.02:
            mlp_takeaway = (
                "Because the nonlinear decoder underperforms ridge on held-out R2, "
                "it does not support pursuing this as the main explanation for "
                "compression in this session."
            )
        else:
            mlp_takeaway = (
                "Within this small tanh-MLP family, the nonlinear decoder is similar "
                "to ridge. This narrow control does not show that nonlinearity in "
                "general cannot recover additional rate information."
            )
        mlp_rate_summary = (
            "The MLP is a small nonlinear upper-bound control, not a preferred "
            "biological model. It reached CV R<sup>2</sup> "
            f"{mlp_metrics['cv_r2_mean']:.2f}, calibration slope "
            f"{mlp_metrics['calibration_slope']:.2f}, and residual slope "
            f"{mlp_metrics['residual_slope']:.2f}, compared with ridge CV "
            f"R<sup>2</sup> {ridge_metrics['cv_r2_mean']:.2f}, calibration slope "
            f"{ridge_metrics['calibration_slope']:.2f}, and residual slope "
            f"{ridge_metrics['residual_slope']:.2f}. {mlp_takeaway}"
        )
    else:
        mlp_rate_summary = "No MLP rate-control result was available."

    model_rows = []
    for label, metrics in [
        ("ridge", q5["model_comparison"]["ridge"]),
        ("MLP", q5["model_comparison"].get("mlp", {})),
        ("multiclass collapse", q5["model_comparison"]["multiclass_collapse"]),
    ]:
        model_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{metric_cell(metrics.get('cv_r2_mean'))}</td>"
            f"<td>{metric_cell(metrics.get('exact_rate_acc'))}</td>"
            f"<td>{metric_cell(metrics.get('calibration_slope'))}</td>"
            f"<td>{metric_cell(metrics.get('residual_slope'))}</td>"
            f"<td>{metric_cell(metrics.get('mae_hz'))}</td>"
            f"<td>{metric_cell(metrics.get('category_collapsed_acc'))}</td>"
            "</tr>"
        )
    rate_model_table = (
        "<table><caption><strong>Table 10.</strong> Held-out rate-readout comparison.</caption><thead><tr><th>readout</th><th>CV R2</th><th>exact-rate acc</th>"
        "<th>calib slope</th><th>resid slope</th><th>MAE (Hz)</th>"
        "<th>category acc</th></tr></thead>"
        f"<tbody>{''.join(model_rows)}</tbody></table>"
    )
    slope_subset_rows = []
    for row in q5["slope_subset_diagnostic"]:
        slope_subset_rows.append(
            "<tr>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{row['n_units']}</td>"
            f"<td>{metric_cell(row['cv_r2_mean'])}</td>"
            f"<td>{metric_cell(row['calibration_slope'])}</td>"
            f"<td>{metric_cell(row['residual_slope'])}</td>"
            f"<td>{metric_cell(row['low_side_mean_residual'])}</td>"
            f"<td>{metric_cell(row['high_side_mean_residual'])}</td>"
            "</tr>"
        )
    slope_subset_table = (
        "<table><caption><strong>Table 13.</strong> Rate-slope subset sensitivity.</caption><thead><tr><th>unit subset</th><th>n units</th><th>CV R2</th>"
        "<th>calib slope</th><th>resid slope</th><th>low-side resid</th>"
        "<th>high-side resid</th></tr></thead>"
        f"<tbody>{''.join(slope_subset_rows)}</tbody></table>"
    )
    single_unit_rows = []
    for row in q5["single_unit_slope_examples"]:
        single_unit_rows.append(
            "<tr>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{row['unit_index']}</td>"
            f"<td>{row['rate_slope_sp_s_hz']:.3f}</td>"
            f"<td>{row['abs_rate_slope_sp_s_hz']:.3f}</td>"
            f"<td>{metric_cell(row['cv_r2_mean'])}</td>"
            f"<td>{metric_cell(row['calibration_slope'])}</td>"
            f"<td>{metric_cell(row['residual_slope'])}</td>"
            "</tr>"
        )
    single_unit_table = (
        "<table><caption><strong>Table 16.</strong> Descriptive single-unit slope-bin examples.</caption><thead><tr><th>example bin</th><th>unit index</th>"
        "<th>rate slope</th><th>|rate slope|</th><th>CV R2</th>"
        "<th>calib slope</th><th>resid slope</th></tr></thead>"
        f"<tbody>{''.join(single_unit_rows)}</tbody></table>"
    )

    def params_text(params: dict) -> str:
        if not params:
            return "n/a"
        return ", ".join(
            f"{escape(key.split('__')[-1])}={escape(format_param_value(value))}"
            for key, value in params.items()
        )

    def counts_text(counts: dict) -> str:
        if not counts:
            return "n/a"
        parts = []
        for key, vals in counts.items():
            items = ", ".join(f"{escape(val)} x{n}" for val, n in vals.items())
            parts.append(f"{escape(key.split('__')[-1])}: {items}")
        return "; ".join(parts)

    def edges_text(edges: dict) -> str:
        hits = []
        for key, vals in edges.items():
            short = key.split("__")[-1]
            if vals["hit_min"]:
                hits.append(f"{short} min")
            if vals["hit_max"]:
                hits.append(f"{short} max")
        return ", ".join(hits) if hits else "none"

    tuning_rows = []
    for label, tuning in q5["hyperparameter_tuning"].items():
        tuning_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{params_text(tuning['final_refit'])}</td>"
            f"<td>{counts_text(tuning['selected_param_counts'])}</td>"
            f"<td>{escape(edges_text(tuning['edge_hits']))}</td>"
            "</tr>"
        )
    tuning_table = (
        "<table><caption><strong>Table 11.</strong> Hyperparameter selections and grid-edge checks.</caption><thead><tr><th>readout</th><th>final refit params</th>"
        "<th>outer-fold selected params</th><th>grid edge hits</th></tr></thead>"
        f"<tbody>{''.join(tuning_rows)}</tbody></table>"
    )
    alpha_path = q5.get("ridge_alpha_path", [])
    if alpha_path:
        ridge_alpha_rows = [row for row in alpha_path if row.get("alpha") is not None]
        ols_alpha = next((row for row in alpha_path if row.get("alpha") is None), None)
        best_alpha = max(ridge_alpha_rows, key=lambda row: row["cv_r2_mean"])
        selected_alpha = next(
            (row for row in alpha_path if row.get("is_selected_main_alpha")), None
        )
        ols_text = ""
        if ols_alpha is not None:
            ols_text = (
                f" The OLS endpoint had CV R<sup>2</sup> "
                f"{ols_alpha['cv_r2_mean']:.2f} and calibration slope "
                f"{ols_alpha['calibration_slope']:.2f}."
            )
        if selected_alpha is None:
            alpha_path_summary = (
                f"Best held-out R2 in the fixed-alpha sweep was alpha "
                f"{best_alpha['alpha']:.3g}.{ols_text}"
            )
        else:
            alpha_path_summary = (
                f"The nested-CV-selected alpha was {selected_alpha['alpha']:.3g}; "
                f"the best fixed-alpha R2 in this sweep was at alpha "
                f"{best_alpha['alpha']:.3g}.{ols_text}"
            )
        alpha_rows = []
        for row in alpha_path:
            alpha_label = "OLS" if row.get("alpha") is None else f"{row['alpha']:.3g}"
            alpha_rows.append(
                "<tr>"
                f"<td>{alpha_label}</td>"
                f"<td>{'main' if row.get('is_selected_main_alpha') else ''}</td>"
                f"<td>{metric_cell(row['cv_r2_mean'])}</td>"
                f"<td>{metric_cell(row['calibration_slope'])}</td>"
                f"<td>{metric_cell(row['residual_slope'])}</td>"
                f"<td>{metric_cell(row['mae_hz'])}</td>"
                f"<td>{metric_cell(row['category_collapsed_acc'])}</td>"
                "</tr>"
            )
        alpha_path_table = (
            "<table><caption><strong>Table 14.</strong> Fixed-alpha and OLS sensitivity.</caption><thead><tr><th>alpha</th><th>selected</th><th>CV R2</th>"
            "<th>calib slope</th><th>resid slope</th><th>MAE (Hz)</th>"
            "<th>category acc</th></tr></thead>"
            f"<tbody>{''.join(alpha_rows)}</tbody></table>"
        )
    else:
        alpha_path_summary = "No fixed-alpha ridge sweep was available."
        alpha_path_table = "<p>No fixed-alpha ridge sweep was available.</p>"
    steepest = q5.get("steepest_unit_ols")
    if steepest:
        steepest_metrics = steepest["decoder_metrics"]
        steepest_unit_interpretation = (
            f"The steepest unit had a raw rate slope of "
            f"{steepest['rate_slope_sp_s_hz']:.2f} sp/s/Hz, but the held-out "
            f"single-unit OLS decoder still had CV R<sup>2</sup> "
            f"{steepest_metrics['cv_r2_mean']:.2f}, calibration slope "
            f"{steepest_metrics['calibration_slope']:.2f}, and residual slope "
            f"{steepest_metrics['residual_slope']:.2f}."
        )
        steepest_unit_table = (
            "<table><caption><strong>Table 15.</strong> Steepest-unit descriptive diagnostic.</caption><thead><tr><th>unit index</th><th>rate slope</th>"
            "<th>encoding R2</th><th>decoder CV R2</th><th>decoder calib slope</th>"
            "<th>decoder resid slope</th></tr></thead><tbody><tr>"
            f"<td>{steepest['unit_index']}</td>"
            f"<td>{steepest['rate_slope_sp_s_hz']:.3f}</td>"
            f"<td>{metric_cell(steepest['encoding_r2'])}</td>"
            f"<td>{metric_cell(steepest_metrics['cv_r2_mean'])}</td>"
            f"<td>{metric_cell(steepest_metrics['calibration_slope'])}</td>"
            f"<td>{metric_cell(steepest_metrics['residual_slope'])}</td>"
            "</tr></tbody></table>"
        )
    else:
        steepest_unit_interpretation = "No steepest-unit OLS diagnostic was available."
        steepest_unit_table = "<p>No steepest-unit OLS diagnostic was available.</p>"
    choice_resid_rate = q5["choice_residualized_rate"]["metrics"]
    rate_self_resid = q5["rate_self_residualized_control"]["metrics"]
    choice_resid_rate_rows = []
    for label, metrics in [
        ("raw ridge", q5["model_comparison"]["ridge"]),
        ("choice-residualized ridge", choice_resid_rate),
        ("rate-self residualized", rate_self_resid),
    ]:
        choice_resid_rate_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{metric_cell(metrics['cv_r2_mean'])}</td>"
            f"<td>{metric_cell(metrics['calibration_slope'])}</td>"
            f"<td>{metric_cell(metrics['residual_slope'])}</td>"
            f"<td>{metric_cell(metrics['mae_hz'])}</td>"
            f"<td>{metric_cell(metrics['category_collapsed_acc'])}</td>"
            "</tr>"
        )
    choice_resid_rate_table = (
        "<table><caption><strong>Table 12.</strong> Choice-residualized and self-residualized rate controls.</caption><thead><tr><th>readout</th><th>CV R2</th>"
        "<th>calib slope</th><th>resid slope</th><th>MAE (Hz)</th>"
        "<th>category acc</th></tr></thead>"
        f"<tbody>{''.join(choice_resid_rate_rows)}</tbody></table>"
    )
    choice_resid_rate_summary = (
        f"Choice-residualized ridge had CV R<sup>2</sup> "
        f"{choice_resid_rate['cv_r2_mean']:.2f} and residual slope "
        f"{choice_resid_rate['residual_slope']:.2f}, compared with raw ridge "
        f"CV R<sup>2</sup> {q5['real_r2_mean']:.2f} and residual slope "
        f"{q5['residual_slope']:.2f}. The rate-self residualization control had "
        f"CV R<sup>2</sup> {rate_self_resid['cv_r2_mean']:.2f}."
    )
    target_rows = []
    for row in q5["rate_target_family"]:
        metrics = row["metrics"]
        target_rows.append(
            "<tr>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{metric_cell(metrics['cv_r2_mean'])}</td>"
            f"<td>{metric_cell(metrics['category_collapsed_acc'])}</td>"
            f"<td>{metric_cell(metrics['same_trial_choice_alignment'])}</td>"
            f"<td>{metric_cell(metrics['right_minus_left_d'])}</td>"
            f"<td>{metric_cell(metrics['mae_evidence'])}</td>"
            "</tr>"
        )
    target_family_table = (
        "<table><caption><strong>Table 17.</strong> Alternative rate/evidence targets.</caption><thead><tr><th>target</th><th>CV R2</th>"
        "<th>category acc</th><th>same-trial choice align</th>"
        "<th>right-left d</th><th>MAE evidence</th></tr></thead>"
        f"<tbody>{''.join(target_rows)}</tbody></table>"
    )
    dim = q5["encoding_dimensionality"]
    dim_rate_best = max(dim["curves"], key=lambda row: row["rate_r2_mean"])
    dim_choice_best = max(dim["curves"], key=lambda row: row["choice_acc_mean"])
    choice_rows = []
    for row in q5["choice_model_comparison"]:
        choice_rows.append(
            "<tr>"
            f"<td>{escape(row['model'])}</td>"
            f"<td>{row['log_loss']:.3f}</td>"
            f"<td>{row['delta_log_loss_vs_bias']:.3f}</td>"
            f"<td>{row['pseudo_r2_vs_bias']:.3f}</td>"
            f"<td>{row['balanced_accuracy']:.2f}</td>"
            "</tr>"
        )
    choice_model_table = (
        "<table><caption><strong>Table 18.</strong> Fully nested choice-model comparison.</caption><thead><tr><th>choice model</th><th>held-out log loss</th>"
        "<th>gain vs bias</th><th>pseudo R2 vs bias</th><th>balanced acc</th></tr></thead>"
        f"<tbody>{''.join(choice_rows)}</tbody></table>"
    )
    best_choice_model = min(
        q5["choice_model_comparison"], key=lambda row: row["log_loss"]
    )
    within_tests = q5["within_rate_choice_tests"]
    within_rows = []
    for label in ["nested ridge"]:
        hard = within_tests["hard_alignment_summary"][label]
        cont = within_tests["continuous_readout_summary"][label]
        within_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{hard['rate_balanced_alignment']:.2f}</td>"
            f"<td>{hard['mean_expected_alignment']:.2f}</td>"
            f"<td>{hard['mean_excess_alignment']:.2f}</td>"
            f"<td>{hard['shuffle_mean']:.2f}</td>"
            f"<td>{hard['shuffle_p']:.3f}</td>"
            f"<td>{cont['mean_right_minus_left_d']:.2f}</td>"
            f"<td>{cont['shuffle_mean']:.2f}</td>"
            f"<td>{cont['shuffle_p']:.3f}</td>"
            "</tr>"
        )
    within_rate_table = (
        "<table><caption><strong>Table 19.</strong> Within-rate alignment tests.</caption><thead><tr><th>readout</th><th>same-trial alignment</th>"
        "<th>expected from rate bin</th><th>excess</th><th>shuffle</th><th>p</th>"
        "<th>right-left d</th><th>shuffle</th>"
        f"<th>p</th></tr></thead><tbody>{''.join(within_rows)}</tbody></table>"
    )
    fixed_rows = []
    for row in within_tests["rate_fixed_choice_models"]:
        fixed_rows.append(
            "<tr>"
            f"<td>{escape(row['model'])}</td>"
            f"<td>{row['log_loss']:.3f}</td>"
            f"<td>{row['delta_log_loss_vs_rate_fe']:.3f}</td>"
            f"<td>{row['pseudo_r2_vs_rate_fe']:.3f}</td>"
            f"<td>{row['balanced_accuracy']:.2f}</td>"
            "</tr>"
        )
    fixed_choice_table = (
        "<table><caption><strong>Table 20.</strong> Rate-fixed fully nested choice models.</caption><thead><tr><th>choice model</th><th>held-out log loss</th>"
        "<th>gain vs rate FE</th><th>pseudo R2 vs rate FE</th>"
        f"<th>balanced acc</th></tr></thead><tbody>{''.join(fixed_rows)}</tbody></table>"
    )
    best_fixed_model = min(
        within_tests["rate_fixed_choice_models"], key=lambda row: row["log_loss"]
    )
    time_choice = q5["time_resolved_choice"]
    best_time_bin = max(
        time_choice["bins"], key=lambda row: row["delta_log_loss_vs_rate_fe"]
    )
    pre_exit_bins = [
        row
        for row in time_choice["pre_exit_bins"]
        if row["delta_log_loss_vs_rate_fe"] is not None
    ]
    best_pre_exit_bin = (
        max(pre_exit_bins, key=lambda row: row["delta_log_loss_vs_rate_fe"])
        if pre_exit_bins
        else None
    )
    exit_latency = time_choice["center_exit_latency"]
    if exit_latency["median_s"] is None:
        exit_sentence = "Center-exit timing was unavailable for this session."
    else:
        exit_sentence = (
            "Center-port exit occurred at median "
            f"{exit_latency['median_s'] * 1000:.0f} ms after first stimulus onset "
            f"(IQR {exit_latency['p25_s'] * 1000:.0f}-"
            f"{exit_latency['p75_s'] * 1000:.0f} ms)."
        )
    if best_pre_exit_bin is None:
        pre_exit_sentence = (
            "No pre-exit bin had enough trials for the censored control."
        )
    else:
        pre_exit_sentence = (
            "In the pre-exit-only control, the largest usable gain was "
            f"{best_pre_exit_bin['delta_log_loss_vs_rate_fe']:.3f} at "
            f"{best_pre_exit_bin['bin_center_ms']:.0f} ms "
            f"(n = {best_pre_exit_bin['n_trials']} trials)."
        )
    time_rows = []
    for row, pre_row in zip(time_choice["bins"], time_choice["pre_exit_bins"]):
        pre_gain = pre_row["delta_log_loss_vs_rate_fe"]
        pre_gain_text = "n/a" if pre_gain is None else f"{pre_gain:.3f}"
        time_rows.append(
            "<tr>"
            f"<td>{row['bin_center_ms']:.0f}</td>"
            f"<td>{row['ridge_rate_cv_r2']:.2f}</td>"
            f"<td>{row['delta_log_loss_vs_rate_fe']:.3f}</td>"
            f"<td>{pre_row['n_trials']}</td>"
            f"<td>{pre_gain_text}</td>"
            f"<td>{row['pseudo_r2_vs_rate_fe']:.3f}</td>"
            f"<td>{row['balanced_accuracy']:.2f}</td>"
            "</tr>"
        )
    time_choice_table = (
        "<table><caption><strong>Table 21.</strong> Time-resolved nested readout and pre-exit censoring.</caption><thead><tr><th>bin center (ms)</th><th>rate CV R2</th>"
        "<th>all-trial gain vs rate FE</th><th>pre-exit n</th>"
        "<th>pre-exit gain vs rate FE</th>"
        "<th>pseudo R2 vs rate FE</th><th>balanced acc</th></tr></thead>"
        f"<tbody>{''.join(time_rows)}</tbody></table>"
    )
    state_choice = q5["behavior_state_choice_controls"]
    if state_choice["models"]:
        state_feature_text = ", ".join(state_choice["state_feature_names"])
        state_rows = []
        for row in state_choice["models"]:
            state_rows.append(
                "<tr>"
                f"<td>{escape(row['model'])}</td>"
                f"<td>{row['log_loss']:.3f}</td>"
                f"<td>{row['delta_log_loss_vs_rate_fe']:.3f}</td>"
                f"<td>{row['pseudo_r2_vs_rate_fe']:.3f}</td>"
                f"<td>{row['balanced_accuracy']:.2f}</td>"
                "</tr>"
            )
        state_choice_table = (
            "<table><caption><strong>Table 22.</strong> State/history proxy sensitivity.</caption><thead><tr><th>choice model</th><th>held-out log loss</th>"
            "<th>gain vs rate FE</th><th>pseudo R2 vs rate FE</th>"
            f"<th>balanced acc</th></tr></thead><tbody>{''.join(state_rows)}</tbody>"
            "</table>"
        )
        state_gain_sentence = (
            "Adding the fully nested ridge readout on top of rate plus "
            "state/history proxies "
            f"improved log loss by {state_choice['ridge_gain_after_rate_state']:.3f}."
        )
    else:
        state_feature_text = "none available"
        state_choice_table = (
            "<p>No non-constant state/history proxy features were available.</p>"
        )
        state_gain_sentence = "No state/history proxy model was fit."

    time_aware = results["time_aware_cv"]
    motion = meta["motion_cache_audit"]
    provenance = meta["provenance"]
    fixed_by_name = {
        row["model"]: row for row in within_tests["rate_fixed_choice_models"]
    }
    nested_gain = fixed_by_name["rate FE + nested ridge"]["delta_log_loss_vs_rate_fe"]
    if nested_gain > 0.01:
        nested_choice_sentence = (
            "The fully nested neural readout adds a modest amount of same-trial "
            "choice/report information beyond rate fixed effects."
        )
    elif nested_gain > 0:
        nested_choice_sentence = (
            "The fully nested neural readout adds only a very small amount of "
            "same-trial choice/report information beyond rate fixed effects."
        )
    else:
        nested_choice_sentence = (
            "The fully nested neural readout does not improve choice/report "
            "prediction beyond rate fixed effects in this session."
        )
    time_aware_rows = "".join(
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{vals['random_mean']:.3f} +/- {np.std(vals['random_folds'], ddof=1):.3f}</td>"
        f"<td>{vals['blocked_mean']:.3f} +/- {np.std(vals['blocked_folds'], ddof=1):.3f}</td>"
        f"<td>{vals['blocked_minus_random']:+.3f}</td>"
        "</tr>"
        for label, vals in [
            ("category balanced accuracy", time_aware["category"]),
            ("choice/report balanced accuracy", time_aware["choice"]),
            ("rate CV R2", time_aware["rate"]),
        ]
    )
    time_aware_table = (
        "<table><caption><strong>Table 4.</strong> Random versus contiguous-fold "
        "held-out performance; cells are fold mean +/- SD.</caption><thead><tr><th>readout</th><th>random folds</th>"
        "<th>contiguous blocks</th><th>blocked - random</th></tr></thead>"
        f"<tbody>{time_aware_rows}</tbody></table>"
    )
    motion_coverage = (
        f"{motion.get('n_finite', 0)}/{motion['n_analysis_trials']} "
        f"({motion.get('coverage_fraction', 0):.1%})"
    )
    software_rows = "".join(
        f"<tr><td>{escape(package)}</td><td>{escape(value)}</td></tr>"
        for package, value in provenance["software_versions"].items()
    )
    artifact_rows = "".join(
        "<tr>"
        f"<td>{escape(label)}</td><td><code>{escape(item['path'])}</code></td>"
        f"<td>{'yes' if item['exists'] else 'no'}</td>"
        f"<td>{escape(item['modified_at_utc'] or 'n/a')}</td>"
        "</tr>"
        for label, item in meta["artifact_inventory"].items()
    )
    blocked_fold_rows = "".join(
        "<tr>"
        f"<td>{cat_fold['fold'] + 1}</td>"
        f"<td>{cat_fold['test_index_min']}-{cat_fold['test_index_max']}</td>"
        f"<td>{cat_fold['n_test']}</td>"
        f"<td>{cat_fold['test_class_counts'].get('0', 0)} / "
        f"{cat_fold['test_class_counts'].get('1', 0)}</td>"
        f"<td>{choice_fold['test_class_counts'].get('0', 0)} / "
        f"{choice_fold['test_class_counts'].get('1', 0)}</td>"
        "</tr>"
        for cat_fold, choice_fold in zip(
            time_aware.get("blocked_folds", []),
            time_aware.get("choice_blocked_folds", []),
        )
    )

    def selected_alpha_text(search_infos):
        counts = {}
        for info in search_infos:
            alpha = info.get("best_params", {}).get("ridge__alpha")
            label = "n/a" if alpha is None else f"{float(alpha):.4g}"
            counts[label] = counts.get(label, 0) + 1
        return ", ".join(f"{label} x{count}" for label, count in counts.items())

    random_blocked_alpha_sentence = (
        "Selected ridge alpha by outer fold: random folds "
        f"{selected_alpha_text(time_aware['rate_random_search'])}; contiguous blocks "
        f"{selected_alpha_text(time_aware['rate_blocked_search'])}."
    )
    review_rows = f"""
<tr><td>1. Squared-error wording</td><td>Confirmed and corrected</td>
<td>Removed the exclusion claim; OLS and this MLP still use squared error.</td>
<td>None.</td><td>Now restricted to compressed held-out rate predictions and evidence against ridge shrinkage.</td>
<td>A loss-aware/noise-aware encoding analysis would be needed for a stronger claim.</td></tr>
<tr><td>2. Rate-readout to choice nesting</td><td>Confirmed and corrected</td>
<td>Outer choice folds now refit/tune ridge; meta-training readouts are inner cross-fitted.</td>
<td>Gain over rate fixed effects is {nested_gain:.3f}; the prior non-nested value was 0.016.</td>
<td>The small gain survives, but remains preliminary and mainly post-exit.</td>
<td>Replicate across sessions and add directional movement features.</td></tr>
<tr><td>3. Scope of the MLP control</td><td>Confirmed and corrected</td>
<td>Recast as a bounded tanh-MLP control; both selected hyperparameters hit upper grid edges.</td>
<td>None.</td><td>No statement about nonlinear decoders in general.</td>
<td>A broader nonlinear search is optional, not required before sharing.</td></tr>
<tr><td>4. Slow-drift vulnerability</td><td>Confirmed and corrected</td>
<td>Added ten contiguous held-out trial blocks and reran the full analysis.</td>
<td>Category {time_aware["category"]["random_mean"]:.3f} to {time_aware["category"]["blocked_mean"]:.3f}; choice/report {time_aware["choice"]["random_mean"]:.3f} to {time_aware["choice"]["blocked_mean"]:.3f}; rate R2 {time_aware["rate"]["random_mean"]:.3f} to {time_aware["rate"]["blocked_mean"]:.3f}.</td>
<td>Slow drift sampled across random folds is not a sufficient explanation for the three readouts.</td>
<td>Blocked CV does not remove all temporal confounding or test forward prediction.</td></tr>
<tr><td>5. Movement confound</td><td>Confirmed but not yet corrected</td>
<td>Recovered and audited scalar motion energy ({motion_coverage}); did not present it as directional control.</td>
<td>No validated movement-residual choice estimate is reported.</td>
<td>Late decoding is called choice/report-related, not a pure decision signal.</td>
<td>Extract directional pre-/peri-/post-exit video features and nest them in the choice analysis.</td></tr>
<tr><td>6. Artifact consolidation</td><td>Partially confirmed</td>
<td>Made this embedded, self-contained HTML the reader artifact and audited stale files.</td>
<td>None.</td><td>The older PowerPoint must not be used for final values or wording.</td>
<td>Update/delete the deck and split the large script only if this analysis remains active.</td></tr>
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Preliminary analysis of stimulus and choice readouts in GRB006 V1</title>
<style>
body {{ max-width: 980px; margin: 2rem auto; padding: 0 1rem; font: 16px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; color: #1f2328; }}
h1, h2, h3 {{ line-height: 1.2; }}
figure {{ margin: 1.5rem 0; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
figcaption {{ color: #555; font-size: 0.9rem; margin-top: 0.35rem; }}
pre {{ background: #f6f8fa; padding: 0.8rem; overflow-x: auto; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; font-size: 0.92rem; }}
caption {{ text-align: left; color: #444; margin-bottom: 0.35rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.55rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
summary {{ cursor: pointer; font-weight: 700; margin: 1rem 0 0.5rem; }}
.callout {{ border-left: 4px solid #57606a; background: #f6f8fa; padding: 0.8rem 1rem; margin: 1rem 0; }}
.answer {{ font-weight: 700; }}
.note {{ color: #555; }}
.label {{ display: inline-block; min-width: 6.2rem; font-weight: 700; color: #34495e; }}
.status {{ font-weight: 700; white-space: nowrap; }}
</style>
</head>
<body>
<h1>Preliminary analysis of stimulus and choice readouts in GRB006 V1</h1>
<p class="note">One recording session: {meta["subject"]} <code>{meta["session"]}</code>;
{meta["n_units"]} well-isolated neurons; {meta["n_trials"]} non-boundary choice trials.<br>
Data acquisition identifier: <code>{meta["session"]}</code>. Full analysis and report
generation: <code>{escape(provenance["generated_at_utc"])}</code>.</p>

<nav class="callout"><strong>Contents</strong>
<ol>
<li><a href="#executive-summary">Executive summary</a></li>
<li><a href="#review-resolution">Review resolution</a></li>
<li><a href="#questions">Scientific questions</a></li>
<li><a href="#data-methods">Data, inclusion, preprocessing, and methods</a></li>
<li><a href="#category-choice">Category and choice/report results</a></li>
<li><a href="#timing-robustness">Timing and robustness</a></li>
<li><a href="#rate-results">Rate decoding and held-out compression</a></li>
<li><a href="#nested-choice">Fully nested rate-readout-to-choice result</a></li>
<li><a href="#movement">Movement-control status</a></li>
<li><a href="#interpretation">Interpretation and limitations</a></li>
<li><a href="#next-steps">Next steps</a></li>
<li><a href="#technical-appendix">Technical appendix and code-path map</a></li>
</ol></nav>

<h2 id="executive-summary">Executive summary</h2>
<div class="callout">
<p><span class="label">Result</span> I am treating this as a preliminary one-session analysis. The main result is that
V1 contains readable sensory category/rate information and strong choice/report-related
activity. The category signal is not simply a choice decoder. The rate code is
readable but compressed; removing ridge shrinkage does not recover a well-calibrated
held-out rate estimate. The animal's choices are not a simple direct readout of the
V1 rate axis.</p>
<p><span class="label">Interpretation</span> V1 makes task-relevant sensory information
available, but the late choice/report signal and the mismatch between sensory readout
and behavior require movement, state, feedback, and downstream-readout explanations
to remain open.</p>
<p><span class="label">Caveat</span> This is one animal and one session. The 1 s population
window extends beyond median center-port exit, and the available video-motion measure
does not distinguish leftward from rightward report movement.</p>
</div>
<ul>
<li><strong>Stimulus information is clear in this session.</strong> Category decoding was
{q1["real_mean"]:.2f} vs {q1["shuffle_mean"]:.2f} shuffle, and continuous rate
prediction reached CV R<sup>2</sup> {q5["real_r2_mean"]:.2f}.</li>
<li><strong>Choice/report information is also strong.</strong> Choice decoded at
{q2["real_mean"]:.2f} vs {q2["shuffle_mean"]:.2f} shuffle in the 1 s post-stimulus
window.</li>
<li><strong>The category-choice confound does not explain everything.</strong>
category remained decodable after removing choice ({q3["cat_resid_mean"]:.2f}), and
choice remained decodable after removing category ({q3["cho_resid_mean"]:.2f}).</li>
<li><strong>The rate compression looks real in this readout.</strong> OLS/tiny-alpha ridge
reduces the compression only modestly and generalizes poorly, and the steepest
single-unit control is also compressed.</li>
<li><strong>The main tension is the relationship to behavior.</strong> The V1 rate readout is
better calibrated to stimulus side than the mouse's behavior, so behavior is not
explained by sensory availability alone.</li>
<li><strong>My current interpretation:</strong> V1 makes task-relevant sensory information
available, while choice-related activity likely mixes sensory readout, movement/report
state, and/or downstream decision feedback.</li>
<li><strong>Next analyses:</strong> replicate across sessions/animals and add stronger
state/movement controls, ideally with an explicit gain or state model.</li>
</ul>

<p class="note">All results are preliminary and from a single mouse/session. Decoder
performance is held-out cross-validated accuracy after repeated class-balanced
subsampling; boxplots summarize resampling or shuffled-label distributions from this
session, not uncertainty across animals.</p>

<h2 id="review-resolution">Review resolution</h2>
<p>Each issue below was checked against the active code path, current outputs, and
recoverable motion artifacts. “Corrected” means the revised analysis or wording is in
this report; it does not imply that every biological confound is resolved.</p>
<table><caption><strong>Table 1.</strong> Review resolution.</caption><thead><tr>
<th>Issue</th><th>Verification status</th><th>Action taken</th>
<th>Effect on result</th><th>Effect on interpretation</th><th>Remaining work</th>
</tr></thead><tbody>{review_rows}</tbody></table>

<h2 id="questions">Scientific questions and analysis goals</h2>
<ol>
<li>Is low-versus-high stimulus category readable from GRB006 V1?</li>
<li>Is left-versus-right choice/report readable, and does either readout survive
fold-safe linear removal of the other variable?</li>
<li>When do category and choice/report information emerge relative to center-port exit?</li>
<li>How accurately does V1 predict nominal flash rate, and how compressed are held-out
predictions toward the center of the rate range?</li>
<li>Does a fully nested neural rate readout add same-trial choice prediction beyond
nominal rate, and is that gain present before report movement?</li>
<li>Do the principal readouts survive contiguous-trial testing that is harder for slow
session drift to exploit?</li>
</ol>

<h2 id="data-methods">Data, inclusion, preprocessing, and methods</h2>
<p><strong>Population matrix.</strong> For trial <em>t</em> and unit <em>u</em>,
<code>X(t,u) = spikes in [first stimulus, first stimulus + 1 s) / 1 s</code>.
The primary dataset contains {meta["n_trials"]} non-boundary choice trials and
{meta["n_units"]} units from {meta["subject"]} {meta["session"]}. Category is low-rate
versus high-rate; choice/report is left versus right; continuous rate is the nominal
visual flash rate in Hz.</p>
<table><caption><strong>Table 2.</strong> Inclusion flow and inferential level.</caption>
<thead><tr><th>Stage</th><th>n</th><th>Rule / role</th></tr></thead><tbody>
<tr><td>Good units</td><td>{meta["n_units"]}</td><td><code>unit_criteria_id={meta["unit_criteria_id"]}</code></td></tr>
<tr><td>Aligned trials after mismatch truncation</td><td>{meta["n_aligned_trials_after_mismatch_truncation"]}</td><td>OBX 505 and Chipmunk 504; truncate to 504</td></tr>
<tr><td>Excluded without a valid choice</td><td>{meta["n_excluded_without_choice"]}</td><td><code>with_choice != 1</code></td></tr>
<tr><td>All valid-choice trials</td><td>{meta["n_choice_trials"]}</td><td>Used for boundary accounting</td></tr>
<tr><td>Boundary choice trials</td><td>{meta["n_boundary_choice_trials"]}</td><td>12 Hz; used only in boundary-transfer analyses</td></tr>
<tr><td>Primary non-boundary choice trials</td><td>{meta["n_trials"]}</td><td>Category, choice, and rate analyses</td></tr>
</tbody></table>
<p><strong>Statistical hierarchy.</strong> Predictive scores are trial-held-out within
one session. Single-unit interaction tests use neurons within that same session.
Resamples, folds, trials, and neurons are not independent animals; none of the
within-session dispersion estimates support population-level inference across mice.</p>
<p><strong>Classification.</strong> Headline binary accuracy repeatedly undersamples the
majority class and uses stratified ten-fold held-out prediction. The score is balanced
accuracy, <code>BA = (sensitivity + specificity) / 2</code>. Shuffled-label fits supply
within-session null distributions; the one-sided finite-null p value is
<code>(1 + number(null >= observed)) / (1 + number(null))</code>. With 100 shuffles,
the minimum is 1/101. These p values are not animal-level inference. Cohen's d is the
difference in group means divided by the pooled within-group standard deviation.</p>
<p><strong>Rate regression.</strong> Ridge fits standardized neural features by minimizing
<code>sum_t (r_t - beta_0 - X_t beta)^2 + alpha ||beta||^2</code>; alpha is selected
inside each training fold. Calibration is summarized by
<code>predicted rate = intercept + slope x true rate</code>; prediction residual is
<code>predicted - true</code>. Outer-fold
<code>R2_fold = 1 - sum_test((r-r_hat)^2) / sum_test((r-mean_test(r))^2)</code>;
the primary value is the unweighted mean over ten folds. MAE and RMSE are computed
over concatenated out-of-fold predictions and are in Hz.</p>
<p><strong>Residualization.</strong> For nuisance variable <code>z</code>, a linear model
<code>X = A z + error</code> is fit on each training fold only, then the fitted nuisance
component is removed from both training and test neural data before decoding.</p>
<p><strong>Fully nested stacked choice model.</strong> Within each outer choice fold,
the neural-to-rate ridge model and alpha selection see only outer-training trials.
Its training-fold readouts are cross-fitted again before the logistic choice model is
fit. The outer choice-test trials are touched only once, for final scoring.</p>
<p><strong>Choice-model score.</strong> Binary log loss is the trial mean
<code>-y log(p) - (1-y) log(1-p)</code>; lower is better. “Gain” is baseline log loss
minus augmented-model log loss on the concatenated outer-fold predictions. Pseudo-R2
is <code>1 - augmented loss / baseline loss</code>. No independent-session standard
error or hypothesis test is attached to this small difference.</p>
<p><strong>Chronological stress test.</strong> The same trial order is split into ten
contiguous held-out blocks. This does not model every form of nonstationarity, but it
directly tests whether shuffled folds hide slow session drift.</p>
<p><strong>Preprocessing order and missing values.</strong> Spike samples are converted
to seconds using the recording sampling rate, counted in half-open time windows, and
divided by window duration to obtain sp/s. No trial or unit outlier trimming is
applied after the stated inclusion rules. Standardization, PCA, nuisance regression,
state-value imputation, model fitting, and hyperparameter selection are fit on
training data only in the analyses where they are used. The six state/history
features may contain missing values; outer-training means are used by
<code>SimpleImputer</code>. The primary neural matrix contains no missing values.</p>
<p><strong>Randomness and model grids.</strong> The common seed is
<code>{RANDOM_STATE}</code>. Ridge alpha spans <code>10^-4</code> through
<code>10^6</code> in 21 logarithmic steps. The bounded MLP uses tanh activations,
LBFGS, hidden layers (4), (8), or (16), L2 alpha <code>10^-4</code> through
<code>10^2</code>, target standardization, and no early stopping. The binary headline
decoder uses L2 logistic regression with the scikit-learn default ten-C grid and an
inner ten-fold search. Per-time-bin descriptive decoders use fixed <code>C=1</code>
to avoid a separate hyperparameter search at every time point.</p>

<h3>Reproducibility and provenance</h3>
<ul>
<li>Live tables: <code>labdata.schema.Dataset</code>, <code>EphysRecording</code>,
<code>SpikeSorting.Unit</code>, and <code>UnitCount.Unit</code>.</li>
<li>Trial/event alignment: <code>src/utils/io_digital_events.py</code> and
<code>src/utils/io_chipmunk_trials.py</code>. A known 505-versus-504 OBX/Chipmunk
trial mismatch is explicitly truncated to 504 before filtering.</li>
<li>Analysis source: <code>scripts/analyses/category_choice_decoding.py</code>, SHA-256
<code>{provenance["script_sha256"]}</code>.</li>
<li>Git state at execution: branch <code>{escape(provenance["git_branch"])}</code>,
commit <code>{escape(provenance["git_commit"])}</code>, working tree
{"dirty" if provenance["working_tree_dirty"] else "clean"}.</li>
<li>Run: {escape(meta["run_mode"])} mode at
<code>{escape(provenance["generated_at_utc"])}</code>. Only a full-mode report is
appropriate for circulation.</li>
</ul>
<table><caption><strong>Table 3.</strong> Software versions used for this run.</caption>
<thead><tr><th>Package</th><th>Version</th></tr></thead><tbody>{software_rows}</tbody></table>

<h2 id="category-choice">1. V1 carries category and choice/report signals</h2>
<p>This first pair of figures establishes the positive controls: the population
contains enough information to read out both the sensory category and the animal's
eventual report. Because category and choice are behaviorally coupled in this
session, these raw decoders demonstrate information content but do not yet determine
whether the signals are separable.</p>
<p class="note">Choice alone predicts category {meta["confound_acc"]:.0%} of the time
in these trials.</p>
<pre>{escape(meta["contingency"])}</pre>

{img("fig1_category_decoding", "train: non-boundary category; test: held-out trials")}
<p><span class="label">Direct result</span> Category is clearly readable. Cross-validated category
accuracy was {q1["real_mean"]:.2f} vs shuffle {q1["shuffle_mean"]:.2f}
(p = {q1["p_perm"]:.3f}). Decoder weights align with single-unit category
discriminability (r = {q1["r_tuning_weight"]:.2f}), which is consistent with the
population decoder using units in the expected direction. Because both quantities use
the same session, this correlation is descriptive rather than an independent validation.</p>

{img("fig2_choice_decoding", "train: non-boundary choice; test: held-out trials")}
<p><span class="label">Direct result</span> Choice/report is even more readable. Choice decoded at
{q2["real_mean"]:.2f} vs shuffle {q2["shuffle_mean"]:.2f}
(p = {q2["p_perm"]:.3f}), and decoder weights align with single-unit choice
discriminability (r = {q2["r_tuning_weight"]:.2f}). Because the 1 s window includes
the report movement, this should be called choice/report-related activity rather than
a pure decision variable.</p>
<p><span class="label">Caveat</span> Neither raw classifier separates sensory,
movement, internal-state, and feedback contributions. Both scores are within-session
information measures.</p>

<h2>2. Category and choice are partly separable</h2>
{img("fig3_residual_decoding", "train/test: held-out trials after fold-safe nuisance removal")}
<p>This is the key confound-control section. If category decoding reflected choice
variance alone, removing choice-related variance should destroy the category readout.
If choice decoding reflected stimulus-category variance alone, removing
category-related variance should destroy the choice readout. Neither happens.</p>
<p><span class="label">Direct result</span> Category after removing choice: {q3["cat_resid_mean"]:.2f} vs shuffle
{q3["cat_resid_shuffle_mean"]:.2f} (p = {q3["cat_resid_p"]:.3f}).
Choice after removing category: {q3["cho_resid_mean"]:.2f} vs shuffle
{q3["cho_resid_shuffle_mean"]:.2f} (p = {q3["cho_resid_p"]:.3f}). A drop from raw
to residualized decoding indicates shared category-choice variance; residual decoding
above shuffle indicates separable information remains.</p>
<p><span class="label">Caveat</span> Linear residualization removes only the component
captured by that fold-specific linear nuisance model. It does not remove all correlated
choice, motion, or state variance.</p>

<details><summary>Sanity check: residualization removes the signal it is told to remove</summary>
{img("fig7_residual_sanity", "train/test: self-residual negative controls")}
<p>Self-removal drops category to {sanity["cat_self_mean"]:.2f} and choice to
{sanity["cho_self_mean"]:.2f}. This negative control asks whether the linear
residualization can remove the exact signal it is told to remove; if self-removal
remained high, the cross-residual result above would be much harder to trust.</p>
</details>

<h3>Population geometry: how distributed are the readout axes?</h3>
{img("fig5_encoding_dimensionality", "PCA dimensionality of rate and choice readouts")}
<p>This dimensionality diagnostic asks how many unsupervised population PCs are
needed for rate and choice readout. PCA finds high-variance neural dimensions; the
ridge decoder axis is different because it is supervised to predict rate. PCA is
refit inside each training fold before decoding, so test trials do not set the PC
axes. Rate peaked at {dim_rate_best["rate_r2_mean"]:.2f} CV R<sup>2</sup>
using {dim_rate_best["n_pcs"]} PCs; choice peaked at
{dim_choice_best["choice_acc_mean"]:.2f} balanced accuracy using
{dim_choice_best["n_pcs"]} PCs. Gradual improvement across many PCs suggests a
distributed code; early saturation suggests a lower-dimensional readout. This is a
population-geometry diagnostic, not a rate-compression control.</p>

{img("fig8_category_axis_projection", "train: non-boundary category; test: category-axis projection")}
<p>The boundary projection asks the same question in task coordinates. A category
axis trained only on non-boundary trials separates low from high stimuli. On 12 Hz
boundary trials, where stimulus category evidence is absent, the same axis shifts
with the animal's choice. That means the sensory category axis is partly aligned with
choice under ambiguity, but it is not reducible to a decoder trained on choice.</p>

<h2 id="timing-robustness">3. Timing and robustness checks</h2>
{img("fig6_timecourse", "train: category or choice; test: held-out same-bin trials")}
<p>The time-course asks whether sensory and report-related signals appear at the same
time. Category and choice both become readable within the stimulus period, with peak
100 ms accuracies of {tc["peak_category"]:.2f} and {tc["peak_choice"]:.2f}. The timing
is important because the animal typically exits the center port around the middle of
the 1 s stimulus window, so late choice information may include report or movement
state.</p>

{img("fig9_time_aware_cv", "random trial folds versus ten contiguous held-out trial blocks; this tests sensitivity to slow session drift")}
<p><span class="label">Result</span> {time_aware["design"]}</p>
{time_aware_table}
<details><summary>Contiguous-fold trial spans and binary class counts</summary>
<table><caption><strong>Table 5.</strong> Each test fold is one contiguous trial interval;
the remaining trials form its training set.</caption><thead><tr><th>Fold</th>
<th>Test trial indices</th><th>n test</th><th>Category low / high</th>
<th>Choice left / right</th></tr></thead><tbody>{blocked_fold_rows}</tbody></table>
</details>
<p>{random_blocked_alpha_sentence} Differences can reflect local rate distributions
and training diversity as well as drift; they are reported rather than interpreted as
a separate biological effect.</p>
<p><span class="label">Interpretation</span> A signal that remains above its chance
level under contiguous-block testing is not explained solely by a slowly drifting
session baseline sampled into both train and test folds. The blocked-minus-random
difference quantifies sensitivity to this stricter split.</p>
<p><span class="label">Caveat</span> Contiguous-block CV still trains on trials both
before and after each held-out block. It is a drift stress test, not a causal or
forward-in-time generalization claim.</p>

{img("fig8_june29_controls", "left train: category within strength; right train: non-boundary category")}
<p>The stimulus-strength control asks whether obvious sensory evidence is readable
earlier or more strongly than near-boundary evidence. Trials are grouped by distance
from the 12 Hz boundary while balancing category and choice. This is a check on the
sensory interpretation of the category decoder: if V1 is carrying stimulus evidence,
stronger stimuli should be easier to read out.</p>
<ul>{strength_items}</ul>
<details><summary>Category x choice cell counts by strength</summary>
<pre>{escape(strength_counts)}</pre>
</details>
<p>The right panel is the direct boundary-transfer control. A category decoder is
trained on non-boundary trials, balanced by category x choice, then applied to 12 Hz
boundary trials where category evidence is absent. Choice alignment was
{controls["cat_boundary_choice_align_mean"]:.2f} vs shuffled-choice
{controls["cat_boundary_choice_align_shuffle_mean"]:.2f}
(p = {controls["cat_boundary_choice_align_p"]:.3f}).</p>

<h2>4. Single-neuron effects mostly look additive, with a small interaction subset</h2>
{img("fig4_interaction", "single-unit category and choice stability plus interaction model")}
<p>This section asks whether the category and choice/report signals are mixed in a
strongly nonlinear way at the single-neuron level. Most single-unit effects are stable
across the other variable: category effects are similar across choices, and choice
effects are similar across categories. That points to substantial additive structure,
with a smaller subset of neurons showing interaction-like mixing.</p>
<p>{q4["n_sig_interaction"]}/{q4["n_units"]} units passed Benjamini-Hochberg FDR
correction for the interaction term. Category effects were similar across choices
(r = {q4["r_category_effect_left_right"]:.2f}); choice effects were similar across
categories (r = {q4["r_choice_effect_low_high"]:.2f}). This test was chosen because
it is the simplest direct per-neuron test of non-additive mixing in the same firing
rate window as the decoders. It assumes independent trial residuals with a correctly
specified linear mean and uses ordinary OLS standard errors; heteroskedastic spike-count
variance is plausible, so q values are screening results rather than definitive unit
classifications. FDR controls the tested unit family only. This is not a complete
nonlinear encoding model.</p>
{sig_table}
{img("fig4_interaction_psths", "first-stimulus-aligned PSTHs for FDR-significant interaction units")}

	<h2 id="rate-results">5. V1 has a readable but compressed held-out rate readout</h2>
	{img("fig5_rate_decoding", "train: stimulus rate; test: held-out trials")}
	<p><span class="label">Direct result</span> The population carries graded rate information, but not as a perfectly calibrated
meter. Continuous ridge prediction was well above shuffle (CV R<sup>2</sup>
{q5["real_r2_mean"]:.2f} vs {q5["shuffle_r2_mean"]:.2f}, p = {q5["r2_p"]:.3f}), yet
the calibration slope was only {q5["calibration_slope"]:.2f}. In plain terms, the
decoder overestimates low rates and underestimates high rates. This compression is
part of the result rather than an artifact corrected by simple target transforms or
by removing ridge regularization.</p>
	<p><span class="label">Interpretation</span> The directly supported object is a
compressed distribution of held-out decoder predictions. This is consistent with a
coarse/noisy rate representation, but it does not by itself establish compression in
the latent biological neural code.</p>
	<p><span class="label">Caveat</span> Measurement noise and squared-error
regression-to-the-mean remain viable explanations. OLS tests shrinkage only; the MLP
tests one bounded nonlinear family only.</p>
	{rate_model_table}
	{tuning_table}
	{img("fig5_mlp_rate_diagnostic", "train: stimulus rate; test: ridge vs MLP readout")}
	<p>{mlp_rate_summary}</p>
	<p class="note">The MLP uses LBFGS and no early stopping; the script does not persist
per-iteration training curves. No convergence warning occurred in the full run. Both
hidden size and alpha selected the upper tested boundary in every outer fold, so the
search is not evidence that this architecture family was exhausted.</p>
	<p>The ridge weights are aligned with per-unit rate slopes
	(r = {q5["r_slope_weight"]:.2f}), suggesting the population readout uses neurons in
	the expected direction rather than an arbitrary axis.</p>

{img("fig5_rate_confusion", "balanced multiclass exact-rate decoder")}
<p>The multiclass decoder reached {q5["multiclass_acc"]:.2f} vs chance
{q5["multiclass_chance"]:.2f}. This panel asks whether exact rate labels are
separable. Exact-rate classification is weak but above chance, and errors are
structured rather than random. Together with the ridge result, this supports a coarse
or compressed rate representation rather than a precise labeled-rate code.</p>

<details><summary>Additional rate-code controls</summary>
	<p>These controls ask whether the residual compression is caused by choice/report
	variance, the unit subset used for decoding, or choosing exact Hz as the regression
	target. They are useful controls rather than primary evidence.</p>
	{img("fig5_choice_residualized_rate", "train: stimulus rate; test: choice-residualized rate readout")}
	<p>The choice-residualized rate control asks whether choice/report variance in the
	1 s fitted neural window contributes to the compressed rate predictions. The
	nuisance model is fit inside each outer training fold, then applied to held-out
	trials before fitting the same ridge rate decoder. {choice_resid_rate_summary} If
	choice residualization keeps similar CV R<sup>2</sup> and residual slope, compression
	is probably not mainly caused by choice/report variance. If CV R<sup>2</sup> remains
	decent but the residual slope becomes less negative, choice/report activity likely
	contributes to compression. If CV R<sup>2</sup> collapses, the original rate readout
	depends strongly on variance shared with choice/report. Because choice is correlated
	with stimulus category and rate, this control is conservative: removing choice can
	also remove stimulus-related variance.</p>
	{choice_resid_rate_table}
	{img("fig5_rate_slope_subset_diagnostic", "train: slope-selected units; test: rate residual compression")}
	{slope_subset_table}
	{img("fig5_ridge_alpha_path", "train: fixed-alpha ridge; test: regularization path")}
	<p>The alpha-path control asks whether the compressed readout is simply caused by
	the high ridge penalty. Units are already standardized, so this is not a raw firing
	rate scale issue; it is a test of the single global L2 penalty on the population
	weight vector. {alpha_path_summary} In this session, the no-shrinkage endpoint and
	the smallest ridge penalties do not recover a calibration slope near 1, and they
	generalize very poorly. That makes high ridge regularization an unlikely primary
	explanation for the compression. OLS isolates the effect of removing shrinkage, but
	it remains a squared-error estimator; this control therefore does not rule out
	loss-function-dependent regression-to-the-mean behavior.</p>
	{alpha_path_table}
	{img("fig5_steepest_unit_ols", "train: steepest single unit; test: raw tuning and OLS readout")}
	<p>The steepest-unit OLS diagnostic asks whether compression is already visible in
	the best individual rate-tuned neuron. The first two panels inspect the raw tuning
	curve and its linear residuals in firing-rate units; the last panel asks what an
	unregularized single-unit decoder predicts on held-out trials.
	{steepest_unit_interpretation} This supports the interpretation that the endpoint
	residuals are tied to neural tuning, noise, or state dependence in the recorded
	activity, not just to population ridge shrinkage.</p>
	{steepest_unit_table}
	{img("fig5_single_unit_slope_examples", "train: one random unit per slope bin; test: single-unit residual compression")}
	<p>The single-unit example panel is descriptive rather than a formal selector: one
	fixed-seed random unit is picked from each full-session slope bin, then that single
	unit alone is evaluated with held-out ridge regression. This lets us compare
	single-unit residual slopes across five levels of rate tuning without mixing many
	cells together.</p>
	{single_unit_table}
	{img("fig5_rate_target_family", "train: rate-target family; test: held-out evidence")}
	<p>This control asks whether exact Hz is the wrong readout target for V1 in this
	task. The linear target is boundary-centered signed distance,
	(rate - 12) / 8. The tanh targets use the same signed distance before passing it
	through <code>tanh((rate - 12) / scale)</code>: scale 2 is strongly saturated and
	category-like, scale 4 is intermediate, and scale 8 is closer to metric evidence.
	The scale is a hypothesis about sensory evidence saturation, not a post-hoc
	calibration fit. A target with higher CV R<sup>2</sup> is easier to read out in
	its own units, but it should be interpreted with category-side accuracy and
	same-trial choice alignment rather than treated as the one true brain model.</p>
	{target_family_table}
</details>

	<h2 id="nested-choice">6. Fully nested rate-readout-to-choice analysis</h2>
	{img("fig5_rate_behavior", "behavior psychometric vs category-collapsed rate decoder")}
	<p>This is the main behavioral tension. The V1 rate readout usually falls on the
correct side of the 12 Hz boundary, but the mouse is much more biased/variable than
that sensory readout. The comparison uses the same non-boundary trials as the neural
decoder, so the mismatch is not just full-session behavior versus a neural subset.
The category-collapsed ridge accuracy was {q5["category_collapsed_acc"]:.2f}.</p>
	{img("fig5_choice_model_comparison", "train: fully nested rate-readout-to-choice stack; test: outer-fold choices")}
	<p><span class="label">Direct result</span> Average curves can look similar for the wrong reason, so the stricter question is
	whether the neural readout predicts the animal's choices in fully held-out outer
	folds. The upstream ridge readout is refit inside each outer fold, and its training
	values are cross-fitted before the choice model is trained. The
	best model here was <strong>{escape(best_choice_model["model"])}</strong>
	(log loss {best_choice_model["log_loss"]:.3f}). {nested_choice_sentence}</p>
	<p><span class="label">Caveat</span> This is a small within-session log-loss gain.
Its ten fold values are correlated because training sets overlap, and it is not a
session- or animal-level effect. Timing and movement determine the biological wording.</p>
	{choice_model_table}
	{img("fig5_within_rate_choice_tests", "rate-conditioned choice alignment using outer-fold neural readouts and fully nested choice scoring")}
		<p>These three rate-conditioned tests separate average psychometric matching from
		same-trial prediction. The hard-alignment panel asks whether the model's high/low
		threshold matches the mouse's choice on the same trials after conditioning on rate.
		The excess-alignment panel subtracts the alignment expected from the mouse and model
		right/high fractions within each rate bin, so high-rate dips caused by opposite
		marginal biases do not get mistaken for true anti-alignment. The continuous panel
		asks whether right-choice trials have higher readout values than left-choice trials
		within each rate. The fixed-effect choice models ask whether a
		fully nested neural readout improves outer-fold choice prediction after the average
		choice fraction at each rate is already modeled. The best rate-conditioned choice model was
	<strong>{escape(best_fixed_model["model"])}</strong>
	(log loss {best_fixed_model["log_loss"]:.3f}).</p>
	{within_rate_table}
	{fixed_choice_table}
	{img("fig5_time_resolved_choice_readout", "fully nested rate-fixed choice models across time; late gains are compared with center-port exit timing")}
	<p>This timing control asks whether the choice-predictive component of the V1 readout
	appears before or after the animal starts reporting. {exit_sentence} The best
	all-trial bin was {best_time_bin["bin_center_ms"]:.0f} ms, with log-loss gain
	{best_time_bin["delta_log_loss_vs_rate_fe"]:.3f} over rate fixed effects and rate
	CV R<sup>2</sup> {best_time_bin["ridge_rate_cv_r2"]:.2f}. {pre_exit_sentence} Early
	gains would favor a sensory-linked readout; late gains would be more consistent
	with report, movement, or downstream decision feedback entering V1.</p>
	{time_choice_table}

<details><summary>Behavior/state controls around realized flashes and port timing</summary>
	{img("fig5_realized_flash_behavior", "behavior and rate readouts by realized flash count")}
	<p>Because the flash trains are Poisson, nominal stimulus rate and realized sensory
	evidence can differ from trial to trial. This follow-up bins the same trials by the
	actual number of flashes before center-port exit and by the total number of flashes
	before response. The movement-only count is kept in the JSON summary, but it is not
	the visual headline because those flashes occur after the animal has already started
	reporting its choice. If the mouse's bias reflects different sensory samples across
	task epochs, the mouse curve should track realized flash count more cleanly than
	nominal rate.</p>
	{img("fig5_behavior_state_choice_controls", "fully nested choice models with rate and state/history proxies; test: outer-fold choices")}
	<p>This state/history control asks whether the ridge readout predicts same-trial
	choice after adding simple trial-level proxies that could explain choice structure:
	{escape(state_feature_text)}. These are not a full video-based movement control, but
	they test whether previous-trial history and port-timing/report variables absorb the
	apparent V1-choice link. {state_gain_sentence}</p>
	{state_choice_table}
</details>

	<h2 id="movement">7. Movement-control status</h2>
<p><span class="label">Result</span> A recoverable June cache contains scalar video
motion energy for {motion_coverage} analysis trials. The source notebook code is
available at <code>{escape(motion["source_ref"])}</code>. It converted the full
640 x 512, 60 Hz rear-view video <code>{escape(motion["video_path"])}</code> to
grayscale, computed the mean absolute frame-to-frame difference without cropping,
masking, PCA, normalization, or smoothing, and averaged that quantity from 0 to 0.5 s
after first stimulus onset. Frame timestamps came from <code>DatasetVideo</code> or the
aligned <code>frames</code> event fallback; feature construction did not use labels.
The finite values were median-split at
{motion.get("median_split_threshold", float("nan")):.3f}. The active report code does
not regenerate this cache, and the first {motion.get("n_missing", 0)} cached trials
have no finite value.</p>
<p class="note">The recovered source checks video, database frame count, and frame-time
count to within one frame, but the cached NPZ does not preserve the exact discrepancy
observed in that run or an explicit dropped-frame mask.</p>
<p><span class="label">Interpretation</span> This cache can support a limited statement
about whether category decoding differs between globally low- and high-motion trials.
It cannot adjudicate whether the late left-versus-right choice decoder reads report
direction, because the scalar metric discards motion direction and covers only part of
the session.</p>
<p><span class="label">Caveat</span> The planned equalized cross-motion asymmetry
artifact is absent, and the existing motion-state files predate the current maintained
script. Movement therefore remains an unresolved confound, not a passed control.</p>

	<h2 id="interpretation">Interpretation</h2>
<p><span class="answer">Interpretation:</span> I think this session argues against the simple
explanation that V1 lacks the task-relevant sensory information. Category and rate
are both readable, and category remains readable after controlling for choice. The
harder question is how to interpret the strong choice/report signal.</p>
<p>The most conservative interpretation is that V1 contains a sensory axis that is
partly aligned with the animal's report, especially under ambiguity, but behavior is
not a clean readout of the V1 rate estimate. That leaves several live possibilities:
downstream circuits may read out V1 with different weights; internal state/history may
dominate behavior on some trials; report/movement signals may enter V1 late in the
trial; or shared gain may reshape V1 activity in a way these simple decoders do not
model.</p>
<h2>Limitations</h2>
<ul>
<li>One session, one animal.</li>
<li>The 1 s window includes movement/report behavior.</li>
<li>Residualization is conservative when variables are correlated.</li>
<li>Rate may still be confounded by total light dose.</li>
<li>Fold and resampling dispersion does not estimate between-session or between-animal uncertainty.</li>
<li>The MLP search ended at the largest hidden layer and alpha in the tested grid.</li>
<li>Contiguous-block CV is a slow-drift stress test, not complete temporal deconfounding.</li>
</ul>

<h2 id="next-steps">Proposed next steps and feedback requested</h2>
<table><caption><strong>Table 6.</strong> Analyses are ordered by their ability to change
the principal interpretation.</caption><thead><tr><th>Analysis</th><th>Failure mode</th>
<th>Result that would change interpretation</th><th>Required before sharing?</th>
<th>Current or future stage</th></tr></thead><tbody>
<tr><td>Directional video movement around center exit</td><td>Late neural readout may encode left/right action</td><td>A pre-exit neural gain that survives directional motion covariates would strengthen a choice-related interpretation</td><td>No, if the present caveat remains visible; yes before claiming a biological choice signal</td><td>Next priority</td></tr>
<tr><td>Across-session/animal replication</td><td>Single-session idiosyncrasy</td><td>Consistent blocked-CV, residual, and timing effects with session/animal-level uncertainty</td><td>Yes before generalization</td><td>Next priority</td></tr>
<tr><td>Poisson GLM encoding model</td><td>Noisy MSE readout may create central tendency</td><td>A validated count-likelihood model that changes endpoint residual structure</td><td>No for this decoding report</td><td>Separate next-stage analysis</td></tr>
<tr><td>Shared-gain/state-dependent model</td><td>Unmodeled state may couple sensory and report axes</td><td>State terms explain held-out choice/report variance without erasing sensory rate information</td><td>No</td><td>Future after motion/replication</td></tr>
</tbody></table>
<p><strong>Feedback requested from Anne:</strong> whether to prioritize directional
video features in this session or replication across sessions first, and whether the
held-out rate-prediction compression is important enough to motivate the separate
Poisson/count-encoding stage.</p>

<h2 id="technical-appendix">Technical appendix: active code paths and artifact audit</h2>
<table><caption><strong>Table 7.</strong> Active path from source data to report values.</caption>
<thead><tr><th>Output</th><th>Active function(s)</th><th>Validation boundary</th></tr></thead><tbody>
<tr><td>Session units/trials</td><td><code>load_session</code></td><td>Live DataJoint plus explicit 505/504 truncation</td></tr>
<tr><td>1 s and time-bin neural features</td><td><code>build_design_matrix</code>, <code>build_timecourse</code></td><td>No missing neural values; first-stimulus alignment</td></tr>
<tr><td>Category and choice/report</td><td><code>analyze_category</code>, <code>analyze_choice</code></td><td>Balanced stratified held-out trials plus shuffles</td></tr>
<tr><td>Cross-residual controls</td><td><code>analyze_residual</code>, <code>analyze_residual_sanity</code></td><td>Nuisance fit inside each fold</td></tr>
<tr><td>Single-unit interaction</td><td><code>analyze_interaction</code></td><td>OLS interaction plus BH-FDR across units</td></tr>
<tr><td>Random vs blocked robustness</td><td><code>analyze_time_aware_cv</code></td><td>Ten shuffled or contiguous outer folds</td></tr>
<tr><td>Rate and bounded controls</td><td><code>analyze_rate</code> and its diagnostic helpers</td><td>Outer ten-fold prediction; inner five-fold tuning</td></tr>
<tr><td>Rate-readout to choice/report</td><td><code>nested_rate_choice_cv</code>, <code>time_resolved_rate_fixed_choice</code></td><td>Outer choice folds plus inner cross-fitted upstream readouts</td></tr>
<tr><td>Final reader artifact</td><td><code>write_report_html</code></td><td>Dynamic values from the full-run result dictionary; PNGs embedded as data URIs</td></tr>
</tbody></table>

<table><caption><strong>Table 8.</strong> Artifact inventory at execution time.</caption>
<thead><tr><th>Artifact</th><th>Path</th><th>Exists</th><th>Modified (UTC)</th></tr></thead>
<tbody>{artifact_rows}</tbody></table>
<p><strong>Active versus exploratory.</strong> The maintained report values come only
from the script/function paths above. <code>notebooks/categorydecoding.ipynb</code> is
not executed by this report. The recoverable motion code exists only at
<code>{escape(motion["source_ref"])}</code> and its cache is disclosed but not merged
into the choice claim. The PowerPoint and its lock file are not validated outputs.</p>
<p><strong>Execution checks.</strong> The full run completed against the live database
with {meta["n_shuffles"]} shuffles and {meta["n_balance_resamples"]} balanced draws.
The only runtime warnings were the known one-trial alignment truncation and
Matplotlib's non-fatal open-figure-count warning. Ruff formatting/lint and
<code>git diff --check</code> passed after the final source edit.</p>
<p class="note">Generated by <code>scripts/analyses/category_choice_decoding.py</code>.</p>
</body>
</html>
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "results_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[report] wrote {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.n_balance_resamples = 4 if args.quick else N_BALANCE_RESAMPLES
    args.tc_resamples = 3 if args.quick else max(5, N_BALANCE_RESAMPLES // 4)
    args.n_shuffles = 3 if args.quick else N_SHUFFLES
    args.timecourse_n_shuffles = 2 if args.quick else TIMECOURSE_N_SHUFFLES
    args.ridge_alphas = np.logspace(-4, 6, 7 if args.quick else 21)
    args.mlp_alphas = np.logspace(-4, 2, 3 if args.quick else 7)
    args.mlp_hidden_layer_sizes = [(4,), (8,)] if args.quick else [(4,), (8,), (16,)]
    args.mlp_max_iter = 1200 if args.quick else 2500
    args.mlp_max_fun = 12000 if args.quick else 20000

    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _set_style(plt)

    units, srate, trials = load_session()
    valid_trials = trials[
        (trials.with_choice == 1) & (trials.stim_category != "boundary")
    ].reset_index(drop=True)
    boundary_trials = trials[
        (trials.with_choice == 1) & (trials.stim_category == "boundary")
    ].reset_index(drop=True)
    print(
        f"{len(units)} units, {len(valid_trials)} valid trials, "
        f"{len(boundary_trials)} boundary choice trials"
    )

    X = build_design_matrix(units, srate, valid_trials)
    X_boundary = build_design_matrix(units, srate, boundary_trials)
    X_time, bin_centers = build_timecourse(units, srate, valid_trials)
    X_time_200, bin_centers_200 = build_timecourse(
        units, srate, valid_trials, n_timepoints=N_TIMEPOINTS_200MS
    )
    y_cat = (valid_trials["stim_category"] == "high_rate").astype(int).to_numpy()
    y_choice = (valid_trials["response"] == 1).astype(int).to_numpy()
    rate = valid_trials["stim_rate_vision"].to_numpy(dtype=float)

    contingency = pd.crosstab(
        pd.Series(y_cat, name="category(0=low,1=high)"),
        pd.Series(y_choice, name="choice(0=left,1=right)"),
    )
    phi = _corr(y_cat.astype(float), y_choice.astype(float))
    confound_acc = float(max(np.mean(y_cat == y_choice), np.mean(y_cat != y_choice)))

    cv_reg = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    time_aware_cv = analyze_time_aware_cv(plt, X, y_cat, y_choice, rate, args)

    results = {
        "meta": {
            "subject": SUBJECT,
            "session": SESSION,
            "n_units": int(len(units)),
            "n_trials": int(len(valid_trials)),
            "n_aligned_trials_after_mismatch_truncation": int(len(trials)),
            "n_choice_trials": int((trials.with_choice == 1).sum()),
            "n_boundary_choice_trials": int(len(boundary_trials)),
            "n_excluded_without_choice": int((trials.with_choice != 1).sum()),
            "unit_criteria_id": int(UNIT_CRITERIA_ID),
            "phi": phi,
            "confound_acc": confound_acc,
            "contingency": contingency.to_string(),
            "n_shuffles": int(args.n_shuffles),
            "n_balance_resamples": int(args.n_balance_resamples),
            "run_mode": "quick" if args.quick else "full",
            "provenance": repository_provenance(),
            "motion_cache_audit": audit_motion_cache(len(valid_trials)),
            "artifact_inventory": artifact_inventory(),
        },
        "q1": analyze_category(plt, X, y_cat, rng, args),
        "q2": analyze_choice(plt, X, y_choice, rng, args),
        "q3": analyze_residual(plt, X, y_cat, y_choice, rng, args),
        "sanity": analyze_residual_sanity(plt, X, y_cat, y_choice, rng, args),
        "q4": analyze_interaction(
            plt, units, srate, valid_trials, X, y_cat, y_choice, args
        ),
        "q5": analyze_rate(
            plt, X, X_time, bin_centers, valid_trials, rate, y_choice, cv_reg, rng, args
        ),
        "tc": analyze_timecourse(
            plt,
            X_time,
            bin_centers,
            X_time_200,
            bin_centers_200,
            y_cat,
            y_choice,
            rng,
            args,
        ),
        "june29_controls": analyze_june29_controls(
            plt,
            X,
            X_time_200,
            bin_centers_200,
            X_boundary,
            valid_trials,
            boundary_trials,
            y_cat,
            y_choice,
            rng,
            args,
        ),
        "time_aware_cv": time_aware_cv,
    }

    if not args.no_save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        (FIGURE_DIR / "results_summary.json").write_text(json.dumps(results, indent=2))

    write_report_html(results, args.no_save)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
