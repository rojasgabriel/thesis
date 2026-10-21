"""Do raw-video components improve prediction beyond motion energy?

Paired comparison: the current model with 10 motion-energy PCs against the
same task/history design with those same PCs plus 10 raw grayscale-video PCs.
Use the fixed 20-unit sample of each mouse, the current first-flash 1-ms grid,
completed trials, response-entry mask, constant-rate null, and exclusions of
fit.py. The sampling unit is the unit; tests are within mouse. Both models
use the same ten whole-trial folds and inner penalty selection. Reuse the
current model's saved folds by unit ID; fit only the added-video model.

Raw PCA and scaling use the original training split, as for motion energy,
not each fold. This is a pilot comparison with fixed preprocessing, not a
fully nested evaluation or a selection of the best PC count. The camera and
full-frame crop match motion energy. Raw frames can contain flash brightness
as well as posture/position; do not read a gain as a pure movement effect.
No main fit is changed. Outputs stay in each session's raw_video_test folder.
Checkpoint reuse requires unchanged inputs; fingerprints guard this test.

    THESIS_CACHE_DIR=figures/cache uv run python -m \
        thesis.ephys.analyses.glm.raw_video_test --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.video_svd import write_video_features
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    hierarchical_interval,
    pooled_points,
    pooled_strip,
)

COMPONENTS = 10


def fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(root: Path) -> Path:
    """Append raw PCs to the saved main design, preserving every existing column."""
    model = cli.PoissonGLM(root=root)
    output = root / "raw_video_test"
    output.mkdir(exist_ok=True)
    raw = output / "raw_video_features.npz"
    if not raw.exists():
        write_video_features(model.windows, raw, raw_video=True, components=COMPONENTS)
    provenance = {
        "windows": fingerprint(model.windows),
        "source_design": fingerprint(model.design),
        "source_metadata": fingerprint(model.design.with_suffix(".json")),
        "raw_video_features": fingerprint(raw),
        "raw_components": COMPONENTS,
    }
    design_path = output / "common_design_raw.npy"
    metadata_path = design_path.with_suffix(".json")
    if design_path.exists() or metadata_path.exists():
        if not (design_path.exists() and metadata_path.exists()):
            raise FileExistsError(
                "Incomplete raw-video design; move it aside before retrying."
            )
        if json.loads(metadata_path.read_text()).get("raw_video_test") != provenance:
            raise ValueError("Raw-video test inputs changed; use a new output folder.")
        return design_path
    prepared = dmat._load_windows(model.windows)
    valid = dmat._valid_bin_mask(prepared)
    train, _, _ = dmat._split_masks(prepared["split"], valid)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    if metadata["video_components"] != COMPONENTS:
        raise ValueError(
            "This comparison needs the current 10-PC motion-energy design."
        )
    with (
        np.load(raw, allow_pickle=False) as data,
        np.load(model.video, allow_pickle=False) as motion,
    ):
        for key in ("frame_indices", "frame_times_s", "frame_trial_row", "frame_split"):
            if not np.array_equal(data[key], motion[key]):
                raise ValueError(f"Raw and motion-energy frame mappings differ: {key}.")
        if json.loads(str(data["metadata_json"]))["feature"] != "raw grayscale frames":
            raise ValueError("Expected raw-video features.")
        scores = data["scores"]
        frame_times = data["frame_times_s"]
        if (
            scores.shape != (len(frame_times), COMPONENTS)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Raw-video scores must be finite, with 10 columns.")
        sample_times = (
            prepared["alignments"][:, None] + prepared["relative_centers"]
        ).ravel()
        sampled = np.column_stack(
            [
                np.interp(sample_times, frame_times, scores[:, i], left=0, right=0)
                for i in range(COMPONENTS)
            ]
        )
        scaled, mean, scale = dmat.training_zscore(sampled, train)
        brightness = [
            float(spearmanr(scores[:, i], data["total"]).statistic)
            for i in range(COMPONENTS)
        ]
        # Raw components can mix animal appearance and illumination. Show all ten.
        with plt.rc_context(FIGURE_STYLE):
            fig, axes = plt.subplots(2, 5, figsize=(8, 3.8), layout="constrained")
            summary = json.loads(str(data["metadata_json"]))
            for i, ax in enumerate(axes.flat):
                image = data["components"][i].reshape(
                    summary["height"], summary["width"]
                )
                limit = np.max(np.abs(image))
                ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_axis_off()
                ax.text(
                    0, 1.03, f"{chr(97 + i)}", weight="bold", transform=ax.transAxes
                )
                ax.text(
                    0.13,
                    1.03,
                    f"raw pc {i + 1}: {100 * data['explained_variance_ratio'][i]:.1f}%",
                    transform=ax.transAxes,
                )
            fig.savefig(output / "raw_components.svg", bbox_inches="tight")
            plt.close(fig)
    matrix = np.lib.format.open_memmap(
        design_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(common), common.shape[1] + COMPONENTS),
    )
    matrix[:, : common.shape[1]] = common
    matrix[:, common.shape[1] :] = scaled
    # The paired baseline must be identical, including its column scaling.
    assert np.array_equal(matrix[:, : common.shape[1]], common)
    np.testing.assert_allclose(scaled[train].mean(0), 0, atol=1e-7)
    np.testing.assert_allclose(scaled[train].std(0), 1, atol=1e-7)
    matrix.flush()
    metadata.update(
        columns=matrix.shape[1],
        video_components=2 * COMPONENTS,
        video_component_candidates=[2 * COMPONENTS],
        raw_video_test=provenance,
        raw_pc_brightness_spearman=brightness,
    )
    metadata["column_names"] += [f"raw_video_pc_{i + 1:03d}" for i in range(COMPONENTS)]
    metadata["training_mean"] += mean.tolist()
    metadata["training_scale"] += scale.tolist()
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return design_path


def compare(root: Path, design: Path, workers: int, unit_set: str = "sample") -> dict:
    model = cli.PoissonGLM(root=root)
    output = root / "raw_video_test" / f"{unit_set}_fit_raw"
    print(
        f"{root.name}: 10 motion-energy + 10 raw-video PCs, {unit_set} units.",
        flush=True,
    )
    fit.crossvalidate(model.windows, design, unit_set, output, 2 * COMPONENTS, workers)
    baseline = {
        item["unit_id"]: item
        for item in fit.load_unit_records(
            model.fit_dir("all") / "checkpoints", "unit_*_folds.json"
        )
    }
    candidate = fit.load_unit_records(output / "checkpoints", "unit_*_folds.json")
    expected = {
        unit["unit_id"]
        for unit in fit.analysis_units(dmat._load_windows(model.windows), unit_set)
    }
    if {item["unit_id"] for item in candidate} != expected:
        raise ValueError(
            "The raw-video comparison did not complete the selected units."
        )
    rows = []
    for item in candidate:
        before = baseline[item["unit_id"]]
        if before["folds_scored"] != item["folds_scored"]:
            raise ValueError(
                "Baseline and raw-video models have different scored folds."
            )
        rows.append(
            {
                "unit_id": item["unit_id"],
                "baseline_d2": before["deviance_explained_mean"],
                "raw_video_d2": item["deviance_explained_mean"],
                "baseline_sem": before["deviance_explained_sem"],
                "raw_video_sem": item["deviance_explained_sem"],
                "fold_differences": [
                    b["test"]["deviance_explained"] - a["test"]["deviance_explained"]
                    for a, b in zip(before["folds"], item["folds"], strict=True)
                ],
            }
        )
    change = np.array([item["raw_video_d2"] - item["baseline_d2"] for item in rows])
    stats = {
        "median_d2_change": float(np.median(change)),
        "units_improved": int((change > 0).sum()),
        "wilcoxon_p": float(wilcoxon(change).pvalue) if np.any(change) else 1.0,
    }
    result = {"session": root.name, "units": rows, "statistics": stats}
    name = "comparison.json" if unit_set == "sample" else "comparison_all.json"
    (root / "raw_video_test" / name).write_text(json.dumps(result, indent=2))
    print(root.name, json.dumps(stats), flush=True)
    return result


def plot(sessions: list[dict], output: Path) -> dict:
    xs = [np.array([u["baseline_d2"] for u in s["units"]]) for s in sessions]
    ys = [np.array([u["raw_video_d2"] for u in s["units"]]) for s in sessions]
    changes = [y - x for x, y in zip(xs, ys, strict=True)]
    subjects = [s["session"].split("_")[0] for s in sessions]
    pooled = hierarchical_interval(changes)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), layout="constrained")
        ax = axes[0]
        for s, x, y in zip(sessions, xs, ys, strict=True):
            ax.errorbar(
                x,
                y,
                xerr=[u["baseline_sem"] for u in s["units"]],
                yerr=[u["raw_video_sem"] for u in s["units"]],
                fmt="none",
                color=MODEL_COLOR,
                alpha=0.2,
                lw=0.6,
            )
        pooled_points(ax, xs, ys, subjects, MODEL_COLOR)
        for label in ax.texts:
            if label.get_text() in subjects:
                label.set_anncoords("axes fraction")
                label.set_position(
                    (0.68, 0.18 if label.get_text() == "GRB006" else 0.10)
                )
        low = min(np.min(np.r_[x, y]) for x, y in zip(xs, ys, strict=True))
        high = max(np.max(np.r_[x, y]) for x, y in zip(xs, ys, strict=True))
        limits = (min(0, low) - 0.01, high + 0.02)
        ax.plot(limits, limits, color=REFERENCE_COLOR, ls="--", lw=0.8)
        ax.set(
            xlim=limits,
            ylim=limits,
            aspect="equal",
            xlabel="10 motion-energy pcs: held-out D²",
            ylabel="+ 10 raw-video pcs: held-out D²",
        )
        pooled_strip(axes[1], 0, changes, subjects, MODEL_COLOR)
        axes[1].axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
        axes[1].set(xticks=[], xlim=(-0.25, 0.48), ylabel="change in held-out D²")
        for letter, axis in zip("ab", axes, strict=True):
            axis.text(-0.15, 1.06, letter, weight="bold", transform=axis.transAxes)
        fig.savefig(output / "raw_video_test.svg", bbox_inches="tight")
        plt.close(fig)
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    results = []
    for session in SESSIONS:
        root = Path("figures/glm") / session
        design = prepare(root)
        if not args.prepare_only:
            results.append(compare(root, design, args.workers))
    if results:
        pooled = plot(results, Path("figures/glm"))
        report = {
            "comparison": "10 motion-energy PCs versus those PCs plus 10 raw-video PCs",
            "preprocessing": "original training split, not nested within folds",
            "pooled_d2_change": pooled,
            "sessions": results,
        }
        Path("figures/glm/raw_video_test.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
