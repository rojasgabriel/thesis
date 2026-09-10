"""Fit mean-centered raw-video SVD on training trials and project all trial frames.

Uses the trial selection and chronological split saved by prepare_v1_glm.
Frames cover the selected first-stimulus windows and their interpolation support.
Spatial area downsampling preserves the full camera view; no motion differencing,
pixel standardization, or neural-response selection is applied. PCA's randomized
SVD fits 200 training-frame axes at 80 by 64 pixels. Component scores are centered
and scaled with training frames only. Validation/test frames are projected and
scaled without refitting. Later validation compares nested prefixes of 10, 25,
50, 100, and 200 components. Whole trials remain the sampling units for later fits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

WIDTH = 80
COMPONENTS = 200


def training_zscore(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scale every column using only training rows."""
    mean = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale == 0)
    ):
        raise ValueError("Training video scores must have finite nonzero variance.")
    return (values - mean) / scale, mean, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alignment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.alignment, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if not metadata.get("video_aligned", False):
            raise ValueError("Preparation has no validated neural-clock video mapping.")
        indices = np.flatnonzero(data["frame_split"] >= 0)
        split = data["frame_split"][indices]
        times = data["frame_times_s"][indices]
        trial_rows = data["frame_trial_row"][indices]
    height = round(WIDTH * metadata["height"] / metadata["width"])
    frame_size = WIDTH * height
    pixels = np.empty((len(indices), frame_size), dtype=np.float32)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        metadata["video_path"],
        "-map",
        "0:v:0",
        "-vf",
        f"scale={WIDTH}:{height}:flags=area",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    # Sequential decoding avoids approximate seeking and checks every frame.
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        selected = 0
        for frame_index in range(metadata["n_frames"]):
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                process.kill()
                raise ValueError(f"Video decode stopped at frame {frame_index}.")
            if selected < len(indices) and frame_index == indices[selected]:
                pixels[selected] = np.frombuffer(raw, dtype=np.uint8) / 255.0
                selected += 1
        if process.stdout.read(1):
            process.kill()
            raise ValueError("Video has more frames than the aligned timestamp array.")
        if process.wait() != 0 or selected != len(indices):
            raise RuntimeError("Video decoding did not complete successfully.")
    train = split == 0
    if COMPONENTS >= min(int(train.sum()), frame_size):
        raise ValueError("Too many SVD components for the training-frame matrix.")
    print(
        f"Decoded {metadata['n_frames']} frames; retained {len(indices)} trial frames.",
        flush=True,
    )
    model = PCA(n_components=COMPONENTS, svd_solver="randomized", random_state=0)
    model.fit(pixels[train])
    scores, score_mean, score_scale = training_zscore(model.transform(pixels), train)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite video scores.")
    summary = {
        "alignment_path": str(args.alignment.resolve()),
        "source": metadata,
        "width": WIDTH,
        "height": height,
        "components": COMPONENTS,
        "component_candidates": [10, 25, 50, 100, 200],
        "score_scaling": "training z-score",
        "training_frames": int(train.sum()),
        "trial_frames": len(indices),
        "training_variance_fraction": float(model.explained_variance_ratio_.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(
            handle,
            allow_pickle=False,
            frame_indices=indices,
            frame_times_s=times,
            frame_trial_row=trial_rows,
            frame_split=split,
            scores=scores,
            score_mean=score_mean,
            score_scale=score_scale,
            components=model.components_,
            mean=model.mean_,
            explained_variance_ratio=model.explained_variance_ratio_,
            metadata_json=json.dumps(summary),
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
