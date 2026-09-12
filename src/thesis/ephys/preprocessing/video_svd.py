"""Fit motion-energy SVD on training trials and project all trial frames.

Uses the trial selection and split saved by prepare_v1_glm. Each retained frame
is the absolute pixel difference from the previous decoded frame when the two
indices are adjacent; the first frame of a gap is left at zero. PCA fits 200
training-frame axes at 80 by 64. Scores are z-scored with training frames only.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from thesis.ephys.preprocessing.prepare_v1_glm import training_zscore

WIDTH = 80
COMPONENTS = 200


def write_motion_energy_features(alignment: Path, output: Path) -> None:
    """Fit motion-energy SVD on training frames and project all trial frames."""
    if output.exists():
        raise FileExistsError(output)
    with np.load(alignment, allow_pickle=False) as data:
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
    consecutive = np.zeros(len(indices), dtype=bool)
    consecutive[1:] = np.diff(indices) == 1
    motion = np.zeros_like(pixels)
    motion[consecutive] = np.abs(np.diff(pixels, axis=0)[consecutive[1:]])
    train = (split == 0) & consecutive
    if COMPONENTS >= min(int(train.sum()), frame_size):
        raise ValueError("Too many SVD components for the training-frame matrix.")
    print(
        f"Decoded {metadata['n_frames']} frames; retained {len(indices)} trial frames.",
        flush=True,
    )
    model = PCA(n_components=COMPONENTS, svd_solver="randomized", random_state=0)
    model.fit(motion[train])
    scores, score_mean, score_scale = training_zscore(model.transform(motion), train)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite video scores.")
    summary = {
        "alignment_path": str(alignment.resolve()),
        "source": metadata,
        "width": WIDTH,
        "height": height,
        "components": COMPONENTS,
        "component_candidates": [10, 25, 50, 100, 200],
        "feature": "absolute frame-to-frame motion energy",
        "training_frames": int(train.sum()),
        "trial_frames": len(indices),
        "training_variance_fraction": float(model.explained_variance_ratio_.sum()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
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
