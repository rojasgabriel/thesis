"""The main GLM fit of each thesis session, computed by `make`."""

from ._base import (  # noqa: F401
    AnalysisFile,
    SpikeSorting,
    UnitCount,
    dj,
    np,
    rojasbowe_schema,
)
from .sessions import ThesisSession  # noqa: F401


def _analysis_files(key: dict, paths: list, dataset_name: str) -> list[dict]:
    """Return AnalysisFile keys of local files; upload the ones not stored yet."""
    from labdata.utils import compute_md5s

    dataset = {
        "subject_name": key["subject_name"],
        "session_name": key["session_name"],
        "dataset_name": dataset_name,
    }
    targets = AnalysisFile().generate_filepaths(paths, dataset)
    missing = []
    for path, target, md5 in zip(paths, targets, compute_md5s(paths), strict=True):
        stored = (AnalysisFile & {"file_path": target}).fetch("file_md5")
        if len(stored) and stored[0] != md5:
            raise ValueError(f"{target} is stored with another md5; delete it first.")
        if not len(stored):
            missing.append(path)
    if missing:
        AnalysisFile().upload_files(missing, dataset)
    return [(AnalysisFile & {"file_path": target}).fetch1("KEY") for target in targets]


def _code_design() -> dict:
    """The design and fit settings that the code at this commit uses."""
    from thesis.ephys.analyses.glm import dmat, fit

    return {
        "l2_penalties": np.asarray(fit.ALPHAS, dtype=float),
        "folds": fit.CV_FOLDS,
        "fold_seed": fit.CV_SEED,
        "split_seed": dmat.SPLIT_SEED,
        "video_components": np.asarray(dmat.VIDEO_COMPONENT_COUNTS, dtype=int),
    }


def _clean_commit() -> str:
    """Return the commit of this checkout; refuse changes that the commit does not hold."""
    import os
    import subprocess
    from pathlib import Path

    if os.environ.get("THESIS_GLM_TERMS"):
        raise RuntimeError("THESIS_GLM_TERMS changes the design; unset it for GLMFit.")
    repository = Path(__file__).resolve().parents[2]
    git = ["git", "-C", str(repository)]
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError(f"{repository} has uncommitted changes; commit them first.")
    return subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()


@rojasbowe_schema
class GLMDesignParam(dj.Lookup):
    """Settings of the main GLM. `make` stops if the code uses other settings.

    The code defines the design; this row records it. A new design is a new
    commit and a new row. Exploratory fits (leak and power tests, grids, raw
    video) stay as scripts.
    """

    definition = """
    glm_design_id                          : tinyint unsigned
    ---
    description                            : varchar(255)
    unit_criteria_id                       : tinyint unsigned  # quality criteria of the units
    stability_param_id                     : tinyint unsigned  # UnitStability parameter set
    terms                                  : varchar(1023)  # task terms in design order; also video PCs and spike history
    l2_penalties                           : longblob  # penalty grid of each unit
    folds                                  : tinyint unsigned  # cross-validation folds over trials
    fold_seed                              : int unsigned
    split_seed                             : int unsigned  # 60/20/20 trial split for the PC count
    video_components                       : longblob  # motion-energy PC counts compared on validation trials
    """
    default_contents = [
        {
            "glm_design_id": 0,
            "description": "design of 2026-10-15 with flash x side (2026-10-29)",
            "unit_criteria_id": 1,
            "stability_param_id": 0,
            "terms": (
                "visual_flash,flash_just_before,flash_after_exit,flash_recovery,"
                "flash_side,hold,fixation,previous_side,session_drift,"
                "response_period,turn_direction"
            ),
            "l2_penalties": np.logspace(-6, 4, 11),
            "folds": 10,
            "fold_seed": 20260913,
            "split_seed": 20260912,
            "video_components": np.array([10]),
        }
    ]


@rojasbowe_schema
class GLMFit(dj.Computed):
    """The main GLM fit of each session

    `make` runs the fit of `glm.fit` (model selection, 10-fold cross-validation,
    final fit) on all GLM units, in results/glm/<session>/glm<id>_<commit>/
    under the working directory. Files of another commit are never reused.
    The design manifest, the validation and cross-validation summaries, and
    the final weights go to the analysis bucket (dataset glm<id>_<commit>).
    Units and trials: those of the fit (glm.fit). In-port and after-exit
    flashes: at least 80 ms after the flash before, with the whole −50 to
    150 ms window in its period (dataset_overview). Excited and suppressed: GLM
    units with that StimulusResponse class (first flash). D² and bits/spike:
    mean over the folds for each unit; median and quartiles over units here.
    Flash weight: the flash kernel at the first flash, 30–120 ms, Δ log rate,
    with its jackknife SE over folds (glm.flash_weight_check).
    """

    definition = """
    -> ThesisSession
    -> GLMDesignParam
    ---
    code_commit                            : char(40)  # git commit of the fit code
    trials                                 : smallint  # completed trials of the fit
    units                                  : smallint  # fitted units (single and multi)
    in_port_flashes                        : smallint
    after_exit_flashes                     : smallint
    excited_units                          : smallint  # StimulusResponse 'excited'
    suppressed_units                       : smallint  # StimulusResponse 'suppressed'
    video_components                       : tinyint  # selected PC count
    median_d2                              : float
    q25_d2                                 : float
    q75_d2                                 : float
    median_bits_per_spike                  : float
    q25_bits_per_spike                     : float
    q75_bits_per_spike                     : float
    -> AnalysisFile.proj(design_file="file_path", design_storage="storage")
    -> AnalysisFile.proj(validation_file="file_path", validation_storage="storage")
    -> AnalysisFile.proj(summary_file="file_path", summary_storage="storage")
    -> AnalysisFile.proj(weights_file="file_path", weights_storage="storage")
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        d2                                 : float  # cross-validated deviance explained
        d2_sem                             : float  # SEM over folds
        bits_per_spike                     : float  # cross-validated
        l2_penalty                         : float  # penalty of the final model
        flash_weight                       : float  # Δ log rate, 30–120 ms after the first flash
        flash_weight_se                    : float  # jackknife SE over folds
        flash_weight_class                 : enum('excited', 'suppressed', 'not responsive')  # ±1.96 SE from 0
        """

    class ExcludedUnit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        reason                             : varchar(255)
        """

    def make_fetch(self, key):
        param = (GLMDesignParam & key).fetch1()
        code = _code_design()
        for name, value in code.items():
            if not np.array_equal(param[name], value):
                raise ValueError(
                    f"{name}: the code uses {value}, the row {param[name]}."
                )
        if (param["unit_criteria_id"], param["stability_param_id"]) != (1, 0):
            raise ValueError("glm.fit selects units with criteria 1 and stability 0.")
        # The sorting of the GLM units (criteria 1, the sorting of fetch_unit_table).
        from thesis.ephys.units import sorting_key

        chosen = sorting_key(key["subject_name"], key["session_name"])
        sorting = (
            SpikeSorting & chosen & (UnitCount & key & {"unit_criteria_id": 1})
        ).fetch("KEY")
        if len(sorting) != 1:
            raise ValueError(f"{key}: {len(sorting)} sortings; expected 1.")
        return param, sorting[0], _clean_commit()

    def make_compute(self, key, param, sorting, commit):
        import json
        import os
        from pathlib import Path

        from thesis.ephys.analyses import dataset_overview as overview
        from thesis.ephys.analyses.glm.cli import PoissonGLM, _artifact_lock
        from thesis.ephys.analyses.glm.fit import excluded_units, load_unit_records
        from thesis.ephys.analyses.glm.flash_weight_check import (
            model_class,
            model_values,
        )
        from thesis.ephys.analyses.glm.peaks import excited_units

        # Read trials and spikes from the database, never from the pickle cache.
        os.environ.pop("THESIS_CACHE_DIR", None)
        subject, session = key["subject_name"], key["session_name"]
        dataset = f"glm{key['glm_design_id']}_{commit[:7]}"
        root = Path("results/glm") / f"{subject}_{session}" / dataset
        model = PoissonGLM(subject=subject, session=session, root=root)
        # submit-glm sets LABDATA_N_JOBS to the CPUs of the Slurm job; srun and
        # sbatch also limit the CPUs of the process.
        workers = int(os.environ.get("LABDATA_N_JOBS", 0)) or (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        )
        with _artifact_lock(root):
            model.prepare()
            model.fit("all", workers)
        manifest = json.loads(model.design.with_suffix(".json").read_text())
        terms = ",".join(item["name"] for item in manifest["task_manifest"])
        if terms != param["terms"]:
            raise ValueError(
                f"The design has the terms {terms}, the row {param['terms']}."
            )

        fit = model.fit_dir("all")
        summary = json.loads((fit / "cross_validation_summary.json").read_text())
        records = load_unit_records(fit / "checkpoints", "unit_*_final.json")
        data = overview.session_data(f"{subject}_{session}", set(), root)
        ids = set(data["ids"].tolist())
        if {r["unit_id"] for r in records} != ids or len(ids) != summary["units"]:
            raise ValueError(f"{fit}: the fitted units differ from the GLM units.")
        if len(data["first"]) != summary["trials"]:
            raise ValueError(f"{fit}: the fitted trials differ from the GLM trials.")
        values = model_values(root)
        weights = dict(zip(values["unit_ids"], values["weights"], strict=True))
        weights_se = dict(zip(values["unit_ids"], values["weights_se"], strict=True))
        classes = dict(
            zip(
                values["unit_ids"],
                model_class(values["weights"], values["weights_se"]),
                strict=True,
            )
        )
        weights_file = fit / "final_weights.npz"
        np.savez_compressed(
            weights_file,
            unit_id=np.array([r["unit_id"] for r in records]),
            l2_penalty=np.array([r["alpha"] for r in records]),
            intercept=np.array([r["intercept"] for r in records]),
            coefficients=np.array([r["coefficients"] for r in records]),
        )
        d2 = np.array([r["cross_validated_deviance_explained"] for r in records])
        bits = np.array([r["cross_validated_bits_per_spike"] for r in records])
        row = {
            **key,
            "code_commit": commit,
            "trials": summary["trials"],
            "units": len(records),
            "in_port_flashes": len(data["still"]),
            "after_exit_flashes": len(data["moving"]),
            "excited_units": len(excited_units(subject, session) & ids),
            "suppressed_units": len(
                excited_units(subject, session, "suppressed") & ids
            ),
            "video_components": summary["components"],
            "median_d2": np.median(d2),
            "q25_d2": np.percentile(d2, 25),
            "q75_d2": np.percentile(d2, 75),
            "median_bits_per_spike": np.median(bits),
            "q25_bits_per_spike": np.percentile(bits, 25),
            "q75_bits_per_spike": np.percentile(bits, 75),
        }
        units = [
            {
                **key,
                **sorting,
                "unit_id": r["unit_id"],
                "d2": r["cross_validated_deviance_explained"],
                "d2_sem": r["cross_validated_deviance_explained_sem"],
                "bits_per_spike": r["cross_validated_bits_per_spike"],
                "l2_penalty": r["alpha"],
                "flash_weight": weights[r["unit_id"]],
                "flash_weight_se": weights_se[r["unit_id"]],
                "flash_weight_class": str(classes[r["unit_id"]]),
            }
            for r in records
        ]
        excluded = [
            {**key, **sorting, "unit_id": unit, "reason": reason}
            for unit, reason in excluded_units(subject, session).items()
        ]
        files = [
            model.design.with_suffix(".json"),
            fit / "validation_summary.json",
            fit / "cross_validation_summary.json",
            weights_file,
        ]
        return row, units, excluded, [str(path) for path in files], dataset

    def make_insert(self, key, row, units, excluded, files, dataset):
        stored = _analysis_files(key, files, dataset)
        for name, file in zip(
            ("design", "validation", "summary", "weights"), stored, strict=True
        ):
            row[f"{name}_file"] = file["file_path"]
            row[f"{name}_storage"] = file["storage"]
        self.insert1(row)
        self.Unit.insert(units)
        self.ExcludedUnit.insert(excluded)
