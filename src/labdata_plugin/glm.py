"""GLM fits for each thesis session."""

from ._base import (  # noqa: F401
    AnalysisFile,
    SpikeSorting,
    UnitCount,
    dj,
    np,
    rojasbowe_schema,
)
from .sessions import ThesisSession  # noqa: F401


def _analysis_files(key: dict, paths: list) -> list[dict]:
    """Return AnalysisFile keys of local files; upload the ones not stored yet."""
    from labdata.utils import compute_md5s

    dataset = {**key, "dataset_name": "glm"}
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


@rojasbowe_schema
class GLMFit(dj.Imported):
    """GLM fits for each session

    Reads results/glm/<session>/ from the working directory; it does not fit.
    The design and cross-validation summaries go to the analysis bucket
    (dataset "glm"). Units and trials: those of the fit (glm.fit). In-port and
    after-exit flashes: at least 80 ms after the flash before, with the whole
    −50 to 150 ms window in its period (dataset_overview). Excited and
    suppressed: GLM units with that StimulusResponse class (first flash), the
    units the GLM analyses use. D² and bits/spike: mean over the 10 trial
    folds for each unit; median and quartiles over units here. Flash weight:
    the flash kernel at the first flash, 30–120 ms, Δ log rate, with its
    jackknife SE over folds (glm.flash_weight_check).
    """

    definition = """
    -> ThesisSession
    ---
    trials                                 : smallint  # completed trials of the fit
    units                                  : smallint  # fitted units (single and multi)
    in_port_flashes                        : smallint
    after_exit_flashes                     : smallint
    excited_units                          : smallint  # StimulusResponse 'excited'
    suppressed_units                       : smallint  # StimulusResponse 'suppressed'
    folds                                  : tinyint
    video_components                       : tinyint
    median_d2                              : float
    q25_d2                                 : float
    q75_d2                                 : float
    median_bits_per_spike                  : float
    q25_bits_per_spike                     : float
    q75_bits_per_spike                     : float
    -> AnalysisFile.proj(design_file="file_path", design_storage="storage")
    -> AnalysisFile.proj(summary_file="file_path", summary_storage="storage")
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

    def make(self, key, **_kwargs):
        import json
        from pathlib import Path

        from thesis.ephys.analyses import dataset_overview as overview
        from thesis.ephys.analyses.glm.fit import excluded_units, load_unit_records
        from thesis.ephys.analyses.glm.flash_weight_check import (
            model_class,
            model_values,
        )
        from thesis.ephys.analyses.glm.peaks import excited_units

        subject, session = key["subject_name"], key["session_name"]
        root = Path("results/glm") / f"{subject}_{session}"
        fit = root / "all_fit_me"
        summary = json.loads((fit / "cross_validation_summary.json").read_text())
        records = load_unit_records(fit / "checkpoints", "unit_*_final.json")
        data = overview.session_data(f"{subject}_{session}", set())
        ids = set(data["ids"].tolist())
        if {r["unit_id"] for r in records} != ids or len(ids) != summary["units"]:
            raise ValueError(f"{fit}: the fitted units differ from the GLM units.")
        if len(data["first"]) != summary["trials"]:
            raise ValueError(f"{fit}: the fitted trials differ from the GLM trials.")
        weights = model_values(root)
        classes = dict(
            zip(
                weights["unit_ids"],
                model_class(weights["weights"], weights["weights_se"]),
                strict=True,
            )
        )
        weight = dict(zip(weights["unit_ids"], weights["weights"], strict=True))
        weight_se = dict(zip(weights["unit_ids"], weights["weights_se"], strict=True))
        # The sorting of the GLM units (criteria 1, as in fetch_unit_table).
        sorting = (SpikeSorting & (UnitCount & key & {"unit_criteria_id": 1})).fetch(
            "KEY"
        )
        if len(sorting) != 1:
            raise ValueError(f"{key}: {len(sorting)} sortings; expected 1.")
        design, cv_summary = _analysis_files(
            key,
            [root / "common_design_me.json", fit / "cross_validation_summary.json"],
        )
        d2 = np.array([r["cross_validated_deviance_explained"] for r in records])
        bits = np.array([r["cross_validated_bits_per_spike"] for r in records])
        self.insert1(
            {
                **key,
                "design_file": design["file_path"],
                "design_storage": design["storage"],
                "summary_file": cv_summary["file_path"],
                "summary_storage": cv_summary["storage"],
                "trials": summary["trials"],
                "units": len(records),
                "in_port_flashes": len(data["still"]),
                "after_exit_flashes": len(data["moving"]),
                "excited_units": len(excited_units(subject, session) & ids),
                "suppressed_units": len(
                    excited_units(subject, session, "suppressed") & ids
                ),
                "folds": summary["folds"],
                "video_components": summary["components"],
                "median_d2": np.median(d2),
                "q25_d2": np.percentile(d2, 25),
                "q75_d2": np.percentile(d2, 75),
                "median_bits_per_spike": np.median(bits),
                "q25_bits_per_spike": np.percentile(bits, 25),
                "q75_bits_per_spike": np.percentile(bits, 75),
            }
        )
        self.Unit.insert(
            {
                **key,
                **sorting[0],
                "unit_id": r["unit_id"],
                "d2": r["cross_validated_deviance_explained"],
                "d2_sem": r["cross_validated_deviance_explained_sem"],
                "bits_per_spike": r["cross_validated_bits_per_spike"],
                "l2_penalty": r["alpha"],
                "flash_weight": weight[r["unit_id"]],
                "flash_weight_se": weight_se[r["unit_id"]],
                "flash_weight_class": str(classes[r["unit_id"]]),
            }
            for r in records
        )
        self.ExcludedUnit.insert(
            {**key, **sorting[0], "unit_id": unit, "reason": reason}
            for unit, reason in excluded_units(subject, session).items()
        )
