"""Cortical layer of each stable unit, from the flash-evoked CSD and the probe plan."""

import json

from ._base import SpikeSorting, UnitMetrics, dj, np, rojasbowe_schema  # noqa: F401
from .units import UnitStability


def _code_settings() -> str:
    from thesis.ephys.analyses.csd import settings

    return json.dumps(settings(), sort_keys=True)


@rojasbowe_schema
class UnitLayerParam(dj.Lookup):
    """Settings of thesis.ephys.analyses.csd. `make` stops if the code uses other settings.

    The row also names the one unit set of a session that gets layers (a
    session has a UnitStability key for each quality criteria). A new method
    is a new commit and a new row.
    """

    definition = """
    layer_param_id                         : tinyint unsigned
    ---
    description                            : varchar(255)
    unit_criteria_id                       : tinyint unsigned  # quality criteria of the units
    stability_param_id                     : tinyint unsigned  # UnitStability parameter set
    settings                               : varchar(2000)  # JSON of the csd constants that change the result
    """
    default_contents = [
        {
            "layer_param_id": 0,
            "description": "first-flash step iCSD, vvasp plan atlas, MUA offset (2026-11-20)",
            "unit_criteria_id": 1,
            "stability_param_id": 0,
            "settings": _code_settings(),
        }
    ]


@rojasbowe_schema
class UnitLayer(dj.Computed):
    """Atlas layer of the stable units of a session (thesis.ephys.analyses.csd).

    The layers are the Allen VISp layers on the planned probe track, shifted
    by the offset of the measured MUA peaks. Only GRB061, GRB064 and GRB068
    share the plan of the stored atlas layers. Use the layers only where
    `reliable` is 1.
    """

    definition = """
    -> UnitStability
    -> UnitLayerParam
    ---
    offset_um                              : float              # median over shanks of (MUA peak - atlas mid layer 5)
    offset_spread_um                       : float              # range of the per-shank values
    reliable                               : tinyint            # spread within the max_offset_spread_um setting
    n_trials                               : smallint unsigned  # first flashes used
    n_rail                                 : smallint unsigned  # first flashes left out at the ADC limit
    """

    class Shank(dj.Part):
        definition = """
        -> master
        shank                              : tinyint unsigned
        ---
        top_um                             : float          # highest row, um from the tip
        bottom_um                          : float          # lowest row
        sink_um = NULL                     : float          # most negative local CSD minimum in the sink window
        sink_lo_um = NULL                  : float          # 95 percent bootstrap interval over trials
        sink_hi_um = NULL                  : float
        sink_found = NULL                  : float          # fraction of bootstrap draws with a sink
        sink_onset_ms = NULL               : float          # first time below 20 percent of the sink minimum
        sink_peak_ms = NULL                : float          # time of the sink minimum
        lfp_dip_um = NULL                  : float          # most negative local LFP minimum in the sink window
        mua_peak_um = NULL                 : float          # largest local maximum of the MUA power
        """

    class Unit(dj.Part):
        definition = """
        -> master
        -> SpikeSorting.Unit
        ---
        layer                              : enum('below L6', 'L6', 'L5', 'L4', 'L2/3', 'L1', 'outside')
        """

    @property
    def key_source(self):
        from thesis.ephys.analyses.csd import PLAN_SUBJECTS

        param = UnitLayerParam.proj(
            criteria="unit_criteria_id", stability="stability_param_id"
        )
        keys = (UnitStability * param) & [
            {"subject_name": subject} for subject in PLAN_SUBJECTS
        ]
        return (
            keys & "unit_criteria_id = criteria" & "stability_param_id = stability"
        ).proj()

    def make(self, key, **_kwargs):
        from thesis.ephys.analyses import csd

        stored = json.loads((UnitLayerParam & key).fetch1("settings"))
        if stored != json.loads(_code_settings()):
            raise RuntimeError(
                f"thesis.ephys.analyses.csd uses other settings than "
                f"UnitLayerParam {key['layer_param_id']}"
            )
        if key["probe_num"] != 0:
            raise ValueError(f"Only probe 0 (imec0) is handled: {key}")

        path = csd.measure(key["subject_name"], key["session_name"])
        shanks, _ = csd.landmarks(path)
        offset, spread = csd.atlas_offset(shanks)
        if not np.isfinite(offset):
            raise ValueError(f"No MUA peak on any shank for {key}")
        stable = (UnitStability.Unit & key & "passes = 1").proj()
        units = (UnitMetrics & stable).fetch("unit_id", "shank", "depth", as_dict=True)
        missing = [
            u["unit_id"] for u in units if u["shank"] is None or u["depth"] is None
        ]
        if missing:
            raise ValueError(f"Units without shank or depth for {key}: {missing}")
        layers = csd.unit_layers(
            [u["shank"] for u in units], [u["depth"] for u in units], offset
        )

        def value(x):
            return None if x is None or not np.isfinite(x) else float(x)

        self.insert1(
            {
                **key,
                "offset_um": offset,
                "offset_spread_um": spread,
                "reliable": int(spread <= stored["max_offset_spread_um"]),
                "n_trials": int(shanks["n_trials"].iloc[0]),
                "n_rail": int(shanks["n_rail"].iloc[0]),
            }
        )
        columns = [
            "top_um",
            "bottom_um",
            "sink_um",
            "sink_lo_um",
            "sink_hi_um",
            "sink_found",
            "sink_onset_ms",
            "sink_peak_ms",
            "lfp_dip_um",
            "mua_peak_um",
        ]
        self.Shank.insert(
            {**key, "shank": int(row["shank"]), **{c: value(row[c]) for c in columns}}
            for _, row in shanks.iterrows()
        )
        self.Unit.insert(
            {**key, "unit_id": int(unit["unit_id"]), "layer": layer}
            for unit, layer in zip(units, layers, strict=True)
        )
