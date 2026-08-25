"""Minimal stub for Chipmunk tables (DataJoint-like join / restrict API)."""

from typing import Any, ClassVar

class ChipmunkMeta(type):
    def __mul__(cls, other: Any) -> Any: ...
    def __and__(cls, other: Any) -> Any: ...

class Chipmunk(metaclass=ChipmunkMeta):
    Trial: ClassVar[Any]
    TrialParameters: ClassVar[Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def trial_query(cls, **key: Any) -> Any: ...
    @classmethod
    def fit_psychometric(
        cls,
        rewarded_modality: str = "visual",
        min_choices: int = 100,
        min_required_stim_values: int = 6,
        **key: Any,
    ) -> dict[str, Any] | None: ...
    @classmethod
    def trial_events(
        cls,
        is_nidq: bool = False,
        observation_window: str = "center_exit",
        **key: Any,
    ) -> list[dict[str, Any]]: ...
    @classmethod
    def fit_psychophysical_kernel(
        cls,
        is_nidq: bool = False,
        observation_window: str = "center_exit",
        **key: Any,
    ) -> dict[str, Any] | None: ...
    def __and__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
