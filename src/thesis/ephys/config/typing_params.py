"""Structured parameter dicts consumed by typed analysis helpers (ty-friendly)."""

from __future__ import annotations

from typing import TypedDict


class PopulationPethParams(TypedDict):
    pre_seconds: float
    post_seconds: float
    binwidth_ms: int
    t_rise: float | None
    t_decay: float | None


class UnitSelectivityParams(TypedDict):
    base_window: tuple[float, float]
    resp_window: tuple[float, float]
    test: str
    correction: str
    alpha: float


class PeakCountParams(TypedDict):
    search_window: tuple[float, float]
    baseline_window: tuple[float, float]
    min_prominence_frac: float
    min_distance_ms: float
    binwidth_ms: float
