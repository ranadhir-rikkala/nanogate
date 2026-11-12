from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanogate.models import RouterScore

UNTRAINED_P_ERROR = 0.5
UNTRAINED_THRESHOLD = 0.25


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def interpolate(xs: Sequence[float], ys: Sequence[float], value: float) -> float:
    if not xs:
        return value
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    index = bisect_right(xs, value)
    left_x, right_x = xs[index - 1], xs[index]
    if right_x == left_x:
        return ys[index]
    weight = (value - left_x) / (right_x - left_x)
    return ys[index - 1] + weight * (ys[index] - ys[index - 1])


class RouterModel:
    def __init__(self, artifact: Mapping[str, Any] | None = None):
        self.artifact = dict(artifact) if artifact else None
        if self.artifact:
            lengths = {len(self.artifact[key]) for key in ("features", "mean", "scale", "weights")}
            if len(lengths) != 1:
                raise ValueError("router artifact has inconsistent feature dimensions")

    @classmethod
    def from_file(cls, path: str | Path) -> "RouterModel":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return cls(json.load(handle))
        except (OSError, ValueError, KeyError):
            return cls(None)

    @property
    def trained(self) -> bool:
        return self.artifact is not None

    @property
    def threshold(self) -> float:
        return float(self.artifact["threshold"]) if self.artifact else UNTRAINED_THRESHOLD

    def raw(self, features: Mapping[str, float]) -> tuple[float, list[tuple[str, float]]]:
        artifact = self.artifact
        total = float(artifact["bias"])
        contributions = []
        for name, mean, scale, weight in zip(
            artifact["features"], artifact["mean"], artifact["scale"], artifact["weights"]
        ):
            contribution = weight * (float(features.get(name, 0.0)) - mean) / (scale or 1.0)
            total += contribution
            contributions.append((name, contribution))
        return sigmoid(total), contributions

    def score(self, features: Mapping[str, float]) -> RouterScore:
        if not self.artifact:
            return RouterScore(UNTRAINED_P_ERROR, UNTRAINED_THRESHOLD, ("untrained_router",), trained=False)
        raw, contributions = self.raw(features)
        calibration = self.artifact.get("calibration") or {}
        p_error = interpolate(calibration.get("x", []), calibration.get("y", []), raw)
        ranked = sorted(contributions, key=lambda pair: -abs(pair[1]))[:3]
        factors = tuple(f"{name}{'+' if value >= 0 else '-'}" for name, value in ranked)
        return RouterScore(min(1.0, max(0.0, p_error)), self.threshold, factors)
