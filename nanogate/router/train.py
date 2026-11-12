from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nanogate.router.features import FEATURES
from nanogate.router.model import RouterModel, sigmoid

MIN_ROWS = 60


def split_of(group: str, seed: int = 0) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "calibrate" if bucket < 8 else "test"


def standardize(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    count = len(rows)
    width = len(rows[0])
    mean = [sum(row[i] for row in rows) / count for i in range(width)]
    scale = [math.sqrt(sum((row[i] - mean[i]) ** 2 for row in rows) / count) or 1.0 for i in range(width)]
    return mean, scale


def fit_logistic(
    rows: Sequence[Sequence[float]], labels: Sequence[int], epochs: int = 400, rate: float = 0.5, l2: float = 0.01
) -> tuple[list[float], float]:
    width, count = len(rows[0]), len(rows)
    weights, bias = [0.0] * width, 0.0
    for _ in range(epochs):
        gradient, bias_gradient = [0.0] * width, 0.0
        for row, label in zip(rows, labels):
            error = sigmoid(bias + sum(w * x for w, x in zip(weights, row))) - label
            bias_gradient += error
            for i in range(width):
                gradient[i] += error * row[i]
        for i in range(width):
            weights[i] -= rate * (gradient[i] / count + l2 * weights[i])
        bias -= rate * bias_gradient / count
    return weights, bias


def isotonic(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    blocks: list[list[float]] = []
    for score, label in sorted(zip(scores, labels)):
        blocks.append([score, score, float(label), 1.0])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] >= blocks[-1][2] / blocks[-1][3]:
            low, high, total, weight = blocks.pop()
            blocks[-1][1] = high
            blocks[-1][2] += total
            blocks[-1][3] += weight
    xs, ys = [], []
    for low, high, total, weight in blocks:
        xs.append((low + high) / 2.0)
        ys.append(total / weight)
    return xs, ys


def pick_threshold(probabilities: Sequence[float], labels: Sequence[int], target_error: float) -> float:
    best, errors = 0.0, 0
    for accepted, (probability, label) in enumerate(sorted(zip(probabilities, labels)), start=1):
        errors += label
        if errors / accepted <= target_error:
            best = probability
    return round(best, 6)


def auc(probabilities: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [p for p, label in zip(probabilities, labels) if label == 1]
    negatives = [p for p, label in zip(probabilities, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


def evaluate(model: RouterModel, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    probabilities = [model.score(row["features"]).p_error for row in rows]
    labels = [int(row["error"]) for row in rows]
    accepted = [label for p, label in zip(probabilities, labels) if p <= model.threshold]
    return {
        "n": len(rows),
        "base_error_rate": round(sum(labels) / len(labels), 4),
        "auc": auc(probabilities, labels),
        "brier": round(sum((p - label) ** 2 for p, label in zip(probabilities, labels)) / len(labels), 4),
        "coverage": round(len(accepted) / len(labels), 4),
        "accepted_error_rate": round(sum(accepted) / len(accepted), 4) if accepted else None,
    }


def train(rows: Iterable[Mapping[str, Any]], target_error: float = 0.1, seed: int = 0) -> dict[str, Any]:
    rows = list(rows)
    if len(rows) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} graded rows, got {len(rows)}")
    parts: dict[str, list[Mapping[str, Any]]] = {"train": [], "calibrate": [], "test": []}
    for index, row in enumerate(rows):
        parts[split_of(str(row.get("group", index)), seed)].append(row)
    if not parts["train"] or not parts["calibrate"]:
        raise ValueError("grouped split left the train or calibration set empty")

    def matrix(part: str) -> list[list[float]]:
        return [[float(row["features"].get(name, 0.0)) for name in FEATURES] for row in parts[part]]

    def labels(part: str) -> list[int]:
        return [int(row["error"]) for row in parts[part]]

    train_x = matrix("train")
    mean, scale = standardize(train_x)
    scaled = [[(value - m) / s for value, m, s in zip(row, mean, scale)] for row in train_x]
    weights, bias = fit_logistic(scaled, labels("train"))
    artifact: dict[str, Any] = {
        "features": list(FEATURES),
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "bias": bias,
        "calibration": {"x": [], "y": []},
        "threshold": 0.0,
    }
    uncalibrated = RouterModel(artifact)
    raw_scores = [uncalibrated.raw(row["features"])[0] for row in parts["calibrate"]]
    xs, ys = isotonic(raw_scores, labels("calibrate"))
    artifact["calibration"] = {"x": xs, "y": ys}
    calibrated = RouterModel(artifact)
    probabilities = [calibrated.score(row["features"]).p_error for row in parts["calibrate"]]
    artifact["threshold"] = pick_threshold(probabilities, labels("calibrate"), target_error)
    artifact["target_error"] = target_error
    artifact["trained_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    artifact["rows"] = {name: len(part) for name, part in parts.items()}
    artifact["held_out"] = evaluate(RouterModel(artifact), parts["test"])
    return artifact


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(artifact: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
