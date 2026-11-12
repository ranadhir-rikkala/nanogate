import random

import pytest

from nanogate.dlp import Scanner
from nanogate.models import Completion, Route
from nanogate.router import FEATURES, RouterModel, build_features, save, train
from nanogate.router.model import interpolate
from nanogate.router.train import auc, isotonic, pick_threshold, split_of


def completion(text="Open Settings.", logprobs=(-0.05, -0.1)):
    return Completion(text=text, model="m", route=Route.LOCAL_SMALL, completion_tokens=len(text.split()), logprobs=logprobs)


def synthetic_rows(count=600, seed=7):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        hard = rng.random() < 0.4
        mean = rng.gauss(-1.6 if hard else -0.3, 0.35)
        features = dict.fromkeys(FEATURES, 0.0)
        features.update(
            logprob_mean=mean,
            logprob_min=mean * 3,
            logprob_low_fraction=min(1.0, max(0.0, -mean / 3)),
            prompt_reasoning_cues=float(hard and rng.random() < 0.7),
            prompt_log_words=rng.uniform(1, 4),
        )
        error = int(rng.random() < (0.75 if hard else 0.05))
        rows.append({"features": features, "error": error, "group": f"g{index // 3}"})
    return rows


def test_features_cover_the_declared_schema():
    features = build_features("Why is 12 * 12 = 144? Explain step by step.", completion(), Scanner().scan("x"), 0.4)
    assert set(features) == set(FEATURES)
    assert features["prompt_reasoning_cues"] >= 2 and features["prompt_questions"] == 1
    assert features["logprob_missing"] == 0.0 and features["cache_similarity"] == 0.4


def test_features_without_a_completion_or_logprobs():
    features = build_features("hello")
    assert features["logprob_missing"] == 1.0 and features["answer_log_tokens"] == 0.0
    hedged = build_features("q", completion("I'm not sure, it depends.", logprobs=()))
    assert hedged["answer_hedges"] == 2.0 and hedged["logprob_missing"] == 1.0


def test_untrained_router_is_pessimistic(tmp_path):
    model = RouterModel.from_file(tmp_path / "missing.json")
    score = model.score(build_features("hi", completion()))
    assert model.trained is False and score.trained is False
    assert score.confident is False


def test_isotonic_output_is_monotone():
    xs, ys = isotonic([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 0, 0, 1, 1])
    assert ys == sorted(ys) and xs == sorted(xs)
    assert interpolate(xs, ys, -1.0) == ys[0] and interpolate(xs, ys, 2.0) == ys[-1]
    assert ys[0] <= interpolate(xs, ys, 0.35) <= ys[-1]


def test_threshold_keeps_accepted_error_within_target():
    probabilities = [0.05, 0.1, 0.15, 0.2, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 0, 1, 1]
    assert pick_threshold(probabilities, labels, target_error=0.0) == 0.2
    assert pick_threshold(probabilities, labels, target_error=0.2) == 0.7
    assert pick_threshold([0.9], [1], target_error=0.1) == 0.0


def test_auc_basics():
    assert auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert auc([0.5, 0.6], [1, 1]) is None


def test_groups_never_straddle_splits():
    assert len({split_of("same-question", seed=3) for _ in range(5)}) == 1
    assert {split_of(f"g{n}") for n in range(200)} == {"train", "calibrate", "test"}


def test_training_learns_signal_and_round_trips(tmp_path):
    artifact = train(synthetic_rows(), target_error=0.1)
    held_out = artifact["held_out"]
    assert held_out["auc"] > 0.8
    assert held_out["accepted_error_rate"] < held_out["base_error_rate"]
    assert 0.0 < artifact["threshold"] < 1.0

    path = tmp_path / "router.json"
    save(artifact, path)
    model = RouterModel.from_file(path)
    easy = dict.fromkeys(FEATURES, 0.0) | {"logprob_mean": -0.1, "logprob_min": -0.3}
    hard = dict.fromkeys(FEATURES, 0.0) | {"logprob_mean": -2.0, "logprob_min": -6.0, "logprob_low_fraction": 0.7}
    assert model.score(easy).p_error < model.score(hard).p_error
    assert model.score(easy).confident and not model.score(hard).confident
    assert len(model.score(hard).top_factors) == 3


def test_training_refuses_tiny_datasets_and_bad_artifacts():
    with pytest.raises(ValueError):
        train(synthetic_rows(10))
    with pytest.raises(ValueError):
        RouterModel({"features": ["a"], "mean": [0, 0], "scale": [1], "weights": [1], "bias": 0, "threshold": 0.2})
