from __future__ import annotations

import math
import re

from nanogate.models import Completion, ScanResult

FEATURES: tuple[str, ...] = (
    "prompt_log_words",
    "prompt_questions",
    "prompt_has_code",
    "prompt_digit_ratio",
    "prompt_reasoning_cues",
    "answer_log_tokens",
    "answer_hedges",
    "logprob_mean",
    "logprob_min",
    "logprob_low_fraction",
    "logprob_missing",
    "data_class_rank",
    "cache_similarity",
)

LOW_LOGPROB = math.log(0.5)
_REASONING = re.compile(
    r"\b(?:why|prove|derive|calculate|compute|compare|explain|step by step|how many|how much|estimate|trade-?offs?)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:i(?:'m| am) not sure|i don't know|i do not know|cannot (?:determine|answer)|unclear|might be|"
    r"it depends|as an ai|no information)\b",
    re.IGNORECASE,
)
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|[{};]\s*$|\bSELECT\b.+\bFROM\b", re.MULTILINE)


def build_features(
    prompt: str,
    completion: Completion | None = None,
    scan: ScanResult | None = None,
    cache_similarity: float | None = None,
) -> dict[str, float]:
    words = prompt.split()
    digits = sum(char.isdigit() for char in prompt)
    answer = completion.text if completion else ""
    logprobs = completion.logprobs if completion else ()
    return {
        "prompt_log_words": math.log1p(len(words)),
        "prompt_questions": float(min(prompt.count("?"), 5)),
        "prompt_has_code": 1.0 if _CODE.search(prompt) else 0.0,
        "prompt_digit_ratio": digits / len(prompt) if prompt else 0.0,
        "prompt_reasoning_cues": float(min(len(_REASONING.findall(prompt)), 5)),
        "answer_log_tokens": math.log1p(completion.completion_tokens if completion else 0),
        "answer_hedges": float(min(len(_HEDGE.findall(answer)), 5)),
        "logprob_mean": sum(logprobs) / len(logprobs) if logprobs else 0.0,
        "logprob_min": max(min(logprobs), -20.0) if logprobs else 0.0,
        "logprob_low_fraction": sum(value < LOW_LOGPROB for value in logprobs) / len(logprobs) if logprobs else 0.0,
        "logprob_missing": 0.0 if logprobs else 1.0,
        "data_class_rank": float(scan.data_class) if scan else 0.0,
        "cache_similarity": float(cache_similarity or 0.0),
    }
