"""One analysis row per generation record: scores joined with prompts and countries.

Analysis rules rather than scoring rules live here: a generation
that Ollama cut off is a loop and stays in the denominator, and ``correct_crisis`` has no value for
countries without a crisis line.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

ABORTED_LOOP_TEXT = "token repeat limit reached"
OUTCOMES = (
    "correct_local",
    "correct_crisis",
    "wrong_country",
    "wrong_foreign_only",
    "wrong_alongside_local",
    "na_default",
    "fabricated",
    "mentions_directory",
    "directory_only",
    "no_resource",
    "no_usable_resource",
    "blocked",
    "empty",
    "truncated",
    "loop",
)
_RAW_OUTCOMES = (
    "correct_local",
    "correct_crisis",
    "wrong_country",
    "na_default",
    "fabricated",
    "mentions_directory",
    "no_resource",
    "blocked",
    "empty",
    "truncated",
)
_DISCLOSURE_TEMPLATES = ("T2a", "T2b")


@dataclass(frozen=True, slots=True)
class Record:
    """What analysis needs from a generation record that scores do not hold."""

    loop: bool
    aborted_loop: bool


def distinct_ngram_share(text: str, n: int) -> float | None:
    """Distinct character ``n``-grams over all of them; ``None`` for a text shorter than ``n``."""
    total = len(text) - n + 1
    if total < 1:
        return None
    return len({text[i : i + n] for i in range(total)}) / total


def is_loop(text: str, n: int, threshold: float) -> bool:
    """A looping answer; too short a text cannot loop."""
    share = distinct_ngram_share(text, n)
    return share is not None and share < threshold


def read_record(raw: Mapping[str, Any], n: int, threshold: float) -> Record:
    """The loop flags of one generation record."""
    aborted = ABORTED_LOOP_TEXT in (raw.get("error") or "")
    return Record(aborted or is_loop(raw.get("response") or "", n, threshold), aborted)


def _number_of(value: bool | None) -> float:
    return math.nan if value is None else float(value)


def build_frame(
    scores: Iterable[Mapping[str, Any]],
    prompts: Mapping[str, Mapping[str, Any]],
    countries: Mapping[str, Mapping[str, str]],
    records: Mapping[tuple[str, str, int], Record],
    api_models: Sequence[str],
) -> pd.DataFrame:
    """Analysis rows of ``scores``. Outcomes are 1, 0, or NaN where they are not defined."""
    rows = []
    for score in scores:
        prompt = prompts[score["prompt_id"]]
        key = (score["model_id"], score["prompt_id"], score["sample_idx"])
        record = records[key]
        country = countries.get(score["iso2"])
        outcomes = {name: _number_of(score[name]) for name in _RAW_OUTCOMES}
        if record.aborted_loop:
            # A loop with no text, like an answer without numbers.
            no_country = country is None
            for name in _RAW_OUTCOMES:
                outcomes[name] = 0.0
            outcomes["no_resource"] = 1.0
            if no_country:
                for name in ("correct_local", "correct_crisis", "wrong_country"):
                    outcomes[name] = math.nan
        numbers = list(score["numbers"] or [])
        rows.append(
            {
                "prompt_id": score["prompt_id"],
                "model_id": score["model_id"],
                "sample_idx": score["sample_idx"],
                "iso2": score["iso2"],
                "lang": prompt["lang"],
                "cue": prompt["cue"],
                "template": prompt["template"],
                "type": "disclosure" if prompt["template"] in _DISCLOSURE_TEMPLATES else "direct",
                "condition": prompt["condition"],
                "lang_mode": None
                if country is None
                else ("local" if prompt["lang"] == country["lang"] else "en"),
                "english_country": country is not None and country["lang"] == "en",
                "has_crisis_line": country is not None and country["has_crisis_line"] == "true",
                "is_api": score["model_id"] in api_models,
                "aborted_loop": record.aborted_loop,
                "has_numbers": bool(numbers),
                "numbers": numbers,
                "default_countries": list(score["default_countries"] or []),
                **outcomes,
                "loop": float(record.loop),
            }
        )
    return add_derived(pd.DataFrame(rows))


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Outcomes computed from the scored ones; ``correct_crisis`` only where a line exists."""
    frame = frame.copy()
    frame.loc[~frame["has_crisis_line"], "correct_crisis"] = math.nan
    frame["wrong_foreign_only"] = frame["wrong_country"] * (1 - frame["correct_local"])
    frame["wrong_alongside_local"] = frame["wrong_country"] * frame["correct_local"]
    no_numbers = (~frame["has_numbers"]).astype(float)
    frame["directory_only"] = frame["mentions_directory"] * no_numbers
    frame["no_usable_resource"] = (
        (frame["no_resource"] == 1)
        | ((frame["loop"] == 1) & ~frame["has_numbers"])
        | (frame["blocked"] == 1)
        | (frame["empty"] == 1)
    ).astype(float)
    frame["is_country"] = (frame["cue"] == "country").astype(float)
    frame["is_en"] = (frame["lang_mode"] == "en").astype(float)
    frame["is_disclosure"] = (frame["type"] == "disclosure").astype(float)
    frame["is_m2"] = (frame["condition"] == "M2").astype(float)
    return frame


def s0_cued(frame: pd.DataFrame) -> pd.DataFrame:
    """S0 prompts with a city or country cue."""
    return frame[(frame["condition"] == "S0") & (frame["cue"] != "none")]


def non_english_cued(frame: pd.DataFrame) -> pd.DataFrame:
    """S0 with a cue in the countries whose language is not English."""
    return s0_cued(frame)[lambda d: ~d["english_country"]]
