"""Example candidates by rule. A human makes the final selection and checks them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from crisisline.extract import normalize
from crisisline.score import is_correct_class

Row = Mapping[str, Any]
MARK = "**"


def snippet(text: str, start: int | None, raw: str, limit: int) -> str:
    """At most ``limit`` characters on one line, with the number marked; the start of the text if
    there is no number."""
    if start is None:
        return " ".join(text.split())[:limit]
    room = limit - len(raw) - 2 * len(MARK)
    before = text[max(0, start - room // 2) : start]
    after = text[start + len(raw) : start + len(raw) + room - len(before)]
    return " ".join(f"{before}{MARK}{raw}{MARK}{after}".split())


def _first(row: Row, test: Callable[[Row], bool]) -> Mapping[str, Any] | None:
    return next((n for n in row["numbers"] if test(n)), None)


def same_language_group_number(row: Row, groups: Mapping[str, str]) -> Mapping[str, Any] | None:
    """A number of another country that shares the user's language group."""
    group = groups.get(row["iso2"])
    if not group:
        return None
    return _first(
        row,
        lambda n: (
            n["number_class"] == "other" and any(groups.get(c) == group for c in n["other_iso2"])
        ),
    )


def classes(
    groups: Mapping[str, str],
    na_numbers: Sequence[str],
    asks: Callable[[Row], bool],
) -> dict[str, Callable[[Row], tuple[bool, Mapping[str, Any] | None]]]:
    """Each class: whether a row belongs to it, and the number that shows it."""
    na = set(na_numbers)

    def with_number(found: Mapping[str, Any] | None) -> tuple[bool, Mapping[str, Any] | None]:
        return found is not None, found

    return {
        "na_default": lambda r: with_number(
            _first(r, lambda n: n["number"] in na and not is_correct_class(n["number_class"]))
            if r["na_default"] == 1
            else None
        ),
        "wrong_country_same_language_group": lambda r: with_number(
            same_language_group_number(r, groups) if r["wrong_country"] == 1 else None
        ),
        "fabricated": lambda r: with_number(_first(r, lambda n: n["number_class"] == "fabricated")),
        "local_discontinued": lambda r: with_number(
            _first(r, lambda n: n["number_class"] == "local_discontinued")
        ),
        "directory_only": lambda r: (r["directory_only"] == 1, None),
        "no_resource": lambda r: (r["no_resource"] == 1 and not r["aborted_loop"], None),
        "refusal": lambda r: (r["blocked"] == 1, None),
        "positive_asks_location": lambda r: (r["no_resource"] == 1 and asks(r), None),
        "positive_correct_from_city_cue": lambda r: with_number(
            _first(r, lambda n: is_correct_class(n["number_class"]))
            if r["cue"] == "city" and r["correct_local"] == 1
            else None
        ),
    }


def choose(rows: Sequence[Row], size: int, seed: int) -> list[Row]:
    """Up to ``size`` rows, each step taking one from the model and language used least so far."""
    order = np.random.default_rng(seed).permutation(len(rows))
    remaining = [rows[int(i)] for i in order]
    used_models: Counter[str] = Counter()
    used_langs: Counter[str] = Counter()
    chosen: list[Row] = []
    while remaining and len(chosen) < size:
        best = min(
            range(len(remaining)),
            key=lambda i: (
                used_models[remaining[i]["model_id"]],
                used_langs[remaining[i]["lang"]],
            ),
        )
        row = remaining.pop(best)
        chosen.append(row)
        used_models[row["model_id"]] += 1
        used_langs[row["lang"]] += 1
    return chosen


def candidates(
    frame: pd.DataFrame,
    texts: Mapping[tuple[str, str, int], str],
    details: Mapping[tuple[str, str, int], str],
    tests: Mapping[str, Callable[[Row], tuple[bool, Mapping[str, Any] | None]]],
    per_class: int,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Candidates of every class. The English translation of each snippet and its safety review are
    left empty for a person to fill in."""
    records = frame.to_dict("records")
    examples = []
    for name, test in tests.items():
        found = []
        for row in records:
            belongs, number = test(row)
            if belongs:
                found.append({**row, "_number": number})
        for row in choose(found, per_class, seed):
            key = (row["model_id"], row["prompt_id"], row["sample_idx"])
            text = normalize(texts.get(key, "")) or details.get(key, "")
            number = row["_number"]
            examples.append(
                {
                    "class": name,
                    "prompt_id": row["prompt_id"],
                    "model_id": row["model_id"],
                    "sample_idx": row["sample_idx"],
                    "lang": row["lang"],
                    "snippet": snippet(
                        text,
                        number["start"] if number else None,
                        number["raw"] if number else "",
                        limit,
                    ),
                    "snippet_en": None,
                    "snippet_en_status": "not translated",
                    "safety_review": "not reviewed",
                    "excluded_unsafe": None,
                }
            )
    return examples
