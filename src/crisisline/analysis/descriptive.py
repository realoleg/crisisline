"""Descriptive measures: none of them is a test of a hypothesis."""

from __future__ import annotations

import re
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from crisisline.analysis.frames import OUTCOMES
from crisisline.unknowns import wilson_interval

CONFIDENCE = 0.95
_QUESTION_END = re.compile(r"[^.!?؟？።।\n]*[?؟？;፧]")


def _rng(seed: int, *key: object) -> np.random.Generator:
    """A generator that depends on the key alone, not on the order of the calls."""
    return np.random.default_rng([seed, zlib.crc32(repr(key).encode())])


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def outcome_share(
    part: pd.DataFrame, outcome: str, replicates: int, seed: int, key: tuple
) -> dict[str, float | int | str] | None:
    """A share with its interval. One response per prompt: Wilson. Several: the
    mean over prompts of the share in each, with a bootstrap over prompts."""
    data = part[part[outcome].notna()]
    if data.empty:
        return None
    per_prompt = data.groupby(["model_id", "prompt_id"], sort=True)[outcome].mean().to_numpy()
    if not data["is_api"].any():
        hits = int(data[outcome].sum())
        low, high = wilson_interval(hits, len(data), CONFIDENCE)
        method = "wilson"
    else:
        rng = _rng(seed, key, outcome)
        draws = rng.integers(len(per_prompt), size=(replicates, len(per_prompt)))
        low, high = _percentile_interval(per_prompt[draws].mean(axis=1))
        method = "bootstrap over prompts"
    return {
        "responses": len(data),
        "prompts": len(per_prompt),
        "share": float(per_prompt.mean()),
        "ci_low": low,
        "ci_high": high,
        "interval": method,
    }


def share_table(
    frame: pd.DataFrame,
    by: Sequence[str],
    replicates: int,
    seed: int,
    outcomes: Sequence[str] = OUTCOMES,
) -> pd.DataFrame:
    """Every outcome in every cell of ``by``, one row each."""
    rows = []
    grouped = frame.assign(lang_mode=frame["lang_mode"].fillna("none")).groupby(list(by))
    for level, part in grouped:
        levels = level if isinstance(level, tuple) else (level,)
        for outcome in outcomes:
            share = outcome_share(part, outcome, replicates, seed, levels)
            if share:
                rows.append({**dict(zip(by, levels, strict=True)), "outcome": outcome, **share})
    return pd.DataFrame(rows)


def foreign_numbers(frame: pd.DataFrame, top: int) -> pd.DataFrame:
    """The numbers most often given as another country's, with those countries."""
    counts: Counter[str] = Counter()
    owners: dict[str, set[str]] = defaultdict(set)
    users: dict[str, Counter[str]] = defaultdict(Counter)
    for iso2, numbers in zip(frame["iso2"], frame["numbers"], strict=True):
        for number in numbers:
            if number["number_class"] == "other":
                counts[number["number"]] += 1
                owners[number["number"]].update(number["other_iso2"])
                users[number["number"]][iso2] += 1
    return pd.DataFrame(
        [
            {
                "number": number,
                "responses": count,
                "belongs_to": ";".join(sorted(owners[number])),
                "top_user_countries": ";".join(f"{c}:{n}" for c, n in users[number].most_common(5)),
            }
            for number, count in counts.most_common(top)
        ]
    )


def outside_scope_map(rows: Sequence[Mapping[str, str]], no_country: str) -> dict[str, str]:
    """Numbers scored for prompts that name no country, to their country outside the 50."""
    return {r["number_norm"]: r["belongs_to_iso2"] for r in rows if r["iso2"] == no_country}


def default_country(
    frame: pd.DataFrame, outside: Mapping[str, str], no_country: str
) -> pd.DataFrame:
    """Per language of prompts that name no country: the share of responses that name a number of
    each country, and of those that name no other country."""
    data = frame[frame["iso2"] == no_country]
    rows = []
    for lang, part in data.groupby("lang"):
        any_count: Counter[str] = Counter()
        only_count: Counter[str] = Counter()
        for scored, numbers in zip(part["default_countries"], part["numbers"], strict=True):
            named = set(scored) | {outside[n["number"]] for n in numbers if n["number"] in outside}
            any_count.update(named)
            if len(named) == 1:
                only_count.update(named)
        rows.extend(
            {
                "lang": lang,
                "country": country,
                "responses": count,
                "share": count / len(part),
                "responses_only_this_country": only_count[country],
                "outside_the_50": country in set(outside.values()),
                "language_responses": len(part),
            }
            for country, count in sorted(any_count.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    return pd.DataFrame(rows)


def asks_location(text: str, words: Sequence[str]) -> bool:
    """A sentence that ends in a question mark and holds one of a language's location words
    (unvalidated)."""
    folded = text.casefold()
    return any(
        any(word.casefold() in sentence for word in words)
        for sentence in _QUESTION_END.findall(folded)
    )


def location_split(
    frame: pd.DataFrame,
    texts: Mapping[tuple[str, str, int], str],
    words: Mapping[str, Sequence[str]],
    no_country: str,
) -> pd.DataFrame:
    """Prompts that name no country, answers with ``no_resource``: asks for the location or nothing,
    per model."""
    data = frame[(frame["iso2"] == no_country) & (frame["no_resource"] == 1)]
    asked = [
        asks_location(texts.get((m, p, s), ""), words[lang])
        for m, p, s, lang in zip(
            data["model_id"], data["prompt_id"], data["sample_idx"], data["lang"], strict=True
        )
    ]
    table = (
        data.assign(asks=np.array(asked, dtype=bool))
        .groupby("model_id")["asks"]
        .agg(no_resource="size", asks_location="sum")
        .reset_index()
    )
    table["nothing"] = table["no_resource"] - table["asks_location"]
    table["validated"] = False
    return table


def georgia_case(
    frame: pd.DataFrame,
    texts: Mapping[tuple[str, str, int], str],
    pattern: re.Pattern[str],
    top: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GE by model and condition, and its most frequent numbers."""
    data = frame[frame["iso2"] == "GE"].copy()
    data["condition_cue"] = data["condition"] + "/" + data["cue"] + "/" + data["lang_mode"]
    data["names_georgia"] = [
        bool(pattern.search(texts.get((m, p, s), "")))
        for m, p, s in zip(data["model_id"], data["prompt_id"], data["sample_idx"], strict=True)
    ]
    table = (
        data.groupby(["model_id", "condition_cue"])
        .agg(
            responses=("prompt_id", "size"),
            correct_local=("correct_local", "mean"),
            wrong_country=("wrong_country", "mean"),
            names_georgia=("names_georgia", "mean"),
        )
        .reset_index()
    )
    counts = Counter(n["number"] for numbers in data["numbers"] for n in numbers)
    numbers = pd.DataFrame(counts.most_common(top), columns=["number", "responses"])
    return table, numbers


def blocked_share(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """A model's blocked share per condition."""
    data = frame[frame["model_id"] == model]
    return (
        data.assign(lang_mode=data["lang_mode"].fillna("none"))
        .groupby(["condition", "lang_mode", "cue", "type"])
        .agg(responses=("blocked", "size"), blocked=("blocked", "sum"))
        .assign(share=lambda t: t["blocked"] / t["responses"])
        .reset_index()
    )


def temperature_comparison(
    main: pd.DataFrame,
    robustness: pd.DataFrame,
    outcomes: Sequence[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Temperature 0.7 (three responses) against 0 on the same prompts, per model and block; the
    interval of the paired difference resamples prompts."""
    rows = []
    for (block, model), part in robustness.groupby(["block", "model_id"]):
        cold = main[(main["model_id"] == model) & main["prompt_id"].isin(part["prompt_id"])]
        cold = cold.set_index("prompt_id")
        for outcome in outcomes:
            warm = part[part[outcome].notna()].groupby("prompt_id")[outcome].mean()
            paired = warm.index.intersection(cold.index[cold[outcome].notna()])
            if not len(paired):
                continue
            base = cold.loc[paired, outcome].to_numpy()
            values = warm.loc[paired].to_numpy()
            rng = _rng(seed, block, model, outcome)
            draws = rng.integers(len(paired), size=(replicates, len(paired)))
            low, high = _percentile_interval((values - base)[draws].mean(axis=1))
            rows.append(
                {
                    "block": block,
                    "model_id": model,
                    "outcome": outcome,
                    "prompts": len(paired),
                    "share_t0": float(base.mean()),
                    "share_t07": float(values.mean()),
                    "difference": float((values - base).mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "prompts_all_three_agree_with_t0": float(np.mean(values == base)),
                }
            )
    return pd.DataFrame(rows)


def error_type(row: Mapping[str, object]) -> str:
    """The one error class of a response, in the order of the error-type figure."""
    order: Sequence[tuple[Callable[[Mapping[str, object]], bool], str]] = (
        (lambda r: r["correct_local"] == 1, "correct number"),
        (lambda r: r["wrong_country"] == 1, "foreign number only"),
        (lambda r: r["fabricated"] == 1, "fabricated number"),
        (lambda r: bool(r["has_numbers"]), "other number"),
        (lambda r: r["mentions_directory"] == 1, "directory only"),
    )
    return next((label for test, label in order if test(row)), "no resource")


ERROR_TYPES = (
    "correct number",
    "foreign number only",
    "fabricated number",
    "other number",
    "directory only",
    "no resource",
)


def error_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Each model's responses to prompts about a country, by error class."""
    data = frame[frame["correct_local"].notna()]
    labels = [error_type(row) for row in data.to_dict("records")]
    table = data.assign(error_type=labels).groupby(["model_id", "error_type"]).size()
    return (
        table.rename("responses")
        .reset_index()
        .assign(
            share=lambda t: t["responses"] / t.groupby("model_id")["responses"].transform("sum")
        )
    )


REFERRAL_CLASSES = (
    "correct_crisis_line",
    "correct_local_not_crisis_line",
    "misrouted",
    "fabricated",
    "unconfirmed_number",
    "nothing",
)


def referral_class(row: Mapping[str, object]) -> str:
    """The one referral outcome of a response to a prompt about a country, by priority: the
    country's crisis line; another correct local number (emergency, youth or text line); a foreign
    number; a fabricated number; a number neither in the ground truth nor confirmed by hand
    (unknown, unverified, another service); nothing (no number: a directory only, no resource, a
    loop, blocked or empty)."""
    order: Sequence[tuple[Callable[[Mapping[str, object]], bool], str]] = (
        (lambda r: r["correct_crisis"] == 1, "correct_crisis_line"),
        (lambda r: r["correct_local"] == 1, "correct_local_not_crisis_line"),
        (lambda r: r["wrong_country"] == 1, "misrouted"),
        (lambda r: r["fabricated"] == 1, "fabricated"),
        (lambda r: bool(r["has_numbers"]), "unconfirmed_number"),
    )
    return next((label for test, label in order if test(row)), "nothing")


def referral_outcomes(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Each model's share of responses in each referral class, and ``correct_local`` (the first two
    classes together), with intervals as in :func:`share_table`. Prompts about a country with a
    crisis line only, so that the classes sum to one over the denominator of ``correct_crisis``.
    The first class is ``correct_crisis`` on these rows and is computed under that name: its share
    and interval are the ones of ``correct_crisis`` in :func:`share_table` by model."""
    data = frame[frame["correct_crisis"].notna()]
    labels = np.array([referral_class(row) for row in data.to_dict("records")])
    indicators = {
        "correct_crisis": labels == REFERRAL_CLASSES[0],
        **{name: labels == name for name in REFERRAL_CLASSES[1:]},
        "correct_local": (data["correct_local"] == 1).to_numpy(),
    }
    classes = data[["model_id", "prompt_id", "lang_mode", "is_api"]].assign(
        **{name: values.astype(float) for name, values in indicators.items()}
    )
    table = share_table(classes, ["model_id"], replicates, seed, list(indicators))
    table["outcome"] = table["outcome"].replace({"correct_crisis": REFERRAL_CLASSES[0]})
    return table.rename(columns={"outcome": "referral"})
