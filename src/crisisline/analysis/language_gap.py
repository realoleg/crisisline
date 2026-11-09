"""The gap between English and local-language prompts beyond ``language_en_vs_local``
(exploratory).

Per model: the model of ``language_en_vs_local`` within one model. Across models: whether the gap
grows as models get weaker, with a model's strength taken from its English prompts. Per language:
whether the gap follows the quality of the translation, and whether the across-model result
survives adjusting for it. Models are GEE logits with country clusters, robust errors and an
independence working correlation, as ``language_en_vs_local``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from crisisline.analysis import hypotheses, intervals
from crisisline.analysis.descriptive import _rng
from crisisline.analysis.frames import non_english_cued

PER_MODEL_FORMULA = "correct_local ~ is_country * is_en + is_disclosure"
STRENGTH_FORMULA = "correct_local ~ is_en + is_en:strength + is_disclosure + C(model_id)"
CHRF_TERM = "local_chrf"
EN_AT_COUNTRY = {"is_en": 1.0, "is_country:is_en": 1.0}


def gap_data(frame: pd.DataFrame) -> pd.DataFrame:
    """The country cue only: the comparison at which language_en_vs_local is stated."""
    return non_english_cued(frame)[lambda d: d["cue"] == "country"]


def _spec(
    name: str,
    formula: str,
    data_of: Callable[[pd.DataFrame], pd.DataFrame],
    contrast: Mapping[str, float],
    effect: str,
) -> hypotheses.GeeSpec:
    return hypotheses.GeeSpec(
        name,
        formula,
        data_of,
        lambda _: dict(contrast),
        hypotheses.TWO_SIDED,
        effect,
        exchangeable=False,
    )


def per_model(
    frame: pd.DataFrame, groups: Mapping[str, Sequence[str]], replicates: int, seed: int
) -> pd.DataFrame:
    """Each model's odds ratio of English against local at the country cue, from the model of
    ``language_en_vs_local`` fitted to that model alone, next to the two shares and their
    difference."""
    group_of = intervals.group_of(groups)
    rows = []
    for model in sorted(frame["model_id"].unique()):
        spec = _spec(
            f"gap {model}",
            PER_MODEL_FORMULA,
            lambda f, m=model: non_english_cued(f)[lambda d: d["model_id"] == m],
            EN_AT_COUNTRY,
            "OR en vs local at cue = country",
        )
        result = hypotheses.run_gee(spec, frame, model, {})
        data = gap_data(frame[frame["model_id"] == model])
        en, local = data[data["lang_mode"] == "en"], data[data["lang_mode"] == "local"]
        shares = {
            name: intervals.share(arm, "correct_local", replicates, seed, (model, name))
            for name, arm in (("en", en), ("local", local))
        }
        gap = intervals.difference(en, local, "correct_local", replicates, seed, (model, "gap"))
        rows.append(
            {
                "model_group": group_of[model],
                "model_id": model,
                "share_en": shares["en"]["estimate"],
                "share_en_ci_low": shares["en"]["ci_low"],
                "share_en_ci_high": shares["en"]["ci_high"],
                "share_local": shares["local"]["estimate"],
                "share_local_ci_low": shares["local"]["ci_low"],
                "share_local_ci_high": shares["local"]["ci_high"],
                "difference_en_minus_local": gap["estimate"],
                "difference_ci_low": gap["ci_low"],
                "difference_ci_high": gap["ci_high"],
                "odds_ratio": result.estimate,
                "or_ci_low": result.ci_low,
                "or_ci_high": result.ci_high,
                "p_two_sided": result.p,
                "converged": result.converged,
                "n": result.n,
                "clusters": result.clusters,
                "note": result.note,
            }
        )
    return pd.DataFrame(rows)


def strength(frame: pd.DataFrame, leave_out_gap_data: bool = False) -> pd.Series:
    """Each model's ``correct_local`` over its English prompts that name a country, every
    condition. ``leave_out_gap_data`` drops the English arm of :func:`gap_data`, so that strength
    does not share responses with the outcome it explains."""
    data = frame[(frame["lang"] == "en") & frame["correct_local"].notna()]
    if leave_out_gap_data:
        data = data.drop(gap_data(data).index)
    units = data.groupby(["model_id", "prompt_id"])["correct_local"].mean()
    return units.groupby("model_id").mean().rename("strength")


@dataclass(frozen=True, slots=True)
class StrengthRow:
    variant: str
    term: str
    estimate: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n: int
    clusters: int
    converged: bool
    note: str


def _fit_row(variant: str, term: str, spec: hypotheses.GeeSpec, data: pd.DataFrame) -> StrengthRow:
    result = hypotheses.run_gee(spec, data, variant, {})
    return StrengthRow(
        variant,
        term,
        result.estimate,
        result.ci_low,
        result.ci_high,
        result.p,
        result.n,
        result.clusters,
        result.converged,
        result.note,
    )


def strength_interaction(
    frame: pd.DataFrame,
    model_strength: pd.Series,
    step: float,
    variant: str,
    chrf: pd.Series | None = None,
    chrf_step: float = 10.0,
) -> pd.DataFrame:
    """``lang_mode x strength`` at the country cue, with model fixed effects (strength itself is a
    property of the model and is absorbed by them). Reported as odds ratios: the interaction per
    ``step`` of strength (below 1: the English advantage grows as strength falls), and the gap at
    the weakest, average and strongest model. With ``chrf`` (per row, the back-translation chrF
    of the local-language prompt), its centred value enters the local arm per ``chrf_step``."""
    data = frame.assign(strength=frame["model_id"].map(model_strength))
    formula = STRENGTH_FORMULA
    if chrf is not None:
        local = gap_data(data)[lambda d: d["lang_mode"] == "local"]
        if chrf.loc[local.index].isna().any():
            raise ValueError("a local-language prompt has no translation chrF")
        centre = chrf.loc[local.index].groupby(local["prompt_id"]).first().mean()
        data[CHRF_TERM] = ((chrf - centre) / chrf_step).where(data["is_en"] == 0, 0.0)
        formula += f" + {CHRF_TERM}"
    values = model_strength.loc[sorted(gap_data(data)["model_id"].unique())]
    contrasts: list[tuple[str, dict[str, float]]] = [
        (f"interaction: OR of en vs local per +{step:g} strength", {"is_en:strength": step}),
        *(
            (f"OR en vs local at strength {s:.3f} ({label})", {"is_en": 1.0, "is_en:strength": s})
            for label, s in (
                ("weakest", values.min()),
                ("mean of models", values.mean()),
                ("strongest", values.max()),
            )
        ),
    ]
    if chrf is not None:
        contrasts.append((f"OR per +{chrf_step:g} chrF, local arm", {CHRF_TERM: 1.0}))
    rows = [
        _fit_row(variant, term, _spec(variant, formula, gap_data, weights, term), data)
        for term, weights in contrasts
    ]
    return pd.DataFrame([asdict(r) for r in rows]).assign(models=len(values))


def leave_one_model_out(
    frame: pd.DataFrame, model_strength: pd.Series, step: float
) -> pd.DataFrame:
    """The interaction refitted without each model in turn: strength varies only between models,
    and robust errors clustered by country do not reflect how few models there are."""
    rows = []
    for model in sorted(gap_data(frame)["model_id"].unique()):
        kept = frame[frame["model_id"] != model]
        data = kept.assign(strength=kept["model_id"].map(model_strength))
        term = f"interaction: OR of en vs local per +{step:g} strength"
        spec = _spec(f"without {model}", STRENGTH_FORMULA, gap_data, {"is_en:strength": step}, term)
        rows.append(_fit_row(f"without {model}", term, spec, data))
    return pd.DataFrame([asdict(r) for r in rows]).assign(models=len(model_strength) - 1)


def translation_quality(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Per language: entries, median back-translation chrF and the share with any QA flag."""
    table = pd.DataFrame(
        {
            "lang": [e["lang"] for e in entries],
            "chrf": [float(e["chrf"]) for e in entries],
            "flagged": [bool(e.get("flags")) for e in entries],
        }
    )
    return (
        table.groupby("lang")
        .agg(
            entries=("chrf", "size"), median_chrf=("chrf", "median"), flag_rate=("flagged", "mean")
        )
        .reset_index()
    )


def prompt_chrf(frame: pd.DataFrame, entries: Sequence[Mapping[str, Any]]) -> pd.Series:
    """Per row of ``frame``, the chrF of the translation its prompt was built from (NaN for
    English prompts and prompts without a translated cue)."""
    by_key = {(e["iso2"], e["cue"], e["template"]): float(e["chrf"]) for e in entries}
    keys = zip(frame["iso2"], frame["cue"], frame["template"], frame["lang_mode"], strict=True)
    return pd.Series(
        [by_key.get((i, c, t), np.nan) if mode == "local" else np.nan for i, c, t, mode in keys],
        index=frame.index,
        dtype=float,
    )


def penalty_by_language(
    frame: pd.DataFrame, quality: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    """Per language, English minus local ``correct_local`` at the country cue, all models, with
    the language's translation QA."""
    data = gap_data(frame)
    local_lang = data[data["lang_mode"] == "local"].groupby("iso2")["lang"].first()
    rows = []
    for lang, part in data.groupby(data["iso2"].map(local_lang), sort=True):
        en, local = part[part["lang_mode"] == "en"], part[part["lang_mode"] == "local"]
        result = intervals.difference(
            en, local, "correct_local", replicates, seed, (lang, "penalty")
        )
        rows.append(
            {
                "lang": lang,
                "countries": part["iso2"].nunique(),
                "share_en": intervals.unit_means(en, "correct_local")["value"].mean(),
                "share_local": intervals.unit_means(local, "correct_local")["value"].mean(),
                "penalty_en_minus_local": result["estimate"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "cluster": result["cluster"],
            }
        )
    return pd.DataFrame(rows).merge(quality, on="lang", how="left")


def correlations(by_language: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Spearman and Pearson correlation of the penalty with each QA measure over languages, with a
    bootstrap interval over languages."""
    rows = []
    y = by_language["penalty_en_minus_local"].to_numpy()
    for measure in ("median_chrf", "flag_rate"):
        x = by_language[measure].to_numpy()
        for method, function in (("spearman", stats.spearmanr), ("pearson", stats.pearsonr)):
            estimate = function(x, y)
            rng = _rng(seed, measure, method)
            draws = rng.integers(len(x), size=(replicates, len(x)))
            values = []
            for row in draws:
                if np.ptp(x[row]) > 0 and np.ptp(y[row]) > 0:
                    values.append(function(x[row], y[row]).statistic)
            low, high = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "measure": measure,
                    "method": method,
                    "languages": len(x),
                    "correlation": float(estimate.statistic),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_two_sided": float(estimate.pvalue),
                }
            )
    return pd.DataFrame(rows)
