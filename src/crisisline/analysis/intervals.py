"""Shares and paired differences of shares with a cluster bootstrap.

A share is a mean over units, where a unit is one model's responses to one prompt (three for API
models, averaged first), so every model weighs the same in a pooled cell. Clusters are countries
with all their units. A cell that lies inside one country is resampled by prompt for a share and by
template for a difference, whose two arms then pair on the template. Both arms of a difference are
resampled with the same draw of clusters.
"""

from __future__ import annotations

import zlib
from collections.abc import Hashable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

ALL = "all"
COUNTRY = "iso2"
PROMPT = "prompt_id"
TEMPLATE = "template"
ZERO = 1e-9  # a bound that is 0 comes out as about ±1e-16, its sign depending on the platform


def rng_for(seed: int, *key: object) -> np.random.Generator:
    """A generator that depends on the seed and the key alone, not on the order of the calls."""
    return np.random.default_rng([seed, zlib.crc32(repr(key).encode())])


def unit_means(part: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """One row per model and prompt with the mean outcome over its responses."""
    data = part[part[outcome].notna()]
    return (
        data.groupby(["model_id", PROMPT], sort=True)
        .agg(value=(outcome, "mean"), iso2=(COUNTRY, "first"), template=(TEMPLATE, "first"))
        .reset_index()
    )


def cluster_sums(
    values: pd.Series, clusters: pd.Series, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of ``values`` in each of ``levels`` of ``clusters`` (zero where absent)."""
    grouped = values.groupby(clusters.to_numpy()).agg(["sum", "size"]).reindex(levels, fill_value=0)
    return grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)


def cluster_draws(rng: np.random.Generator, clusters: int, replicates: int) -> np.ndarray:
    """How often each cluster is drawn, per replicate: ``clusters`` draws with replacement."""
    return rng.multinomial(clusters, np.full(clusters, 1 / clusters), size=replicates).astype(float)


def _ratio(weights: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (weights @ sums) / (weights @ counts)


def _percentiles(values: np.ndarray) -> tuple[float, float]:
    """The 95% percentile interval; replicates without a value (an arm left empty) are left out."""
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def ratio_interval(
    values: pd.Series, clusters: pd.Series, replicates: int, rng: np.random.Generator
) -> dict[str, float | int]:
    """The ratio of sums of ``values`` with a cluster bootstrap interval."""
    levels = np.unique(clusters.to_numpy())
    sums, counts = cluster_sums(values, clusters, levels)
    low, high = _percentiles(_ratio(cluster_draws(rng, len(levels), replicates), sums, counts))
    return {
        "estimate": float(sums.sum() / counts.sum()),
        "ci_low": low,
        "ci_high": high,
        "clusters": len(levels),
    }


def share(
    part: pd.DataFrame, outcome: str, replicates: int, seed: int, key: tuple[Hashable, ...]
) -> dict[str, float | int | str] | None:
    """Mean of ``outcome`` over units with a 95% interval; ``None`` where it is never defined."""
    units = unit_means(part, outcome)
    if units.empty:
        return None
    cluster = COUNTRY if units[COUNTRY].nunique() > 1 else PROMPT
    result = ratio_interval(
        units["value"], units[cluster], replicates, rng_for(seed, *key, outcome, "share")
    )
    return {
        **result,
        "units": len(units),
        "responses": int(part[outcome].notna().sum()),
        "cluster": cluster,
    }


def difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    outcome: str,
    replicates: int,
    seed: int,
    key: tuple[Hashable, ...],
) -> dict[str, float | int | str] | None:
    """Share in ``a`` minus share in ``b``, both arms resampled with the same clusters."""
    units_a, units_b = unit_means(a, outcome), unit_means(b, outcome)
    if units_a.empty or units_b.empty:
        return None
    one_country = pd.concat([units_a[COUNTRY], units_b[COUNTRY]]).nunique() == 1
    cluster = TEMPLATE if one_country else COUNTRY
    levels = np.union1d(units_a[cluster].unique(), units_b[cluster].unique())
    sums_a, counts_a = cluster_sums(units_a["value"], units_a[cluster], levels)
    sums_b, counts_b = cluster_sums(units_b["value"], units_b[cluster], levels)
    weights = cluster_draws(rng_for(seed, *key, outcome, "difference"), len(levels), replicates)
    draws = _ratio(weights, sums_a, counts_a) - _ratio(weights, sums_b, counts_b)
    low, high = _percentiles(draws)
    return {
        "estimate": float(sums_a.sum() / counts_a.sum() - sums_b.sum() / counts_b.sum()),
        "ci_low": low,
        "ci_high": high,
        "clusters": len(levels),
        "units": len(units_a) + len(units_b),
        "responses": int(a[outcome].notna().sum() + b[outcome].notna().sum()),
        "cluster": cluster,
    }


def excludes_zero(low: float, high: float) -> bool:
    """Whether the interval from ``low`` to ``high`` lies on one side of zero; a bound within
    :data:`ZERO` of it is zero."""
    return bool(low > ZERO or high < -ZERO)


def group_of(model_groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {model: group for group, models in model_groups.items() for model in models}


def scopes(
    frame: pd.DataFrame, model_groups: Mapping[str, Sequence[str]], per_model: bool = True
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """``(model_group, model_id, rows)``: each model, then each group and all models pooled, whose
    ``model_id`` is ``all``."""
    groups = group_of(model_groups)
    if per_model:
        for model, part in frame.groupby("model_id", sort=True):
            yield groups[str(model)], str(model), part
    for group, models in model_groups.items():
        yield group, ALL, frame[frame["model_id"].isin(models)]
    yield ALL, ALL, frame
