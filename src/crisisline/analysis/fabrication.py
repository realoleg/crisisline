"""The fabrication share per model: the stratified estimator of ``unknowns --estimate``.

Within a stratum a model's share is taken over its mentions of labelled pairs, and strata are summed
with that model's own mention weights. A stratum where the model has no labelled mention has no
share: the estimate is then reported twice, with that share at 0 and at 1. Pairs are resampled
within sampled strata; the head is a census.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crisisline.score import FABRICATED
from crisisline.unknowns import in_stratum


@dataclass(frozen=True, slots=True)
class ModelStratum:
    """One stratum: per model, mentions over all pairs, and per labelled pair and model."""

    name: str
    census: bool
    all_mentions: np.ndarray  # models
    labelled_mentions: np.ndarray  # labelled pairs x models
    fabricated: np.ndarray  # labelled pairs


def build_strata(
    specs: Sequence[Mapping[str, object]],
    flags: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, str]]],
    mentions: Mapping[str, Mapping[str, int]],
    models: Sequence[str],
    label: str = FABRICATED,
) -> list[ModelStratum]:
    """Strata from the config of ``unknowns.estimate``. ``mentions`` maps a pair to its mentions per
    model; ``labels`` maps a stratum's path to the rows of its labelling file. ``fabricated`` marks
    the pairs with ``label``."""
    strata = []
    for spec in specs:
        members = {
            row["pair_id"]
            for row in flags
            if in_stratum(row, spec["bands"], spec.get("impossible"))
        }
        rows = [r for r in labels[str(spec["path"])] if r["pair_id"] in members]
        matrix = np.array(
            [[mentions[r["pair_id"]].get(m, 0) for m in models] for r in rows], dtype=float
        )
        totals = np.zeros(len(models))
        for pair in members:
            totals += [mentions[pair].get(m, 0) for m in models]
        strata.append(
            ModelStratum(
                str(spec["name"]),
                len(rows) == len(members),
                totals,
                matrix,
                np.array([r["label"] == label for r in rows]),
            )
        )
    return strata


def _estimate(
    strata: Sequence[ModelStratum], weights: Sequence[np.ndarray], empty_value: float
) -> np.ndarray:
    """Shares per replicate (rows) and model (columns); ``weights[h]`` counts each labelled pair of
    stratum ``h`` in each replicate."""
    total = sum(s.all_mentions for s in strata)
    shares = stratum_shares(strata, weights, empty_value)
    result = 0.0
    for h, stratum in enumerate(strata):
        share = shares[:, h]
        result = result + np.where(stratum.all_mentions > 0, share, 0.0) * stratum.all_mentions
    return result / np.where(total > 0, total, np.nan)


def per_model(
    strata: Sequence[ModelStratum],
    models: Sequence[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Lower and upper estimate per model, each with its 95% percentile interval."""
    point = [np.ones((1, s.labelled_mentions.shape[0])) for s in strata]
    draws = replicate_weights(strata, replicates, seed)
    rows = []
    low_point, high_point = _estimate(strata, point, 0.0)[0], _estimate(strata, point, 1.0)[0]
    low_draws, high_draws = _estimate(strata, draws, 0.0), _estimate(strata, draws, 1.0)
    for j, model in enumerate(models):
        empty = [
            s.name for s in strata if s.all_mentions[j] > 0 and s.labelled_mentions[:, j].sum() == 0
        ]
        rows.append(
            {
                "model_id": model,
                "share_lower": float(low_point[j]),
                "ci_lower_low": float(np.percentile(low_draws[:, j], 2.5)),
                "ci_lower_high": float(np.percentile(low_draws[:, j], 97.5)),
                "share_upper": float(high_point[j]),
                "ci_upper_low": float(np.percentile(high_draws[:, j], 2.5)),
                "ci_upper_high": float(np.percentile(high_draws[:, j], 97.5)),
                "mentions": int(sum(s.all_mentions[j] for s in strata)),
                "mentions_in_empty_strata": int(
                    sum(s.all_mentions[j] for s in strata if s.name in empty)
                ),
                "empty_strata": "; ".join(empty),
            }
        )
    return pd.DataFrame(rows)


def collapsed_intervals(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of :func:`per_model` where an interval is a single point: few or no labelled mentions
    leave nothing to resample, which is not precision. Report these as a range of the two bounds."""
    collapsed = np.isclose(table["ci_lower_low"], table["ci_lower_high"]) | np.isclose(
        table["ci_upper_low"], table["ci_upper_high"]
    )
    return table[collapsed]


def cells(strata: Sequence[ModelStratum], models: Sequence[str]) -> pd.DataFrame:
    """M, L and F of the estimator, per model and stratum."""
    rows = []
    for s in strata:
        for j, model in enumerate(models):
            column = s.labelled_mentions[:, j]
            labelled = float(column.sum())
            fabricated = float(column[s.fabricated].sum())
            rows.append(
                {
                    "model_id": model,
                    "stratum": s.name,
                    "mentions": int(s.all_mentions[j]),
                    "labelled_mentions": int(labelled),
                    "fabricated_mentions": int(fabricated),
                    "share": fabricated / labelled if labelled else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# Reader terms (exploratory): the same strata and labels, other denominators.


def stratum_index(
    specs: Sequence[Mapping[str, object]], flags: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Pair id to the index of its stratum in ``specs``, as :func:`build_strata` assigns them."""
    found = {}
    for h, spec in enumerate(specs):
        for row in flags:
            if in_stratum(row, spec["bands"], spec.get("impossible")):
                found[row["pair_id"]] = h
    return found


@dataclass(frozen=True, slots=True)
class Responses:
    """One domain's responses: all of them, those holding a pair labelled fabricated, and among
    the rest the distinct patterns of unlabelled pairs per stratum with how many have each."""

    total: int
    known: int
    patterns: np.ndarray  # patterns x strata
    counts: np.ndarray  # patterns


def replicate_weights(
    strata: Sequence[ModelStratum], replicates: int, seed: int
) -> list[np.ndarray]:
    """Per stratum, how often each labelled pair is drawn in each replicate (a census: once)."""
    rng = np.random.default_rng(seed)
    weights = []
    for s in strata:
        n = s.labelled_mentions.shape[0]
        if s.census:
            weights.append(np.ones((replicates, n)))
        else:
            weights.append(rng.multinomial(n, np.full(n, 1 / n), size=replicates).astype(float))
    return weights


def stratum_shares(
    strata: Sequence[ModelStratum], weights: Sequence[np.ndarray], empty_value: float
) -> np.ndarray:
    """Share of marked mentions among labelled ones: replicates x strata x domains."""
    shares = []
    for s, w in zip(strata, weights, strict=True):
        labelled = w @ s.labelled_mentions
        marked = w @ (s.labelled_mentions * s.fabricated[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            shares.append(np.where(labelled > 0, marked / labelled, empty_value))
    return np.stack(shares, axis=1)


def _shares_by_reader(
    fabricated: Sequence[ModelStratum],
    not_numbers: Sequence[ModelStratum],
    numbers: np.ndarray,
    responses: Sequence[Responses],
    weights: Sequence[np.ndarray],
    empty_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicates x domains: fabricated over all numbers given, and responses with at least one
    fabricated number. Unlabelled pairs of a response are fabricated independently, each with its
    stratum's share."""
    mentions = np.stack([s.all_mentions for s in fabricated])  # strata x domains
    f = stratum_shares(fabricated, weights, empty_value)
    n = stratum_shares(not_numbers, weights, 0.0)
    of_numbers = (f * mentions).sum(axis=1) / (numbers - (n * mentions).sum(axis=1))
    of_responses = np.empty_like(of_numbers)
    for d, r in enumerate(responses):
        clean = np.prod((1 - f[:, None, :, d]) ** r.patterns[None, :, :], axis=2)
        of_responses[:, d] = (r.known + (1 - clean) @ r.counts) / r.total
    return of_numbers, of_responses


def reader_terms(
    fabricated: Sequence[ModelStratum],
    not_numbers: Sequence[ModelStratum],
    numbers: np.ndarray,
    responses: Sequence[Responses],
    domains: Sequence[object],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Per domain, a lower and an upper estimate (a stratum without a labelled mention of the
    domain at 0 and at 1; its ``not_a_number`` share is 0 in both), each with a 95% percentile
    interval. ``fabricated`` and ``not_numbers`` are :func:`build_strata` of the same pairs with
    the two labels; ``numbers`` counts every number each domain gives."""
    point = [np.ones((1, s.labelled_mentions.shape[0])) for s in fabricated]
    draws = replicate_weights(fabricated, replicates, seed)
    columns: dict[str, np.ndarray] = {}
    for bound, empty in (("lower", 0.0), ("upper", 1.0)):
        estimate = _shares_by_reader(fabricated, not_numbers, numbers, responses, point, empty)
        spread = _shares_by_reader(fabricated, not_numbers, numbers, responses, draws, empty)
        for k, measure in enumerate(("of_numbers", "of_responses")):
            columns[f"{measure}_{bound}"] = estimate[k][0]
            columns[f"{measure}_{bound}_ci_low"] = np.percentile(spread[k], 2.5, axis=0)
            columns[f"{measure}_{bound}_ci_high"] = np.percentile(spread[k], 97.5, axis=0)
    rows = []
    for d, domain in enumerate(domains):
        empty = [
            s.name
            for s in fabricated
            if s.all_mentions[d] > 0 and not s.labelled_mentions[:, d].any()
        ]
        rows.append(
            {
                "domain": domain,
                "numbers": int(numbers[d]),
                "unknown_mentions": int(sum(s.all_mentions[d] for s in fabricated)),
                "responses": responses[d].total,
                "responses_with_labelled_fabricated": responses[d].known,
                **{name: float(values[d]) for name, values in columns.items()},
                "empty_strata": "; ".join(empty),
            }
        )
    return pd.DataFrame(rows)
