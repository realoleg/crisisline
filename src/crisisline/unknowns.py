"""Blind labelling files for unknown numbers and the estimate of fabricated mentions.

:func:`collect_pairs` turns scores into ``(iso2, number)`` pairs. :func:`check_number` asks
libphonenumber whether a number can exist; :func:`group_pairs` joins forms of one number within a
country; :func:`context` and :func:`blind_contexts` give the text a labeller reads, with no trace
of the model behind it; :func:`draw_sample` draws from a band of pairs. :func:`stratum` and
:func:`fabrication_estimate` turn the finished labels into the stratified share of fabricated
mentions.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import phonenumbers
from phonenumbers import NumberParseException, ValidationResult, shortnumberinfo

from crisisline.audit import response_id
from crisisline.score import FABRICATED, UNKNOWN

LABEL_COLUMNS = (
    "belongs_to_iso2",
    "label",
    "type",
    "name",
    "status",
    "scope",
    "hours",
    "operator",
    "source_url",
    "quote",
    "verified_on",
    "decision",
    "note",
)
_PAIR_COLUMNS = ("pair_id", "iso2", "number_norm", "number_as_written", "responses", "group_id")
HEAD_COLUMNS = (*_PAIR_COLUMNS, "context_1", "context_2", "context_3", *LABEL_COLUMNS)
SAMPLE_COLUMNS = (*_PAIR_COLUMNS, "context_1", *LABEL_COLUMNS)
FLAG_COLUMNS = (
    "pair_id",
    "iso2",
    "number_norm",
    "number_as_written",
    "responses",
    "band",
    "impossible",
    "check",
)
IMPOSSIBLE = frozenset({"too_short", "too_long", "invalid_length", "impossible_everywhere"})
# libphonenumber's validation results and parse errors, by the reason a labeller reads. It raises
# NOT_A_NUMBER for a digit string too short to parse ("1", "+12"), so that counts as too_short.
_REASONS = {
    ValidationResult.TOO_SHORT: "too_short",
    ValidationResult.TOO_LONG: "too_long",
    ValidationResult.INVALID_LENGTH: "invalid_length",
    ValidationResult.INVALID_COUNTRY_CODE: "impossible_everywhere",
}
_PARSE_ERRORS = {
    NumberParseException.TOO_SHORT_NSN: "too_short",
    NumberParseException.TOO_SHORT_AFTER_IDD: "too_short",
    NumberParseException.TOO_LONG: "too_long",
    NumberParseException.INVALID_COUNTRY_CODE: "impossible_everywhere",
    NumberParseException.NOT_A_NUMBER: "too_short",
}
_BLANK_LINE = re.compile(r"\n[ \t\r]*\n")
_MODEL_PLACEHOLDER = "[model]"


@dataclass(frozen=True, slots=True)
class Mention:
    """The first occurrence of a number in one response; ``start`` is its offset in the
    normalised text."""

    response_id: str
    raw: str
    start: int


@dataclass(slots=True)
class Pair:
    """A number unknown to the ground truth, given for one country, with its responses."""

    iso2: str
    number: str
    mentions: list[Mention] = field(default_factory=list)

    @property
    def pair_id(self) -> str:
        return f"{self.iso2}:{self.number}"

    @property
    def responses(self) -> int:
        return len(self.mentions)

    @property
    def written(self) -> str:
        """The most common written form; of equally common forms, the first in sort order."""
        counts = Counter(m.raw for m in self.mentions)
        return min(counts, key=lambda raw: (-counts[raw], raw))


def band_of(responses: int, bands: Sequence[Mapping[str, Any]]) -> str:
    """The name of the band that holds ``responses``; a band whose ``high`` is empty has no upper
    bound."""
    return next(b["name"] for b in bands if b["low"] <= responses <= (b["high"] or responses))


def flag_row(pair: Pair, bands: Sequence[Mapping[str, Any]], check: str) -> dict[str, Any]:
    """A row of ``unknowns_flags.csv``: the pair's band and libphonenumber's ``check`` of it."""
    return {
        "pair_id": pair.pair_id,
        "iso2": pair.iso2,
        "number_norm": pair.number,
        "number_as_written": pair.written,
        "responses": pair.responses,
        "band": band_of(pair.responses, bands),
        "impossible": "yes" if check in IMPOSSIBLE else "no",
        "check": check,
    }


def complete_flags(
    flags: Sequence[Mapping[str, Any]],
    pairs: Iterable[Pair],
    bands: Sequence[Mapping[str, Any]],
    no_country_iso2: str,
) -> list[Mapping[str, Any]]:
    """``flags`` and a :func:`flag_row` for every pair it lacks, such as the numbers of a model
    run after the file was written, so that every pair falls in a stratum."""
    known = {row["pair_id"] for row in flags}
    missing = [
        flag_row(p, bands, check_number(p.number, p.iso2, no_country_iso2))
        for p in pairs
        if p.pair_id not in known
    ]
    return [*flags, *missing]


def in_stratum(row: Mapping[str, Any], bands: Collection[str], impossible: str | None) -> bool:
    """Whether a row of ``unknowns_flags.csv`` is in a stratum: one of its ``bands``, with the
    flag ``impossible`` if one is given."""
    return row["band"] in bands and impossible in (None, row["impossible"])


def pair_order(pair: Pair) -> tuple[str, int, str]:
    """Country, then most responses first, then the number."""
    return (pair.iso2, -pair.responses, pair.number)


def collect_pairs(scores: Iterable[Mapping[str, Any]]) -> list[Pair]:
    """Unknown numbers in rows of a scores file, as pairs in :func:`pair_order`."""
    pairs: dict[tuple[str, str], Pair] = {}
    for row in scores:
        rid = response_id(row)
        for number in row["numbers"] or ():
            if number["number_class"] == UNKNOWN:
                key = (row["iso2"], number["number"])
                pair = pairs.setdefault(key, Pair(*key))
                pair.mentions.append(Mention(rid, number["raw"], number["start"]))
    return sorted(pairs.values(), key=pair_order)


def _check_in_region(number: str, region: str | None) -> str:
    try:
        parsed = phonenumbers.parse(number, region)
    except NumberParseException as exc:
        return _PARSE_ERRORS[exc.error_type]
    if region and shortnumberinfo.is_valid_short_number_for_region(parsed, region):
        return "short_code"
    return _REASONS.get(phonenumbers.is_possible_number_with_reason(parsed), "possible")


def check_number(number: str, iso2: str, no_country_iso2: str) -> str:
    """What libphonenumber says of a number in the user's country: ``possible``, ``short_code``
    (a known one) or the reason it is impossible. With no country, the number is impossible only
    if it is so in every region and no region knows it as a short code."""
    if iso2 != no_country_iso2:
        return _check_in_region(number, iso2)
    if number.startswith("+"):
        check = _check_in_region(number, None)
        return "impossible_everywhere" if check in IMPOSSIBLE else "possible_somewhere"
    possible = any(
        _check_in_region(number, region) not in IMPOSSIBLE
        for region in sorted(phonenumbers.SUPPORTED_REGIONS)
    )
    return "possible_somewhere" if possible else "impossible_everywhere"


def _forms(pair: Pair, no_country_iso2: str) -> set[str]:
    """The number's digits, with and without the country code or the trunk prefix."""
    digits = pair.number.lstrip("+")
    forms = {digits}
    if pair.iso2 == no_country_iso2:
        return forms
    code = str(phonenumbers.country_code_for_region(pair.iso2))
    metadata = phonenumbers.PhoneMetadata.metadata_for_region(pair.iso2)
    trunk = (metadata.national_prefix if metadata else None) or ""
    if pair.number.startswith("+") and digits.startswith(code):
        forms.add(digits[len(code) :])
    elif trunk and digits.startswith(trunk):
        forms.add(digits[len(trunk) :])
    return forms


def _same_number(a: Pair, b: Pair, no_country_iso2: str) -> bool:
    """Equal digits in some form, or, if the two share a response, one form ending another."""
    forms_a, forms_b = _forms(a, no_country_iso2), _forms(b, no_country_iso2)
    if forms_a & forms_b:
        return True
    if not {m.response_id for m in a.mentions} & {m.response_id for m in b.mentions}:
        return False
    return any(
        min(len(x), len(y)) >= 3 and (x.endswith(y) or y.endswith(x))
        for x in forms_a
        for y in forms_b
    )


def group_pairs(pairs: Sequence[Pair], no_country_iso2: str) -> dict[str, str]:
    """Pair id to group id, for pairs that are forms of one number within a country. A pair with
    no other form has no group."""
    parent = {p.pair_id: p.pair_id for p in pairs}

    def find(pair_id: str) -> str:
        while parent[pair_id] != pair_id:
            parent[pair_id] = parent[parent[pair_id]]
            pair_id = parent[pair_id]
        return pair_id

    by_country: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_country.setdefault(pair.iso2, []).append(pair)
    for members in by_country.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if _same_number(a, b, no_country_iso2):
                    parent[find(a.pair_id)] = find(b.pair_id)
    roots = Counter(find(p.pair_id) for p in pairs)
    groups: dict[str, str] = {}
    ids: dict[str, str] = {}
    for pair in pairs:
        root = find(pair.pair_id)
        if roots[root] > 1:
            groups[pair.pair_id] = ids.setdefault(root, f"G{len(ids) + 1:04d}")
    return groups


def context(text: str, start: int, end: int) -> str:
    """The whole paragraph holding ``text[start:end]``, not truncated. If the number is in its
    first line, the line above the paragraph comes too: it usually names the service."""
    before = [m for m in _BLANK_LINE.finditer(text, 0, start)]
    paragraph_start = before[-1].end() if before else 0
    after = _BLANK_LINE.search(text, end)
    paragraph_end = after.start() if after else len(text)
    paragraph = text[paragraph_start:paragraph_end]
    newline = text.find("\n", paragraph_start, paragraph_end)
    if before and (newline < 0 or start < newline):
        heading = text[: before[-1].start()].rsplit("\n", 1)[-1]
        paragraph = f"{heading}\n{paragraph}"
    return paragraph.replace("\r", "").strip()


def blind_contexts(
    pair: Pair,
    texts: Mapping[str, str],
    size: int,
    seed: int,
    model_names: re.Pattern[str],
) -> list[str]:
    """Up to ``size`` distinct contexts of a pair from random responses, model names replaced. The
    draw depends only on the seed and the pair."""
    mentions = sorted(pair.mentions, key=lambda m: m.response_id)
    random.Random(f"{seed}:{pair.pair_id}").shuffle(mentions)
    found: dict[str, None] = {}
    for mention in mentions:
        if len(found) == size:
            break
        text = context(texts[mention.response_id], mention.start, mention.start + len(mention.raw))
        found[model_names.sub(_MODEL_PLACEHOLDER, text)] = None
    return list(found)


def draw_sample(
    pairs: Sequence[Pair],
    exclude: Collection[str],
    low: int,
    high: int,
    size: int,
    seed: int,
) -> list[Pair]:
    """Up to ``size`` random pairs with ``low`` to ``high`` responses, except ``exclude`` (pair
    ids), in :func:`pair_order`."""
    population = sorted(
        (p for p in pairs if low <= p.responses <= high and p.pair_id not in exclude),
        key=lambda p: p.pair_id,
    )
    chosen = random.Random(seed).sample(population, min(size, len(population)))
    return sorted(chosen, key=pair_order)


def allocate(total: int, weights: Sequence[int]) -> list[int]:
    """``total`` split in proportion to ``weights`` by largest remainder; ties go to the first.
    Nothing to split when every weight is zero."""
    all_weight = sum(weights)
    if not all_weight:
        return [0] * len(weights)
    shares = [total * weight / all_weight for weight in weights]
    sizes = [int(share) for share in shares]
    order = sorted(range(len(shares)), key=lambda k: (sizes[k] - shares[k], k))
    for k in order[: total - sum(sizes)]:
        sizes[k] += 1
    return sizes


def labelling_row(
    pair: Pair,
    group: str,
    contexts: Sequence[str],
    size: int,
) -> dict[str, str | int]:
    """A row of a labelling file, with ``size`` context columns and empty label columns."""
    return {
        "pair_id": pair.pair_id,
        "iso2": pair.iso2,
        "number_norm": pair.number,
        "number_as_written": pair.written,
        "responses": pair.responses,
        "group_id": group,
        **{f"context_{k + 1}": contexts[k] if k < len(contexts) else "" for k in range(size)},
        **dict.fromkeys(LABEL_COLUMNS, ""),
    }


@dataclass(frozen=True, slots=True)
class Stratum:
    """A set of pairs of known size, with the labelled ones as ``(responses, label)``."""

    name: str
    pairs: int
    mentions: int
    labelled: tuple[tuple[int, str], ...]

    @property
    def is_census(self) -> bool:
        return len(self.labelled) == self.pairs


def stratum(
    name: str,
    flags: Iterable[Mapping[str, str]],
    labelled: Iterable[Mapping[str, str]],
    bands: Collection[str],
    impossible: str | None = None,
) -> Stratum:
    """The pairs of ``flags`` in ``bands``, with this flag if one is given, and the rows of
    ``labelled`` among them."""
    members = {
        row["pair_id"]: int(row["responses"]) for row in flags if in_stratum(row, bands, impossible)
    }
    rows = tuple((members[r["pair_id"]], r["label"]) for r in labelled if r["pair_id"] in members)
    return Stratum(name, len(members), sum(members.values()), rows)


def _fabricated_share(labelled: Sequence[tuple[int, str]]) -> float:
    """The share of mentions whose pair is labelled fabricated."""
    fabricated = sum(responses for responses, label in labelled if label == FABRICATED)
    return fabricated / sum(responses for responses, _ in labelled)


def wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval of a proportion."""
    if total == 0:
        raise ValueError("a Wilson interval needs at least one trial")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    share = successes / total
    denominator = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / denominator
    margin = z * (share * (1 - share) / total + z**2 / (4 * total**2)) ** 0.5 / denominator
    # The bounds at 0 and at total successes are exactly 0 and 1; centre - margin cancels
    # to a 1-ulp residue whose sign depends on the platform's libm.
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == total else min(1.0, centre + margin)
    return low, high


def _stratum_summary(s: Stratum, total_mentions: int, confidence: float) -> dict[str, Any]:
    labels: dict[str, dict[str, int]] = {}
    for responses, label in s.labelled:
        entry = labels.setdefault(label, {"pairs": 0, "mentions": 0})
        entry["pairs"] += 1
        entry["mentions"] += responses
    fabricated = labels.get(FABRICATED, {"pairs": 0})["pairs"]
    wilson = None if s.is_census else wilson_interval(fabricated, len(s.labelled), confidence)
    return {
        "name": s.name,
        "pairs": s.pairs,
        "mentions": s.mentions,
        "weight": s.mentions / total_mentions,
        "labelled_pairs": len(s.labelled),
        "pair_share": fabricated / len(s.labelled),
        "pair_share_wilson": wilson,
        "mention_share": _fabricated_share(s.labelled),
        "labels": dict(sorted(labels.items())),
    }


def fabrication_estimate(
    strata: Sequence[Stratum], replicates: int, seed: int, confidence: float
) -> dict[str, Any]:
    """The share of mentions labelled fabricated: per stratum, then summed with
    mention weights. The percentile interval resamples pairs within each sampled stratum; a census
    has no sampling error."""
    if empty := [s.name for s in strata if not s.labelled]:
        raise ValueError(f"strata without a labelled pair: {', '.join(empty)}")
    total = sum(s.mentions for s in strata)
    census = sum(s.mentions / total * _fabricated_share(s.labelled) for s in strata if s.is_census)
    sampled = [s for s in strata if not s.is_census]
    rng = random.Random(seed)
    draws = sorted(
        census
        + sum(
            s.mentions / total * _fabricated_share(rng.choices(s.labelled, k=len(s.labelled)))
            for s in sampled
        )
        for _ in range(replicates)
    )
    tail = (1 - confidence) / 2
    return {
        "share": sum(s.mentions / total * _fabricated_share(s.labelled) for s in strata),
        "interval": (
            draws[round(tail * (replicates - 1))],
            draws[round((1 - tail) * (replicates - 1))],
        ),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "pairs": sum(s.pairs for s in strata),
        "mentions": total,
        "strata": [_stratum_summary(s, total, confidence) for s in strata],
    }
