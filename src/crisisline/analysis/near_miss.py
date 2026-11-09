"""Long numbers outside the ground truth that are one or two keystrokes from a real number of the
same country (exploratory).

A number is a near-miss if its digits are within Levenshtein distance ``max_distance`` of a form of
a ground-truth number of its country, or one adjacent transposition away (with the default distance
of 2 a transposition is already within it; it is reported as its own kind). Forms of a ground-truth
number: its digits as stored, and for an E.164 number also the national number with and without
the trunk prefix. The chance baseline is the probability that a random number of the same length
passes the same test: digits after the calling code (E.164) or after the first digit are uniform.
It is exact: every string within the distance of a form is enumerated, not sampled.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import phonenumbers

DIGITS = "0123456789"


@dataclass(frozen=True, slots=True)
class Target:
    """A ground-truth number of a country, with the words that name its service."""

    iso2: str
    number: str
    forms: tuple[str, ...]
    type: str
    status: str
    name: str
    words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Match:
    target: Target
    distance: int
    kind: str


def digits(number: str) -> str:
    return re.sub(r"\D", "", number)


def forms(number: str, iso2: str) -> tuple[str, ...]:
    """Digits as stored; for E.164 also the national number and the national number with the
    trunk prefix of ``iso2``."""
    found = {digits(number)}
    if number.startswith("+"):
        parsed = phonenumbers.parse(number)
        national = str(parsed.national_number)
        metadata = phonenumbers.PhoneMetadata.metadata_for_region(iso2)
        trunk = (metadata.national_prefix if metadata else None) or ""
        found |= {national, trunk + national}
    return tuple(sorted(found))


def levenshtein(a: str, b: str, limit: int) -> int:
    """Edit distance, or ``limit + 1`` as soon as it is known to exceed ``limit``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return min(previous[-1], limit + 1)


def is_adjacent_transposition(a: str, b: str) -> bool:
    """``b`` is ``a`` with two neighbouring, different digits swapped."""
    if len(a) != len(b):
        return False
    diff = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    if len(diff) != 2:
        return False
    i, j = diff
    return j == i + 1 and a[i] == b[j] and a[j] == b[i]


def name_words(name: str, operator: str, generic: Iterable[str], min_chars: int) -> tuple[str, ...]:
    """Casefolded words of a service's name and operator, without parenthetical qualifiers and
    generic words, and the name itself as one phrase."""
    stop = {w.casefold() for w in generic}
    name, operator = (re.sub(r"\([^)]*\)", " ", text).casefold() for text in (name, operator))
    phrase = " ".join(name.split())
    words = {
        w
        for w in re.findall(r"[^\W\d_]+", f"{name} {operator}")
        if len(w) >= min_chars and w not in stop
    }
    if len(phrase) >= min_chars and phrase not in stop:
        words.add(phrase)
    return tuple(sorted(words))


def targets(
    rows: Sequence[Mapping[str, str]], generic: Iterable[str], min_chars: int
) -> dict[str, list[Target]]:
    """Accepted ground-truth numbers per country, every status."""
    generic = tuple(generic)
    found: dict[str, list[Target]] = {}
    for row in rows:
        if row["decision"] != "accept":
            continue
        found.setdefault(row["iso2"], []).append(
            Target(
                row["iso2"],
                row["number_norm"],
                forms(row["number_norm"], row["iso2"]),
                row["type"],
                row["status"],
                row["name"],
                name_words(row["name"], row["operator"], generic, min_chars),
            )
        )
    return found


def match(number: str, candidates: Sequence[Target], max_distance: int) -> Match | None:
    """The closest target within ``max_distance`` or one transposition; ties go to the first. A
    target that has the number as one of its forms is the number itself, not a near-miss."""
    best: Match | None = None
    mine = digits(number)
    for target in candidates:
        distances = [(levenshtein(mine, form, max_distance), form) for form in target.forms]
        if any(distance == 0 for distance, _ in distances):
            continue
        for distance, form in distances:
            if is_adjacent_transposition(mine, form):
                found = Match(target, 2, "transposition")
            elif distance <= max_distance:
                found = Match(target, distance, f"distance {distance}")
            else:
                continue
            if best is None or found.distance < best.distance:
                best = found
    return best


def _one_edit(text: str) -> set[str]:
    found = set()
    for i in range(len(text) + 1):
        for d in DIGITS:
            found.add(text[:i] + d + text[i:])
        if i < len(text):
            found.add(text[:i] + text[i + 1 :])
            for d in DIGITS:
                found.add(text[:i] + d + text[i + 1 :])
    return found


@lru_cache(maxsize=64)
def ball(text: str, radius: int) -> frozenset[str]:
    """Every digit string within Levenshtein distance ``radius`` of ``text``."""
    found = {text}
    frontier = {text}
    for _ in range(radius):
        frontier = {n for t in frontier for n in _one_edit(t)} - found
        found |= frontier
    return frozenset(found)


def random_prefix(number: str) -> str:
    """The part a random number of the same country keeps: the calling code, or the first digit."""
    if number.startswith("+"):
        try:
            return str(phonenumbers.parse(number).country_code)
        except phonenumbers.NumberParseException:
            pass
    return digits(number)[:1]


def chance(number: str, candidates: Sequence[Target], max_distance: int) -> float:
    """Probability that a random number with the same length and prefix is a near-miss of any of
    ``candidates``. Needs ``max_distance >= 2``: the ball then holds every transposition too."""
    if max_distance < 2:
        raise ValueError("the exact chance baseline needs max_distance >= 2")
    length, prefix = len(digits(number)), random_prefix(number)
    own = {f for t in candidates for f in t.forms}
    hits: set[str] = set()
    for target in candidates:
        for form in target.forms:
            if abs(len(form) - length) <= max_distance:
                hits |= {
                    s
                    for s in ball(form, max_distance)
                    if len(s) == length and s.startswith(prefix) and s not in own
                }
    return len(hits) / 10 ** (length - len(prefix))


_OTHER_NUMBER = re.compile(r"\d{3}")


def number_context(text: str, start: int, end: int) -> str:
    """The line holding ``text[start:end]`` and the lines above it up to a blank line or a line
    with another number: in a list, the name a number belongs to, not its neighbours' names."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    lines = [text[line_start : len(text) if line_end < 0 else line_end]]
    above = text[: max(line_start - 1, 0)].split("\n") if line_start else []
    for line in reversed(above):
        if not line.strip() or _OTHER_NUMBER.search(line):
            break
        lines.insert(0, line)
    return "\n".join(lines)


def names_service(context: str, target: Target) -> bool:
    """Whether the paragraph around the number holds a word of the target's name (unvalidated)."""
    folded = context.casefold()
    return any(word in folded for word in target.words)


def noted_numbers(note: str, iso2: str) -> list[str]:
    """Numbers written after the last `` is `` of a labeller's note ("real number of X is ...")."""
    tail = note.rsplit(" is ", 1)[-1]
    found = []
    for part in re.split(r"/|,| and | or ", tail):
        raw = re.sub(r"[^\d+*]", "", part)
        if not raw:
            continue
        if len(digits(raw)) < 7:
            found.append(raw)
            continue
        try:
            parsed = phonenumbers.parse(raw, iso2)
            found.append(phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164))
        except phonenumbers.NumberParseException:
            found.append(raw)
    return found
