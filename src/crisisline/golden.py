"""Comparison of an output of ``crisisline reproduce`` with its reference copy under
``checks/golden``.

Numbers that are not integers agree within :data:`RTOL` and :data:`ATOL`; integers, booleans,
strings and the layout of the file must be equal.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Iterator
from typing import TypeGuard

RTOL = 1e-4  # passes a last-digit difference at six significant digits, fails one in the fourth
ATOL = 1e-9  # a value that is 0 comes out as about ±1e-16 on some platforms

# Literals as Python and pandas write them. A phone number ("+4930...", "0800...") is a string.
INTEGER = re.compile(r"-?(?:0|[1-9]\d*)")
NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")

Value = int | float | str


def cell(text: str) -> Value:
    """A CSV cell as an integer, a float or the string itself."""
    if INTEGER.fullmatch(text):
        return int(text)
    if NUMBER.fullmatch(text):
        return float(text)
    return text


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def same(expected: object, actual: object) -> bool:
    """Within tolerance if both are numbers and one is a float; otherwise equal and of the same
    type, so ``True`` is not ``1``."""
    if _number(expected) and _number(actual) and float in (type(expected), type(actual)):
        return math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL)
    return type(expected) is type(actual) and expected == actual


def csv_differences(expected: str, actual: str) -> list[str]:
    """Cells of two CSV tables that are not :func:`same`, by line and column."""
    old, new = (list(csv.reader(io.StringIO(text))) for text in (expected, actual))
    if old[0] != new[0]:
        return [f"header: {old[0]} != {new[0]}"]
    if len(old) != len(new):
        return [f"{len(new) - 1} rows, expected {len(old) - 1}"]
    differences = []
    for line, (a, b) in enumerate(zip(old[1:], new[1:], strict=True), start=2):
        if len(a) != len(b):
            differences.append(f"line {line}: {len(b)} cells, expected {len(a)}")
            continue
        for column, x, y in zip(old[0], a, b, strict=True):
            if not same(cell(x), cell(y)):
                differences.append(f"line {line}, {column}: {y}, expected {x}")
    return differences


def json_differences(expected: str, actual: str) -> list[str]:
    """Values of two JSON documents that are not :func:`same`, by their path."""
    return list(_walk(json.loads(expected), json.loads(actual), ""))


def _walk(expected: object, actual: object, where: str) -> Iterator[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            yield f"{where or '/'}: keys {sorted(actual)}, expected {sorted(expected)}"
            return
        for key in expected:
            yield from _walk(expected[key], actual[key], f"{where}/{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{where or '/'}: {len(actual)} items, expected {len(expected)}"
            return
        for i, (x, y) in enumerate(zip(expected, actual, strict=True)):
            yield from _walk(x, y, f"{where}/{i}")
    elif not same(expected, actual):
        yield f"{where or '/'}: {actual!r}, expected {expected!r}"


def text_differences(expected: str, actual: str, unchecked: str | None = None) -> list[str]:
    """Lines of two texts that differ. A line that matches the regex ``unchecked`` in both is not
    compared, but must be there."""
    old, new = expected.splitlines(), actual.splitlines()
    if len(old) != len(new):
        return [f"{len(new)} lines, expected {len(old)}"]
    pattern = re.compile(unchecked) if unchecked else None
    return [
        f"line {line}: {y!r}, expected {x!r}"
        for line, (x, y) in enumerate(zip(old, new, strict=True), start=1)
        if x != y and not (pattern and pattern.search(x) and pattern.search(y))
    ]


def differences(suffix: str, expected: str, actual: str, unchecked: str | None = None) -> list[str]:
    """Differences between a reference copy and an output, by the output's file suffix."""
    if suffix == ".csv":
        return csv_differences(expected, actual)
    if suffix == ".json":
        return json_differences(expected, actual)
    return text_differences(expected, actual, unchecked)
