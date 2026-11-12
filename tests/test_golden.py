"""The comparison of reproduce outputs with their reference copies."""

import json

import pytest

from crisisline.golden import csv_differences, differences, json_differences, same

TABLE = "model_id,share,ci_low,responses,excludes_zero\nm-1,0.123456,0,900,False\n"


def test_floats_that_differ_by_rounding_noise_are_the_same() -> None:
    assert same(0.1 + 0.2, 0.3)
    assert same(0.3, 0.3 + 1e-16)
    assert same(0, -1.11022e-16)


@pytest.mark.parametrize(("a", "b"), [(0.123456, 0.123457), (9.99999, 9.99998), (100000, 100001.0)])
def test_a_last_digit_difference_at_six_significant_digits_is_the_same(a: float, b: float) -> None:
    assert same(a, b)


@pytest.mark.parametrize(("a", "b"), [(0.123456, 0.123556), (1234.0, 1235.0), (9.999, 9.998)])
def test_a_change_in_the_fourth_significant_digit_is_not(a: float, b: float) -> None:
    assert not same(a, b)


def test_integers_booleans_and_strings_must_be_equal() -> None:
    assert not same(900, 901)
    assert not same(True, False)
    assert not same(True, 1)
    assert not same("+4930", "+4931")
    assert not same("0.5", 0.5)


def test_csv_with_noise_below_tolerance_passes() -> None:
    noisy = TABLE.replace(",0,900", ",-1.11022e-16,900").replace("0.123456", "0.1234560000000001")
    assert csv_differences(TABLE, noisy) == []
    assert csv_differences(TABLE, TABLE.replace("0.123456", "0.123457")) == []


def test_csv_reports_a_changed_digit_by_line_and_column() -> None:
    changed = TABLE.replace("0.123456", "0.123556")
    assert csv_differences(TABLE, changed) == ["line 2, share: 0.123556, expected 0.123456"]


def test_csv_flipped_boolean_fails() -> None:
    flipped = TABLE.replace("False", "True")
    assert csv_differences(TABLE, flipped) == ["line 2, excludes_zero: True, expected False"]


def test_csv_phone_numbers_are_strings() -> None:
    table = "number\n+18002738255\n0800213103\n"
    assert csv_differences(table, "number\n+18002738256\n800213103\n") == [
        "line 2, number: +18002738256, expected +18002738255",
        "line 3, number: 800213103, expected 0800213103",
    ]


@pytest.mark.parametrize(
    ("actual", "problem"),
    [
        ("model_id,share\nm-1,0.1\n", "header"),
        (TABLE + "m-2,0.5,0,900,False\n", "2 rows, expected 1"),
        ("model_id,share,ci_low,responses,excludes_zero\nm-1,0.123456,0,900\n", "4 cells"),
    ],
)
def test_csv_layout_must_match(actual: str, problem: str) -> None:
    (found,) = csv_differences(TABLE, actual)
    assert problem in found


def test_json_compares_nested_values() -> None:
    expected = {"share": 0.52, "interval": [0.50, 0.54], "pairs": 8478, "wilson": None}
    noisy = {**expected, "share": 0.52 + 1e-16}
    changed = {**expected, "interval": [0.50, 0.5401], "pairs": 8479}
    assert json_differences(json.dumps(expected), json.dumps(noisy)) == []
    assert json_differences(json.dumps(expected), json.dumps(changed)) == [
        "/interval/1: 0.5401, expected 0.54",
        "/pairs: 8479, expected 8478",
    ]


def test_text_lines_are_compared_exactly_except_unchecked_ones() -> None:
    expected = "# Notes\n- Bootstrap: 132 of 2000 failed.\n- Range 11.7-12.5%.\n"
    other_run = expected.replace("132", "134")
    assert differences(".md", expected, other_run, "^- Bootstrap: ") == []
    assert differences(".md", expected, other_run) == [
        "line 2: '- Bootstrap: 134 of 2000 failed.', expected '- Bootstrap: 132 of 2000 failed.'"
    ]
    changed = other_run.replace("12.5", "12.6")
    assert len(differences(".md", expected, changed, "^- Bootstrap: ")) == 1


def test_an_unchecked_line_must_still_be_there() -> None:
    expected = "# Notes\n- Bootstrap: 132 of 2000 failed.\n"
    assert differences(".md", expected, "# Notes\n", "^- Bootstrap: ") == ["1 lines, expected 2"]
    assert differences(".md", expected, "# Notes\n- Something else\n", "^- Bootstrap: ") != []
