"""Tests for ground-truth validation.

Each rule is tested on a minimal valid dataset (passes) and on one mutation of it (fails).
"""

from __future__ import annotations

import pytest

from crisisline import groundtruth as gt
from crisisline.groundtruth import Issue, Severity


def country(**overrides: str) -> dict[str, str]:
    row = {
        "iso2": "US",
        "name_en": "United States",
        "who_region": "AMR",
        "lang": "en",
        "city_en": "Washington",
        "country_local": "United States",
        "city_local": "Washington",
        "lang_group": "en",
        "has_crisis_line": "true",
    }
    return row | overrides


def resource(**overrides: str) -> dict[str, str]:
    row = {
        "iso2": "US",
        "type": "emergency",
        "name": "Emergency",
        "number": "911",
        "number_norm": "911",
        "status": "active",
        "scope": "national",
        "hours": "24/7",
        "operator": "",
        "source_url": "https://example.org/",
        "quote": "",
        "verified_on": "2026-10-28",
        "decision": "accept",
        "note": "",
    }
    return row | overrides


def shared(number_norm: str, iso2_list: str) -> dict[str, str]:
    return {"number_norm": number_norm, "iso2_list": iso2_list}


@pytest.fixture
def countries() -> list[dict[str, str]]:
    return [country(), country(iso2="CA", name_en="Canada", city_en="Ottawa", city_local="Ottawa")]


@pytest.fixture
def resources() -> list[dict[str, str]]:
    return [
        resource(),
        resource(iso2="CA"),
        resource(type="crisis", name="988 Lifeline", number="988", number_norm="988"),
        resource(iso2="CA", type="crisis", name="9-8-8", number="988", number_norm="988"),
        resource(
            type="crisis",
            name="Lifeline (old)",
            number="1-800-273-8255",
            number_norm="+18002738255",
            status="outdated_functional",
        ),
    ]


@pytest.fixture
def shared_rows() -> list[dict[str, str]]:
    return [shared("911", "CA;US"), shared("988", "CA;US")]


def rules(issues: list[Issue]) -> set[str]:
    return {i.rule for i in issues}


def test_valid_dataset_has_no_issues(
    countries: list[dict[str, str]],
    resources: list[dict[str, str]],
    shared_rows: list[dict[str, str]],
) -> None:
    assert gt.validate(countries, resources, shared_rows, []) == []


# required


def test_required_passes_with_blank_optional_fields() -> None:
    assert gt.check_required([resource(hours="", note="")], gt.RESOURCE_REQUIRED, "r") == []


def test_required_fails_on_blank_field() -> None:
    issues = gt.check_required([resource(source_url=" ")], gt.RESOURCE_REQUIRED, "r")
    assert [(i.line, i.message) for i in issues] == [(2, "missing source_url")]


def test_columns_fail_on_missing_column(
    countries: list[dict[str, str]],
    resources: list[dict[str, str]],
    shared_rows: list[dict[str, str]],
) -> None:
    for row in resources:
        del row["decision"]
    issues = gt.validate(countries, resources, shared_rows, [])
    assert [(i.rule, i.message) for i in issues] == [("required", "missing column decision")]


# enums


def test_enums_pass(countries: list[dict[str, str]], resources: list[dict[str, str]]) -> None:
    assert gt.check_enums(countries, gt.COUNTRY_ENUMS, "c") == []
    assert gt.check_enums(resources, gt.RESOURCE_ENUMS, "r") == []


@pytest.mark.parametrize("field", ["type", "status", "scope", "decision"])
def test_resource_enum_fails(field: str) -> None:
    issues = gt.check_enums([resource(**{field: "bogus"})], gt.RESOURCE_ENUMS, "r")
    assert len(issues) == 1
    assert issues[0].message.startswith(f"{field}='bogus'")


@pytest.mark.parametrize("field", ["who_region", "has_crisis_line"])
def test_country_enum_fails(field: str) -> None:
    issues = gt.check_enums([country(**{field: "bogus"})], gt.COUNTRY_ENUMS, "c")
    assert len(issues) == 1
    assert issues[0].message.startswith(f"{field}='bogus'")


# unknown iso2


def test_known_iso2_passes(
    countries: list[dict[str, str]], resources: list[dict[str, str]]
) -> None:
    assert gt.check_known_iso2(countries, resources) == []


def test_unknown_iso2_fails(countries: list[dict[str, str]]) -> None:
    issues = gt.check_known_iso2(countries, [resource(iso2="XX")])
    assert rules(issues) == {"unknown_iso2"}


# duplicates


def test_same_number_in_two_countries_is_not_a_duplicate(
    resources: list[dict[str, str]],
) -> None:
    assert gt.check_duplicates(resources) == []


def test_duplicate_iso2_number_fails(resources: list[dict[str, str]]) -> None:
    issues = gt.check_duplicates([*resources, resource(name="Emergency again")])
    assert [(i.rule, i.line) for i in issues] == [("duplicate", len(resources) + 2)]


# verified_on


def test_verified_on_passes() -> None:
    assert gt.check_verified_on([resource(verified_on="2026-02-28")]) == []


@pytest.mark.parametrize("value", ["28.10.2026", "2026-10-28T00:00", "2026-2-28", "2026-02-30"])
def test_verified_on_fails(value: str) -> None:
    issues = gt.check_verified_on([resource(verified_on=value)])
    assert [i.rule for i in issues] == ["verified_on"]


# number_norm


@pytest.mark.parametrize("number_norm", ["112", "116123", "+18002738255", "+448081640123"])
def test_number_norm_passes(number_norm: str) -> None:
    assert gt.number_norm_problem(number_norm) is None


@pytest.mark.parametrize(
    ("number_norm", "problem"),
    [
        ("+44123", "not a valid number"),
        ("+4408081640123", "not canonical E.164"),
        ("+abc", "does not parse"),
        ("0800 123", "must be digits only"),
        ("١١٢", "must be digits only"),
    ],
)
def test_number_norm_fails(number_norm: str, problem: str) -> None:
    issues = gt.check_number_norm([resource(number_norm=number_norm)])
    assert len(issues) == 1
    assert problem in issues[0].message


# emergency present


def test_emergency_present_passes(
    countries: list[dict[str, str]], resources: list[dict[str, str]]
) -> None:
    assert gt.check_emergency_present(countries, resources) == []


@pytest.mark.parametrize("override", [{"type": "crisis"}, {"decision": "question"}])
def test_emergency_missing_fails(
    countries: list[dict[str, str]],
    resources: list[dict[str, str]],
    override: dict[str, str],
) -> None:
    resources[1] |= override  # CA's only emergency row
    issues = gt.check_emergency_present(countries, resources)
    assert [(i.rule, i.message) for i in issues] == [
        ("emergency_missing", "CA has no emergency row")
    ]


# has_crisis_line


def test_has_crisis_line_passes_both_ways(resources: list[dict[str, str]]) -> None:
    countries = [country(), country(iso2="CA", has_crisis_line="false")]
    without_ca_crisis = [r for r in resources if not (r["iso2"] == "CA" and r["type"] == "crisis")]
    assert gt.check_has_crisis_line(countries, without_ca_crisis) == []


def test_has_crisis_line_true_without_rows_fails(resources: list[dict[str, str]]) -> None:
    emergency_only = [r for r in resources if r["type"] == "emergency"]
    issues = gt.check_has_crisis_line([country()], emergency_only)
    assert rules(issues) == {"has_crisis_line"}


def test_has_crisis_line_false_with_rows_fails(resources: list[dict[str, str]]) -> None:
    issues = gt.check_has_crisis_line([country(has_crisis_line="false")], resources)
    assert rules(issues) == {"has_crisis_line"}


def test_discontinued_crisis_row_does_not_count(resources: list[dict[str, str]]) -> None:
    rows = [resource(), resource(type="crisis", number_norm="1393", status="discontinued")]
    assert gt.check_has_crisis_line([country(has_crisis_line="false")], rows) == []


# shared numbers


def test_shared_numbers_recomputed(resources: list[dict[str, str]]) -> None:
    assert gt.shared_numbers(resources) == {
        "911": frozenset({"US", "CA"}),
        "988": frozenset({"US", "CA"}),
    }


def test_shared_numbers_pass(
    resources: list[dict[str, str]], shared_rows: list[dict[str, str]]
) -> None:
    assert gt.check_shared_numbers(resources, shared_rows) == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([shared("911", "CA;US")], "988 is shared by ['CA', 'US'] but not listed"),
        ([*map(shared, ["911", "988", "112"], ["CA;US", "CA;US", "CA;US"])], "112 is not shared"),
        ([shared("911", "US"), shared("988", "CA;US")], "911 lists ['US'], resources give"),
    ],
    ids=["missing", "extra", "mismatch"],
)
def test_shared_numbers_fail(
    resources: list[dict[str, str]], rows: list[dict[str, str]], message: str
) -> None:
    issues = gt.check_shared_numbers(resources, rows)
    assert len(issues) == 1
    assert issues[0].message.startswith(message)


# emergency recognised by phonenumbers (warning)


def test_emergency_recognised_passes(resources: list[dict[str, str]]) -> None:
    assert gt.check_emergency_recognised(resources) == []


def test_emergency_unrecognised_warns() -> None:
    issues = gt.check_emergency_recognised([resource(iso2="PE", number_norm="106")])
    assert [(i.severity, i.rule) for i in issues] == [(Severity.WARNING, "emergency_unrecognised")]


# unknown_labels.csv


def label(iso2: str = "US", number_norm: str = "1414", value: str = "fabricated") -> dict[str, str]:
    return {"iso2": iso2, "number_norm": number_norm, "label": value, "source_url": "", "note": ""}


def test_labels_pass(
    countries: list[dict[str, str]],
    resources: list[dict[str, str]],
    shared_rows: list[dict[str, str]],
) -> None:
    labels = [
        label(),
        label(number_norm="988", value="valid"),
        label("CA", value="other_service"),
        label("GB", value="not_a_number"),
        label("DE", value="unverified"),
    ]
    assert gt.validate(countries, resources, shared_rows, labels) == []


def test_label_required_and_enum_fail() -> None:
    assert rules(gt.check_required([label(number_norm="")], gt.LABEL_REQUIRED, "l")) == {"required"}
    assert rules(gt.check_enums([label(value="wrong")], gt.LABEL_ENUMS, "l")) == {"enum"}


def test_duplicate_label_fails() -> None:
    issues = gt.check_duplicates([label(), label(value="other_service")], gt.LABELS_FILE)
    assert [(i.rule, i.file) for i in issues] == [("duplicate", gt.LABELS_FILE)]


def test_valid_label_without_accepted_row_fails(resources: list[dict[str, str]]) -> None:
    rejected = [*resources, resource(number="1414", number_norm="1414", decision="reject")]
    issues = gt.check_valid_labels(rejected, [label(value="valid")])
    assert [i.rule for i in issues] == ["valid_label"]
