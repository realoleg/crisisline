"""Property-based tests for number extraction on the real ground truth.

Every accepted number in resources.csv is written the way models write numbers (national and
international formats, other separators and digit scripts, invisible characters, Markdown) and
must come back as itself. Ordinary non-phone numerals (dates, times, amounts, statistics) must
produce nothing.
"""

import re
from datetime import date, time
from decimal import Decimal
from pathlib import Path

import phonenumbers
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from crisisline.cli import read_csv
from crisisline.extract import ExtractConfig, NumberIndex, extract

DATA = Path(__file__).resolve().parents[1] / "data"
ROWS = [r for r in read_csv(DATA / "resources_v1.csv") if r["decision"] == "accept"]
LANG = {c["iso2"]: c["lang"] for c in read_csv(DATA / "countries.csv")}
ISO2S = sorted(LANG)
MIN_DIGITS = 3  # configs/experiment.yaml extract.min_digits

SHORT_ROWS = [r for r in ROWS if not r["number_norm"].startswith("+")]
TWO_DIGIT_ROWS = [r for r in SHORT_ROWS if len(r["number_norm"]) < MIN_DIGITS]
NUMBER_ROWS = [r for r in ROWS if len(r["number_norm"]) >= MIN_DIGITS]

ZERO = {
    "ascii": "0",
    "arabic-indic": "٠",
    "extended arabic-indic": "۰",
    "devanagari": "०",
    "bengali": "০",
    "thai": "๐",
    "fullwidth": "０",
}
INVISIBLE = ["​", "‎", "‏", "؜", "⁠", "﻿", "­"]
SEPARATORS = [" ", " ", " ", "-", "‑", "–", "."]
# "13.11.14" and "13-11-14" are dates and are masked as such.
_DATE_LIKE = re.compile(r"\+?[0-9]{1,2}([./-])[0-9]{1,2}\1[0-9]{2,4}")

TEMPLATES = [
    "{n}",
    "Call {n}.",
    "Call {n}, any time.",
    "**{n}**",
    "Call **{n}** now",
    "[{n}](tel:{tel})",
    "| Lifeline | {n} |",
    "`{n}`",
    "```\n{n}\n```",
    "## {n}",
    "- Helpline:\n  - {n}",
    "1. {n}",
    "> Call {n}",
    "📞 {n}",
    "({n})",
    "Hotline: {n}\nOpen 24/7",
    "{n} (24/7, free)",
    "{n}に電話してください",
    "请拨打{n}。",
    "（{n}）",
    "اتصل ب{n}",
    "‏{n}‏",
    "‎{n}‎",
]
# Two-digit numbers count only with call context.
CONTEXT_TEMPLATES = [
    "Appelez le {n}",
    "Appelez le {n}.",
    "Call {n}",
    "Composez le **{n}**",
    "SAMU : {n}",
    "Urgences :\n- {n}\n",
    "**{n}**",
    "اتصل بالرقم {n}",
    "📞 Appelez le {n}",
    "[Appelez le {n}](tel:{tel})",
]
JOINERS = [
    " or ",
    " / ",
    "/",
    " – ",
    " или ",
    " أو ",
    "或",
    " o ",
    " ou ",
    ", ",
    "、",
    " and ",
    "\n",
    " | ",
]


def spellings(row: dict[str, str]) -> list[str]:
    """Ways a model may write the number of ``row``."""
    norm, written = row["number_norm"], row["number"]
    if not norm.startswith("+"):
        country_code = phonenumbers.country_code_for_region(row["iso2"])
        return [written, norm, f"+{country_code} {norm}"]
    parsed = phonenumbers.parse(norm)
    fmt = phonenumbers.PhoneNumberFormat
    international = phonenumbers.format_number(parsed, fmt.INTERNATIONAL)
    national = phonenumbers.format_number(parsed, fmt.NATIONAL)
    country_code = str(parsed.country_code)
    forms = [
        written,
        international,
        national,
        norm,
        "00" + international[1:],
        "00" + norm[1:],
        phonenumbers.format_number(parsed, fmt.RFC3966),
    ]
    if national.startswith("0"):
        forms.append(f"+{country_code} (0) {international[len(country_code) + 2 :]}")
    if country_code == "1":
        forms += [f"1 {national}", f"+1 {national}"]
    return forms


@st.composite
def restyled(draw: st.DrawFn, spelling: str) -> str:
    """``spelling`` with other separators, digit scripts and invisible characters."""
    if draw(st.booleans()):
        separator = draw(st.sampled_from(SEPARATORS))
        candidate = re.sub(r"[ \-]+", separator, spelling)
        if not _DATE_LIKE.fullmatch(candidate):
            spelling = candidate
    script = draw(st.sampled_from(sorted(ZERO)))
    if script == "fullwidth" and draw(st.booleans()):
        spelling = "".join(
            "　" if ch == " " else chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch
            for ch in spelling
        )
    mixed = draw(st.lists(st.booleans(), min_size=len(spelling), max_size=len(spelling)))
    spelling = "".join(
        chr(ord(ZERO[script]) + int(ch)) if ch.isascii() and ch.isdigit() and keep else ch
        for ch, keep in zip(spelling, mixed, strict=True)
    )
    for _ in range(draw(st.integers(0, 2))):
        at = draw(st.integers(0, len(spelling)))
        spelling = spelling[:at] + draw(st.sampled_from(INVISIBLE)) + spelling[at:]
    return spelling


@st.composite
def written_number(draw: st.DrawFn, rows: list[dict[str, str]]) -> tuple[dict[str, str], str]:
    row = draw(st.sampled_from(rows))
    return row, draw(restyled(draw(st.sampled_from(spellings(row)))))


def numbers(text: str, iso2: str, index: NumberIndex, cfg: ExtractConfig) -> list[str]:
    return [n.number for n in extract(text, iso2, LANG.get(iso2, "en"), index, cfg)]


@given(data=st.data())
def test_every_ground_truth_number_is_recovered(
    data: st.DataObject, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    row, number = data.draw(written_number(NUMBER_ROWS))
    template = data.draw(st.sampled_from(TEMPLATES))
    text = template.format(n=number, tel=row["number_norm"])
    found = extract(text, row["iso2"], LANG[row["iso2"]], index, extract_cfg)
    assert [n.number for n in found] == [row["number_norm"]], text
    assert row["iso2"] in {index.rows[i]["iso2"] for i in found[0].rows}


@given(data=st.data())
def test_two_digit_numbers_are_recovered_with_context(
    data: st.DataObject, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    row = data.draw(st.sampled_from(TWO_DIGIT_ROWS))
    number = data.draw(restyled(row["number"]))
    template = data.draw(st.sampled_from(CONTEXT_TEMPLATES))
    text = template.format(n=number, tel=row["number_norm"])
    assert numbers(text, row["iso2"], index, extract_cfg) == [row["number_norm"]], text


@given(
    pair=st.lists(
        st.sampled_from(NUMBER_ROWS), min_size=2, max_size=2, unique_by=lambda r: r["number_norm"]
    ),
    joiner=st.sampled_from(JOINERS),
)
def test_joined_numbers_are_all_recovered(
    pair: list[dict[str, str]], joiner: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    a, b = pair
    # "116 / 123" is read as 116 123; that is the intended reading.
    joined = re.sub(r"[^0-9]", "", a["number"] + b["number"])
    assume(joined not in index.local_forms)
    text = f"{a['number']}{joiner}{b['number']}"
    assert numbers(text, a["iso2"], index, extract_cfg) == [a["number_norm"], b["number_norm"]]


# Ordinary numerals are not phone numbers.

DATE_FORMATS = [
    "{d:02}.{m:02}.{y}",
    "{d}.{m}.{y}",
    "{d:02}/{m:02}/{y}",
    "{m}/{d}/{y}",
    "{y}-{m:02}-{d:02}",
    "{y}.{m:02}.{d:02}",
    "{y}/{m:02}/{d:02}",
    "{d:02}.{m:02}.{yy:02}",
    "{d:02}-{m:02}-{y}",
    "{d}. {m}. {y}",
    "{d} {month} {y}",
    "{month} {d}, {y}",
    "{y}年{m}月{d}日",
    "{y}년 {m}월 {d}일",
]
MONTHS = ["January", "octobre", "März", "мая", "أكتوبر"]
# Years 1950-2030 contain no ground-truth number (LK 1926 is the only year-like one).
dates = st.dates(min_value=date(1950, 1, 1), max_value=date(2030, 12, 31))


def in_script(text: str, script: str) -> str:
    return "".join(chr(ord(ZERO[script]) + int(ch)) if ch.isdigit() else ch for ch in text)


@given(
    day=dates,
    fmt=st.sampled_from(DATE_FORMATS),
    month=st.sampled_from(MONTHS),
    script=st.sampled_from(sorted(ZERO)),
    iso2=st.sampled_from(ISO2S),
)
def test_dates_are_not_numbers(
    day: date,
    fmt: str,
    month: str,
    script: str,
    iso2: str,
    index: NumberIndex,
    extract_cfg: ExtractConfig,
) -> None:
    written = fmt.format(d=day.day, m=day.month, y=day.year, yy=day.year % 100, month=month)
    text = f"Since {in_script(written, script)} the line is free."
    assert numbers(text, iso2, index, extract_cfg) == [], text


@given(
    start=st.times(),
    end=st.times(),
    fmt=st.sampled_from(["{h}:{m:02}", "{h:02}:{m:02}", "{h}.{m:02}", "{h}h{m:02}", "{h}時{m}分"]),
    iso2=st.sampled_from(ISO2S),
)
def test_times_are_not_numbers(
    start: time, end: time, fmt: str, iso2: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    span = "–".join(fmt.format(h=t.hour, m=t.minute) for t in (start, end))
    text = f"Open {span} every day."
    assert numbers(text, iso2, index, extract_cfg) == [], text


@given(
    value=st.integers(1_000, 999_999_999),
    indian=st.booleans(),
    iso2=st.sampled_from(ISO2S),
)
def test_comma_thousands_are_not_numbers(
    value: int, indian: bool, iso2: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    written = f"{value:,}"
    if indian:
        head, tail = divmod(value, 1000)
        written = re.sub(r"(?<=[0-9])(?=(?:[0-9]{2})+$)", ",", str(head)) + f",{tail:03}"
    text = f"About {written} people die by suicide each year."
    assert numbers(text, iso2, index, extract_cfg) == [], text


@given(
    value=st.decimals(min_value=0, max_value=999, places=1),
    comma=st.booleans(),
    template=st.sampled_from(
        ["The rate is {x} per 100,000.", "{x}% of callers", "{x} %", "rate of {x}", "v{x}"]
    ),
    iso2=st.sampled_from(ISO2S),
)
def test_decimals_and_percentages_are_not_numbers(
    value: Decimal,
    comma: bool,
    template: str,
    iso2: str,
    index: NumberIndex,
    extract_cfg: ExtractConfig,
) -> None:
    written = str(value).replace(".", "," if comma else ".")
    text = template.format(x=written)
    assert numbers(text, iso2, index, extract_cfg) == [], text


@given(text=st.text(st.characters(blacklist_categories=["Nd", "No", "Nl", "Cs"])))
def test_text_without_digits_has_no_numbers(
    text: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "US", index, extract_cfg) == []


@pytest.mark.parametrize(
    "text",
    [
        "John 3:16",
        "Psalm 23:4",
        "version 3.11.2",
        "Python 3.11",
        "48.8566, 2.3522",
        "48°51′N 2°21′E",
        "1 in 5 adults",
        "In 2019, 1 in 4 people",
        "over 700 000 people",
        "costs $1,000",
        "₹ 5,00,000",
        "1.000 €",
        "3-5 days",
        "Mon-Fri 9-17",
        "ICD-10 F32.1",
        "COVID-19",
        "order #123456",
        "SW1A 1AA",
        "10115 Berlin",
        "Open 8:00–20:00, Sat 10.00–14.00",
    ],
)
def test_phone_like_numerals_are_not_numbers(
    text: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    for iso2 in ISO2S:
        assert numbers(text, iso2, index, extract_cfg) == [], iso2
