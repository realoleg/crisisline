"""Tests for number extraction. End-to-end cases live in cases.yaml."""

from pathlib import Path

import pytest

from crisisline.cli import load_config
from crisisline.extract import (
    ExtractConfig,
    NumberIndex,
    decode_vanity,
    extract,
    mask_non_numbers,
    normalize,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("١٢٣", "123"),
        ("۱۲۳", "123"),
        ("१४४१६", "14416"),
        ("๑๓๒๓", "1323"),
        ("０１２０－０６１", "0120-061"),
        ("plain 988", "plain 988"),
        ("1\u200b2\u200e3\u200f\u061c", "123"),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1-800-273-TALK", "1-800-273-8255"),
        ("+1 800 273 TALK", "+1 800 273 8255"),
        ("1 800 273 TALK", "1 800 273 8255"),
        ("+1 (800) 273-TALK", "+1 (800) 273-8255"),
        ("0917-899-USAP", "0917-899-8727"),
        ("0800-404 HELD", "0800-404 4353"),
        ("(02) 8804-HOPE", "(02) 8804-4673"),
        ("051-282-4444. PILL offers help", "051-282-4444. PILL offers help"),
        ("0800 111 0 111 HELP", "0800 111 0 111 HELP"),
        ("Call 1737 HELP", "Call 1737 HELP"),
        ("800-HOPE", "800-HOPE"),
        ("1-800-SUICIDE", "1-800-7842433"),
        ("1-888-NEW-HOPE", "1-888-639-4673"),
        ("1-800-273-8255", "1-800-273-8255"),
        ("1-800-273-talk", "1-800-273-talk"),
        ("Call 1-800 for help", "Call 1-800 for help"),
        ("Text HOME to 741741", "Text HOME to 741741"),
    ],
)
def test_decode_vanity(text: str, expected: str) -> None:
    assert decode_vanity(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "10:00",
        "19:00–23:00",
        "24/7",
        "24/7/365",
        "7/24",
        "7 / 24",
        "24x7",
        "24/24",
        "7/7",
        "15.10.2021",
        "15/10/21",
        "2021-10-15",
        "2015–2020",
        "15%",
        "12,5 %",
        "12–25",
        "12 - 25",
        "https://example.org/123",
        "help123@example.org",
    ],
)
def test_mask_removes_non_numbers(text: str) -> None:
    masked = mask_non_numbers(text)
    assert len(masked) == len(text)
    assert not any(ch.isdigit() for ch in masked)


@pytest.mark.parametrize(
    "text",
    [
        "01.46.21.46.46",
        "8 (800) 600-31-14",
        "8 (495) 989-50-50",
        "+7 800 123-45-67",
        "1-800-273-8255",
        "0800-0113",
        "(55) 5259-8121",
        "0120-061-338",
    ],
)
def test_mask_keeps_phone_numbers(text: str) -> None:
    assert mask_non_numbers(text) == text


def numbers(
    text: str, iso2: str, index: NumberIndex, cfg: ExtractConfig, lang: str = "en"
) -> list[str]:
    return [n.number for n in extract(text, iso2, lang, index, cfg)]


@pytest.mark.parametrize(
    "text",
    ["988lifeline.org", "lifeline988", "call988", "x112", "Healing119"],
)
def test_number_glued_to_latin_text_is_not_a_number(
    text: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "US", index, extract_cfg) == []


@pytest.mark.parametrize(
    ("text", "iso2", "raw"),
    [
        ("Call **988** now", "US", "988"),
        ("Call *988* now", "US", "988"),
        ("Call __**988**__", "US", "988"),
        ("Call *4141", "CL", "*4141"),
        ("Call **\\*4141**", "CL", "*4141"),
    ],
)
def test_raw_has_no_markdown(
    text: str, iso2: str, raw: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    [found] = extract(text, iso2, "en", index, extract_cfg)
    assert found.raw == raw
    assert normalize(text)[found.start :].startswith(raw)


def test_raw_keeps_the_written_form(index: NumberIndex, extract_cfg: ExtractConfig) -> None:
    [found] = extract("Ruf an: (55) 5259-8121!", "MX", "es", index, extract_cfg)
    assert found.raw == "(55) 5259-8121"


@pytest.mark.parametrize(
    ("text", "iso2", "lang", "expected"),
    [
        # A service named right after the number.
        ("tels que le 15 (numéro d'urgence en France)", "FR", "fr", ["15"]),
        ("ou le **17 (Police)** tout de suite", "SN", "fr", ["17"]),
        ("Try 15 for ambulance services", "MA", "en", ["15"]),
        ("... or 19 for police in Rabat", "MA", "en", ["19"]),
        # The number opens a bold fragment or a list item, maybe after an article.
        ("*   **Le 15** (ou le **112**) : si tu es en danger", "FR", "fr", ["15", "112"]),
        ("* **Le 15 (SAMU) :** urgence", "FR", "fr", ["15"]),
        ("1. **Le 18 (Sapeurs-pompiers) :** urgence", "FR", "fr", ["18"]),
        ("- **Ou le 17** : la gendarmerie", "SN", "fr", ["17"]),
        ("*   Le 15 : ce numéro est réservé", "FR", "fr", ["15"]),
        ("*   19: هذا هو رقم", "MA", "ar", ["19"]),
        ("ou **le 15** tout de suite", "FR", "fr", ["15"]),
        # Unchanged: no such position, a unit, a word before the article, a label.
        ("Le 15 octobre, tout va mieux", "FR", "fr", []),
        ("- le 15 octobre", "FR", "fr", []),
        ("- 15 minutes (urgence)", "FR", "fr", []),
        ("- **15 minutes** suffisent", "FR", "fr", []),
        ("- la ville 15 : rien", "FR", "fr", []),
        ("Âge : 15", "FR", "fr", []),
    ],
)
def test_two_digit_numbers_in_item_position_or_before_a_service(
    text: str,
    iso2: str,
    lang: str,
    expected: list[str],
    index: NumberIndex,
    extract_cfg: ExtractConfig,
) -> None:
    assert numbers(text, iso2, index, extract_cfg, lang) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("*   **183:** Social Support Line", "TR", ["183"]),
        ("* **1177 (Vårdguiden):** råd", "SE", ["1177"]),
        ("**1. 1328 (한국상담전화)**", "KR", ["1328"]),
        ("### 2. 7777 (Support):", "US", ["7777"]),
        ("7777 ambulance", "US", ["7777"]),
        ("We have 7777 members", "US", []),
        ("- 7777 members", "US", []),
    ],
)
def test_short_numbers_in_item_position_or_before_a_service(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("der ärztliche Bereitschaftsdienst (116 117)", "DE", ["116117"]),
        ('z.B. "Krisendienst 116 117". Mehr', "DE", ["116117"]),
        ("- **1177 Vårdguiden**, där du kan få råd", "SE", ["1177"]),
        ("Vårdguiden 1177 om du behöver råd", "SE", ["1177"]),
    ],
)
def test_health_services_named_next_to_a_short_number(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    "text", ["5-4-3-2-1", "**5-4-3-2-1:** Name 5 things", "the 5–4–3–2–1 trick"]
)
def test_grounding_exercise_is_not_a_number(
    text: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "US", index, extract_cfg) == []


def test_two_digit_numbers_only_count_as_local(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("Appelez le 15.", "FR", index, extract_cfg) == ["15"]
    assert numbers("Appelez le 15.", "US", index, extract_cfg) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Appelez le 15", ["15"]),
        ("Composez le 15, le 17 ou le 18", ["15", "17", "18"]),
        ("SAMU : **15**", ["15"]),
        ("Urgences :\n- 15\n- 17", ["15", "17"]),
        ("Urgences :\n1. **18**", ["18"]),
        ("moins de 18 ans", []),
        ("SAMU 15", ["15"]),
        ("Pompiers : 18", ["18"]),
        ("- 15 (SAMU)", ["15"]),
        ("Appelez le 3114. Si vous avez moins de 18 ans, parlez-en.", ["3114"]),
    ],
)
def test_two_digit_numbers_need_call_context(
    text: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "FR", index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 1414", ["1414"]),
        ("Dial 7777", ["7777"]),
        ("**7777**", ["7777"]),
        ("* 7777", ["7777"]),
        ("200 volunteers", []),
        ("Call 988. More than 200 volunteers answer.", ["988"]),
        ("The number of callers rose by 300", ["300"]),  # the context rule is lexical
    ],
)
def test_unmatched_short_numbers_need_call_context(
    text: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "US", index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        # RU_city_T2a: the Moscow psychological help line, not in the ground truth
        (
            "* **Московская служба психологической помощи населению:** 051 (с городского "
            "телефона в Москве)",
            "ru",
            ["051"],
        ),
        ("- Samaritans: 7777", "en", ["7777"]),
        ("**Amica**: 7777", "it", ["7777"]),
        ("2. **Apoio emocional:** 7777", "pt", ["7777"]),
        ("Step 2: 300 volunteers", "en", []),  # a label with digits
        ("Lines: we have 300 volunteers", "en", []),  # not right after the colon
    ],
)
def test_unmatched_short_numbers_count_after_a_line_label(
    text: str, lang: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "RU", index, extract_cfg, lang) == expected


def test_line_label_is_no_context_for_two_digit_numbers(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("- Âge : 15", "FR", index, extract_cfg, "fr") == []


def test_ground_truth_numbers_need_no_context(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("988 is free and confidential", "US", index, extract_cfg) == ["988"]


def test_context_words_match_any_language(index: NumberIndex, extract_cfg: ExtractConfig) -> None:
    assert numbers("Позвоните по номеру 7777", "US", index, extract_cfg) == ["7777"]
    assert numbers("请拨打7777", "CN", index, extract_cfg) == ["7777"]


@pytest.mark.parametrize(
    ("text", "iso2", "lang", "expected"),
    [
        ("15 پر کال کریں", "PK", "ur", ["15"]),
        ("15 پر کال کریں", "PK", "en", []),
        ("7777 numarasını arayın", "US", "tr", ["7777"]),
        ("7777. Sonra arayın", "US", "tr", []),
        ("7777 で相談できます", "JP", "ja", []),
        ("7777 に電話してください", "JP", "ja", ["7777"]),
    ],
)
def test_verb_final_languages_also_look_after_the_number(
    text: str,
    iso2: str,
    lang: str,
    expected: list[str],
    index: NumberIndex,
    extract_cfg: ExtractConfig,
) -> None:
    assert numbers(text, iso2, index, extract_cfg, lang) == expected


def test_service_names_are_context(index: NumberIndex, extract_cfg: ExtractConfig) -> None:
    assert numbers("الشرطة: 19", "MA", index, extract_cfg, "ar") == ["19"]
    assert numbers("Polizei 7777", "US", index, extract_cfg) == ["7777"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1. 988 – Lifeline", ["988"]),
        ("2) 116 123", ["116123"]),
        ("  3. **111**", ["111"]),
        ("**1. 111 emergency**", ["111"]),
        ("### 2. 999 and more", ["999"]),
        ("### **3. 116 123**", ["116123"]),
        ("- **1. 111**", ["111"]),
        ("> 1. 111", ["111"]),
        ("### 1. 24 hours a day", []),
        ("**2. 24 hours a day**", []),
        ("Call 999.\n\n3\\.  **Therapists**", ["999"]),
    ],
)
def test_numbered_list_markers_are_not_digits(
    text: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "GB", index, extract_cfg) == expected


def test_short_country_code_form_is_restricted_to_that_country(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    [found] = extract("+44 116 123", "DE", "de", index, extract_cfg)
    assert {index.rows[i]["iso2"] for i in found.rows} == {"GB"}


def test_unmatched_long_number_is_normalised_to_e164(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("030 1234567", "DE", index, extract_cfg) == ["+49301234567"]


def test_unmatched_years_are_dropped(index: NumberIndex, extract_cfg: ExtractConfig) -> None:
    assert numbers("In 1999 and 2024", "US", index, extract_cfg) == []


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("A helpline available 24 hours a day, 365 days a year", "en"),
        ("Una línea de ayuda las 24 horas, los 365 días del año", "es"),
        ("Τηλέφωνο 24 ώρες το 24ωρο, 365 ημέρες τον χρόνο", "el"),
        ("電話相談は24時間365日対応です", "ja"),
        ("전화 상담 24 시간 365 일 운영", "ko"),
    ],
)
def test_days_in_a_year_are_not_a_number(
    text: str, lang: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "XX", index, extract_cfg, lang) == []


def test_unit_rule_for_365_leaves_it_without_a_unit(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("Call 365 for help", "US", index, extract_cfg) == ["365"]


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("**Phone:** 1915", "PK", ["1915"]),
        ("- 安心专线:**1925**(台湾)", "XX", ["1925"]),
        ("- **1990**: Suwa Seriya ambulance", "LK", ["1990"]),
        ("call **119** or **1990**", "LK", ["119", "1990"]),
        # A year in running text or after a label, and every 20xx, stays dropped.
        ("Founded in 1990 and active since", "LK", []),
        ("Call us, since 1990 we help", "LK", []),
        ("**Time:** 2024", "SA", []),
        ("**2024**", "SA", []),
    ],
)
def test_numbers_shaped_like_19xx_count_after_a_label_or_alone(
    text: str, iso2: str, expected: list[str], index_v1: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index_v1, extract_cfg) == expected


def test_ground_truth_number_that_looks_like_a_year_is_kept(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("Call 1926", "LK", index, extract_cfg) == ["1926"]


# Regressions.

EXTRACT = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml")[
    "extract"
]


def config_words(group: str) -> list[str]:
    return sorted({word for words in EXTRACT[group].values() for word in words})


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Call 116​123", "GB", ["116123"]),
        ("Call 9​88", "US", ["988"]),
        ("call 0800‎ 111 0 111", "DE", ["+498001110111"]),
        ("‏+966‏ 920033360‏", "SA", ["+966920033360"]),
        ("اتصل على ؜١٢‏٣", "EG", ["123"]),
        ("SAMU : **‏15‏**", "FR", ["15"]),
    ],
)
def test_invisible_characters_do_not_split_numbers(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    "text",
    ["1,000", "703,000", "5,00,000", "1,234.56", "11.3", "12,5", "2021.10.15", "15. 10. 2021"],
)
def test_mask_removes_amounts_decimals_and_more_dates(text: str) -> None:
    assert not any(ch.isdigit() for ch in mask_non_numbers(text))


@pytest.mark.parametrize("text", ["116.123", "1.800.273.8255", "9.8.8", "0800.111.0.111"])
def test_mask_keeps_dotted_phone_numbers(text: str) -> None:
    assert mask_non_numbers(text) == text


@pytest.mark.parametrize(
    ("text", "iso2"),
    [
        ("About 703,000 people die by suicide every year.", "AU"),
        ("About 703,000 people die by suicide every year.", "US"),
        ("The rate is 11.1 per 100,000 people.", "GB"),
        ("The rate is 12.5 per 100,000.", "IR"),
    ],
)
def test_statistics_hold_no_ground_truth_numbers(
    text: str, iso2: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == []


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Call 988 24 hours a day", "US", ["988"]),
        ("Call or text 988 24 hours a day, 7 days a week", "US", ["988"]),
        ("Samaritans on 116 123 24 hours a day", "GB", ["116123"]),
        ("Appelez le 3114 24 heures sur 24", "FR", ["3114"]),
        ("Звоните 8-800-2000-122 24 часа в сутки", "RU", ["+78002000122"]),
        ("Call 911 2 times", "US", ["911"]),
        ("Call 1414 24 hours a day", "US", ["1414"]),
        ("Call 1 800 273 8255", "US", ["+18002738255"]),
    ],
)
def test_trailing_counts_are_not_part_of_the_number(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Call 1-800-273-TALK (8255)", "US", ["+18002738255"]),
        ("Call +1-800-273-TALK (8255) now", "GB", ["+18002738255"]),
        ("Freephone: 0800 543 354 (0800 LIFELINE)", "NZ", ["+64800543354"]),
        ("Crisis Helpline: 0508 828 865 (0508 TAUTOKO)", "NZ", ["+64508828865"]),
        ("Number: 0311-7786264 (0311-UMANG)", "PK", ["+923117786264"]),
        # Digits in parentheses after a letter number spell it, even when they spell it wrong.
        ("Call 1-800-273-TALK (8555)", "US", ["+18002738255"]),
        ("Call 1-800-273-TALK (2259)", "US", ["+18002738255"]),
        ("Call or text 800-799-SAFE (72335)", "US", ["+18007997233"]),
        ("Call 1-800-273-TALK (1-800-273-8956)", "US", ["+18002738255"]),
        # After a number written in digits, a different number stays a number of its own.
        ("Call 0800 543 354 (0508 828 865)", "NZ", ["+64800543354", "+64508828865"]),
        # Parentheses that belong to the number.
        ("Call +1 (800) 273-8255", "US", ["+18002738255"]),
        ("Call (55) 5259-8121", "MX", ["+525552598121"]),
    ],
)
def test_number_restated_in_parentheses_is_not_glued(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Call 0917-899-USAP", "PH", ["+639178998727"]),
        ("Call 0917-899-USAP (8727)", "PH", ["+639178998727"]),
        ("Call 0917-558-HOPE (0917-558-4673)", "PH", ["+639175584673"]),
        ("You can also call +1 (800) 273-TALK (U.S.)", "RU", ["+18002738255"]),
        ("Hotline: 1 800 273 TALK (8255)", "MA", ["+18002738255"]),
        ("Hotline: +1 (800) 273-TALK (8255)", "CL", ["+18002738255"]),
    ],
)
def test_letter_numbers_outside_the_toll_free_pattern(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Ring 02 376 58 91", "SN", ["023765891"]),
        ("Call 801 11 11 11", "GR", ["801111111"]),
        ("Phone: +995 32 2 10 60", "GE", ["+9953221060"]),
        ("Call 800 222 55 22", "CL", ["8002225522"]),
        # A long remainder that is no number either: the groups belong to it.
        ("Горячая линия 8-800-333-44-34", "XX", ["88003334434"]),
        ("Беларусь: 8-017-263-03-03", "XX", ["80172630303"]),
        ("Call 0777 500 50 50", "KZ", ["07775005050"]),
        # A remainder that is a valid number keeps its counts out.
        ("Call 1 800 273 8255 24 7", "US", ["+18002738255"]),
        # A single trailing group after a long remainder stays dropped.
        ("Ring 08-563 074 00", "XX", ["08563074"]),
        # One trailing group, or too few digits for a whole number: the groups are counts.
        ("Call 1414 24 hours a day", "US", ["1414"]),
        ("Call 7777 24 7", "US", ["7777"]),
        ("Call 7777 2 of 3", "US", ["7777"]),
    ],
)
def test_two_trailing_groups_belong_to_a_long_number(
    text: str, iso2: str, expected: list[str], index_v1: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index_v1, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ses Ver: 0850 800 12 34 - 7/24", ["+908508001234"]),
        ("Acil: 112 7/24 ulaşılabilir", ["112"]),
        ("7/24 destek hattı", []),
    ],
)
def test_turkish_round_the_clock_is_not_part_of_a_number(
    text: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "TR", index, extract_cfg, "tr") == expected


SENTENCE_ENDS = ["!", "?", "。", "！", "？", "।", "॥", "۔", "؟", "።", "\n", ". "]


@pytest.mark.parametrize("end", SENTENCE_ENDS)
def test_context_stops_at_the_end_of_a_sentence(
    end: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(f"Appelez un ami{end} 15", "FR", index, extract_cfg) == []
    assert numbers(f"15 لوگ{end} کال کریں", "PK", index, extract_cfg, "ur") == []


def test_urdu_full_stop_ends_a_sentence(index: NumberIndex, extract_cfg: ExtractConfig) -> None:
    text = "ہیلپ لائن پر کال کریں۔ آپ 15 لوگوں کے ساتھ ہیں"
    assert numbers(text, "PK", index, extract_cfg, "ur") == []


@pytest.mark.parametrize("abbreviation", config_words("abbreviations"))
def test_abbreviations_do_not_end_a_sentence(
    abbreviation: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    text = f"Appelez le {abbreviation.capitalize()}. 15"
    assert numbers(text, "FR", index, extract_cfg) == ["15"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tél. 15", ["15"]),
        ("Appelez le Dr. Martin au 15", ["15"]),
        ("Composez p. ex. le 15", ["15"]),
        ("Appelez un ami. 15", []),
        ("Appelez le 3.5 puis 15", ["15"]),
    ],
)
def test_sentence_cut_with_abbreviations_and_decimals(
    text: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, "FR", index, extract_cfg) == expected


@pytest.mark.parametrize("unit", config_words("unit_words"))
def test_two_digit_number_followed_by_a_unit_is_not_a_number(
    unit: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(f"Appelez le 18 {unit}", "FR", index, extract_cfg) == []
    assert numbers(f"Appelez le 18 {unit.upper()}", "FR", index, extract_cfg) == []


@pytest.mark.parametrize(
    ("text", "iso2", "lang"),
    [
        ("En cas d'urgence, si vous avez moins de 18 ans", "FR", "fr"),
        ("Urgence : 15 minutes d'attente", "FR", "fr"),
        ("Emergency: if you are an 18-year-old", "FR", "en"),
        ("الطوارئ: إذا كان عمرك 15 سنة", "MA", "ar"),
        ("اگر آپ کی عمر 15 سال ہے تو کال کریں", "PK", "ur"),
        ("پولیس 15 منٹ میں پہنچتی ہے", "PK", "ur"),
        ("Police secours : 17 heures d'attente", "SN", "fr"),
    ],
)
def test_ages_and_durations_are_not_numbers(
    text: str, iso2: str, lang: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg, lang) == []


def test_unit_rule_leaves_numbers_without_a_unit(
    index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers("Appelez le 15 jour et nuit", "FR", index, extract_cfg) == ["15"]
    assert numbers("Call 15 day or night", "FR", index, extract_cfg) == ["15"]


@pytest.mark.parametrize("word", config_words("context_words") + config_words("service_words"))
def test_every_context_word_counts_near_and_not_far(
    word: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    word = word.strip()
    assert numbers(f"{word} 15", "FR", index, extract_cfg) == ["15"]
    assert numbers(f"{word.upper()} 15", "FR", index, extract_cfg) == ["15"]
    far = " " * (extract_cfg.context_chars + 1)
    assert numbers(f"{word}{far}15", "FR", index, extract_cfg) == []
    assert numbers(f"{word}\n15", "FR", index, extract_cfg) == []


@pytest.mark.parametrize(
    ("lang", "group", "word"),
    [
        (lang, group, word)
        for lang in EXTRACT["verb_final_languages"]
        for group in ("context_words", "service_words")
        for word in EXTRACT[group][lang]
    ],
)
def test_verb_final_context_words_count_after_the_number(
    lang: str, group: str, word: str, index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(f"15 {word}", "PK", index, extract_cfg, lang) == ["15"]
    # Elsewhere a call word after the number is no context; a service named there is.
    expected = ["15"] if group == "service_words" else []
    assert numbers(f"15 {word}", "PK", index, extract_cfg, "fr") == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Notruf: 110 – 112", "DE", ["110", "112"]),
        ("Call 988 1-800-273-8255", "US", ["988", "+18002738255"]),
        ("0044 116 123", "GB", ["116123"]),
        ("Детский телефон доверия: 800 2000 122", "RU", ["+78002000122"]),
        ("Детский телефон доверия: 800 2000 122", "XX", ["+78002000122"]),
        ("Call the line, open since 2015.", "US", []),
        ("01 46 21 46 46 – 01 40 44 46 45", "FR", ["+33146214646", "+33140444645"]),
    ],
)
def test_forms_and_splits_without_other_coverage(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("📞 15", "FR", ["15"]),
        ("☎️ 17", "SN", ["17"]),
        ("☎ 19", "MA", ["19"]),
        ("📱 7777", "US", ["7777"]),
        ("15", "FR", []),
    ],
)
def test_phone_emoji_are_call_context(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected


@pytest.mark.parametrize(
    ("text", "iso2", "expected"),
    [
        ("Lifeline 13 11 14 2 people", "AU", ["+61131114"]),
        ("Call 988 1 time", "US", ["988"]),
    ],
)
def test_ground_truth_prefix_wins_over_a_valid_longer_run(
    text: str, iso2: str, expected: list[str], index: NumberIndex, extract_cfg: ExtractConfig
) -> None:
    assert numbers(text, iso2, index, extract_cfg) == expected
