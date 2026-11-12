"""Prompt assembly and translation.

Translation units are whole user prompts (cue + template) for every non-English country. Each
one is translated, back-translated by a model of another vendor and checked by
:func:`qa_flags`, always over the whole language file (:func:`recheck`), so flags do not depend
on the order in which entries were filled. ``build_prompts`` assembles the full prompt matrix
from the configs, the ground truth and the translations. Everything here is pure
except :func:`translate`, which calls the injected client.
"""

from __future__ import annotations

import dataclasses
import json
import re
import statistics
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sacrebleu.metrics import CHRF

from crisisline.groundtruth import CRISIS_TYPES
from crisisline.llm import ChatClient, Completion, CompletionError, ModelSpec, Result

Row = Mapping[str, str]
Key = tuple[str, str, str]  # (iso2, cue, template)

ENGLISH = "en"
CUES = ("city", "country")
NO_CUE = "none"
ROUND_THE_CLOCK = "24/7"
FAILED_FLAGS = frozenset({Result.CONTENT_FILTER.value, Result.EMPTY.value})
# Flags of the API outcome, set at translation time; every other flag is a QA check.
OUTCOME_FLAGS = FAILED_FLAGS | {f"back_{flag}" for flag in FAILED_FLAGS}
ADDED_CONTENT = "added_content"

# The translator answers in JSON with this one field (structured outputs), so a help note or
# any other text outside the translation has no place to go.
TRANSLATION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"translation": {"type": "string"}},
            "required": ["translation"],
            "additionalProperties": False,
        },
    },
}

_CHRF = CHRF()
# Typographic quotes the back-translator writes where the English source has ASCII ones.
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


class MalformedTranslation(CompletionError):
    """The translator's answer is not the JSON object it was asked for; a rerun may succeed."""


@dataclass(frozen=True, slots=True)
class Unit:
    """One string to translate: a whole English user prompt for one country."""

    iso2: str
    lang: str
    cue: str
    template: str
    source_en: str
    country_en: str
    name: str | None  # local country or city name the translation must contain, maybe inflected

    @property
    def key(self) -> Key:
        return (self.iso2, self.cue, self.template)


@dataclass(frozen=True, slots=True)
class Translation:
    """One entry of ``data/translations/{lang}.yaml``."""

    iso2: str
    cue: str
    template: str
    source_en: str
    translation: str
    back_translation: str
    chrf: float | None  # None when there is no translation to back-translate
    flags: list[str] = field(default_factory=list)
    qa: str = "auto"
    translator: dict[str, str | None] = field(default_factory=dict)
    back_translator: dict[str, str | None] = field(default_factory=dict)
    date: str = ""


def entry_key(entry: Mapping[str, Any]) -> Key:
    return (entry["iso2"], entry["cue"], entry["template"])


def user_prompt(template_text: str, cue_sentence: str | None) -> str:
    """The cue sentence, if any, then one space and the template."""
    return f"{cue_sentence} {template_text}" if cue_sentence else template_text


def english_name(country: Row, cfg: Mapping[str, Any]) -> str:
    """The country's name in English running text, with the article where English requires it."""
    return cfg["country_en"].get(country["iso2"], country["name_en"])


def english_prompt(cfg: Mapping[str, Any], country: Row, cue: str, template: str) -> str:
    text = cfg["templates"][template]["text"]
    if cue == NO_CUE:
        return text
    sentence = cfg["cues"][cue].format(city=country["city_en"], country=english_name(country, cfg))
    return user_prompt(text, sentence)


def translation_units(countries: Sequence[Row], cfg: Mapping[str, Any]) -> list[Unit]:
    """cue ∈ {city, country, none} × every template, for each non-English country."""
    local_name = {"city": "city_local", "country": "country_local"}
    return [
        Unit(
            iso2=country["iso2"],
            lang=country["lang"],
            cue=cue,
            template=template,
            source_en=english_prompt(cfg, country, cue, template),
            country_en=english_name(country, cfg),
            name=country[local_name[cue]] if cue in local_name else None,
        )
        for country in countries
        if country["lang"] != ENGLISH
        for cue in (*CUES, NO_CUE)
        for template in cfg["templates"]
    ]


def translator_messages(unit: Unit, cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    settings = cfg["translation"]
    notes = [
        settings["language_notes"].get(unit.lang),
        settings["country_notes"].get(unit.iso2),
    ]
    user = settings["translator_user"].format(
        language=settings["languages"][unit.lang],
        country_en=unit.country_en,
        notes="".join(f" {note}" for note in notes if note),
        names=settings["names"][unit.cue].format(name=unit.name) if unit.name else "",
        source=unit.source_en,
    )
    return [
        {"role": "system", "content": settings["translator_system"]},
        {"role": "user", "content": user},
    ]


def back_translator_messages(lang: str, text: str, cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    settings = cfg["translation"]
    user = settings["back_translator_user"].format(language=settings["languages"][lang], text=text)
    return [
        {"role": "system", "content": settings["back_translator_system"]},
        {"role": "user", "content": user},
    ]


def translate(
    unit: Unit,
    client: ChatClient,
    translator: ModelSpec,
    back_translator: ModelSpec,
    params: Mapping[str, Any],
    cfg: Mapping[str, Any],
    today: str,
) -> Translation:
    """Translate and back-translate one unit; QA flags are added by :func:`merge`.

    The translator answers in JSON (:data:`TRANSLATION_FORMAT`). A translation declined by a
    content filter or returned empty is kept, without a back-translation, and flagged
    ``content_filter`` or ``empty``; a back-translation that fails so is flagged
    ``back_content_filter`` or ``back_empty``. Raises :class:`crisisline.llm.CompletionError`
    when a call fails technically, :class:`MalformedTranslation` when the answer is not the
    JSON asked for.
    """
    forward = client.complete(
        translator,
        translator_messages(unit, cfg),
        {**params, "response_format": TRANSLATION_FORMAT},
    )
    result, text = forward.result, forward.text
    if result == Result.OK:
        text = parse_translation(text)
        result = Result.OK if text else Result.EMPTY
    fields = {
        "iso2": unit.iso2,
        "cue": unit.cue,
        "template": unit.template,
        "source_en": unit.source_en,
        "translation": text,
        "translator": _model_record(translator, forward),
        "date": today,
    }
    if result != Result.OK:
        return Translation(**fields, back_translation="", chrf=None, flags=[result.value])
    back = client.complete(back_translator, back_translator_messages(unit.lang, text, cfg), params)
    return Translation(
        **fields,
        back_translation=back.text,
        chrf=round(chrf(back.text, unit.source_en), 2),
        flags=[] if back.result == Result.OK else [f"back_{back.result.value}"],
        back_translator=_model_record(back_translator, back),
    )


def parse_translation(text: str) -> str:
    """The ``translation`` field of the translator's JSON answer, stripped."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedTranslation(f"translator answer is not JSON: {text[:200]!r}") from exc
    if not (
        isinstance(answer, dict)
        and answer.keys() == {"translation"}
        and isinstance(answer["translation"], str)
    ):
        raise MalformedTranslation(f"translator answer is not {{translation}}: {text[:200]!r}")
    return answer["translation"].strip()


def failed(entry: Mapping[str, Any]) -> bool:
    """The translation itself was declined or empty (not just its back-translation)."""
    return bool(FAILED_FLAGS & set(entry.get("flags") or ()))


def _model_record(spec: ModelSpec, completion: Completion) -> dict[str, str | None]:
    """Model and provider as returned by the API, and the reasoning effort requested."""
    return {
        "model": completion.model,
        "provider": completion.provider,
        "reasoning_effort": (spec.reasoning or {}).get("effort"),
    }


def _chrf_text(text: str) -> str:
    return " ".join(text.translate(_QUOTES).split())


def chrf(hypothesis: str, reference: str) -> float:
    """Sentence-level chrF (sacrebleu defaults: character 6-grams, β = 2) after mapping
    typographic quotes to ASCII and collapsing whitespace on both sides."""
    return _CHRF.sentence_score(_chrf_text(hypothesis), [_chrf_text(reference)]).score


def sentence_count(text: str, ends: str) -> int:
    pieces = re.split(f"[{re.escape(ends)}]+", unicodedata.normalize("NFKC", text))
    return sum(1 for piece in pieces if piece.strip())


def in_scripts(text: str, scripts: Sequence[str]) -> bool:
    """Every letter's Unicode name starts with one of ``scripts`` (``LATIN``, ``CJK``, ...)."""
    return all(
        unicodedata.name(char, "").startswith(tuple(scripts))
        for char in unicodedata.normalize("NFKC", text)
        if unicodedata.category(char).startswith("L")
    )


def length_ratio(entry: Mapping[str, Any]) -> float:
    return len(entry["translation"]) / len(entry["source_en"])


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def has_name(text: str, name: str, lang: str, qa_cfg: Mapping[str, Any]) -> bool:
    """``text`` contains a case-folded stem of ``name``, so inflected names count."""
    trim = qa_cfg["name_stem_trim"].get(lang, 0)
    stems = qa_cfg["name_stems"].get(name) or [name[: len(name) - trim]]
    return any(_fold(stem) in _fold(text) for stem in stems)


def added_content(text: str, source: str, patterns: Mapping[str, str]) -> bool:
    """``text`` matches one of ``patterns`` that ``source`` does not."""
    return any(
        re.search(pattern, text) and not re.search(pattern, source) for pattern in patterns.values()
    )


def qa_flags(
    entry: Mapping[str, Any], unit: Unit, median_ratio: float, qa_cfg: Mapping[str, Any]
) -> list[str]:
    """QA flags of one translation; empty if it passes every check.

    ``median_ratio`` is the median translation/source length ratio of the language.
    """
    text = entry["translation"]
    ends = qa_cfg["sentence_ends"] + qa_cfg["sentence_ends_by_lang"].get(unit.lang, "")
    factor = qa_cfg["length_ratio_factor"]
    relative = length_ratio(entry) / median_ratio
    checks = {
        "low_chrf": entry["chrf"] < qa_cfg["chrf_threshold"],
        "missing_name": unit.name is not None and not has_name(text, unit.name, unit.lang, qa_cfg),
        "sentence_count": sentence_count(text, ends) != sentence_count(unit.source_en, ends),
        "wrong_script": not in_scripts(text, qa_cfg["scripts"][unit.lang]),
        "length_ratio": not 1 / factor <= relative <= factor,
        ADDED_CONTENT: added_content(text, unit.source_en, qa_cfg["added_content"]),
    }
    disabled = set(qa_cfg["disabled_checks"].get(unit.lang, []))
    if unknown := disabled - checks.keys():
        raise ValueError(f"translation.disabled_checks.{unit.lang}: unknown {sorted(unknown)}")
    return [name for name, failed in checks.items() if failed and name not in disabled]


def pending(
    units: Iterable[Unit], existing: Mapping[Key, Mapping[str, Any]], force: bool
) -> list[Unit]:
    """Units to translate: missing ones, and ``auto`` ones with ``force``. Never manual/native."""
    return [
        u for u in units if u.key not in existing or (force and existing[u.key]["qa"] == "auto")
    ]


def merge(
    units: Sequence[Unit],
    existing: Sequence[Mapping[str, Any]],
    new: Sequence[Translation],
    qa_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Language file after a run: ``new`` entries replace or join ``existing`` ones in unit order,
    then the whole file is rechecked (:func:`recheck`).

    Entries for keys that are no longer units go last, in their original order.
    """
    by_key: dict[Key, dict[str, Any]] = {entry_key(e): dict(e) for e in existing}
    by_key.update((entry_key(e), e) for e in map(dataclasses.asdict, new))
    unit_keys = {u.key for u in units}
    ordered = [by_key[u.key] for u in units if u.key in by_key]
    return recheck(units, ordered + [e for k, e in by_key.items() if k not in unit_keys], qa_cfg)


def recheck(
    units: Sequence[Unit], entries: Sequence[Mapping[str, Any]], qa_cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """``entries`` of one language with chrF and QA flags recomputed, without API calls.

    chrF is recomputed from the stored back-translation; QA flags use the median length ratio
    over every entry with a translation, so they depend only on the file, not on the order in
    which it was filled. Outcome flags (:data:`OUTCOME_FLAGS`) are kept. Translations and ``qa``
    are never changed, but manual and native entries are rechecked too, so a flag describes
    the current text. Declined or empty entries, and entries for keys that are no longer units,
    are returned unchanged.
    """
    unit_by_key = {u.key: u for u in units}
    out = [dict(e) for e in entries]
    checked = [e for e in out if entry_key(e) in unit_by_key and not failed(e) and e["translation"]]
    if not checked:
        return out
    for entry in checked:
        entry["chrf"] = round(chrf(entry["back_translation"], entry["source_en"]), 2)
    median_ratio = statistics.median(length_ratio(e) for e in checked)
    for entry in checked:
        entry["flags"] = [
            *(f for f in entry["flags"] if f in OUTCOME_FLAGS),
            *qa_flags(entry, unit_by_key[entry_key(entry)], median_ratio, qa_cfg),
        ]
    return out


def _crisis_candidates(rows: Sequence[Row]) -> list[Row]:
    return [
        r
        for r in rows
        if r["decision"] == "accept" and r["status"] == "active" and r["type"] in CRISIS_TYPES
    ]


def _rank(row: Row) -> tuple[bool, bool, bool]:
    return (
        row["type"] != "crisis",
        row["scope"] != "national",
        ROUND_THE_CLOCK not in row["hours"],
    )


def crisis_line(rows: Sequence[Row]) -> Row | None:
    """The country's crisis line for M2: first accepted active crisis or 116 123 row by rank.

    Rank: ``crisis`` before ``emotional_116123``, national before regional, hours with 24/7
    first, then file order.
    """
    return min(_crisis_candidates(rows), key=_rank, default=None)


def emergency_numbers(rows: Sequence[Row]) -> list[str]:
    """Every accepted active emergency number of the country, as written, in file order."""
    return [
        r["number"]
        for r in rows
        if r["decision"] == "accept" and r["status"] == "active" and r["type"] == "emergency"
    ]


@dataclass(frozen=True, slots=True)
class M2Resources:
    """The resources an M2 system prompt names, and the rule that chose the crisis line."""

    iso2: str
    crisis: Row | None  # None for countries without a crisis line
    emergency: list[str]
    rule: str  # "override", "rank: ..." or "no_crisis_line"
    name: str | None = None  # how the system prompt names the crisis line, if not as written
    display: str | None = None  # how the system prompt shows the crisis number, if not as written

    @property
    def prompt_name(self) -> str:
        return self.name or (self.crisis["name"] if self.crisis else "")

    @property
    def prompt_number(self) -> str:
        return self.display or (self.crisis["number"] if self.crisis else "")


def m2_resources(country: Row, rows: Sequence[Row], cfg: Mapping[str, Any]) -> M2Resources:
    """Crisis line from ``m2_overrides`` or by :func:`crisis_line`; every emergency number."""
    iso2 = country["iso2"]
    emergency = emergency_numbers(rows)
    if not emergency:
        raise ValueError(f"{iso2}: no accepted active emergency number for M2")
    override = cfg["m2_overrides"].get(iso2)
    if country["has_crisis_line"] != "true":
        if override is not None:
            raise ValueError(f"m2_overrides.{iso2}: the country has no crisis line")
        return M2Resources(iso2, None, emergency, "no_crisis_line")
    candidates = _crisis_candidates(rows)
    if override is not None:
        line = next((r for r in candidates if r["number"] == override["number"]), None)
        if line is None:
            raise ValueError(
                f"m2_overrides.{iso2}: {override['number']!r} is not an accepted active crisis or "
                "116 123 number of the country"
            )
        return M2Resources(
            iso2, line, emergency, "override", override.get("name"), override.get("display")
        )
    line = crisis_line(rows)
    if line is None:
        raise ValueError(f"{iso2}: has_crisis_line is true but no crisis line for M2")
    hours = ROUND_THE_CLOCK if ROUND_THE_CLOCK in line["hours"] else "not 24/7"
    rule = f"rank: {line['type']}, {line['scope']}, {hours}"
    if (ties := sum(_rank(r) == _rank(line) for r in candidates)) > 1:
        rule += f", first of {ties} in file order"
    return M2Resources(iso2, line, emergency, rule)


def m2_table(
    countries: Sequence[Row], resources: Sequence[Row], cfg: Mapping[str, Any]
) -> dict[str, M2Resources]:
    rows_by_iso2: dict[str, list[Row]] = {}
    for row in resources:
        rows_by_iso2.setdefault(row["iso2"], []).append(row)
    return {c["iso2"]: m2_resources(c, rows_by_iso2.get(c["iso2"], []), cfg) for c in countries}


M2_COLUMNS = (
    "iso2",
    "rule",
    "crisis_name",
    "crisis_number",
    "prompt_name",
    "prompt_number",
    "crisis_hours",
    "emergency_numbers",
)


def m2_rows(table: Iterable[M2Resources], cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    """Rows of ``m2_resources.csv``, for review of what every M2 system prompt names."""
    separator = cfg["system"]["M2"]["emergency_separator"]
    return [
        {
            "iso2": m2.iso2,
            "rule": m2.rule,
            "crisis_name": m2.crisis["name"] if m2.crisis else "",
            "crisis_number": m2.crisis["number"] if m2.crisis else "",
            "prompt_name": m2.prompt_name,
            "prompt_number": m2.prompt_number,
            "crisis_hours": m2.crisis["hours"] if m2.crisis else "",
            "emergency_numbers": separator.join(m2.emergency),
        }
        for m2 in table
    ]


def m2_system(country: Row, m2: M2Resources, cfg: Mapping[str, Any]) -> str:
    settings = cfg["system"]["M2"]
    fields = {
        "country_en": english_name(country, cfg),
        "emergency_label": settings["emergency_label"]["one" if len(m2.emergency) == 1 else "many"],
        "emergency_number": settings["emergency_separator"].join(m2.emergency),
    }
    if m2.crisis is None:
        return settings["emergency_only"].format(**fields)
    line = m2.crisis
    hours = settings["hours"].format(hours=line["hours"]) if line["hours"] else ""
    return settings["crisis_line"].format(
        **fields, crisis_name=m2.prompt_name, crisis_number=m2.prompt_number, hours=hours
    )


class BuildError(ValueError):
    """Translations are missing, stale or flagged; ``problems`` lists every one."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__(f"{len(problems)} translation problems")
        self.problems = list(problems)


def none_sources(countries: Sequence[Row], cfg: Mapping[str, Any]) -> dict[str, str]:
    """Language → country whose cue-free translations serve as ``XX_{lang}_none`` prompts."""
    by_lang: dict[str, list[str]] = {}
    for country in countries:
        if country["lang"] != ENGLISH:
            by_lang.setdefault(country["lang"], []).append(country["iso2"])
    sources = {}
    for lang, iso2s in by_lang.items():
        if len(iso2s) == 1:
            sources[lang] = iso2s[0]
        elif cfg["none_source"].get(lang) in iso2s:
            sources[lang] = cfg["none_source"][lang]
        else:
            raise ValueError(f"prompts.yaml: none_source has no country of {lang} for {iso2s}")
    return sources


def build_prompts(
    countries: Sequence[Row],
    m2: Mapping[str, M2Resources],
    cfg: Mapping[str, Any],
    translations: Mapping[Key, Mapping[str, Any]],
    no_country_iso2: str,
) -> list[dict[str, Any]]:
    """The full prompt matrix in a fixed order; ``m2`` is :func:`m2_table`.

    Raises :class:`BuildError` if a non-English prompt has no translation, its translation was
    made from a different English source, or it is flagged ``added_content``.
    """
    templates = list(cfg["templates"])
    problems: list[str] = []

    def local(country: Row, cue: str, template: str) -> tuple[str, str | None]:
        """User prompt in the country's language and its qa level (None for English)."""
        source = english_prompt(cfg, country, cue, template)
        if country["lang"] == ENGLISH:
            return source, None
        key = (country["iso2"], cue, template)
        entry = translations.get(key)
        if entry is None:
            problems.append(f"{country['lang']}: no translation for {key}")
            return source, None
        if entry["source_en"] != source:
            problems.append(f"{country['lang']}: stale translation for {key}")
        if not entry["translation"] or (entry["qa"] == "auto" and failed(entry)):
            problems.append(f"{country['lang']}: declined or empty translation for {key}")
        if ADDED_CONTENT in (entry.get("flags") or ()):
            problems.append(
                f"{country['lang']}: translation for {key} has added content; remove it by "
                "hand, set qa: manual and run crisisline translate --recheck"
            )
        return entry["translation"], entry["qa"]

    def record(
        iso2: str,
        lang: str,
        cue: str,
        template: str,
        condition: str,
        system: str | None,
        user: str,
        qa: str | None,
    ) -> dict[str, Any]:
        tag = cue if condition == "S0" else condition
        return {
            "prompt_id": f"{iso2}_{lang}_{tag}_{template}",
            "iso2": iso2,
            "lang": lang,
            "cue": cue,
            "template": template,
            "condition": condition,
            "system": system,
            "user": user,
            "qa": qa,
        }

    prompts = []
    for country in countries:
        iso2, lang = country["iso2"], country["lang"]
        for prompt_lang in dict.fromkeys((lang, ENGLISH)):
            for cue in CUES:
                for template in templates:
                    if prompt_lang == ENGLISH:
                        user, qa = english_prompt(cfg, country, cue, template), None
                    else:
                        user, qa = local(country, cue, template)
                    prompts.append(record(iso2, prompt_lang, cue, template, "S0", None, user, qa))

    by_iso2 = {c["iso2"]: c for c in countries}
    sources: list[tuple[str, str | None]] = [(ENGLISH, None), *none_sources(countries, cfg).items()]
    for lang, source_iso2 in sources:
        for template in templates:
            if source_iso2 is None:
                user, qa = cfg["templates"][template]["text"], None
            else:
                user, qa = local(by_iso2[source_iso2], NO_CUE, template)
            prompts.append(record(no_country_iso2, lang, NO_CUE, template, "S0", None, user, qa))

    for condition in ("M1", "M2"):
        for country in countries:
            if condition == "M1":
                system = cfg["system"]["M1"].format(country_en=english_name(country, cfg))
            else:
                system = m2_system(country, m2[country["iso2"]], cfg)
            for template in templates:
                user, qa = local(country, NO_CUE, template)
                prompts.append(
                    record(
                        country["iso2"],
                        country["lang"],
                        NO_CUE,
                        template,
                        condition,
                        system,
                        user,
                        qa,
                    )
                )

    if problems:
        raise BuildError(list(dict.fromkeys(problems)))
    return prompts
