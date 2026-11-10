"""The prompt subset behind every value of the paper's outputs, in words, with the number of
countries it covers counted from ``data/countries.csv``. A value about the crisis line covers only
the countries that have one."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

CUED = "S0 with a city or country cue"
COUNTRY_CUE = "S0 with a country cue"
ABOUT_A_COUNTRY = "prompts about a country (S0 with a city or country cue, M1, M2)"
NO_COUNTRY = "prompts that name no country (S0 without a cue)"
ALL_LANGUAGES = "all prompt languages"
MAIN_RUN = "main run"
NO_LINE = "countries of the prompts without a national crisis line"
CONDITIONS = {
    "S0-country": COUNTRY_CUE,
    "M1": "M1 (location in the system prompt)",
    "M2": "M2 (location and numbers in the system prompt)",
}
CRISIS_OUTCOMES = ("correct_crisis", "correct_crisis_line")
_WITHOUT_COUNTRY = re.compile(r"^without ([A-Z]{2})$")


def countries(n: int) -> str:
    return f"{n} country" if n == 1 else f"{n} countries"


def about_crisis_line(outcome: str) -> bool:
    return outcome in CRISIS_OUTCOMES


@dataclass(frozen=True, slots=True)
class Country:
    iso2: str
    name: str
    region: str
    english: bool
    crisis_line: bool


@dataclass(frozen=True, slots=True)
class Scope:
    """The countries of the prompts, the countries of the temperature block, the number of prompt
    languages and the models left out of the grounding comparison."""

    countries: tuple[Country, ...]
    temperature_block: frozenset[str]
    languages: int
    grounding_excluded: tuple[str, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, str]],
        temperature_block: Collection[str],
        grounding_excluded: Sequence[str],
    ) -> Scope:
        countries = tuple(
            Country(
                r["iso2"],
                r["name_en"],
                r["who_region"],
                r["lang"] == "en",
                r["has_crisis_line"] == "true",
            )
            for r in rows
        )
        languages = len({r["lang"] for r in rows} | {"en"})
        return cls(countries, frozenset(temperature_block), languages, tuple(grounding_excluded))

    def count(
        self,
        outcome: str = "",
        *,
        non_english: bool = False,
        region: str | None = None,
        within: Collection[str] | None = None,
        without: Collection[str] = (),
    ) -> str:
        """``N countries``, ``N countries with a crisis line`` for an outcome about it."""
        crisis = about_crisis_line(outcome)
        n = sum(
            (c.crisis_line or not crisis)
            and not (non_english and c.english)
            and (region is None or c.region == region)
            and (within is None or c.iso2 in within)
            and c.iso2 not in without
            for c in self.countries
        )
        return countries(n) + (" with a crisis line" if crisis else "")

    def name(self, iso2: str) -> str:
        return next(c.name for c in self.countries if c.iso2 == iso2)


# Figures and the tables built from the same data.


def referral(scope: Scope) -> str:
    return f"{ABOUT_A_COUNTRY}, {scope.count('correct_crisis')}, {ALL_LANGUAGES}"


def condition(scope: Scope, measure: str, outcome: str) -> str:
    """A condition of the grounding comparison, or a difference ``A - B`` of two."""
    arms = " minus ".join(CONDITIONS[m] for m in measure.split(" - "))
    return f"{arms}, local-language prompts, {scope.count(outcome)}"


def language_by_model(scope: Scope, prompt_language: str) -> str:
    language = "English" if prompt_language == "en" else "local-language"
    return (
        f"{COUNTRY_CUE}, {scope.count(non_english=True)} whose language is not English, "
        f"{language} prompts"
    )


def language_by_model_pair(subset: str) -> str:
    """The English-minus-local difference over the subset of one of its two arms."""
    return subset.removesuffix(", English prompts").removesuffix(", local-language prompts") + (
        ", English minus local-language prompts"
    )


def language(lang: str, n: int) -> str:
    return f"{CUED}, prompts in {lang}, {countries(n)} where it is the local language"


def language_gap(lang: str, n: int) -> str:
    return (
        f"{COUNTRY_CUE}, {countries(n)} where {lang} is the local language, English and "
        "local-language prompts"
    )


def translations(lang: str) -> str:
    return f"all prompts translated into {lang}"


def region(scope: Scope, who_region: str, outcome: str) -> str:
    return f"{CUED}, {ALL_LANGUAGES}, {scope.count(outcome, region=who_region)} in {who_region}"


def all_regions(scope: Scope, outcome: str) -> str:
    return f"{CUED}, {ALL_LANGUAGES}, {scope.count(outcome)}"


def local_languages(scope: Scope, outcome: str = "") -> str:
    return f"{CUED}, local-language prompts, {scope.count(outcome)}"


def local_language_country(name: str) -> str:
    return f"{CUED}, local-language prompts, {name}"


def country(name: str) -> str:
    return f"{CUED}, {ALL_LANGUAGES}, {name}"


def cue(scope: Scope, cue_name: str) -> str:
    return f"S0 with a {cue_name} cue, {ALL_LANGUAGES}, {scope.count()}"


def template(scope: Scope, name: str, outcome: str) -> str:
    templates = "all templates" if name == "all" else f"template {name}"
    return f"{CUED}, {ALL_LANGUAGES}, {scope.count(outcome)}, {templates}"


def contrast(scope: Scope, name: str, template_name: str, outcome: str) -> str:
    """A template contrast of the analysis, by its name there; pooled over templates, the subset
    of the same contrast elsewhere."""
    templates = "" if template_name == "all" else f", template {template_name}"
    if name.startswith("en - local"):
        arms = (
            f"{COUNTRY_CUE}, {scope.count(outcome, non_english=True)} whose language is not "
            "English, English minus local-language prompts"
        )
    elif name == "country - city, S0":
        arms = f"S0, country minus city cue, {ALL_LANGUAGES}, {scope.count(outcome)}"
    elif name.endswith(", local"):
        measure = name.removesuffix(", local")
        arms = condition(scope, measure, outcome)
    elif name.endswith(", S0"):
        groups = name.removesuffix(", S0").replace(" - ", " minus ")
        arms = f"{CUED}, {ALL_LANGUAGES}, {scope.count(outcome)}, {groups}"
    else:
        raise ValueError(f"no subset for the contrast {name!r}")
    return f"{arms}{templates}"


def georgia(condition_cue: str) -> str:
    """``S0/city/en`` and the like: condition, cue and prompt language."""
    name, cue_name, mode = condition_cue.split("/")
    prompts = "English" if mode == "en" else "local-language"
    setting = f"S0 with a {cue_name} cue" if name == "S0" else CONDITIONS[name]
    return f"Georgia, {setting}, {prompts} prompts"


# The no-country prompts.

NORTH_AMERICAN = (
    "English prompts that name no country, responses with a number, main run and temperature-0.7 "
    "block"
)
NORTH_AMERICAN_BLOCKS = {
    "main": "English prompts that name no country, responses with a number, main run",
    "robustness": (
        "English prompts that name no country, responses with a number, temperature-0.7 block "
        "(local models)"
    ),
}


def asks_location(scope: Scope) -> str:
    return f"{NO_COUNTRY}, all {scope.languages} prompt languages, {MAIN_RUN}"


def default_country(lang: str) -> str:
    return f"{NO_COUNTRY} in {lang}, {MAIN_RUN}"


# The tests and their sensitivity analyses.


def comparison(scope: Scope, comparison_id: str, analysis: str) -> str:
    """The data of a model-based comparison under one of its analyses."""
    if comparison_id == "us_default":
        base = NORTH_AMERICAN
        return f"{base}, without truncated responses" if analysis == "without truncated" else base
    crisis = "correct_crisis" if analysis == "only countries with a crisis line" else ""
    without = _WITHOUT_COUNTRY.match(analysis)
    left_out = [without[1]] if without else []
    if comparison_id == "cue_country_vs_city":
        subset = f"{CUED}, {ALL_LANGUAGES}, {scope.count(crisis, without=left_out)}"
    elif comparison_id == "language_en_vs_local":
        countries = scope.count(crisis, non_english=True, without=left_out)
        subset = f"{COUNTRY_CUE}, {countries} whose language is not English, English and local"
    elif comparison_id == "model_size":
        subset = f"Qwen models, {CUED}, {ALL_LANGUAGES}, {scope.count(crisis)}"
    elif comparison_id == "grounding_verified_number":
        subset = f"{COUNTRY_CUE} and M2, local-language prompts, {scope.count(crisis)}"
        if not analysis.endswith("all models") and scope.grounding_excluded:
            subset += f", without {', '.join(scope.grounding_excluded)}"
    else:
        raise ValueError(f"no subset for the comparison {comparison_id!r}")
    if analysis == "without truncated":
        subset += ", without truncated responses"
    elif analysis.endswith(" without blocked"):
        subset += f", without blocked responses of {analysis.removesuffix(' without blocked')}"
    return subset


# Temperature, fabrication, models.


def temperature(scope: Scope, block: str, outcome: str) -> str:
    if block == "none_en":
        return "English prompts that name no country, local models"
    countries = scope.count(outcome, within=scope.temperature_block)
    return f"{CUED}, {ALL_LANGUAGES}, {countries} of the temperature block"


FABRICATION_READER = "all responses of the main run, numbers as scored before labels"
UNKNOWN_MENTIONS = (
    "mentions of numbers outside ground truth v1 before labelling (unknown numbers), main run"
)


def near_miss(min_digits: int) -> str:
    return (
        f"numbers of {min_digits} or more digits outside the ground truth, or labelled fabricated, "
        "another service or unverified (ground truth v2), main run"
    )


MODELS = "main run, all prompts"
