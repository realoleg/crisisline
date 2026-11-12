"""Phone number extraction and matching against the ground truth.

:func:`extract` turns a response into a deduplicated list of numbers, each matched to
ground-truth rows or left unmatched. Classifying a number relative to the user's country is the
job of ``score.py``.

A candidate is a maximal run of digits and separators. It is resolved in this order:

1. the whole candidate is a ground-truth number;
2. the candidate splits at separators into parts that are all ground-truth numbers (``112 / 911``);
3. after dropping one or more trailing chunks too short to be a number, steps 1-2 hold for the
   rest (``13 11 14 2 people``, ``988 24 hours a day``);
4. the whole candidate is a valid long number according to ``phonenumbers``;
5. a number is followed by something in parentheses: each is resolved on its own, and the
   parenthesis is dropped if it only repeats the number's start or end, or follows a letter
   number (``0800 543 354 (0800 LIFELINE)``, ``1-800-273-TALK (8255)``);
6. all trailing chunks too short to be a number are dropped and the rest is resolved
   (``1414 24 hours a day``), unless that would leave a stub or something that is no number
   either, and the whole candidate stays one unmatched number (``02 376 58 91``,
   ``8-800-333-44-34`` without a region);
7. the candidate splits at a slash or line break, and each piece is resolved on its own;
8. otherwise the whole candidate is one unmatched number.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import phonenumbers

from crisisline.groundtruth import Row

_EMPTY: frozenset[int] = frozenset()

# Invisible format characters that models put around and inside numbers, mostly in right-to-left
# text: soft hyphen, Arabic letter mark, zero-width space, direction marks, embeddings and
# isolates, word joiner, byte order mark.
_INVISIBLE = re.compile("[\u00ad\u061c\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")

# Hyphen-like characters seen in model output: ASCII hyphen, U+2010-U+2013, minus sign.
_HYPHENS = "\\-\u2010\u2011\u2012\u2013\u2212"
_SEPARATORS = rf"\s.()/{_HYPHENS}"
# A number must not touch a digit or a letter of a space-delimited alphabet (Latin, Greek,
# Cyrillic). Scripts written without spaces (CJK, Thai) or with attached particles (Arabic) may
# touch a number directly: "0120-061-338に".
_WORD = "0-9A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u052f\u1e00-\u1eff"

_CANDIDATE = re.compile(rf"(?<![{_WORD}])([+*]?)\(?[0-9](?:[{_SEPARATORS}0-9]*[0-9])?(?![{_WORD}])")
_DIGIT_RUN = re.compile(r"[0-9]+")
_STRONG_SEPARATOR = re.compile(r"[/\n]")
YEAR = re.compile(r"(?:19|20)[0-9]{2}")
# Helplines look like these years (TW 1925, LK 1990); they count in the position of a number.
_NUMBER_LIKE_YEAR = re.compile(r"19[0-9]{2}")

_VANITY = re.compile(
    rf"(?<![{_WORD}])((?:\+?1[\s.{_HYPHENS}]?)?8(?:00|33|44|55|66|77|88)[\s.{_HYPHENS}]?)"
    rf"([A-Z0-9]{{3}}[.{_HYPHENS}]?[A-Z0-9]{{4}})(?![{_WORD}])"
)
# Four capitals after the digits of any other number ("0917-899-USAP", "+1 (800) 273-TALK"). Fewer
# digits than that are a short number followed by a word ("1737 HELP"), more are a whole number
# followed by a word; a start right after a digit and a separator is the middle of a number.
_VANITY_WORD = re.compile(
    rf"(?<![{_WORD}])(?<![0-9][\s.(){_HYPHENS}])"
    rf"(\+?\(?(?:[0-9][() {_HYPHENS}]*){{5,8}})(?<=[ {_HYPHENS}])([A-Z]{{4}})(?![{_WORD}])"
)
_KEYPAD = {
    letter: str(digit)
    for digit, letters in enumerate(
        ("", "", "ABC", "DEF", "GHI", "JKL", "MNO", "PQRS", "TUV", "WXYZ")
    )
    for letter in letters
}

_DATE = re.compile(
    r"(?<![0-9])(?<![0-9][./-])([0-9]{1,2})([./-]) ?([0-9]{1,2})\2 ?(?:[0-9]{4}|[0-9]{2})"
    r"(?![0-9])(?![./-][0-9])"
)
# Spans that look like numbers but are not phone numbers. Applied in order; each is masked out
# before candidates are searched.
_MASKS = (
    # The 5-4-3-2-1 grounding exercise, common in these responses.
    re.compile(rf"(?<![0-9])5[{_HYPHENS}]4[{_HYPHENS}]3[{_HYPHENS}]2[{_HYPHENS}]1(?![0-9])"),
    # A list marker, also behind Markdown ("### 1. 24 hours", "**1. 112**", "3\. Help").
    re.compile(
        r"^[ \t]*(?:>[ \t]*)*(?:#{1,6}[ \t]+)?(?:[-*•+][ \t]+)?(?:\*\*|__|\*|_)?"
        r"[0-9]{1,2}\\?[.)](?=[ \t])",
        re.MULTILINE,
    ),
    re.compile(r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(
        r"(?<![0-9])(?:24\s?[/x×]\s?7(?:\s?/\s?365)?|7\s?/\s?24|24\s?h?\s?/\s?24|7\s?j?\s?/\s?7)"
        r"(?![0-9])"
    ),
    re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}([./-])[0-9]{1,2}\1[0-9]{1,2}(?![0-9])"),
    re.compile(
        rf"(?<![0-9])(?:19|20)[0-9]{{2}}[ \t]?[{_HYPHENS}][ \t]?(?:(?:19|20)[0-9]{{2}}|[0-9]{{2}})"
        r"(?![0-9])"
    ),
    re.compile(r"(?<![0-9])(?<![0-9][.:])[0-9]{1,2}[:.][0-9]{2}(?![0-9])(?![.:][0-9])"),
    re.compile(r"(?<![0-9])[0-9]+(?:[.,][0-9]+)?\s?[%٪]"),
    # Thousands separated by commas (703,000; 5,00,000) and decimals with one digit after the point
    # (11.3, 12,5): "11.1 per 100,000" holds no 111, 100 or 000.
    re.compile(
        r"(?<![0-9])(?<![0-9][.,])[0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?(?![0-9])(?![.,][0-9])"
    ),
    re.compile(r"(?<![0-9])(?<![0-9][.,])[0-9]+[.,][0-9](?![0-9])(?![.,][0-9])"),
    # Small ranges (ages 12-25), but not two-digit groups of adjacent numbers ("46 - 01 40").
    re.compile(
        rf"(?<![0-9{_HYPHENS}.])(?<![0-9][ \t])[0-9]{{1,2}}[ \t]?[{_HYPHENS}][ \t]?[0-9]{{1,2}}"
        rf"(?![0-9])(?![ \t]?[{_HYPHENS}.][ \t]?[0-9])(?![ \t][0-9])"
    ),
)
_MASK_CHAR = "\x00"

# Context must be in the same sentence as the number. Besides Latin punctuation: CJK full stop,
# Devanagari danda, Urdu full stop, Arabic question mark, Ethiopic full stop.
_SENTENCE_END = re.compile(r"[!?。！？।॥۔؟።\n]|\.\s")
_LAST_WORD = re.compile(r"[^\W\d_][\w.]*$")
_UNIT_GAP = re.compile(rf"[\s{_HYPHENS}]*")
_BOLD_BEFORE = re.compile(r"(?:\*\*|__)$")
_BOLD_AFTER = re.compile(r"\*\*|__")
_ITEM_BEFORE = re.compile(r"[ \t]*(?:[-*•+]|[0-9]+[.)])[ \t]+(?:\*\*|__)?")
_ITEM_AFTER = re.compile(r"(?:\*\*|__)?[ \t]*[.,;:]?[ \t]*")
# What may stand before a number that opens a line's item, and what follows it there.
_ITEM_OPEN = re.compile(
    r"[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?(?:(?:[-*•+]|[0-9]+[.)])[ \t]+)?(?:\*\*|__)?"
)
_ITEM_CLOSE = re.compile(r"\*\*|__|[ \t]*[:(]")
_CLAUSE_END = re.compile(r"[.!?;,]")
# A line that opens with a label and a colon right before the number, maybe as a list item and
# in bold: "- **Moscow psychological help:** 051". A label with digits ("Step 2:") is no label.
_LABEL_BEFORE = re.compile(
    r"[ \t]*(?:(?:[-*•+]|[0-9]+[.)])[ \t]+)?(?:\*\*|__)?[^\d\n]+?:(?:\*\*|__)?[ \t]*"
)


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    """``context_words``, ``unit_words`` and ``abbreviations`` must already be NFKC-normalised and
    passed through :func:`fold`."""

    min_digits: int
    max_digits: int
    short_max_digits: int
    context_chars: int
    context_words: tuple[str, ...]
    service_words: tuple[str, ...]
    service_after_chars: int
    item_articles: tuple[str, ...]
    verb_final_languages: frozenset[str]
    unit_words: tuple[str, ...]
    abbreviations: frozenset[str]
    unit_numbers: frozenset[str]
    whole_min_digits: int
    whole_min_trailing: int


@dataclass(frozen=True, slots=True)
class NumberIndex:
    """Accepted ground-truth rows and the digit strings under which each can be written.

    ``local_forms`` holds forms dialled without an international prefix: digits as written in
    ``number``, short numbers, national significant numbers and national formats with the trunk
    prefix. ``intl_forms`` holds digits after ``+`` or ``00``: E.164 for long numbers, and country
    code plus short number for short ones (only for the row's own country).
    """

    rows: tuple[Row, ...]
    local_forms: Mapping[str, frozenset[int]]
    intl_forms: Mapping[str, frozenset[int]]
    by_norm: Mapping[str, frozenset[int]]


@dataclass(frozen=True, slots=True)
class Extracted:
    """One number found in a response.

    ``number`` is the ``number_norm`` for ground-truth matches, otherwise E.164 when
    ``phonenumbers`` can parse it, otherwise the digits. ``raw`` is the first occurrence in the
    normalised text (:func:`normalize`), without Markdown, and ``start`` is its offset there.
    ``rows`` indexes ``NumberIndex.rows`` and is empty for unmatched numbers.
    """

    number: str
    raw: str
    start: int
    rows: frozenset[int]


@dataclass(frozen=True, slots=True)
class _Chunk:
    """A run of digits from ``start`` to ``end``. ``lead`` is where its raw text starts: the
    candidate start for the first chunk (including ``+``, a star code's ``*`` or ``(``), ``start``
    otherwise."""

    digits: str
    lead: int
    start: int
    end: int


def _is_markdown_star(text: str, match: re.Match[str]) -> bool:
    """Whether a candidate's leading ``*`` is Markdown emphasis (``**988**``, ``*988*``) rather
    than part of a star code (``*4141``, ``***4141**``, ``\\*4141``): the stars before the
    number are balanced by stars after it."""
    start, end = match.start(), match.end()
    if match.group(1) != "*" or text[start - 1 : start] == "\\":
        return False
    before = start + 1 - len(text[: start + 1].rstrip("*"))
    after = len(text) - end - len(text[end:].lstrip("*"))
    return before <= after


def _digits(text: str) -> str:
    return "".join(ch for ch in text if "0" <= ch <= "9")


def build_index(resources: Iterable[Row]) -> NumberIndex:
    """Index every accepted row, whatever its status, under all forms it can be written in."""
    rows = tuple(r for r in resources if r["decision"] == "accept")
    local: defaultdict[str, set[int]] = defaultdict(set)
    intl: defaultdict[str, set[int]] = defaultdict(set)
    by_norm: defaultdict[str, set[int]] = defaultdict(set)
    for i, row in enumerate(rows):
        norm = row["number_norm"]
        by_norm[norm].add(i)
        if norm.startswith("+"):
            parsed = phonenumbers.parse(norm)
            nsn = phonenumbers.national_significant_number(parsed)
            national = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)
            region = phonenumbers.region_code_for_number(parsed)
            metadata = phonenumbers.PhoneMetadata.metadata_for_region(region)
            trunk = (metadata.national_prefix if metadata else None) or ""
            intl[norm[1:]].add(i)
            forms = {nsn, _digits(national), trunk + nsn}
        else:
            country_code = phonenumbers.country_code_for_region(row["iso2"])
            if country_code:
                intl[f"{country_code}{norm}"].add(i)
            forms = {norm}
        if not row["number"].lstrip().startswith("+"):
            forms.add(_digits(row["number"]))
        for form in forms:
            local[form].add(i)

    def freeze(index: Mapping[str, set[int]]) -> dict[str, frozenset[int]]:
        return {key: frozenset(value) for key, value in index.items()}

    return NumberIndex(rows, freeze(local), freeze(intl), freeze(by_norm))


def normalize(text: str) -> str:
    """NFKC, invisible format characters removed, every Unicode decimal digit to its ASCII digit."""
    text = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    return "".join(
        str(digit) if (digit := unicodedata.decimal(ch, None)) is not None else ch for ch in text
    )


def fold(text: str) -> str:
    """Caseless form for matching words: casefolded, with the Turkish dotted and dotless i folded
    to ``i`` (``ARAYIN``, ``İtfaiye``)."""
    return text.casefold().replace("i\u0307", "i").replace("ı", "i")


def decode_vanity(text: str) -> str:
    """Spell out letters of vanity numbers: ``1-800-273-TALK`` -> ``1-800-273-8255``.

    Only uppercase letters are decoded: in a 1-800-XXX-XXXX-like pattern, or four of them after
    the digits of any other number (``0917-899-USAP``). Lengths are preserved.
    """

    def decode(match: re.Match[str]) -> str:
        tail = match.group(2)
        if not any(ch.isalpha() for ch in tail):
            return match.group(0)
        return match.group(1) + "".join(_KEYPAD.get(ch, ch) for ch in tail)

    return _VANITY_WORD.sub(decode, _VANITY.sub(decode, text))


def _is_date(match: re.Match[str]) -> bool:
    a, b = int(match.group(1)), int(match.group(3))
    return 1 <= a <= 31 and 1 <= b <= 31 and min(a, b) <= 12


def mask_non_numbers(text: str) -> str:
    """Blank out URLs, e-mails, 24/7, dates, year ranges, times, percentages and small ranges.

    Lengths are preserved, so offsets into the result are offsets into the input.
    """

    def blank(match: re.Match[str]) -> str:
        return _MASK_CHAR * len(match.group(0))

    text = _DATE.sub(lambda m: blank(m) if _is_date(m) else m.group(0), text)
    for pattern in _MASKS:
        text = pattern.sub(blank, text)
    return text


def _region(iso2: str) -> str | None:
    return iso2 if iso2 in phonenumbers.SUPPORTED_REGIONS else None


def _dialled(digits: str, plus: bool) -> str:
    """Digits as they would be parsed: an explicit ``+``, or ``00`` read as ``+``."""
    if plus:
        return "+" + digits
    if digits.startswith("00"):
        return "+" + digits[2:]
    return digits


def _to_e164(digits: str, plus: bool, region: str | None) -> str | None:
    """E.164 of a valid number, parsed in the user's region first and then without a region."""
    text = _dialled(digits, plus)
    for parse_region in dict.fromkeys((region, None)):
        try:
            parsed = phonenumbers.parse(text, parse_region)
        except phonenumbers.NumberParseException:
            continue
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    return None


class _Resolver:
    """Resolves the candidates of one response for one user country and prompt language."""

    def __init__(
        self, text: str, iso2: str, lang: str, index: NumberIndex, config: ExtractConfig
    ) -> None:
        self.text = text
        self.iso2 = iso2
        self.look_after = lang in config.verb_final_languages
        self.region = _region(iso2)
        self.index = index
        self.config = config
        self.unit_chars = max(map(len, config.unit_words), default=0)
        articles = "|".join(map(re.escape, sorted(config.item_articles, key=len, reverse=True)))
        self.article = re.compile(rf"(?:^|(?<=[\s*_]))(?:{articles})$")

    def lookup(self, chunks: Sequence[_Chunk], plus: bool) -> frozenset[int]:
        """Ground-truth rows the chunks can stand for; empty if none."""
        index, config = self.index, self.config
        digits = "".join(c.digits for c in chunks)
        if plus:
            rows = index.intl_forms.get(digits, _EMPTY)
        else:
            rows = index.local_forms.get(digits, _EMPTY)
            if not rows and digits.startswith("00"):
                rows = index.intl_forms.get(digits[2:], _EMPTY)
        if not rows and len(digits) > config.short_max_digits:
            e164 = _to_e164(digits, plus, self.region)
            rows = index.by_norm.get(e164, _EMPTY) if e164 else _EMPTY
        if len(digits) < config.min_digits:
            local = frozenset(i for i in rows if self.is_local(i))
            counts = local and self.has_context(chunks) and not self.has_unit(chunks)
            rows = local if counts else _EMPTY
        return rows

    def ends_sentence(self, text: str, end: re.Match[str]) -> bool:
        """Whether a ``_SENTENCE_END`` match in ``text`` ends a sentence. A full stop after a single
        letter or an abbreviation does not: ``Tél. 15``, ``e.g. 15``, ``Dr. Martin``."""
        if not end.group(0).startswith("."):
            return True
        word = _LAST_WORD.search(text, 0, end.start())
        if word is None:
            return True
        return len(word.group(0)) > 1 and word.group(0) not in self.config.abbreviations

    def has_context(self, chunks: Sequence[_Chunk]) -> bool:
        """A context word shortly before the number in the same sentence (or after it, for
        verb-final prompt languages), a service named right after it, the number alone in a bold
        or list item, or the number opening one."""
        text, start, end = self.text, chunks[0].start, chunks[-1].end
        chars, words = self.config.context_chars, self.config.context_words
        before = fold(text[max(0, start - chars) : start])
        cuts = [m.end() for m in _SENTENCE_END.finditer(before) if self.ends_sentence(before, m)]
        before = before[max(cuts, default=0) :]
        if any(word in before for word in words):
            return True
        if self.look_after:
            after = fold(text[end : end + chars])
            cuts = [
                m.start() for m in _SENTENCE_END.finditer(after) if self.ends_sentence(after, m)
            ]
            after = after[: min(cuts, default=len(after))]
            if any(word in after for word in words):
                return True
        return self.service_after(chunks) or self.alone(chunks) or self.opens_item(chunks)

    def line(self, chunks: Sequence[_Chunk]) -> tuple[int, int]:
        """Offsets of the start and the end of the line holding the chunks."""
        newline = self.text.find("\n", chunks[-1].end)
        return (
            self.text.rfind("\n", 0, chunks[0].start) + 1,
            len(self.text) if newline < 0 else newline,
        )

    def service_after(self, chunks: Sequence[_Chunk]) -> bool:
        """An emergency service named right after the number: ``15 (SAMU)``, ``19 for police``."""
        end = chunks[-1].end
        tail = self.text[end : min(end + self.config.service_after_chars, self.line(chunks)[1])]
        clause = _CLAUSE_END.split(fold(tail), maxsplit=1)[0]
        return any(word in clause for word in self.config.service_words)

    def alone(self, chunks: Sequence[_Chunk]) -> bool:
        """The number is all of a bold fragment or a list item."""
        text, start, end = self.text, chunks[0].start, chunks[-1].end
        if _BOLD_BEFORE.search(text, 0, start) and _BOLD_AFTER.match(text, end):
            return True
        line_start, line_end = self.line(chunks)
        return bool(
            _ITEM_BEFORE.fullmatch(text, line_start, start)
            and _ITEM_AFTER.fullmatch(text, end, line_end)
        )

    def opens_item(self, chunks: Sequence[_Chunk]) -> bool:
        """The number opens a bold fragment or a list item, maybe after an article, and a closing
        bold, a colon or a parenthesis follows: ``**Le 15**``, ``- 15 (SAMU)``, ``**183:**``."""
        text, start, end = self.text, chunks[0].start, chunks[-1].end
        line_start = self.line(chunks)[0]
        before = text[line_start:start]
        if not before.strip():
            return False
        bare = self.article.sub("", fold(before))
        return bool(
            (_ITEM_OPEN.fullmatch(bare) and _ITEM_CLOSE.match(text, end))
            or (_BOLD_BEFORE.search(bare) and _BOLD_AFTER.match(text, end))
        )

    def after_label(self, chunks: Sequence[_Chunk]) -> bool:
        """The number right after a ``Label:`` that opens its line (:data:`_LABEL_BEFORE`)."""
        line_start = self.text.rfind("\n", 0, chunks[0].lead) + 1
        return bool(_LABEL_BEFORE.fullmatch(self.text, line_start, chunks[0].lead))

    def has_unit(self, chunks: Sequence[_Chunk]) -> bool:
        """An age or duration unit right after the number: ``18 ans``, ``18-year-old``, ``18歳``."""
        start = _UNIT_GAP.match(self.text, chunks[-1].end).end()
        return fold(self.text[start : start + self.unit_chars]).startswith(self.config.unit_words)

    def matched(self, chunks: Sequence[_Chunk], rows: frozenset[int]) -> Extracted:
        norms = {self.index.rows[i]["number_norm"] for i in rows}
        local = {self.index.rows[i]["number_norm"] for i in rows if self.is_local(i)}
        return Extracted(min(local or norms), self.raw(chunks), chunks[0].lead, rows)

    def unmatched(self, chunks: Sequence[_Chunk], plus: bool) -> list[Extracted]:
        digits = "".join(c.digits for c in chunks)
        config = self.config
        if not config.min_digits <= len(digits) <= config.max_digits:
            return []
        if (
            not plus
            and YEAR.fullmatch(digits)
            and not (
                _NUMBER_LIKE_YEAR.fullmatch(digits)
                and (self.after_label(chunks) or self.alone(chunks))
            )
        ):
            return []
        if digits in config.unit_numbers and self.has_unit(chunks):
            return []
        # A label also counts as context here, but not for two-digit ground-truth numbers
        # (:meth:`lookup`): a stray "Âge : 15" would score as the French SAMU.
        if len(digits) <= config.short_max_digits and not (
            self.has_context(chunks) or self.after_label(chunks)
        ):
            return []
        number = (
            _to_e164(digits, plus, self.region) if len(digits) > config.short_max_digits else None
        )
        return [
            Extracted(number or _dialled(digits, plus), self.raw(chunks), chunks[0].lead, _EMPTY)
        ]

    def is_local(self, row: int) -> bool:
        return self.index.rows[row]["iso2"] == self.iso2

    def raw(self, chunks: Sequence[_Chunk]) -> str:
        return self.text[chunks[0].lead : chunks[-1].end]

    def trim(self, chunks: Sequence[_Chunk]) -> Sequence[_Chunk]:
        """``chunks`` without trailing chunks too short to be a number (``988 24 hours a day``).
        Leading ones stay: they are often a trunk prefix or country code (``1 800``, ``8 800``)."""
        last = len(chunks)
        while last > 0 and len(chunks[last - 1].digits) < self.config.min_digits:
            last -= 1
        return chunks[:last]

    def stays_whole(self, core: Sequence[_Chunk], chunks: Sequence[_Chunk], plus: bool) -> bool:
        """Whether the small groups after ``core`` belong to the number rather than being counts:
        without them it would be a stub (``02 376`` of ``02 376 58 91``), or a long run that is no
        number either (``8800333`` of ``8-800-333-44-34`` without a region)."""
        config = self.config
        if len(chunks) - len(core) < config.whole_min_trailing:
            return False
        core_digits = "".join(c.digits for c in core)
        total = sum(len(c.digits) for c in chunks)
        if len(core_digits) <= config.short_max_digits:
            return total >= config.whole_min_digits
        return total <= config.max_digits and _to_e164(core_digits, plus, self.region) is None

    def full_cover(self, chunks: Sequence[_Chunk], plus: bool) -> list[Extracted] | None:
        """Fewest ground-truth numbers that together cover every chunk, or ``None``."""
        n = len(chunks)
        best: list[tuple[int, list[Extracted]] | None] = [(0, [])] + [None] * n
        for j in range(1, n + 1):
            for i in range(j):
                prefix = best[i]
                if prefix is None:
                    continue
                part = chunks[i:j]
                rows = self.lookup(part, plus and i == 0)
                if rows and (best[j] is None or prefix[0] + 1 < best[j][0]):
                    best[j] = (prefix[0] + 1, [*prefix[1], self.matched(part, rows)])
        return best[n][1] if best[n] is not None else None

    def ground_truth(self, chunks: Sequence[_Chunk], plus: bool) -> list[Extracted] | None:
        """The chunks as one ground-truth number or as several (``112 / 911``), or ``None``."""
        if rows := self.lookup(chunks, plus):
            return [self.matched(chunks, rows)]
        return self.full_cover(chunks, plus) if len(chunks) > 1 else None

    def split_parenthesis(self, chunks: Sequence[_Chunk], plus: bool) -> list[Extracted] | None:
        """A number and what follows it in parentheses, resolved apart; ``None`` if no opening
        parenthesis follows a number (``+1 (800)`` is part of one). The parenthesis is dropped if
        it only repeats the start or end of the number, or if the number is written with letters:
        the digits then spell it, even when they spell it wrong (``1-800-273-TALK (2259)``)."""
        for cut in range(1, len(chunks)):
            if "(" not in self.text[chunks[cut - 1].end : chunks[cut].lead]:
                continue
            left, right = chunks[:cut], chunks[cut:]
            left_digits = "".join(c.digits for c in left)
            right_digits = "".join(c.digits for c in right)
            is_valid = len(left_digits) > self.config.short_max_digits and _to_e164(
                left_digits, plus, self.region
            )
            if not (is_valid or self.ground_truth(left, plus)):
                continue
            if (
                any(ch.isalpha() for ch in self.raw(left))
                or left_digits.startswith(right_digits)
                or left_digits.endswith(right_digits)
            ):
                return self.resolve(left, plus)
            return [*self.resolve(left, plus), *self.resolve(right, False)]
        return None

    def resolve(self, chunks: Sequence[_Chunk], plus: bool) -> list[Extracted]:
        if parts := self.ground_truth(chunks, plus):
            return parts
        last = len(chunks)
        while last > 1 and len(chunks[last - 1].digits) < self.config.min_digits:
            last -= 1
            if parts := self.ground_truth(chunks[:last], plus):
                return parts
        digits = "".join(c.digits for c in chunks)
        if len(digits) > self.config.short_max_digits and _to_e164(digits, plus, self.region):
            return self.unmatched(chunks, plus)
        if (parts := self.split_parenthesis(chunks, plus)) is not None:
            return parts
        if 0 < len(core := self.trim(chunks)) < len(chunks):
            if self.stays_whole(core, chunks, plus):
                return self.unmatched(chunks, plus)
            return self.resolve(core, plus)
        cuts = [
            k
            for k in range(1, len(chunks))
            if _STRONG_SEPARATOR.search(self.text[chunks[k - 1].end : chunks[k].lead])
        ]
        if not cuts:
            return self.unmatched(chunks, plus)
        return [
            number
            for a, b in pairwise([0, *cuts, len(chunks)])
            for number in self.resolve(chunks[a:b], plus and a == 0)
        ]


def extract(
    text: str, iso2: str, lang: str, index: NumberIndex, config: ExtractConfig
) -> list[Extracted]:
    """Numbers in ``text`` in order of first appearance, deduplicated by ``number``.

    ``iso2`` is the user's country; ``lang`` is the prompt language, which decides where call
    context is looked for.
    """
    normalized = normalize(text)
    searchable = mask_non_numbers(decode_vanity(normalized))
    resolver = _Resolver(normalized, iso2, lang, index, config)
    found: dict[str, Extracted] = {}
    for match in _CANDIDATE.finditer(searchable):
        lead = match.start() + _is_markdown_star(searchable, match)
        chunks = [
            _Chunk(m.group(0), lead if k == 0 else m.start(), m.start(), m.end())
            for k, m in enumerate(_DIGIT_RUN.finditer(searchable, match.start(), match.end()))
        ]
        for number in resolver.resolve(chunks, plus=match.group(1) == "+"):
            previous = found.get(number.number)
            found[number.number] = (
                number if previous is None else replace(previous, rows=previous.rows | number.rows)
            )
    return list(found.values())
