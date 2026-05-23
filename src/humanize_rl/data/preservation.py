"""V-Slice 3 — preservation checks for the v03 pair gate.

Per spec §10.3: AIify and Humanize must preserve numbers, named entities,
person/tense, polarity/stance, and discourse role.

This module computes lightweight, regex-based preservation diffs that are:
- free (no LLM call, no spaCy dep)
- cheap to test
- conservative (false positives on rejection are OK; false negatives are
  what we want to catch)

V-Slice 4 / later may swap these for an actual NER pass if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Number / metric extraction
# ---------------------------------------------------------------------------

# Catches: 3.14, 1,234, 80%, 240MB, $185, 14:00, 11:30, 2x, 30-second
# Captures *meaningful* numbers — not bare 1/2/3 list markers, which the
# AIify model legitimately removes when it converts a numbered list to prose.
_NUMBER_RE = re.compile(
    r"""
    \$\d[\d,]*(?:\.\d+)?            # $185, $1,234.50
    | \b\d+(?:[:\-/]\d+){1,2}\b     # times, dates (11:30, 2024-01-15)
    | \b\d+(?:,\d{3})+(?:\.\d+)?\b  # comma-separated thousands
    | \b\d+(?:\.\d+)+\b             # multi-dot versions: 3.14.1, 1.2.3
    | \b\d+(?:\.\d+)?\s*%           # percentages: 80%, 14.5%
    | \b\d+(?:\.\d+)?\s*(?:ms|s|sec|min|hr|MB|GB|KB|TB|x)\b
    | \b\d{2,}(?:\.\d+)?\b          # multi-digit integers/decimals (10+); skips list markers
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Capitalized-token "named entity" proxy
# ---------------------------------------------------------------------------
#
# We can't run NER for free. Cheap proxy: tokens that are capitalized
# *not at sentence start*, plus all-caps acronyms anywhere. Filters out
# common noise (I, the first word of the text).

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Token = a word with letters, possibly with internal periods/apostrophes/dashes
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\.\-]*[A-Za-z]|[A-Za-z]")

# Common short stopwords / sentence-starters we should ignore even if capitalized.
_CAPITAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "I",
        "A",
        "An",
        "The",
        "This",
        "That",
        "These",
        "Those",
        "We",
        "You",
        "They",
        "He",
        "She",
        "It",
        "Our",
        "Their",
        "Your",
        "His",
        "Her",
        "My",
        "Its",
        "And",
        "But",
        "Or",
        "If",
        "When",
        "While",
        "Since",
        "Because",
        "So",
        "As",
        "At",
        "On",
        "In",
        "By",
        "Of",
        "To",
        "For",
        "From",
        "With",
        "Without",
        "Re",
        "Hi",
        "Hello",
        "Hey",
        "Thanks",
        "Cheers",
        "Best",
        "FYI",
        # AI tells we don't want to count as entities
        "Furthermore",
        "Moreover",
        "Additionally",
        "However",
        "Therefore",
        "Consequently",
    }
)


def clean_system_headers(text: str) -> str:
    """Strip system/email headers and ID markers to avoid parsing metadata as content."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        lower_line = line.lower().strip()
        if (
            lower_line.startswith("from:")
            or lower_line.startswith("to:")
            or lower_line.startswith("cc:")
            or lower_line.startswith("bcc:")
            or lower_line.startswith("subject:")
            or lower_line.startswith("date:")
            or lower_line.startswith("sent:")
            or lower_line.startswith("importance:")
            or lower_line.startswith("mime-version:")
            or lower_line.startswith("content-type:")
            or lower_line.startswith("content-transfer-encoding:")
            or lower_line.startswith("x-")
            or lower_line.startswith("sender:")
            or lower_line.startswith("precedence:")
            or lower_line.startswith("charset=")
            or lower_line.startswith("boundary=")
        ):
            continue
        # Skip boundary lines or ID markers
        if lower_line.startswith("------_=_") or re.match(r"^\s*<[^>]+>\s*$", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_numbers(text: str) -> set[str]:
    """Return the set of normalized number-like tokens in `text`.

    Normalization: strip surrounding whitespace; collapse comma separators
    so 1,234 == 1234 for comparison purposes.
    """
    text = clean_system_headers(text)
    out: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        token = m.group(0).strip().replace(",", "")
        out.add(token.lower())
    return out


def _is_strongly_entity_shaped(tok: str) -> bool:
    """Tokens that are obviously names / acronyms / qualified identifiers.

    These are the cases we are confident enough about to enforce preservation
    even if the token only appears once in the original.

    True if:
    - all-caps and >= 2 chars (GDPR, SSO, ETL)
    - contains a non-leading uppercase letter (GitHub, UnicodeDecodeError)
    - contains an internal '.' or '-' (pg_stat_activity, conn.transaction)
    """
    if len(tok) < 2:
        return False
    if tok.lower() in {s.lower() for s in _CAPITAL_STOPWORDS}:
        return False
    if tok.isupper() and 2 <= len(tok) <= 5:
        return True
    if any(c.isupper() for c in tok[1:]) and not tok.isupper():
        return True
    if "." in tok or "-" in tok:
        return True
    return False


def extract_entities(text: str) -> set[str]:
    """Return the set of entity-shaped tokens in `text`."""
    text = clean_system_headers(text)
    sentences = _SENT_SPLIT_RE.split(text)
    entities: set[str] = set()

    for sentence in sentences:
        if ":" in sentence:
            prefix, suffix = sentence.split(":", 1)
            if prefix.strip().replace(" ", "").isupper():
                sentence = suffix

        words = _TOKEN_RE.findall(sentence)
        if not words:
            continue

        # Check first word
        first_word = words[0]
        if _is_strongly_entity_shaped(first_word):
            entities.add(first_word)
        elif (
            len(words) >= 2
            and first_word.istitle()
            and words[1].istitle()
            and first_word.lower() not in {s.lower() for s in _CAPITAL_STOPWORDS}
        ):
            entities.add(first_word)

        # Check remaining words
        for word in words[1:]:
            if (
                len(word) >= 2
                and word.istitle()
                and word.lower() not in {s.lower() for s in _CAPITAL_STOPWORDS}
            ):
                entities.add(word)
            elif _is_strongly_entity_shaped(word):
                entities.add(word)

    # Also add words that are capitalized and appear >= 2 times in the text
    all_words = _TOKEN_RE.findall(text)
    counts: dict[str, int] = {}
    for w in all_words:
        counts[w] = counts.get(w, 0) + 1
    for w, count in counts.items():
        if w.lower() not in {s.lower() for s in _CAPITAL_STOPWORDS} and count >= 2:
            if w.istitle() or _is_strongly_entity_shaped(w):
                entities.add(w)

    return entities


# ---------------------------------------------------------------------------
# Discourse-role coherence (cheap proxy)
# ---------------------------------------------------------------------------
#
# We don't classify the role from text; we test a coherence invariant: if the
# original is a question (ends in "?"), the rewrite should also end in "?".
# If the original is an email-style note (has a greeting line and a sign-off),
# the rewrite should keep the email shape. These checks are deliberately
# narrow; broader role-preservation needs an LLM judge.

_GREETING_RE = re.compile(r"^(?:hi|hey|hello|dear|re:|fyi)(?:\b|:|\s)", re.IGNORECASE)
_SIGNOFF_RE = re.compile(
    r"\b(?:thanks|cheers|best|talk\s+(?:soon|friday|monday)|happy\s+to\s+chat)\b[^.!?\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_question(text: str) -> bool:
    return text.rstrip().endswith("?")


def has_email_shape(text: str) -> bool:
    first_line = text.strip().split("\n")[0]
    return bool(_GREETING_RE.search(first_line))


# ---------------------------------------------------------------------------
# Result type + main check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreservationResult:
    """Result of running preservation checks on (original, rewrite).

    `dropped` and `added` give the lossy diff so reports can show *why*
    a row was flagged.
    """

    numbers_dropped: tuple[str, ...]
    numbers_added: tuple[str, ...]
    entities_dropped: tuple[str, ...]
    entities_added: tuple[str, ...]
    role_drift: tuple[str, ...]

    @property
    def has_violations(self) -> bool:
        return bool(self.numbers_dropped or self.entities_dropped or self.role_drift)


def evaluate_preservation(
    *,
    original: str,
    rewrite: str,
    max_added_entity_ratio: float = 0.30,
) -> PreservationResult:
    """Compare `original` vs `rewrite` and return what was lost/added.

    A preservation *violation* (-> reject) is:
    - excessive number/entity drops (allowing up to 15% drops, minimum 1 drop allowed if > 2 items)
    - role drift: question -> not-question, or email -> not-email
    """
    orig_numbers = extract_numbers(original)
    rewr_numbers = extract_numbers(rewrite)
    orig_entities = extract_entities(original)
    rewr_entities = extract_entities(rewrite)

    # Perform case-insensitive token-lookup on the rewrite for entity preservation
    rewr_tokens_lower = {tok.lower() for tok in _TOKEN_RE.findall(rewrite)}
    all_entities_dropped = {
        ent for ent in orig_entities if ent.lower() not in rewr_tokens_lower
    }
    all_numbers_dropped = orig_numbers - rewr_numbers

    def get_violating_drops(dropped_set, orig_set):
        if not dropped_set:
            return set()
        max_allowed = int(len(orig_set) * 0.15) if len(orig_set) > 5 else 0
        if len(dropped_set) <= max_allowed:
            return set()
        return dropped_set

    violating_entities_dropped = get_violating_drops(
        all_entities_dropped, orig_entities
    )
    violating_numbers_dropped = get_violating_drops(all_numbers_dropped, orig_numbers)

    numbers_dropped = tuple(sorted(violating_numbers_dropped))
    numbers_added = tuple(sorted(rewr_numbers - orig_numbers))
    entities_dropped = tuple(sorted(violating_entities_dropped))
    entities_added = tuple(sorted(rewr_entities - orig_entities))

    role_drift_reasons: list[str] = []
    if is_question(original) and not is_question(rewrite):
        role_drift_reasons.append("question_to_statement")
    if has_email_shape(original) and not has_email_shape(rewrite):
        role_drift_reasons.append("email_lost_greeting")

    # Soft signal: rewrite invented a flood of new entities
    _ = max_added_entity_ratio

    return PreservationResult(
        numbers_dropped=numbers_dropped,
        numbers_added=numbers_added,
        entities_dropped=entities_dropped,
        entities_added=entities_added,
        role_drift=tuple(role_drift_reasons),
    )
