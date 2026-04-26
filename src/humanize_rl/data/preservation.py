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


def extract_numbers(text: str) -> set[str]:
    """Return the set of normalized number-like tokens in `text`.

    Normalization: strip surrounding whitespace; collapse comma separators
    so 1,234 == 1234 for comparison purposes.
    """
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
    if tok in _CAPITAL_STOPWORDS:
        return False
    if tok.isupper() and len(tok) >= 2:
        return True
    if any(c.isupper() for c in tok[1:]):
        return True
    if "." in tok or "-" in tok:
        return True
    return False


def extract_entities(text: str) -> set[str]:
    """Return the set of entity-shaped tokens in `text`.

    Two paths:
    1. Strongly-shaped tokens (acronyms, CamelCase, dotted/hyphenated
       identifiers) are always counted, even with a single mention.
    2. Plain capitalized words (Sarah, Datadog, Mike, Postgres) are only
       counted if they appear at least twice in the text. This avoids
       firing on sentence-starters ("Spent", "Turned", "Quick", "Reply",
       "Better") that the AIify model legitimately rephrases.

    Trade-off: a single-mention proper name in the original will not
    trigger preservation enforcement for that token. Acceptable for V-Slice
    3 — the strong-shape path still catches every product/tool/file/error
    name, which is where we care most about preservation.
    """
    tokens = _TOKEN_RE.findall(text)
    entities: set[str] = set()
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    for tok, count in counts.items():
        if _is_strongly_entity_shaped(tok):
            entities.add(tok)
            continue
        if (
            tok not in _CAPITAL_STOPWORDS
            and len(tok) >= 2
            and tok[0].isupper()
            and count >= 2
        ):
            entities.add(tok)
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
    - ANY number from the original missing in the rewrite
    - ANY entity from the original missing in the rewrite
    - role drift: question -> not-question, or email -> not-email

    Numbers added or entities added are usually fine (the rewrite may use
    "the team" instead of "Sarah's team"), but we still surface the diff
    for inspection.
    """
    orig_numbers = extract_numbers(original)
    rewr_numbers = extract_numbers(rewrite)
    orig_entities = extract_entities(original)
    rewr_entities = extract_entities(rewrite)

    numbers_dropped = tuple(sorted(orig_numbers - rewr_numbers))
    numbers_added = tuple(sorted(rewr_numbers - orig_numbers))
    entities_dropped = tuple(sorted(orig_entities - rewr_entities))
    entities_added = tuple(sorted(rewr_entities - orig_entities))

    role_drift_reasons: list[str] = []
    if is_question(original) and not is_question(rewrite):
        role_drift_reasons.append("question_to_statement")
    if has_email_shape(original) and not has_email_shape(rewrite):
        role_drift_reasons.append("email_lost_greeting")

    # Soft signal: rewrite invented a flood of new entities (often means
    # the model hallucinated new actors). We don't fail on this; just flag.
    _ = max_added_entity_ratio  # reserved for future flag-only signal

    return PreservationResult(
        numbers_dropped=numbers_dropped,
        numbers_added=numbers_added,
        entities_dropped=entities_dropped,
        entities_added=entities_added,
        role_drift=tuple(role_drift_reasons),
    )
