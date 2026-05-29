"""Deterministic reward checks for RL task responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from humanize_rl.reward.tasks import RLTask

OPTION_MENU_RE = re.compile(
    r"^(?:option|version|alternative)\s*[A-C1-9]?\s*[:.)-]|"
    r"\b(?:here are|below are)\s+(?:a few|some|two|three|\d+)\b|"
    r"\b(?:option one|option two|version a|version b)\b",
    re.IGNORECASE | re.MULTILINE,
)
WRAPPER_RE = re.compile(
    r"^\s*(?:sure[,.!]?\s+)?(?:here(?:'s| is)|below is|i rewrote|i've rewritten|"
    r"this version|a more natural version|certainly[,.!])\b",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"\[[^\]\n]{1,60}\]|<[^>\n]{1,60}>|\{\{[^}\n]{1,60}\}\}")
SUBJECT_RE = re.compile(r"(?im)^\s*subject\s*:")
SIGNOFF_RE = re.compile(
    r"(?im)^\s*(?:best|regards|kind regards|warm regards|thanks|thank you|sincerely|cheers),?\s*$"
)
AI_TELL_PHRASES: tuple[str, ...] = (
    "certainly",
    "of course",
    "great question",
    "i'd be happy to",
    "it is worth noting",
    "it's worth noting",
    "furthermore",
    "moreover",
    "in conclusion",
    "please don't hesitate",
    "please do not hesitate",
    "i hope this email finds you well",
    "seamless",
    "unlock",
    "empower",
    "mission-critical",
)
COMMON_FAKE_NAMES: tuple[str, ...] = (
    "Alice",
    "Bob",
    "Charlie",
    "Jane",
    "John",
    "Marcus",
    "Sarah",
)
REFUSAL_RE = re.compile(
    r"(?i)^\s*(?:i\s+)?(?:can't|cannot|won't|am unable to|i'm unable to)\b"
)
MARKDOWN_LIST_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
NUMBER_RE = re.compile(r"\b(?:\d+[\d,.:/-]*|[A-Z]+-\d+)\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z][A-Z0-9_]{2,})\b")
WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")

PENALTIES: dict[str, float] = {
    "option_menu": -0.40,
    "wrapper_phrase": -0.20,
    "invented_detail": -0.50,
    "placeholder_disallowed": -0.35,
    "placeholder_required": -0.30,
    "missing_number": -0.40,
    "missing_entity": -0.40,
    "subject_line": -0.20,
    "signoff": -0.15,
    "too_long": -0.30,
    "too_short": -0.15,
    "sentence_count": -0.20,
    "ai_tell_phrase": -0.25,
    "refusal": -0.40,
    "wrong_format_markdown": -0.20,
    "missing_required_fact": -0.35,
    "forbidden_fact": -0.50,
}

# Stop-entities are tokens that match the ENTITY_RE [A-Z][a-z]+ heuristic
# but are not real entities. Adding sentence-start salutations / openers
# here prevents the missing_entity false positive documented in the
# Slice 4 audit (docs/plans/gemma4_rl_modal_stable_training_continuation.md
# Slice 4 follow-up): the SFT model legitimately drops "Please" or "Hi"
# while preserving STRIPE_WEBHOOK_SECRET, and was being penalised for it.
STOP_ENTITIES: set[str] = {
    # Pronouns / articles
    "I",
    "We",
    "You",
    "They",
    "The",
    "This",
    "That",
    "It",
    "A",
    "An",
    "No",
    # Common imperative-instruction starters
    "Use",
    "Return",
    "Write",
    "Clean",
    "Make",
    "Explain",
    "Subject",
    # Email / chat salutations and sign-offs — added per Slice 4 audit.
    "Please",
    "Hi",
    "Hello",
    "Hey",
    "Dear",
    "Greetings",
    "Thanks",
    "Thank",
    "Regards",
    "Best",
    "Sincerely",
    "Cheers",
    "Apologies",
    "Sorry",
    # Temporal openers commonly mis-captured as entities.
    "Today",
    "Yesterday",
    "Tomorrow",
    "Tonight",
    "Morning",
    "Afternoon",
    "Evening",
    # Other common openers.
    "Yes",
    "Sure",
    "Note",
}


@dataclass(frozen=True)
class CheckDiagnostic:
    """Structured result for one deterministic check."""

    name: str
    passed: bool
    penalty: float = 0.0
    message: str = ""
    matches: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CheckReport:
    """All deterministic diagnostics for a task response."""

    diagnostics: list[CheckDiagnostic]

    @property
    def total_penalty(self) -> float:
        return sum(diagnostic.penalty for diagnostic in self.diagnostics)

    @property
    def failed(self) -> list[CheckDiagnostic]:
        return [diagnostic for diagnostic in self.diagnostics if not diagnostic.passed]

    def by_name(self) -> dict[str, CheckDiagnostic]:
        return {diagnostic.name: diagnostic for diagnostic in self.diagnostics}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    return len(
        [
            match.group(0)
            for match in SENTENCE_RE.finditer(text.strip())
            if match.group(0).strip()
        ]
    )


def extract_numbers(text: str) -> set[str]:
    return {match.group(0) for match in NUMBER_RE.finditer(text)}


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in ENTITY_RE.finditer(text):
        entity = match.group(0).strip()
        if entity not in STOP_ENTITIES and not entity.isdigit():
            entities.add(entity)
    return entities


def _context_text(task: RLTask) -> str:
    return "\n".join(
        [task.instruction, task.input_text, "\n".join(task.required_facts)]
    ).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _fact_is_present(response: str, fact: str) -> bool:
    if _contains_phrase(response, fact):
        return True
    tokens = [
        token.lower()
        for token in WORD_RE.findall(fact)
        if len(token) > 2 and token.lower() not in {"the", "and", "for", "with"}
    ]
    if not tokens:
        return True
    response_tokens = {token.lower() for token in WORD_RE.findall(response)}
    return all(token in response_tokens for token in tokens)


def _diagnostic(name: str, matches: list[str], message: str = "") -> CheckDiagnostic:
    passed = not matches
    return CheckDiagnostic(
        name=name,
        passed=passed,
        penalty=0.0 if passed else PENALTIES[name],
        message=message,
        matches=matches,
    )


def check_option_menu(response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in OPTION_MENU_RE.finditer(response)]
    return _diagnostic("option_menu", matches, "Response offers multiple options.")


def check_wrapper_phrase(response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in WRAPPER_RE.finditer(response)]
    return _diagnostic("wrapper_phrase", matches, "Response starts with wrapper text.")


def check_placeholders(task: RLTask, response: str) -> list[CheckDiagnostic]:
    placeholders = [match.group(0) for match in PLACEHOLDER_RE.finditer(response)]
    diagnostics: list[CheckDiagnostic] = []
    if task.constraints.allow_placeholders:
        diagnostics.append(CheckDiagnostic("placeholder_disallowed", True))
    else:
        diagnostics.append(
            _diagnostic(
                "placeholder_disallowed",
                placeholders,
                "Response uses placeholders when concrete answer is required.",
            )
        )

    if task.constraints.require_placeholders_for_missing_specifics:
        diagnostics.append(
            CheckDiagnostic(
                name="placeholder_required",
                passed=bool(placeholders),
                penalty=0.0 if placeholders else PENALTIES["placeholder_required"],
                message="Response must use a placeholder for missing specifics.",
                matches=[] if placeholders else ["missing_placeholder"],
            )
        )
    else:
        diagnostics.append(CheckDiagnostic("placeholder_required", True))
    return diagnostics


def check_subject_line(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in SUBJECT_RE.finditer(response)]
    if not task.constraints.no_subject_line:
        return CheckDiagnostic("subject_line", True, matches=matches)
    return _diagnostic("subject_line", matches, "Subject line is forbidden.")


def check_signoff(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in SIGNOFF_RE.finditer(response)]
    if not task.constraints.no_signoff:
        return CheckDiagnostic("signoff", True, matches=matches)
    return _diagnostic("signoff", matches, "Signoff is forbidden.")


def check_length(task: RLTask, response: str) -> list[CheckDiagnostic]:
    count = word_count(response)
    diagnostics: list[CheckDiagnostic] = []
    max_words = task.constraints.max_words
    if max_words is not None and count > int(max_words * 1.5):
        diagnostics.append(
            CheckDiagnostic(
                "too_long",
                False,
                PENALTIES["too_long"],
                f"{count}>{max_words} words",
                [str(count)],
            )
        )
    else:
        diagnostics.append(CheckDiagnostic("too_long", True, message=f"words={count}"))

    min_words = task.constraints.min_words
    if min_words is not None and count < min_words:
        diagnostics.append(
            CheckDiagnostic(
                "too_short",
                False,
                PENALTIES["too_short"],
                f"{count}<{min_words} words",
                [str(count)],
            )
        )
    else:
        diagnostics.append(CheckDiagnostic("too_short", True, message=f"words={count}"))

    exact_sentences = task.constraints.exact_sentences
    actual_sentences = sentence_count(response)
    if exact_sentences is not None and actual_sentences != exact_sentences:
        diagnostics.append(
            CheckDiagnostic(
                "sentence_count",
                False,
                PENALTIES["sentence_count"],
                f"{actual_sentences}!={exact_sentences} sentences",
                [str(actual_sentences)],
            )
        )
    else:
        diagnostics.append(
            CheckDiagnostic(
                "sentence_count", True, message=f"sentences={actual_sentences}"
            )
        )
    return diagnostics


def check_numbers(task: RLTask, response: str) -> CheckDiagnostic:
    if not task.constraints.preserve_numbers:
        return CheckDiagnostic("missing_number", True)
    expected = extract_numbers(task.input_text + "\n" + "\n".join(task.required_facts))
    actual = extract_numbers(response)
    missing = sorted(expected - actual)
    return _diagnostic("missing_number", missing, "Response dropped required numbers.")


def check_entities(task: RLTask, response: str) -> CheckDiagnostic:
    if not task.constraints.preserve_entities:
        return CheckDiagnostic("missing_entity", True)
    expected = extract_entities(task.input_text + "\n" + "\n".join(task.required_facts))
    actual = extract_entities(response)
    missing = sorted(expected - actual)
    return _diagnostic("missing_entity", missing, "Response dropped required entities.")


def check_ai_tells(task: RLTask, response: str) -> CheckDiagnostic:
    phrase_pool = set(AI_TELL_PHRASES) | {
        phrase.lower() for phrase in task.forbidden_phrases
    }
    matches = sorted(
        {
            phrase
            for phrase in phrase_pool
            if phrase and _contains_phrase(response, phrase)
        }
    )
    return _diagnostic(
        "ai_tell_phrase", matches, "Response contains known AI-tell phrases."
    )


def check_invented_details(task: RLTask, response: str) -> CheckDiagnostic:
    context = _context_text(task).lower()
    matches: list[str] = []
    for name in COMMON_FAKE_NAMES:
        if (
            re.search(rf"\b{re.escape(name)}\b", response)
            and name.lower() not in context
        ):
            matches.append(name)
    for entity in extract_entities(response):
        if entity.lower() not in context and entity in COMMON_FAKE_NAMES:
            matches.append(entity)
    return _diagnostic(
        "invented_detail",
        sorted(set(matches)),
        "Response invented unsupported specifics.",
    )


def check_required_and_forbidden_facts(
    task: RLTask, response: str
) -> list[CheckDiagnostic]:
    missing = [
        fact for fact in task.required_facts if not _fact_is_present(response, fact)
    ]
    forbidden = [
        fact for fact in task.forbidden_facts if _contains_phrase(response, fact)
    ]
    return [
        _diagnostic(
            "missing_required_fact", missing, "Response dropped required facts."
        ),
        _diagnostic("forbidden_fact", forbidden, "Response added forbidden facts."),
    ]


def check_refusal(response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in REFUSAL_RE.finditer(response)]
    return _diagnostic("refusal", matches, "Response refuses a harmless writing task.")


def check_markdown(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in MARKDOWN_LIST_RE.finditer(response)]
    if task.constraints.allow_markdown:
        return CheckDiagnostic("wrong_format_markdown", True, matches=matches)
    return _diagnostic("wrong_format_markdown", matches, "Markdown list is forbidden.")


def run_deterministic_checks(task: RLTask, response: str) -> CheckReport:
    """Run all deterministic hard checks for one task response."""
    diagnostics: list[CheckDiagnostic] = [
        check_option_menu(response),
        check_wrapper_phrase(response),
        check_subject_line(task, response),
        check_signoff(task, response),
        check_numbers(task, response),
        check_entities(task, response),
        check_ai_tells(task, response),
        check_invented_details(task, response),
        check_refusal(response),
        check_markdown(task, response),
    ]
    diagnostics.extend(check_placeholders(task, response))
    diagnostics.extend(check_length(task, response))
    diagnostics.extend(check_required_and_forbidden_facts(task, response))
    return CheckReport(diagnostics=diagnostics)
