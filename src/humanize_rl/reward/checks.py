"""Deterministic reward checks for RL task responses."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from math import ceil, floor

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
SUBJECT_LINE_RE = re.compile(r"(?im)^\s*(?:\*\*)?subject(?:\*\*)?\s*:\s*.*(?:\n|$)")
SUBJECT_HEADING_RE = re.compile(
    r"(?m)\A\s*(?!P\.?S\.?\b)([A-Z][^\n]{3,140}:[^\n]{0,120})\s*(?:\n|$)"
)
SALUTATION_RE = re.compile(r"(?im)\A\s*(?:dear\s+[^,\n]{1,80}|to whom it may concern),")
SALUTATION_LINE_RE = re.compile(
    r"(?im)^\s*(?:dear|hi|hello|hey|greetings)\s+[^,\n]{1,100},?\s*(?:\n|$)"
)
INLINE_SALUTATION_RE = re.compile(
    r"(?i)\b(?:dear|hi|hello|hey|greetings)\s+[^,\n']{1,100},"
)
SIGNOFF_RE = re.compile(
    r"(?im)^\s*(?:best|best regards|regards|kind regards|warm regards|"
    r"warmly|thanks|thank you|sincerely|cheers|love|your friend),?\s*$"
)
INLINE_SIGNOFF_RE = re.compile(
    r"(?i)\b(?:best regards|kind regards|warm regards|warmly|sincerely|"
    r"cheers|love),\s+[A-Z][^\n]{1,160}\.?\s*$|"
    r"\bthanks for (?:listening|understanding|your patience|sticking around)\.?\s*$|"
    r"\b(?:thanks|thank you) for (?:your )?"
    r"(?:feedback|collaboration|time|attention|input|help|support|"
    r"consideration|review|clarity|thoughts)"
    r"(?: and [a-z ]{2,40})?\.?\s*$"
)
INSTRUCTION_LEAK_RE = re.compile(
    r"\b(?:source message|your draft|the draft|begin writing|"
    r"provide the details in the following format|use the following format|"
    r"ensure (?:the )?(?:output|response|draft)|"
    r"verify (?:the )?(?:output|response)|"
    r"the (?:output|response|prompt|writer) must|"
    r"all constraints must)\b",
    re.IGNORECASE,
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
    "i hope this message finds you well",
    "i am writing to",
    "i'm writing to",
    "we are writing to",
    "we're writing to",
    "i look forward to your feedback",
    "thanks for listening",
    "let's break this down",
    "let's dive",
    "let's jump right in",
    "think about it",
    "here's a recipe",
    "magical world",
    "must-watch",
    "highly recommend",
    "seamless",
    "unlock",
    "empower",
    "mission-critical",
    "move forward effectively",
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
HEADING_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s+\S|[A-Z][A-Za-z0-9 ,/&-]{2,60}:)\s*$")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
HASHTAG_RE = re.compile(r"(?m)(?:^|\s)(#[A-Za-z][\w-]{1,60})")
ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z][A-Z']{3,}\b")
ALL_CAPS_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Z']{3,}\b[\s,.!?;:-]*){3,}")
ALL_CAPS_LINE_RE = re.compile(r"(?m)^[^a-z\n]{12,}$")
TEMPORAL_DETAIL_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|yesterday|tonight|morning|afternoon|evening|"
    r"noon|midnight|q[1-4])\b",
    re.IGNORECASE,
)
UNSUPPORTED_NEGATION_RE = re.compile(
    r"\b(?:i didn't|i did not|i don't care|i do not care|i'm not|i am not|"
    r"no excuses|no more excuses|if you say no|i'll fire|i will fire|"
    r"i'll pay|i will pay|lose it forever|take it or leave it|"
    r"kill(?:ed|s|ing)?|drunk|refunds?|delete it|nobody asked|no one asked)\b",
    re.IGNORECASE,
)
FAKE_CASUAL_RE = re.compile(
    r"\b(?:we got(?: us)?|you guys|we're good|we are good|good stuff|"
    r"project stuff|technical stuff|personal and professional stuff|"
    r"doing you a solid|stuff)\b",
    re.IGNORECASE,
)
VAGUE_SUBSTITUTION_RE = re.compile(
    r"\b(?:stuff|thing|things|something|somebody|someone)\b",
    re.IGNORECASE,
)
BROKEN_INFORMAL_GRAMMAR_RE = re.compile(
    r"\b(?:we working|we tell you|when stuff good|stuff good again|"
    r"we got us|we at [A-Z][A-Za-z0-9& .'-]{2,80} want you at|"
    r"we think we should do some stuff|before day ends)\b",
    re.IGNORECASE,
)
REGISTER_MISMATCH_RE = re.compile(
    r"\b(?:cut the crap|doing you a solid|screw this|screw it|"
    r"what the hell|shut up|bullshit|crap)\b",
    re.IGNORECASE,
)
THANKS_PADDING_RE = re.compile(
    r"(?i)(?:^|[.!?;]\s+)((?:thanks|thank you|appreciate you)"
    r"(?:\s+for\s+(?:looking|looking out|listening|checking in|the heads up|"
    r"your time|your feedback|your help|your patience|your support|"
    r"your collaboration))?[.!]?)\s*$"
)
NUMBER_RE = re.compile(r"\b(?:\d+\s?(?:am|pm)|\d+[\d,.:/-]*|[A-Z]+-\d+)\b", re.I)
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)*|[A-Z][A-Z0-9_]{2,})\b")
WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
CONTRACTION_RE = re.compile(
    r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b",
    re.IGNORECASE,
)
DEFAULT_TARGET_TOLERANCE = 0.15
REPETITION_NGRAM_SIZE = 4
REPETITION_MIN_NGRAMS = 4
REPETITION_MAX_TOP_COUNT = 2
REPETITION_MAX_DENSITY = 0.20
LOW_SOURCE_OVERLAP_MIN_SOURCE_TOKENS = 20
LOW_SOURCE_OVERLAP_MIN_SHARED_TOKENS = 3
LOW_SOURCE_OVERLAP_MIN_RATIO = 0.12
LOW_SPECIFICITY_MIN_SOURCE_TOKENS = 6
LOW_SPECIFICITY_MAX_SHARED_RATIO = 0.35
ROMANCE_TASK_RE = re.compile(
    r"\b(?:romance|romantic|rom-com|love stor(?:y|ies))\b",
    re.IGNORECASE,
)
FILM_TASK_RE = re.compile(r"\b(?:film|movie|cinema)\b", re.IGNORECASE)
RECOMMENDATION_TASK_RE = re.compile(
    r"\b(?:recommend|recommendation|personal recommendation)\b",
    re.IGNORECASE,
)
UPLIFTING_TASK_RE = re.compile(
    r"\b(?:uplifting|hopeful|heartwarming|feel-good)\b",
    re.IGNORECASE,
)
KNOWN_UPLIFTING_ROMANCE_TITLES: tuple[str, ...] = (
    "About Time",
    "Amelie",
    "Before Sunrise",
    "Casablanca",
    "Crazy Rich Asians",
    "Ever After",
    "Love Actually",
    "Notting Hill",
    "Pride and Prejudice",
    "Roman Holiday",
    "Sleepless in Seattle",
    "The Big Sick",
    "The Notebook",
    "The Princess Bride",
    "The Shop Around the Corner",
    "When Harry Met Sally",
    "You've Got Mail",
)
UNSUITABLE_UPLIFTING_ROMANCE_TITLES: tuple[str, ...] = (
    "12 Years a Slave",
    "1984",
    "A Child's Journey to the Sea",
    "A Night at the Opera",
    "A Quiet Place",
    "All the President's Men",
    "Bad Boys 3",
    "Forrest Gump",
    "High School Musical",
    "Inception",
    "Star Trek",
    "Star Wars",
    "The Color Purple",
    "The Dark Knight",
    "The Godfather",
    "The Good Diver",
    "The Tree of Life",
    "This Is Your Life",
)

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
    "salutation": -0.15,
    "too_long": -0.30,
    "too_short": -0.15,
    "sentence_count": -0.20,
    "ai_tell_phrase": -0.25,
    "refusal": -0.40,
    "wrong_format_markdown": -0.20,
    "missing_required_fact": -0.35,
    "forbidden_fact": -0.50,
    "repetition": -0.40,
    "missing_must_include_phrase": -0.35,
    "forbidden_phrase": -0.35,
    "forbidden_opener": -0.25,
    "em_dash": -0.20,
    "wrong_format_bullets": -0.20,
    "wrong_format_heading": -0.20,
    "paragraph_count": -0.20,
    "sentence_window": -0.20,
    "missing_contraction": -0.20,
    "unsuitable_recommendation": -0.60,
    "emoji": -0.30,
    "all_caps": -0.30,
    "hashtag": -0.20,
    "invented_number": -0.45,
    "invented_temporal_detail": -0.35,
    "unsupported_negation": -0.45,
    "low_source_overlap": -0.40,
    "instruction_leak": -0.30,
    "fake_casual_phrase": -0.35,
    "low_specificity_substitution": -0.50,
    "broken_informal_grammar": -0.35,
    "register_mismatch": -0.35,
    "thanks_padding": -0.20,
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
    "Do",
    "First",
    "Firstly",
    "Second",
    "Secondly",
    "Third",
    "Thirdly",
    # Common imperative-instruction starters
    "Use",
    "Return",
    "Send",
    "Write",
    "Rewrite",
    "Clean",
    "Make",
    "Explain",
    "Subject",
    "Slack",
    "Email",
    "Colleague",
    "Project",
    "Date",
    "User",
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
    # Sentence-start discourse / pronoun fragments, not entities.
    "And",
    "As",
    "After",
    "Before",
    "But",
    "During",
    "For",
    "Furthermore",
    "Given",
    "He",
    "Her",
    "His",
    "However",
    "In",
    "Let",
    "My",
    "Our",
    "She",
    "Should",
    "These",
    "Those",
    "To",
    "What",
    "Your",
    # Generic subject-line nouns.
    "Discussion",
    "Compliance",
    "Invitation",
    "Key",
    "Areas",
    "Focus",
    "Meeting",
    "Proposal",
    "Regarding",
    "Request",
    "Review",
    "Urgent",
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
    "Understanding",
}

REQUIRED_FACT_DISCOURSE_STOPWORDS: set[str] = {
    "additionally",
    "after",
    "as",
    "before",
    "during",
    "first",
    "firstly",
    "furthermore",
    "given",
    "however",
    "in",
    "let",
    "please",
    "second",
    "secondly",
    "should",
    "these",
    "third",
    "thirdly",
    "those",
    "what",
}

REQUIRED_FACT_STOPWORDS: set[str] = {
    "dear",
    "greetings",
    "hello",
    "hey",
    "hi",
    "rewrite",
    "write",
    "return",
    "slack",
    "email",
    "message",
    "colleague",
    "project",
    "date",
    "user",
    "subject",
} | REQUIRED_FACT_DISCOURSE_STOPWORDS

SOURCE_OVERLAP_STOPWORDS: set[str] = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "before",
    "being",
    "casual",
    "channel",
    "colleague",
    "direct",
    "email",
    "everyone",
    "following",
    "formal",
    "from",
    "further",
    "have",
    "human",
    "inform",
    "into",
    "just",
    "know",
    "less",
    "like",
    "message",
    "more",
    "natural",
    "only",
    "please",
    "return",
    "rewrite",
    "slack",
    "sound",
    "sounds",
    "source",
    "stiff",
    "subject",
    "team",
    "that",
    "this",
    "under",
    "update",
    "want",
    "with",
    "words",
    "write",
    "writing",
    "your",
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


def word_bounds(task: RLTask) -> tuple[int | None, int | None]:
    """Return active lower/upper word bounds, including target_words windows."""
    constraints = task.constraints
    lower = constraints.min_words
    upper = constraints.max_words

    if constraints.target_words is not None:
        tolerance = (
            constraints.target_tolerance
            if constraints.target_tolerance is not None
            else DEFAULT_TARGET_TOLERANCE
        )
        target_lower = max(1, floor(constraints.target_words * (1.0 - tolerance)))
        target_upper = max(
            target_lower, ceil(constraints.target_words * (1.0 + tolerance))
        )
        lower = target_lower if lower is None else max(lower, target_lower)
        upper = target_upper if upper is None else min(upper, target_upper)

    return lower, upper


def sentence_count(text: str) -> int:
    return len(
        [
            match.group(0)
            for match in SENTENCE_RE.finditer(text.strip())
            if match.group(0).strip()
        ]
    )


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()])


def extract_numbers(text: str) -> set[str]:
    return {match.group(0) for match in NUMBER_RE.finditer(text)}


def extract_temporal_details(text: str) -> set[str]:
    return {match.group(0) for match in TEMPORAL_DETAIL_RE.finditer(text)}


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in ENTITY_RE.finditer(text):
        entity = match.group(0).strip()
        parts = entity.split()
        while parts and parts[0] in STOP_ENTITIES:
            parts = parts[1:]
        entity = " ".join(parts)
        if entity and entity not in STOP_ENTITIES and not entity.isdigit():
            entities.add(entity)
    return entities


def _strip_scaffold_text(text: str) -> str:
    without_subjects = SUBJECT_LINE_RE.sub("\n", text)
    without_salutation_lines = SALUTATION_LINE_RE.sub("\n", without_subjects)
    return INLINE_SALUTATION_RE.sub("", without_salutation_lines)


def _ignored_scaffold_text(text: str) -> str:
    matches: list[str] = []
    matches.extend(match.group(0) for match in SUBJECT_LINE_RE.finditer(text))
    matches.extend(match.group(0) for match in SALUTATION_LINE_RE.finditer(text))
    matches.extend(match.group(0) for match in INLINE_SALUTATION_RE.finditer(text))
    return "\n".join(matches)


def _normalized_phrase(text: str) -> str:
    return " ".join(token.lower() for token in WORD_RE.findall(text))


def _context_text(task: RLTask) -> str:
    return "\n".join(
        [task.instruction, task.input_text, "\n".join(task.required_facts)]
    ).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _has_negated_style_permission(
    context: str, targets: tuple[str, ...], *, window: int = 160
) -> bool:
    target_re = "|".join(targets)
    negative_re = (
        r"\b(?:no|without|remove|omit|avoid|do not|don't|must not|"
        r"never|not use|don't include|do not include)\b"
        rf"[^.\n]{{0,{window}}}\b(?:{target_re})\b"
    )
    must_not_re = rf"\bmust_not\b\s*:?\s*[^.\n]{{0,{window}}}\b(?:{target_re})\b"
    return bool(
        re.search(negative_re, context, re.I) or re.search(must_not_re, context, re.I)
    )


def _has_positive_style_permission(
    context: str, targets: tuple[str, ...], *, window: int = 80
) -> bool:
    target_re = "|".join(targets)
    positive_re = (
        r"\b(?:include|use|add|keep|start with|open with|write|make|respond)\b"
        rf"[^.\n]{{0,{window}}}\b(?:{target_re})\b"
    )
    return bool(re.search(positive_re, context, re.I))


def _allows_emoji(task: RLTask) -> bool:
    context = _context_text(task)
    targets = (r"emoji(?:s)?", r"emoticon(?:s)?")
    if _has_negated_style_permission(context, targets):
        return False
    return _has_positive_style_permission(context, targets)


def _allows_hashtags(task: RLTask) -> bool:
    context = _context_text(task)
    targets = (r"hashtag(?:s)?", r"#")
    if _has_negated_style_permission(context, targets):
        return False
    return _has_positive_style_permission(context, targets)


def _allows_all_caps(task: RLTask) -> bool:
    context = _context_text(task)
    targets = (r"all caps", r"all-caps", r"uppercase", r"capital letters", r"shouty")
    if _has_negated_style_permission(context, targets):
        return False
    return _has_positive_style_permission(context, targets)


def _allows_salutation(task: RLTask) -> bool:
    context = _context_text(task)
    targets = (
        r"greeting(?:s)?",
        r"salutation(?:s)?",
        r"dear",
        r"sign-?off formalit(?:y|ies)",
    )
    if _has_negated_style_permission(context, targets):
        return False
    return _has_positive_style_permission(context, targets, window=80)


def _allows_register_mismatch_phrasing(task: RLTask) -> bool:
    context = _context_text(task)
    targets = (
        r"slang",
        r"swear(?:ing)?",
        r"profan(?:e|ity)",
        r"rough language",
        r"aggressive",
    )
    if _has_negated_style_permission(context, targets):
        return False
    return _has_positive_style_permission(context, targets, window=120)


def _allows_thanks_padding(task: RLTask) -> bool:
    if not task.constraints.no_signoff:
        return True
    context = _context_text(task)
    targets = (r"thank(?:s| you)?", r"appreciat(?:e|ion)")
    if _has_negated_style_permission(context, targets):
        return False
    if re.search(r"\b(?:thank|thanks|appreciat(?:e|ion))\b", context, re.I):
        return True
    return _has_positive_style_permission(context, targets, window=120)


def _matched_phrases(text: str, phrases: list[str]) -> list[str]:
    return sorted(
        {phrase for phrase in phrases if phrase and _contains_phrase(text, phrase)}
    )


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


def _is_scaffold_required_fact(fact: str) -> bool:
    lowered = fact.lower()
    if "\n" in fact and re.search(r"\b(?:dear|subject)\b", lowered):
        return True
    tokens = [token.lower() for token in WORD_RE.findall(fact)]
    if not tokens:
        return True
    return all(token in REQUIRED_FACT_STOPWORDS for token in tokens)


def _fact_only_in_ignored_scaffold(task: RLTask, fact: str) -> bool:
    normalized = _normalized_phrase(fact)
    if not normalized:
        return True
    ignored = _normalized_phrase(_ignored_scaffold_text(task.input_text))
    if normalized not in ignored:
        return False
    remaining = _normalized_phrase(_strip_scaffold_text(task.input_text))
    return normalized not in remaining


def _active_required_facts(task: RLTask) -> list[str]:
    return [
        fact
        for fact in task.required_facts
        if not _is_scaffold_required_fact(fact)
        and not _fact_only_in_ignored_scaffold(task, fact)
    ]


def _entity_source_text(task: RLTask) -> str:
    return (
        _strip_scaffold_text(task.input_text)
        + "\n"
        + "\n".join(_active_required_facts(task))
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_RE.findall(_strip_scaffold_text(text))
        if len(token) > 3
        and not token.isdigit()
        and token.lower() not in SOURCE_OVERLAP_STOPWORDS
    }


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
    matches.extend(
        match.group(1).strip()
        for match in SUBJECT_HEADING_RE.finditer(response)
        if not SUBJECT_RE.match(match.group(1).strip())
    )
    if not task.constraints.no_subject_line:
        return CheckDiagnostic("subject_line", True, matches=matches)
    return _diagnostic(
        "subject_line",
        sorted(set(matches)),
        "Subject line or title-like heading is forbidden.",
    )


def check_signoff(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in SIGNOFF_RE.finditer(response)]
    matches.extend(
        match.group(0).strip() for match in INLINE_SIGNOFF_RE.finditer(response)
    )
    if not task.constraints.no_signoff:
        return CheckDiagnostic("signoff", True, matches=matches)
    return _diagnostic("signoff", sorted(set(matches)), "Signoff is forbidden.")


def check_salutation(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(0).strip() for match in SALUTATION_RE.finditer(response)]
    if _allows_salutation(task):
        return CheckDiagnostic("salutation", True, matches=matches)
    return _diagnostic(
        "salutation",
        sorted(set(matches)),
        "Formal salutation is not requested.",
    )


def _expects_direct_content(task: RLTask) -> bool:
    return task.constraints.return_only_answer and task.mode != "multi_constraint_compose"


def check_instruction_leak(task: RLTask, response: str) -> CheckDiagnostic:
    if not _expects_direct_content(task):
        return CheckDiagnostic("instruction_leak", True)
    matches = [
        match.group(0).strip() for match in INSTRUCTION_LEAK_RE.finditer(response)
    ]
    return _diagnostic(
        "instruction_leak",
        sorted(set(matches), key=str.lower),
        "Response leaks instructions or talks about the draft/source instead of answering.",
    )


def check_length(task: RLTask, response: str) -> list[CheckDiagnostic]:
    count = word_count(response)
    diagnostics: list[CheckDiagnostic] = []
    lower_words, upper_words = word_bounds(task)
    if upper_words is not None and count > int(ceil(upper_words * 1.5)):
        diagnostics.append(
            CheckDiagnostic(
                "too_long",
                False,
                PENALTIES["too_long"],
                f"{count}>{upper_words} words",
                [str(count)],
            )
        )
    else:
        diagnostics.append(CheckDiagnostic("too_long", True, message=f"words={count}"))

    if lower_words is not None and count < lower_words:
        diagnostics.append(
            CheckDiagnostic(
                "too_short",
                False,
                PENALTIES["too_short"],
                f"{count}<{lower_words} words",
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


def check_invented_numbers(task: RLTask, response: str) -> CheckDiagnostic:
    if not task.constraints.preserve_numbers:
        return CheckDiagnostic("invented_number", True)
    expected = extract_numbers(_entity_source_text(task))
    actual = extract_numbers(response)
    invented = sorted(actual - expected)
    return _diagnostic("invented_number", invented, "Response invented numbers.")


def check_invented_temporal_details(task: RLTask, response: str) -> CheckDiagnostic:
    expected = {
        detail.lower() for detail in extract_temporal_details(_entity_source_text(task))
    }
    actual = extract_temporal_details(response)
    invented = sorted(
        {detail for detail in actual if detail.lower() not in expected},
        key=str.lower,
    )
    return _diagnostic(
        "invented_temporal_detail",
        invented,
        "Response invented time/date details.",
    )


def check_entities(task: RLTask, response: str) -> CheckDiagnostic:
    if not task.constraints.preserve_entities:
        return CheckDiagnostic("missing_entity", True)
    expected = extract_entities(_entity_source_text(task))
    actual = extract_entities(response)
    missing = sorted(expected - actual)
    return _diagnostic("missing_entity", missing, "Response dropped required entities.")


def check_unsupported_negation(task: RLTask, response: str) -> CheckDiagnostic:
    source = _entity_source_text(task).lower()
    matches = sorted(
        {
            match.group(0)
            for match in UNSUPPORTED_NEGATION_RE.finditer(response)
            if match.group(0).lower() not in source
        },
        key=str.lower,
    )
    return _diagnostic(
        "unsupported_negation",
        matches,
        "Response added unsupported hostile or contradictory phrasing.",
    )


def check_source_overlap(task: RLTask, response: str) -> CheckDiagnostic:
    source_tokens = _content_tokens(_entity_source_text(task))
    if len(source_tokens) < LOW_SOURCE_OVERLAP_MIN_SOURCE_TOKENS:
        return CheckDiagnostic(
            "low_source_overlap", True, message=f"source_tokens={len(source_tokens)}"
        )

    response_tokens = _content_tokens(response)
    shared = source_tokens & response_tokens
    ratio = len(shared) / len(source_tokens)
    passed = (
        len(shared) >= LOW_SOURCE_OVERLAP_MIN_SHARED_TOKENS
        or ratio >= LOW_SOURCE_OVERLAP_MIN_RATIO
    )
    return CheckDiagnostic(
        name="low_source_overlap",
        passed=passed,
        penalty=0.0 if passed else PENALTIES["low_source_overlap"],
        message=f"shared={len(shared)}/{len(source_tokens)} ratio={ratio:.3f}",
        matches=[] if passed else [f"{len(shared)}/{len(source_tokens)}"],
    )


def check_fake_casual_phrase(task: RLTask, response: str) -> CheckDiagnostic:
    source = _context_text(task).lower()
    matches = sorted(
        {
            match.group(0).lower()
            for match in FAKE_CASUAL_RE.finditer(response)
            if match.group(0).lower() not in source
            or match.group(0).lower() in {"stuff", "we got", "we got us"}
        }
    )
    return _diagnostic(
        "fake_casual_phrase",
        matches,
        "Response uses fake-casual filler instead of plain human prose.",
    )


def check_low_specificity_substitution(task: RLTask, response: str) -> CheckDiagnostic:
    source_text = _entity_source_text(task)
    source_tokens = _content_tokens(source_text)
    if len(source_tokens) < LOW_SPECIFICITY_MIN_SOURCE_TOKENS:
        return CheckDiagnostic(
            "low_specificity_substitution",
            True,
            message=f"source_tokens={len(source_tokens)}",
        )

    source_lower = source_text.lower()
    vague_terms = sorted(
        {
            match.group(0).lower()
            for match in VAGUE_SUBSTITUTION_RE.finditer(response)
            if match.group(0).lower() not in source_lower
        }
    )
    if not vague_terms:
        return CheckDiagnostic("low_specificity_substitution", True)

    response_tokens = _content_tokens(response)
    shared = source_tokens & response_tokens
    ratio = len(shared) / len(source_tokens)
    concrete_facts = bool(
        task.required_facts
        or extract_entities(source_text)
        or extract_numbers(source_text)
        or extract_temporal_details(source_text)
    )
    failed = "stuff" in vague_terms or (
        concrete_facts and ratio < LOW_SPECIFICITY_MAX_SHARED_RATIO
    )
    return CheckDiagnostic(
        name="low_specificity_substitution",
        passed=not failed,
        penalty=0.0 if not failed else PENALTIES["low_specificity_substitution"],
        message=f"shared={len(shared)}/{len(source_tokens)} ratio={ratio:.3f}",
        matches=[] if not failed else vague_terms,
    )


def check_broken_informal_grammar(response: str) -> CheckDiagnostic:
    matches = [
        match.group(0).strip()
        for match in BROKEN_INFORMAL_GRAMMAR_RE.finditer(response)
    ]
    return _diagnostic(
        "broken_informal_grammar",
        sorted(set(matches), key=str.lower),
        "Response uses broken casual grammar as a human-sounding shortcut.",
    )


def check_register_mismatch(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [
        match.group(0).strip().lower()
        for match in REGISTER_MISMATCH_RE.finditer(response)
    ]
    if _allows_register_mismatch_phrasing(task):
        return CheckDiagnostic("register_mismatch", True, matches=matches)
    return _diagnostic(
        "register_mismatch",
        sorted(set(matches)),
        "Response uses slang or aggressive phrasing the task did not request.",
    )


def check_thanks_padding(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(1).strip() for match in THANKS_PADDING_RE.finditer(response)]
    if _allows_thanks_padding(task):
        return CheckDiagnostic("thanks_padding", True, matches=matches)
    return _diagnostic(
        "thanks_padding",
        sorted(set(matches), key=str.lower),
        "Response adds an unsupported thanks-style closing.",
    )


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
        fact
        for fact in _active_required_facts(task)
        if not _fact_is_present(response, fact)
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
    if task.constraints.allow_markdown and not task.constraints.no_markdown:
        return CheckDiagnostic("wrong_format_markdown", True, matches=matches)
    return _diagnostic("wrong_format_markdown", matches, "Markdown list is forbidden.")


def check_emoji(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(0) for match in EMOJI_RE.finditer(response)]
    if _allows_emoji(task):
        return CheckDiagnostic("emoji", True, matches=matches)
    return _diagnostic("emoji", sorted(set(matches)), "Emoji is not requested.")


def check_hashtags(task: RLTask, response: str) -> CheckDiagnostic:
    matches = [match.group(1) for match in HASHTAG_RE.finditer(response)]
    if _allows_hashtags(task):
        return CheckDiagnostic("hashtag", True, matches=matches)
    return _diagnostic("hashtag", sorted(set(matches)), "Hashtags are not requested.")


def check_all_caps(task: RLTask, response: str) -> CheckDiagnostic:
    if _allows_all_caps(task):
        return CheckDiagnostic("all_caps", True)

    matches: list[str] = []
    matches.extend(
        match.group(0).strip(" \t\n.,!?;:-")
        for match in ALL_CAPS_PHRASE_RE.finditer(response)
    )
    for line in response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        words = ALL_CAPS_WORD_RE.findall(stripped)
        letters = [char for char in stripped if char.isalpha()]
        uppercase_ratio = (
            sum(1 for char in letters if char.isupper()) / len(letters)
            if letters
            else 0.0
        )
        if len(words) >= 3 and uppercase_ratio >= 0.65:
            matches.append(stripped)
    for match in ALL_CAPS_LINE_RE.finditer(response):
        stripped = match.group(0).strip()
        if stripped and stripped not in matches:
            words = ALL_CAPS_WORD_RE.findall(stripped)
            if len(words) >= 3:
                matches.append(stripped)

    return _diagnostic("all_caps", matches, "Response uses shouty all-caps text.")


def check_v03_phrase_constraints(task: RLTask, response: str) -> list[CheckDiagnostic]:
    constraints = task.constraints
    missing_include = [
        phrase
        for phrase in constraints.must_include_phrases
        if not _contains_phrase(response, phrase)
    ]
    forbidden_pool = [
        *constraints.must_not_include_phrases,
        *constraints.forbidden_phrases_global,
    ]
    forbidden = _matched_phrases(response, forbidden_pool)
    openers = [
        opener
        for opener in constraints.forbidden_openers
        if response.lstrip().lower().startswith(opener.lower())
    ]
    em_dash_matches = ["—"] if "—" in response else []

    return [
        _diagnostic(
            "missing_must_include_phrase",
            missing_include,
            "Response missed required phrases.",
        ),
        _diagnostic("forbidden_phrase", forbidden, "Response used forbidden phrases."),
        _diagnostic("forbidden_opener", openers, "Response used a forbidden opener."),
        _diagnostic("em_dash", em_dash_matches, "Response used an em dash."),
    ]


def check_v03_structure(task: RLTask, response: str) -> list[CheckDiagnostic]:
    constraints = task.constraints
    diagnostics: list[CheckDiagnostic] = []
    bullets = [match.group(0).strip() for match in MARKDOWN_LIST_RE.finditer(response)]
    headings = [match.group(0).strip() for match in HEADING_RE.finditer(response)]

    if constraints.allow_bullets:
        diagnostics.append(
            CheckDiagnostic("wrong_format_bullets", True, matches=bullets)
        )
    else:
        diagnostics.append(
            _diagnostic("wrong_format_bullets", bullets, "Bullets are forbidden.")
        )

    if constraints.allow_headings:
        diagnostics.append(
            CheckDiagnostic("wrong_format_heading", True, matches=headings)
        )
    else:
        diagnostics.append(
            _diagnostic("wrong_format_heading", headings, "Headings are forbidden.")
        )

    paragraphs = paragraph_count(response)
    paragraph_failures: list[str] = []
    if (
        constraints.min_paragraphs is not None
        and paragraphs < constraints.min_paragraphs
    ):
        paragraph_failures.append(str(paragraphs))
    if (
        constraints.max_paragraphs is not None
        and paragraphs > constraints.max_paragraphs
    ):
        paragraph_failures.append(str(paragraphs))
    diagnostics.append(
        _diagnostic(
            "paragraph_count",
            paragraph_failures,
            f"paragraphs={paragraphs}",
        )
    )

    sentences = sentence_count(response)
    sentence_failures: list[str] = []
    if constraints.min_sentences is not None and sentences < constraints.min_sentences:
        sentence_failures.append(str(sentences))
    if constraints.max_sentences is not None and sentences > constraints.max_sentences:
        sentence_failures.append(str(sentences))
    diagnostics.append(
        _diagnostic("sentence_window", sentence_failures, f"sentences={sentences}")
    )

    missing_headings = [
        heading
        for heading in constraints.required_section_headings
        if not re.search(
            rf"(?im)^\s*{re.escape(heading)}\s*:?\s*$",
            response,
        )
    ]
    if missing_headings:
        diagnostics.append(
            _diagnostic(
                "wrong_format_heading",
                missing_headings,
                "Response missed required headings.",
            )
        )
    return diagnostics


def check_contractions(task: RLTask, response: str) -> CheckDiagnostic:
    required = task.constraints.min_contraction_count
    if required is None:
        return CheckDiagnostic("missing_contraction", True)
    actual = len(CONTRACTION_RE.findall(response))
    if actual >= required:
        return CheckDiagnostic(
            "missing_contraction", True, message=f"contractions={actual}"
        )
    return CheckDiagnostic(
        "missing_contraction",
        False,
        PENALTIES["missing_contraction"],
        f"{actual}<{required} contractions",
        [str(actual)],
    )


def _is_romance_recommendation_task(task: RLTask) -> bool:
    context = _context_text(task)
    return (
        bool(ROMANCE_TASK_RE.search(context))
        and bool(FILM_TASK_RE.search(context))
        and (
            bool(RECOMMENDATION_TASK_RE.search(context))
            or bool(UPLIFTING_TASK_RE.search(context))
        )
    )


def check_recommendation_suitability(task: RLTask, response: str) -> CheckDiagnostic:
    if not _is_romance_recommendation_task(task):
        return CheckDiagnostic("unsuitable_recommendation", True)

    unsuitable = [
        title
        for title in UNSUITABLE_UPLIFTING_ROMANCE_TITLES
        if _contains_phrase(response, title)
    ]
    has_known_good = any(
        _contains_phrase(response, title) for title in KNOWN_UPLIFTING_ROMANCE_TITLES
    )
    matches = unsuitable
    if not has_known_good:
        matches = [*matches, "missing_known_uplifting_romance_film"]

    return _diagnostic(
        "unsuitable_recommendation",
        sorted(set(matches)),
        "Response recommended or cited a film that does not fit the task.",
    )


def check_repetition(response: str) -> CheckDiagnostic:
    tokens = [token.lower() for token in WORD_RE.findall(response)]
    if len(tokens) < REPETITION_NGRAM_SIZE + REPETITION_MIN_NGRAMS - 1:
        return CheckDiagnostic("repetition", True, message=f"words={len(tokens)}")

    ngrams = [
        tuple(tokens[index : index + REPETITION_NGRAM_SIZE])
        for index in range(len(tokens) - REPETITION_NGRAM_SIZE + 1)
    ]
    counts = Counter(ngrams)
    top_ngram, top_count = counts.most_common(1)[0]
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    density = repeated / len(ngrams)
    passed = top_count <= REPETITION_MAX_TOP_COUNT and density <= REPETITION_MAX_DENSITY
    message = f"top_count={top_count} density={density:.3f}"
    return CheckDiagnostic(
        name="repetition",
        passed=passed,
        penalty=0.0 if passed else PENALTIES["repetition"],
        message=message,
        matches=[] if passed else [" ".join(top_ngram)],
    )


def run_deterministic_checks(task: RLTask, response: str) -> CheckReport:
    """Run all deterministic hard checks for one task response."""
    diagnostics: list[CheckDiagnostic] = [
        check_option_menu(response),
        check_wrapper_phrase(response),
        check_subject_line(task, response),
        check_signoff(task, response),
        check_salutation(task, response),
        check_instruction_leak(task, response),
        check_numbers(task, response),
        check_invented_numbers(task, response),
        check_invented_temporal_details(task, response),
        check_entities(task, response),
        check_unsupported_negation(task, response),
        check_source_overlap(task, response),
        check_fake_casual_phrase(task, response),
        check_low_specificity_substitution(task, response),
        check_broken_informal_grammar(response),
        check_register_mismatch(task, response),
        check_thanks_padding(task, response),
        check_ai_tells(task, response),
        check_invented_details(task, response),
        check_refusal(response),
        check_markdown(task, response),
        check_emoji(task, response),
        check_hashtags(task, response),
        check_all_caps(task, response),
        check_contractions(task, response),
        check_recommendation_suitability(task, response),
        check_repetition(response),
    ]
    diagnostics.extend(check_v03_phrase_constraints(task, response))
    diagnostics.extend(check_v03_structure(task, response))
    diagnostics.extend(check_placeholders(task, response))
    diagnostics.extend(check_length(task, response))
    diagnostics.extend(check_required_and_forbidden_facts(task, response))
    return CheckReport(diagnostics=diagnostics)
