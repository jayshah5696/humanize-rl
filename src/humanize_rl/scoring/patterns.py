"""Regex pattern library for Layer 1 deterministic scoring.

Sources:
- humanizer skill (30+ AI tells)
- claudiness (per-model .style files)
- rival.tips (32 dims, 178 models)
- Wikipedia:Signs of AI writing
"""

import re

# ---------------------------------------------------------------------------
# 1. Sycophantic / chatbot openers
# ---------------------------------------------------------------------------

OPENER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^certainly[!.]",
        r"^absolutely[!.]",
        r"^great question[!.]",
        r"^good question[!.]",
        r"^excellent question[!.]",
        r"^interesting question[!.]",
        r"^of course[!,.]",
        r"^sure[!,.]",
        r"^happy to help",
        r"^that'?s (?:a |an )?(?:great|good|excellent|interesting|thoughtful|wonderful|fantastic|really important|important) (?:question|point|topic|observation)",
        r"^what (?:a |an )?(?:great|good|excellent|interesting|thoughtful) (?:question|point)",
        r"^i'?d be happy to\b",
        r"^i appreciate (?:you|your|the|that|this)",
        r"^thank you for (?:raising|asking|bringing|sharing|this)",
        r"^(?:absolutely|definitely)[!,] let'?s",
        r"^let me (?:walk you through|break this down|help you|provide|share)",
        r"^i'?m glad you (?:asked|brought|raised|mentioned)",
    ]
]

# ---------------------------------------------------------------------------
# 2. Hedging phrases
# ---------------------------------------------------------------------------

HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bit'?s worth (?:noting|mentioning|considering|pointing out|examining)\b",
        r"\bit'?s important to (?:note|recognize|acknowledge|understand|remember|consider|mention)\b",
        r"\bit'?s (?:also )?worth (?:noting|mentioning|considering)\b",
        r"\bit'?s essential to (?:note|recognize|understand|consider)\b",
        r"\bit'?s crucial to (?:note|remember|understand)\b",
        r"\bmay potentially\b",
        r"\bcould be considered\b",
        r"\bone might argue\b",
        r"\bto be (?:fair|clear|honest|precise)\b",
        r"\bi should (?:note|mention|point out|clarify|add)\b",
        r"\bit'?s important to acknowledge\b",
        r"\bit'?s also worth mentioning\b",
    ]
]

# ---------------------------------------------------------------------------
# 3. Contraction patterns
# ---------------------------------------------------------------------------

CONTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:don't|won't|can't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b",
        r"\b(?:couldn't|wouldn't|shouldn't|didn't|doesn't)\b",
        r"\b(?:it's|that's|there's|here's|what's|who's|let's)\b",
        r"\b(?:i'm|i've|i'd|i'll)\b",
        r"\b(?:we're|we've|we'd|we'll)\b",
        r"\b(?:they're|they've|they'd|they'll)\b",
        r"\b(?:you're|you've|you'd|you'll)\b",
        r"\b(?:he's|she's)\b",
    ]
]

# ---------------------------------------------------------------------------
# 4. Closing / sign-off patterns
# ---------------------------------------------------------------------------

CLOSING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i hope this (?:helps|gives|provides|was helpful)",
        r"let me know if you (?:have|need|want|would like|'d like)",
        r"feel free to (?:ask|reach out|contact|let me know)",
        r"(?:don't|do not) hesitate to\b",
        r"in conclusion[,.]",
        r"to summarize[,.]",
        r"to sum up[,.]",
        r"in summary[,.]",
        r"happy to (?:discuss|elaborate|help|clarify) further",
        r"if you (?:need|want|have) (?:any )?(?:more|further|additional)",
    ]
]

# ---------------------------------------------------------------------------
# 5. Transition / filler phrases
# ---------------------------------------------------------------------------

TRANSITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\badditionally\b",
        r"\bin addition\b",
        r"\bconsequently\b",
        r"\bit is (?:important|worth|crucial|essential) to (?:note|recognize|understand|consider)\b",
        r"\bcannot be overstated\b",
        r"\bprofound impact\b",
        r"\bcritically important\b",
    ]
]

# ---------------------------------------------------------------------------
# 6. Bullet / list line detection
# ---------------------------------------------------------------------------

BULLET_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•]|\d+[.)]) \s*", re.MULTILINE
)
