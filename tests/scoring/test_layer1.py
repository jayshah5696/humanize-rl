"""Tests for Layer 1 deterministic scoring.

TDD: these tests define the contract. Implementation follows.
Each dimension: 0.0 = clearly AI, 1.0 = clearly human.
"""

import pytest

from humanize_rl.scoring.aggregator import HumannessResult, score_text
from humanize_rl.scoring.layer1 import (
    score_closing,
    score_contractions,
    score_em_dash,
    score_hedging,
    score_list_overuse,
    score_opener,
    score_sentence_variance,
    score_transitions,
)

# ---------------------------------------------------------------------------
# Dimension 1: opener_pattern
# ---------------------------------------------------------------------------

class TestOpenerPattern:
    """Sycophantic / chatbot openers → 0.0, normal openers → 1.0."""

    @pytest.mark.parametrize("text", [
        "Certainly! Here's a comprehensive overview of microservices.",
        "Absolutely! Let's dive into the world of containerization.",
        "Great question! Understanding database indexing is crucial.",
        "That's an excellent point about error handling.",
        "I'd be happy to help you understand the key principles.",
        "I appreciate you raising this important topic.",
        "That's a fantastic question! Testing strategies are crucial.",
    ])
    def test_ai_openers_score_low(self, text: str) -> None:
        assert score_opener(text) <= 0.1

    @pytest.mark.parametrize("text", [
        "I spent three weeks debugging a memory leak in our Go service.",
        "The dirty secret of microservices is most teams don't need them.",
        "We switched from Kubernetes to just running stuff on EC2.",
        "Look, I don't care what the benchmarks say.",
        "Writing good error messages is a design problem.",
    ])
    def test_human_openers_score_high(self, text: str) -> None:
        assert score_opener(text) >= 0.9


# ---------------------------------------------------------------------------
# Dimension 2: hedging_density
# ---------------------------------------------------------------------------

class TestHedgingDensity:
    """Heavy hedging → low score, no hedging → high score."""

    def test_heavy_hedging_scores_low(self) -> None:
        text = (
            "It's worth noting that this approach could be considered "
            "effective. It's important to note that one might argue "
            "there are potential benefits. To be fair, the results "
            "may potentially exceed expectations."
        )
        assert score_hedging(text) <= 0.4

    def test_no_hedging_scores_high(self) -> None:
        text = (
            "We deployed the fix on Tuesday. Response times dropped "
            "from 800ms to 120ms. The team moved on to the next ticket."
        )
        assert score_hedging(text) >= 0.8

    def test_moderate_hedging_scores_mid(self) -> None:
        text = (
            "It's worth noting that the migration took longer than expected. "
            "The database handled 50,000 rows per second during the transfer."
        )
        score = score_hedging(text)
        assert 0.3 <= score <= 0.9


# ---------------------------------------------------------------------------
# Dimension 3: list_overuse
# ---------------------------------------------------------------------------

class TestListOveruse:
    """Bullet-heavy text → low score, prose → high score."""

    def test_bullet_heavy_text_scores_low(self) -> None:
        text = (
            "Key benefits:\n"
            "- Scalability: Each service scales independently\n"
            "- Flexibility: Teams choose different technologies\n"
            "- Resilience: Failure doesn't cascade\n"
            "- Portability: Deploy anywhere\n"
            "- Efficiency: Lower resource overhead\n"
            "- Consistency: Same environment everywhere\n"
        )
        assert score_list_overuse(text) <= 0.3

    def test_prose_scores_high(self) -> None:
        text = (
            "We tried microservices for about six months. The overhead "
            "killed us. Every change required updating three services, "
            "two API contracts, and a shared library. We went back to "
            "a monolith and shipped twice as fast."
        )
        assert score_list_overuse(text) >= 0.8

    def test_numbered_list_also_detected(self) -> None:
        text = (
            "Here are the steps:\n"
            "1. Clone the repository\n"
            "2. Install dependencies\n"
            "3. Configure environment variables\n"
            "4. Run the migration\n"
            "5. Start the server\n"
        )
        assert score_list_overuse(text) <= 0.4


# ---------------------------------------------------------------------------
# Dimension 4: sentence_variance
# ---------------------------------------------------------------------------

class TestSentenceVariance:
    """Uniform sentence lengths (AI) → low, varied (human) → high."""

    def test_uniform_lengths_score_low(self) -> None:
        # All sentences ~10 words — unnaturally uniform
        text = (
            "The system handles requests from multiple client applications. "
            "Each request passes through the authentication middleware first. "
            "The middleware validates the token against our database. "
            "Valid tokens proceed to the appropriate service handler. "
            "Invalid tokens receive a standard four oh three response. "
            "The handler processes the request and returns results."
        )
        assert score_sentence_variance(text) <= 0.5

    def test_varied_lengths_score_high(self) -> None:
        # Mix of short and long — natural human writing
        text = (
            "It broke. The whole thing just stopped working at 3am on a "
            "Saturday, which is exactly when you don't want your payment "
            "processing pipeline to die. I spent two hours on the phone "
            "with AWS support. Turns out it was a misconfigured security "
            "group that had been wrong for months but only mattered when "
            "we hit a specific traffic pattern. Two lines in a YAML file. "
            "Fixed. Back to bed."
        )
        assert score_sentence_variance(text) >= 0.6


# ---------------------------------------------------------------------------
# Dimension 5: contractions
# ---------------------------------------------------------------------------

class TestContractions:
    """Informal text without contractions → low, with → high."""

    def test_no_contractions_in_casual_text_scores_low(self) -> None:
        text = (
            "I do not think this is a good approach. We should not deploy "
            "this to production. It does not handle edge cases and we have "
            "not tested it properly. I would not recommend this."
        )
        assert score_contractions(text) <= 0.4

    def test_natural_contractions_score_high(self) -> None:
        text = (
            "I don't think this is a good approach. We shouldn't deploy "
            "this to production. It doesn't handle edge cases and we "
            "haven't tested it properly. I wouldn't recommend this."
        )
        assert score_contractions(text) >= 0.7


# ---------------------------------------------------------------------------
# Dimension 6: closing_pattern
# ---------------------------------------------------------------------------

class TestClosingPattern:
    """AI sign-off phrases → 0.0, normal endings → 1.0."""

    @pytest.mark.parametrize("text", [
        "Some content here.\n\nI hope this helps! Let me know if you have any other questions.",
        "Some content here.\n\nFeel free to ask if you'd like to dive deeper into any specific area.",
        "Some content here.\n\nIn conclusion, the best approach depends on your specific needs.",
        "Some content here.\n\nLet me know if you need more specific guidance on any of these areas.",
        "Some content here.\n\nDon't hesitate to reach out if you have further questions.",
        "Some content here.\n\nIn summary, effective caching requires careful thought.",
    ])
    def test_ai_closings_score_low(self, text: str) -> None:
        assert score_closing(text) <= 0.1

    @pytest.mark.parametrize("text", [
        "Some content here.\n\nWe shipped the sort.",
        "Some content here.\n\nBoring technology wins again.",
        "Fixed. Back to bed.",
        "That's when I stopped taking job requirements literally.",
    ])
    def test_human_closings_score_high(self, text: str) -> None:
        assert score_closing(text) >= 0.9


# ---------------------------------------------------------------------------
# Dimension 7: em_dash_density
# ---------------------------------------------------------------------------

class TestEmDashDensity:
    """Heavy em-dash use → low score, sparse/none → high score."""

    def test_heavy_em_dash_scores_low(self) -> None:
        text = (
            "The problem — and this is the part nobody talks about — "
            "is systemic. It didn't die of natural causes — it was "
            "bought out. The team — all twelve of them — knew this "
            "was coming — but said nothing."
        )
        assert score_em_dash(text) <= 0.3

    def test_no_em_dash_scores_high(self) -> None:
        text = (
            "We deployed on Tuesday. Response times dropped from 800ms "
            "to 120ms. The team moved on to the next ticket. Nobody "
            "mentioned the outage again."
        )
        assert score_em_dash(text) >= 0.8


# ---------------------------------------------------------------------------
# Dimension 8: transition_overuse
# ---------------------------------------------------------------------------

class TestTransitionOveruse:
    """Academic transitions → low score, natural flow → high."""

    def test_transition_heavy_scores_low(self) -> None:
        text = (
            "Furthermore, the system provides excellent performance. "
            "Moreover, the architecture is designed for scalability. "
            "Additionally, the team has implemented comprehensive testing. "
            "Consequently, the deployment process is streamlined. "
            "In addition, monitoring has been significantly enhanced."
        )
        assert score_transitions(text) <= 0.3

    def test_natural_flow_scores_high(self) -> None:
        text = (
            "We tried three different approaches before landing on this one. "
            "The first was too slow, the second leaked memory, and the third "
            "worked but was impossible to test. So we wrote our own."
        )
        assert score_transitions(text) >= 0.8


# ---------------------------------------------------------------------------
# Aggregator: score_text
# ---------------------------------------------------------------------------

class TestAggregator:
    """End-to-end: known AI text scores low, known human text scores high."""

    def test_returns_humanness_result(self) -> None:
        result = score_text("Hello world, this is a test.")
        assert isinstance(result, HumannessResult)
        assert 0.0 <= result.overall <= 1.0
        assert len(result.per_dim) == 8

    def test_obvious_ai_text_scores_below_threshold(self) -> None:
        ai_text = (
            "Certainly! Here's a comprehensive overview of microservices "
            "architecture. Microservices represent a fundamental shift in "
            "how we think about software design.\n\n"
            "The key benefits include:\n"
            "- **Scalability:** Each service can be scaled independently.\n"
            "- **Flexibility:** Teams can choose different technologies.\n"
            "- **Resilience:** Failure in one service doesn't cascade.\n\n"
            "Furthermore, it's worth noting that microservices also come "
            "with challenges. Moreover, the ecosystem has matured considerably. "
            "Additionally, teams need robust monitoring solutions.\n\n"
            "In conclusion, microservices represent a paradigm shift. "
            "I hope this helps! Let me know if you have any other questions."
        )
        result = score_text(ai_text)
        assert result.overall < 0.4, f"AI text scored {result.overall}, expected < 0.4"

    def test_obvious_human_text_scores_above_threshold(self) -> None:
        human_text = (
            "I spent three weeks debugging a memory leak in our Go service. "
            "Turned out it was a goroutine that never got canceled because "
            "someone forgot a ctx.Done() check in a for-select loop. The "
            "fix was two lines. I wanted to scream."
        )
        result = score_text(human_text)
        assert result.overall > 0.7, f"Human text scored {result.overall}, expected > 0.7"

    def test_per_dim_keys_correct(self) -> None:
        result = score_text("Test text for key checking purposes.")
        expected_keys = {
            "opener_pattern",
            "hedging_density",
            "list_overuse",
            "sentence_variance",
            "contractions",
            "closing_pattern",
            "em_dash_density",
            "transition_overuse",
        }
        assert set(result.per_dim.keys()) == expected_keys
