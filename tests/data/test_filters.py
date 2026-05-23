import pytest
from humanize_rl.data.filters import LangDetectFilter, TokenPriorPerplexityFilter, SpaCyActionFilter

def test_lang_detect_filter():
    filt = LangDetectFilter()
    # Should accept English
    assert filt.filter("Write a short email to my manager about a blocker.") is True
    # Should reject non-English (e.g. French, Spanish)
    assert filt.filter("Écrivez un e-mail court à mon responsable.") is False
    assert filt.filter("Escribe un correo electrónico corto a mi gerente.") is False

def test_token_prior_perplexity_filter():
    filt = TokenPriorPerplexityFilter()
    # Normal English should pass
    assert filt.filter("Write a short email to my manager about a blocker.") is True
    # Gibberish should be rejected
    assert filt.filter("asdfghjkl qwertyuiop zxcvbnm qwert yuiop") is False
    # Extreme repetition should be rejected
    assert filt.filter("the the the the the the the the the the the the the") is False

def test_spacy_action_filter():
    filt = SpaCyActionFilter()
    # Valid writing instructions should pass
    assert filt.filter("Write a professional email explaining the caching mechanism.") is True
    assert filt.filter("Draft a blog post about our new feature.") is True
    assert filt.filter("Summarize this document in a short paragraph.") is True
    
    # Non-writing tasks should be rejected
    assert filt.filter("Calculate the sum of all prime numbers under 100.") is False
    assert filt.filter("How do I install python on macOS?") is True  # Wait, is this a guide/tutorial explanation? Let's check how we handle it.
    assert filt.filter("2 + 2 = ?") is False
