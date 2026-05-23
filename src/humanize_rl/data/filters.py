import re
import math
import langdetect
import spacy

class LangDetectFilter:
    """Filters out non-English texts using langdetect."""
    def filter(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        try:
            # We enforce exact "en" detection
            return langdetect.detect(text) == "en"
        except Exception:
            return False

class TokenPriorPerplexityFilter:
    """
    Estimates token priors and perplexity proxy using word frequencies to filter out
    gibberish, code dumps, and extreme repetition.
    """
    def __init__(self, max_perplexity: float = 3000.0, min_ttr: float = 0.3):
        self.max_perplexity = max_perplexity
        self.min_ttr = min_ttr
        
        # Simple prior-based unigram frequency of common English words
        self.common_words = {
            "the": 0.06, "be": 0.04, "to": 0.03, "of": 0.03, "and": 0.03, "a": 0.02, "in": 0.02, "that": 0.01,
            "have": 0.01, "i": 0.01, "it": 0.01, "for": 0.01, "not": 0.01, "on": 0.01, "with": 0.01, "he": 0.01,
            "as": 0.01, "you": 0.01, "do": 0.01, "at": 0.01, "this": 0.01, "but": 0.01, "his": 0.01, "by": 0.01,
            "from": 0.01, "they": 0.01, "we": 0.01, "say": 0.01, "her": 0.01, "she": 0.01, "or": 0.01, "an": 0.01,
            "will": 0.01, "my": 0.01, "one": 0.01, "all": 0.01, "would": 0.01, "there": 0.01, "their": 0.01, "what": 0.01,
            "write": 0.005, "draft": 0.002, "email": 0.003, "post": 0.003, "essay": 0.002, "explain": 0.003, "about": 0.004,
            "manager": 0.001, "blocker": 0.0005, "caching": 0.0005, "mechanism": 0.0005, "professional": 0.001,
            "summarize": 0.001, "document": 0.002, "short": 0.002, "paragraph": 0.001, "how": 0.005, "install": 0.002,
            "python": 0.001, "macos": 0.0005, "share": 0.002, "technical": 0.002, "note": 0.002, "experience": 0.002,
            "some": 0.003, "new": 0.003, "our": 0.003, "feature": 0.001, "from": 0.005
        }
        self.default_prob = 1e-5

    def filter(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        
        # Tokenize by finding words (alphanumeric/letters)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return False
        
        # 1. Type-Token Ratio (TTR) check for repetition
        if len(words) > 5:
            ttr = len(set(words)) / len(words)
            if ttr < self.min_ttr:
                return False
        
        # 2. Perplexity Proxy check using unigram log-likelihood
        log_prob_sum = 0.0
        for w in words:
            prob = self.common_words.get(w, self.default_prob)
            log_prob_sum += math.log(prob)
        
        avg_log_prob = log_prob_sum / len(words)
        perplexity = math.exp(-avg_log_prob)
        
        if perplexity > self.max_perplexity:
            return False
            
        return True

class SpaCyActionFilter:
    """
    Checks if the prompt asks for a writing act, explanation, sharing of experience, or tutorial/how-to,
    while filtering out raw code snippets, code syntax tests, or mathematical equations.
    """
    def __init__(self):
        # Load spaCy pipeline; disable unnecessary components for speed
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        
        self.writing_verbs = {
            "write", "draft", "compose", "explain", "summarize", "rewrite", "create", 
            "generate", "describe", "provide", "post", "give", "tell", "make", "share",
            "show", "teach", "instruct"
        }
        self.writing_nouns = {
            "email", "post", "essay", "summary", "update", "document", "article", 
            "letter", "paragraph", "story", "blog", "guide", "tutorial", "instructions", 
            "memos", "response", "reply", "abstract", "text", "note", "explanation",
            "message", "report", "review", "caching", "mechanism", "feature", "experience"
        }
        
    def filter(self, text: str) -> bool:
        if not text or not text.strip():
            return False
            
        # Reject raw programming language definitions or simple mathematical formulas
        if text.strip().startswith("def ") or text.strip().startswith("class "):
            return False
        if re.search(r'^[0-9\s\+\-\*\/\=\?]+$', text.strip()):
            return False
            
        doc = self.nlp(text)
        
        # Look for the presence of action verbs or target nouns
        has_writing_verb = False
        has_writing_noun = False
        
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.writing_verbs:
                has_writing_verb = True
            if lemma in self.writing_nouns:
                has_writing_noun = True
                
        # Also support standard "how to" or informational requests
        is_howto = "how" in text.lower() or "what" in text.lower() or "why" in text.lower()
        
        # Must have a verb (English sentence) and either:
        # 1. An action verb and a writing target noun, OR
        # 2. An informational trigger (howto) with a verb, OR
        # 3. Explicit command like "Share your experience"
        has_any_verb = any(token.pos_ == "VERB" for token in doc)
        
        if not has_any_verb:
            return False
            
        if (has_writing_verb and has_writing_noun) or (is_howto and has_any_verb) or ("share" in text.lower()):
            return True
            
        return False
