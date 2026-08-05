# src/ai_prompter.py

import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

from config import GENERATION_MODEL

load_dotenv()
# timeout: hard cap so a hung request cannot freeze the app for minutes.
# max_retries=0: the SDK silently retries 429s/5xx twice with exponential
# backoff by DEFAULT. Stacked under our own retry loops, a rate-limited
# card could fire many hidden HTTP requests and hang for a minute+.
# We handle retries ourselves (fast-failing) — the SDK must not.
_client = None


def _get_client():
    """
    Lazily construct the Groq client.

    Constructing it at MODULE IMPORT time meant a missing/invalid
    GROQ_API_KEY crashed the entire app with a traceback before anything
    rendered. Now the error surfaces through the normal error-handling
    path with a readable message.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it in Streamlit: "
                "\u22ee \u2192 Settings \u2192 Secrets."
            )
        _client = Groq(api_key=api_key, timeout=25.0, max_retries=0)
    return _client

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ----------------------------------------------------------------------
# Italian colloquialisms, fillers, and set idioms that learners decode
# wrong by default. Add entries as you find more. Format:
#   expression -> (literal meaning, what it actually conveys)
# ----------------------------------------------------------------------
ITALIAN_COLLOQUIALISMS = {
    "magari": (
        "maybe (lit.)",
        "context-dependent. On its own with rising tone = 'I wish!' (wistful longing). "
        "Mid-sentence = 'maybe / perhaps' (hedging). 'Magari!' as exclamation = 'if only!'."
    ),
    "boh": (
        "(none)",
        "filler meaning 'dunno' / 'no idea' — verbal shrug. Often paired with raised shoulders."
    ),
    "mica": (
        "(emphatic negation)",
        "'not at all' / 'hardly' — strengthens negation. 'Non è mica facile' = 'it's not easy AT ALL'. "
        "Sometimes used alone in questions: 'Mica male' = 'not bad'."
    ),
    "dai": (
        "give (imperative, lit.)",
        "colloquial 'come on!' / 'really?' / 'no way!' — encouragement, mild disbelief, or playful protest."
    ),
    "allora": (
        "then (lit.)",
        "discourse marker: 'so...' / 'well then...' — opens, transitions, or summarises. Constant in speech."
    ),
    "cioè": (
        "that is",
        "filler 'I mean' / 'like' — extremely common in informal speech, often overused by younger speakers."
    ),
    "insomma": (
        "in short",
        "'so-so' / 'meh' / 'well...' — depending on tone, can mean 'not really great' or 'to wrap up'."
    ),
    "beh": (
        "(none)",
        "'well...' — hesitation/thinking marker, like English 'well' or 'um'."
    ),
    "tipo": (
        "type (lit.)",
        "filler 'like' — used identically to English 'like' in casual speech. 'Era tipo enorme' = 'it was, like, huge'."
    ),
    "comunque": (
        "anyway",
        "discourse marker — wrapping up, shifting topic, or conceding a point."
    ),
    "in bocca al lupo": (
        "in the wolf's mouth (lit.)",
        "'good luck!' — set idiom. Correct response is 'crepi (il lupo)' (may the wolf die), NOT 'grazie'."
    ),
    "non vedo l'ora": (
        "I don't see the hour (lit.)",
        "'I can't wait' — set idiom expressing eager anticipation."
    ),
    "ma dai": (
        "but give (lit.)",
        "'come on!' / 'no way!' — strong disbelief or playful protest."
    ),
    "che palle": (
        "what balls (lit.)",
        "vulgar but very common: 'how annoying' / 'what a pain in the neck'."
    ),
    "figurati": (
        "imagine yourself (lit.)",
        "'don't mention it' / 'no problem' / 'imagine that!' — polite dismissal or sarcastic surprise."
    ),
    "mannaggia": (
        "(none)",
        "mild expletive: 'damn!' / 'darn!' — frustration."
    ),
    "macché": (
        "(none)",
        "'no way!' / 'as if!' — strong dismissive negation. 'Macché stanco, sto benissimo!' = 'Tired?! I feel great!'."
    ),
    "meno male": (
        "less bad (lit.)",
        "'thank goodness' / 'good thing' — relief."
    ),
    "ecco": (
        "(none)",
        "'here / there' (presenting) OR 'exactly' (confirming) OR 'so...' (transitioning). Highly context-driven."
    ),
}

# ----------------------------------------------------------------------
# Italian "spelling traps" — near-homophones Whisper confuses, AND
# pairs learners get wrong. These drive the post-processing safety net.
# Format: tuples of confusable forms.
# ----------------------------------------------------------------------
ITALIAN_SPELLING_TRAPS = [
    # Accent-only minimal pairs (meaning changes)
    ("e", "è"),          # and / is
    ("la", "là"),        # the/her / there
    ("li", "lì"),        # them / there
    ("si", "sì"),        # oneself / yes
    ("da", "dà"),        # from / gives
    ("se", "sé"),        # if / oneself
    ("ne", "né"),        # of-it / neither
    ("papa", "papà"),    # pope / dad
    ("ancora", "ancóra"),  # anchor / still (vowel + stress)
    # h-drop pairs (verb avere — Whisper routinely loses the h)
    ("anno", "hanno"),   # year / they have
    ("a", "ha"),         # to / has
    ("o", "ho"),         # or / I have
    ("ai", "hai"),       # to-the / you have
    # Geminate (double consonant) pairs — real meaning change
    ("capelli", "cappelli"),  # hair / hats
    ("pena", "penna"),        # pain/sorrow / pen
    ("nono", "nonno"),        # ninth / grandfather
    ("sete", "sette"),        # thirst / seven
    ("caro", "carro"),        # dear / cart
    ("casa", "cassa"),        # house / crate/cash-register
    ("note", "notte"),        # notes / night
    ("pala", "palla"),        # shovel / ball
    ("sono", "sonno"),        # I am / sleep
    ("copia", "coppia"),      # copy / couple
]

# ----------------------------------------------------------------------
# Lexical sets that drive target-word categorisation.
# ----------------------------------------------------------------------
ITALIAN_ARTICLES = {
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una", "un'", "l'",
}

ITALIAN_PREPOSITIONS = {
    # simple
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    # articulated
    "del", "dello", "della", "dei", "degli", "delle", "dell'",
    "al", "allo", "alla", "ai", "agli", "alle", "all'",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle", "dall'",
    "nel", "nello", "nella", "nei", "negli", "nelle", "nell'",
    "col", "sul", "sullo", "sulla", "sui", "sugli", "sulle", "sull'",
}

ITALIAN_CLITICS = {
    # direct + indirect + reflexive
    "mi", "ti", "ci", "vi", "si", "lo", "la", "li", "le", "gli", "ne",
    # stressed / combined forms
    "me", "te", "se", "ce", "ve",
    # combined clitics
    "glielo", "gliela", "glieli", "gliele", "gliene",
    "celo", "cela", "celi", "cele", "cene",
    "melo", "mela", "meli", "mele", "mene",
    "telo", "tela", "teli", "tele", "tene",
    "velo", "vela", "veli", "vele", "vene",
}

ITALIAN_TIME_WORDS = {
    "oggi", "ieri", "domani", "dopodomani", "ieri sera",
    "adesso", "ora", "subito", "prima", "dopo",
    "mattina", "pomeriggio", "sera", "notte",
    "settimana", "mese", "anno", "secolo",
    "sempre", "mai", "spesso", "raramente", "talvolta", "qualche volta",
    "presto", "tardi",
}

ITALIAN_NEGATIONS = {
    "non", "mica", "neanche", "neppure", "nemmeno",
    "niente", "nulla", "nessuno", "nessuna", "mai", "né",
}

ITALIAN_QUANTIFIERS = {
    "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci",
    "cento", "mille", "milione",
    "molto", "molti", "molta", "molte",
    "poco", "pochi", "poca", "poche",
    "tanto", "tanti", "tanta", "tante",
    "tutto", "tutti", "tutta", "tutte",
    "qualche", "alcuni", "alcune",
    "ogni", "nessun", "nessuna",
    "abbastanza", "troppo", "troppi", "troppa", "troppe",
}

CONJUGATION_HINT_KEYWORDS = (
    "verb", "verbo", "conj", "subj", "congiunt", "cond",
    "imperf", "passato", "futuro", "presente", "trapass",
    "imperativ", "gerund", "particip",
)


class _TruncatedResponseError(Exception):
    """Raised when the LLM response was cut off at max_completion_tokens."""
    pass


def _classify_llm_error(exc):
    """
    Classify an exception from an LLM call into a category we can
    react to differently:
      - 'rate_limit'   : HTTP 429 or similar, should back off briefly
      - 'transient'    : network blip, timeout, 5xx — retry
      - 'json_parse'   : malformed JSON in response — retry
      - 'auth'         : 401/403 — don't retry, surface it
      - 'unknown'      : anything else
    Returns (category, short_description).
    """
    import json as _json
    s = str(exc)
    low = s.lower()
    if isinstance(exc, _TruncatedResponseError):
        return ("truncated", f"Truncated: {s[:120]}")
    if isinstance(exc, _json.JSONDecodeError):
        return ("json_parse", f"Malformed JSON: {s[:120]}")
    # Groq sometimes returns HTTP 400 with json_validate_failed — treat
    # like a JSON parse error and retry.
    if "json_validate_failed" in low or "failed to generate json" in low:
        return ("json_parse", f"LLM failed to produce valid JSON: {s[:120]}")
    # Groq retires models periodically (llama-3.3-70b was decommissioned in
    # June 2026). Retrying is pointless — the fix is a one-line config edit.
    if ("decommissioned" in low or "model_not_found" in low
            or ("model" in low and "does not exist" in low)
            or ("model" in low and "not found" in low)):
        return ("model_gone", f"Model unavailable/retired: {s[:160]}")
    if "429" in s or "rate limit" in low or "too many requests" in low:
        return ("rate_limit", f"Rate limit hit: {s[:120]}")
    if ("401" in s or "403" in s or "unauthorized" in low
            or "invalid api key" in low
            or "groq_api_key is not set" in low):
        return ("auth", f"Auth error (key missing/invalid?): {s[:120]}")
    if "500" in s or "502" in s or "503" in s or "504" in s:
        return ("transient", f"Server 5xx: {s[:120]}")
    if "timeout" in low or "timed out" in low:
        return ("transient", f"Timeout: {s[:120]}")
    if "connection" in low or "network" in low:
        return ("transient", f"Network: {s[:120]}")
    return ("unknown", f"Unknown error ({type(exc).__name__}): {s[:200]}")


def _generate_minimal_dictation(italian_text, english, hint=""):
    """
    Bare-bones fallback exercise. Requests the smallest useful JSON so it
    can't truncate: a natural Italian sentence, its English translation,
    and three plausible English distractors. No elaborate word breakdown,
    grammar note, or expression note — those are what blow the token
    budget on the full generator. Used only when the rich generator has
    already failed, to keep the session moving instead of showing an error.
    """
    target = (italian_text or "").strip()
    hint_part = f' (hint: {hint})' if hint else ''
    prompt = f"""
    Create a SHORT, natural Italian sentence (4-9 words) that uses the word
    or phrase "{target}"{hint_part}, meaning roughly "{english}".

    Then give its correct English translation and 3 plausible but WRONG
    English translations (same length, grammatical, genuinely tempting for
    a learner — vary the target word's grammar/meaning, not random words).

    Output ONLY this JSON:
    {{
        "italian": "<the Italian sentence with correct accents>",
        "english_correct": "<correct English translation>",
        "english_distractors": ["<wrong 1>", "<wrong 2>", "<wrong 3>"]
    }}
    """
    try:
        data = _call_llm_json(prompt, use_json_mode=True)
    except Exception as e:
        # One more go without json-mode (parses client-side)
        try:
            data = _call_llm_json(prompt, use_json_mode=False)
        except Exception as e2:
            logging.warning(f"Minimal dictation fallback also failed: {e2}")
            return None

    it = (data.get("italian") or target).strip()
    correct = (data.get("english_correct") or english or "").strip()
    distractors = data.get("english_distractors") or []
    # Ensure exactly 3 distractors; pad defensively if the model gave fewer
    distractors = [d for d in distractors if isinstance(d, str) and d.strip()][:3]
    while len(distractors) < 3:
        distractors.append(f"(alternative meaning {len(distractors) + 1})")

    return {
        "exercise_type":   "listen",
        "italian":         it,
        "english_correct": correct,
        "english_distractors": distractors,
        "word_breakdown":  [],
        "grammar_point":   {},
        "expression_note": {},
        "target_category": "general",
        "_minimal_fallback": True,
    }


def _wait_before_retry(category, attempt):
    """
    Brief, non-blocking-feeling wait between retries.

    Rate-limit retries used to wait 15-45s, but that locks Streamlit's
    spinner for an eternity from the user's point of view. We now
    bail out immediately on rate limits (the caller surfaces a clear
    "wait 60s" message) and only sleep briefly for genuine transient
    network errors that will likely clear on the next attempt.
    """
    import time as _time
    if category == "transient":
        _time.sleep(0.3 + attempt * 0.2)  # 0.3s, 0.5s, 0.7s
    elif category == "json_parse":
        _time.sleep(0.2)  # quick retry is fine for malformed JSON
    # No sleep for rate_limit, auth, truncated, or unknown


# Module-level: the category of the most recent generation failure
# (or None for success). Read by main_app.py to show context-specific
# user messages — e.g. "Wait 60 seconds" for rate limits vs a generic
# "try again" for other failures.
_last_error_category = None


def _record_last_error(category):
    global _last_error_category
    _last_error_category = category


def get_last_error_category():
    """Read the category of the most recent generation failure."""
    return _last_error_category


def _extract_json_from_text(text: str):
    """
    Robustly extract a JSON object from model output that may include
    code fences, prose, or thinking preamble. Used as the fallback when
    Groq's server-side json_object validation rejects a response.
    """
    if not text:
        raise ValueError("Empty text")
    t = text.strip()
    # Strip markdown code fences
    if "```" in t:
        parts = t.split("```")
        # take the largest fenced block, dropping a leading 'json' tag
        candidates = []
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                candidates.append(p)
        if candidates:
            t = max(candidates, key=len)
    # Slice from first brace to last brace
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # maybe it's a top-level list
        start = t.find("[")
        end = t.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in text")
    return json.loads(t[start:end + 1])


def _call_llm_json(prompt: str, use_json_mode: bool = True):
    """
    One LLM call returning parsed JSON.

    use_json_mode=True  → Groq's response_format json_object (server
                          validates; can 400 with json_validate_failed).
    use_json_mode=False → plain completion + client-side extraction,
                          which can never 400 on validation. Used as the
                          fallback after a json_validate failure.
    """
    try:
        from config import REASONING_EFFORT, MAX_GEN_TOKENS
    except ImportError:
        REASONING_EFFORT, MAX_GEN_TOKENS = "low", 2048

    kwargs = {
        "messages": [{'role': 'user', 'content': prompt}],
        "model": GENERATION_MODEL,
        "max_completion_tokens": MAX_GEN_TOKENS,
    }
    # Reasoning models (gpt-oss family) default to "medium" effort and
    # burn 1-2k hidden thinking tokens per call — quota poison on the
    # free tier. Force low effort where supported.
    if REASONING_EFFORT and GENERATION_MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = REASONING_EFFORT
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _get_client().chat.completions.create(**kwargs)
    choice = response.choices[0]
    raw = choice.message.content

    # Detect truncation: if the model hit the token ceiling, the JSON is
    # incomplete and will fail to parse. Raise a specific error so the
    # retry loop can react (retry once at low effort usually fits; the
    # caller also has a guaranteed-simple fallback).
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise _TruncatedResponseError(
            "Response hit max_completion_tokens (truncated JSON)."
        )

    if not raw or not raw.strip():
        raise ValueError("Empty response body from LLM")
    if use_json_mode:
        return _unwrap_json_response(json.loads(raw))
    return _unwrap_json_response(_extract_json_from_text(raw))


def _unwrap_json_response(raw_data):
    """
    Groq's JSON mode usually returns a dict, but occasionally the model
    wraps the response in a list like [{"italian": ...}] instead of just
    {"italian": ...}. Unwrap single-element lists; otherwise return the
    first item with expected keys, or the first dict.
    """
    if isinstance(raw_data, list):
        if not raw_data:
            return {}
        if len(raw_data) == 1 and isinstance(raw_data[0], dict):
            return raw_data[0]
        # Multiple items — pick the first with the expected keys
        for item in raw_data:
            if isinstance(item, dict) and ("italian" in item or "hanzi" in item or "english_prompt" in item):
                return item
        if isinstance(raw_data[0], dict):
            return raw_data[0]
        return {}
    return raw_data if isinstance(raw_data, dict) else {}


def _classify_target(italian_text: str, english: str, hint: str = "") -> str:
    """
    Best-effort categorisation of the target word/phrase so the AI can build
    distractors that attack the right learning axis.
    """
    if not italian_text:
        return "general"

    word = italian_text.strip().lower()
    hint_lc = (hint or "").lower()

    # Multi-word entries — likely an idiom/set phrase; let "general" handle
    if " " in word and word not in ITALIAN_CLITICS:
        # check if it's a known colloquialism
        if word in ITALIAN_COLLOQUIALISMS:
            return "expression"
        return "general"

    if word in ITALIAN_ARTICLES:
        return "article"
    if word in ITALIAN_PREPOSITIONS:
        return "preposition"
    if word in ITALIAN_CLITICS:
        return "clitic"
    if word in ITALIAN_NEGATIONS:
        return "negation"
    if word in ITALIAN_TIME_WORDS:
        return "time"
    if word in ITALIAN_QUANTIFIERS:
        return "quantifier"
    if word in ITALIAN_COLLOQUIALISMS:
        return "expression"

    # Verb form detection — via hint, or by infinitive ending
    if any(k in hint_lc for k in CONJUGATION_HINT_KEYWORDS):
        return "verb_form"
    if word.endswith(("are", "ere", "ire")) and len(word) > 3:
        return "verb_form"  # likely an infinitive

    return "content_word"


def _find_relevant_colloquialisms(text: str):
    """Return colloquialism entries present in the sentence, for prompt context."""
    text_lc = (text or "").lower()
    return {k: v for k, v in ITALIAN_COLLOQUIALISMS.items() if k in text_lc}


def _detect_spelling_traps(text: str):
    """
    Find which spelling-trap pairs appear in the sentence (token-level).
    Used to harden distractors against Whisper-style mishearing.
    """
    if not text:
        return []
    # tokenise crudely on whitespace + strip punctuation
    import re as _re
    tokens = [t.lower().strip(".,;:!?\"'’") for t in _re.split(r"\s+", text)]
    present = []
    for pair in ITALIAN_SPELLING_TRAPS:
        if any(tok in pair for tok in tokens):
            present.append(pair)
    return present


# ======================================================================
# Bulletproof Italian verb-to-subject agreement verifier.
#
# Used as a post-generation safety net to catch the LLM's most common
# mistake: emitting an Italian sentence whose verb conjugation doesn't
# match the English translation's subject pronoun.
#
# Example bug we want to catch:
#     Italian:  "Sono a casa."        (verb 'sono' = 1sg OR 3pl)
#     English:  "He/She is at home."  (subject 'He/She' = 3sg)  ← WRONG
#
# Approach: find the main finite verb in the Italian sentence, look up
# what subject(s) it allows, then check the English starts with one of
# those subjects. We bail out (return "consistent") on anything
# ambiguous — better to let a questionable sentence through than to
# wrongly reject a correct one.
# ======================================================================

# Italian words that can precede the main verb and should be skipped.
# Adverbs, conjunctions, negation, proclitic pronouns, common adverbials.
_PRE_VERB_SKIP = {
    # negation & emphasis
    "non", "mai", "mica", "nemmeno", "neanche", "neppure",
    # adverbs of time/frequency/manner
    "anche", "ancora", "appena", "comunque", "domani", "ieri", "oggi",
    "forse", "magari", "sempre", "spesso", "qui", "qua", "lì", "là",
    "presto", "tardi", "subito", "adesso", "ora", "prima", "dopo",
    "molto", "poco", "tanto", "troppo", "abbastanza", "raramente",
    "talvolta", "improvvisamente", "purtroppo", "fortunatamente",
    "veramente", "davvero", "proprio", "solo", "soltanto", "quasi",
    # conjunctions
    "e", "ma", "però", "quindi", "allora", "perciò", "infatti",
    "perché", "dunque", "cioè", "anzi", "tuttavia",
    # proclitic pronouns (don't change verb person)
    "mi", "ti", "ci", "vi", "si", "lo", "la", "li", "le", "gli", "ne",
    "me", "te", "se", "ce", "ve",
    "glielo", "gliela", "glieli", "gliele", "gliene",
    "melo", "mela", "meli", "mele", "mene",
    "telo", "tela", "teli", "tele", "tene",
    "celo", "cela", "celi", "cele", "cene",
    "velo", "vela", "veli", "vele", "vene",
    # explicit subject pronouns — they DO indicate person, but the verb
    # right after will also match, so skipping them is safe.
    "io", "tu", "lui", "lei", "egli", "ella", "esso", "essa",
    "noi", "voi", "loro", "essi", "esse",
    # common articles/prepositions/quantifiers that could start a sentence
    # but aren't verbs ("Il bambino mangia", "Tutti dormono")
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "un'", "l'",
    "del", "dello", "della", "dei", "degli", "delle", "dell'",
    "al", "allo", "alla", "ai", "agli", "alle", "all'",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle", "dall'",
    "nel", "nello", "nella", "nei", "negli", "nelle", "nell'",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle", "sull'",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "tutto", "tutti", "tutta", "tutte", "qualche", "alcuni", "alcune",
    "ogni", "molto", "molti", "molta", "molte",
    # interjections
    "ah", "oh", "ehi", "ehilà", "boh", "beh", "mah", "dai",
}

# Hard-coded high-frequency irregular forms (essere, avere, andare, fare,
# dire, dare, stare, sapere, volere, potere, dovere, venire, etc.).
# Maps Italian form → allowed English subjects.
_IRREGULAR_VERBS = {
    # essere (to be)
    "sono":    ("I", "They"),
    "sei":     ("You",),
    "è":       ("He", "She", "It", "He/She"),
    "siamo":   ("We",),
    "siete":   ("You",),
    # 'sono' is 3pl too — already covered above
    "ero":     ("I",),
    "eri":     ("You",),
    "era":     ("He", "She", "It", "He/She"),
    "eravamo": ("We",),
    "eravate": ("You",),
    "erano":   ("They",),
    "sarò":    ("I",),
    "sarai":   ("You",),
    "sarà":    ("He", "She", "It", "He/She"),
    "saremo":  ("We",),
    "sarete":  ("You",),
    "saranno": ("They",),
    "sia":     ("I", "He", "She", "It", "He/She"),  # 1sg, 2sg, 3sg congiuntivo
    "siano":   ("They",),
    "fossi":   ("I", "You"),  # 1sg or 2sg congiuntivo imperfetto
    "fosse":   ("He", "She", "It", "He/She"),
    "fossero": ("They",),
    "sarei":   ("I",),
    "saresti": ("You",),
    "sarebbe": ("He", "She", "It", "He/She"),
    "saremmo": ("We",),
    "sareste": ("You",),
    "sarebbero": ("They",),

    # avere (to have)
    "ho":      ("I",),
    "hai":     ("You",),
    "ha":      ("He", "She", "It", "He/She"),
    "abbiamo": ("We",),
    "avete":   ("You",),
    "hanno":   ("They",),
    "avevo":   ("I",),
    "avevi":   ("You",),
    "aveva":   ("He", "She", "It", "He/She"),
    "avevamo": ("We",),
    "avevate": ("You",),
    "avevano": ("They",),
    "avrò":    ("I",),
    "avrai":   ("You",),
    "avrà":    ("He", "She", "It", "He/She"),
    "avremo":  ("We",),
    "avrete":  ("You",),
    "avranno": ("They",),
    "abbia":   ("I", "He", "She", "It", "He/She"),
    "abbiano": ("They",),
    "avessi":  ("I", "You"),
    "avesse":  ("He", "She", "It", "He/She"),
    "avessero": ("They",),
    "avrei":   ("I",),
    "avresti": ("You",),
    "avrebbe": ("He", "She", "It", "He/She"),
    "avremmo": ("We",),
    "avreste": ("You",),
    "avrebbero": ("They",),

    # andare (to go) — irregular present
    "vado":    ("I",),
    "vai":     ("You",),
    "va":      ("He", "She", "It", "He/She"),
    "andiamo": ("We",),
    "andate":  ("You",),
    "vanno":   ("They",),

    # fare (to do/make) — irregular present
    "faccio":  ("I",),
    "fai":     ("You",),
    "fa":      ("He", "She", "It", "He/She"),
    "facciamo": ("We",),
    "fate":    ("You",),
    "fanno":   ("They",),

    # dire (to say)
    "dico":    ("I",),
    "dici":    ("You",),
    "dice":    ("He", "She", "It", "He/She"),
    "diciamo": ("We",),
    "dite":    ("You",),
    "dicono":  ("They",),

    # dare (to give)
    "do":      ("I",),
    "dai":     ("You",),  # NOTE: also "dai" = "come on!" or "from the"
    "dà":      ("He", "She", "It", "He/She"),
    "diamo":   ("We",),
    "date":    ("You",),
    "danno":   ("They",),

    # stare (to be/stay)
    "sto":     ("I",),
    "stai":    ("You",),
    "sta":     ("He", "She", "It", "He/She"),
    "stiamo":  ("We",),
    "state":   ("You",),
    "stanno":  ("They",),

    # sapere (to know)
    "so":      ("I",),
    "sai":     ("You",),
    "sa":      ("He", "She", "It", "He/She"),
    "sappiamo": ("We",),
    "sapete":  ("You",),
    "sanno":   ("They",),

    # volere (to want)
    "voglio":  ("I",),
    "vuoi":    ("You",),
    "vuole":   ("He", "She", "It", "He/She"),
    "vogliamo": ("We",),
    "volete":  ("You",),
    "vogliono": ("They",),

    # potere (can/be able)
    "posso":   ("I",),
    "puoi":    ("You",),
    "può":     ("He", "She", "It", "He/She"),
    "possiamo": ("We",),
    "potete":  ("You",),
    "possono": ("They",),

    # dovere (must/have to)
    "devo":    ("I",),
    "devi":    ("You",),
    "deve":    ("He", "She", "It", "He/She"),
    "dobbiamo": ("We",),
    "dovete":  ("You",),
    "devono":  ("They",),

    # venire (to come)
    "vengo":   ("I",),
    "vieni":   ("You",),
    "viene":   ("He", "She", "It", "He/She"),
    "veniamo": ("We",),
    "venite":  ("You",),
    "vengono": ("They",),

    # uscire (to go out)
    "esco":    ("I",),
    "esci":    ("You",),
    "esce":    ("He", "She", "It", "He/She"),
    "usciamo": ("We",),
    "uscite":  ("You",),
    "escono":  ("They",),

    # tenere (to hold/keep)
    "tengo":   ("I",),
    "tieni":   ("You",),
    "tiene":   ("He", "She", "It", "He/She"),
    "teniamo": ("We",),
    "tenete":  ("You",),
    "tengono": ("They",),

    # piacere (to like — usually 3sg/3pl)
    "piace":   ("He", "She", "It", "He/She"),
    "piacciono": ("They",),
}


def _looks_like_finite_verb(token: str):
    """
    Detect whether a token looks like a finite (conjugated) Italian verb
    based on its ending. Returns the allowed English subjects tuple,
    or None if the token isn't unambiguously a verb form.

    Designed to be CONSERVATIVE: returns None on anything ambiguous
    rather than risk a false positive. The LLM's prompt does the
    heavy lifting; this is the safety net for unambiguous cases.
    """
    if not token:
        return None
    tok = token.lower()

    # 1. Check the irregular-form lookup first — this catches short forms
    #    like 'è', 'ha', 'ho' that wouldn't pass length heuristics below.
    if tok in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[tok]

    # Past this point we need length >= 3 to apply ending heuristics safely.
    if len(tok) < 3:
        return None

    # 2. Generic verb-ending heuristics for regular conjugations.
    #    We're looking for endings that ONLY appear on verbs (not nouns
    #    or adjectives) and that unambiguously mark person.
    #
    #    The tricky thing: -o, -a, -i, -e are also noun/adjective endings,
    #    so we can't trust them alone. -iamo, -ate/-ete/-ite, -ano/-ono
    #    are nearly verb-exclusive and safe to use.

    # 1pl -iamo (parliamo, mangiamo, dormiamo, capiamo)
    if tok.endswith("iamo") and len(tok) >= 5:
        return ("We",)

    # 2pl -ate / -ete / -ite (verb 2pl present indicative)
    # NB: this also matches the past participle 'state', but 'state' is
    # in _IRREGULAR_VERBS so it's caught first.
    # Avoid matching obvious nouns/adjectives like 'estate' (summer) —
    # but those usually wouldn't start a clause as a verb anyway.
    if (tok.endswith(("ate", "ete", "ite"))
            and len(tok) >= 4
            and tok not in {"estate", "etichette", "polite", "elite", "limite"}):
        return ("You",)

    # 3pl -ano / -ono (parlano, mangiano, dormono, capiscono)
    if tok.endswith(("ano", "ono")) and len(tok) >= 4:
        # Exclude common non-verb words ending in -ano/-ono
        if tok not in {"piano", "umano", "lontano", "ano", "mano", "anno",
                       "soprano", "italiano", "americano", "africano",
                       "siciliano", "tono", "sono", "buono", "uomo"}:
            return ("They",)

    # 1sg congiuntivo / imperfetto endings -assi/-essi/-issi (be conservative)
    # Skip — too risky given short words.

    # 2sg imperfetto -avi/-evi/-ivi
    if tok.endswith(("avi", "evi", "ivi")) and len(tok) >= 4:
        return ("You",)

    # 1sg imperfetto -avo/-evo/-ivo (parlavo, leggevo, dormivo)
    if tok.endswith(("avo", "evo", "ivo")) and len(tok) >= 4:
        # Skip ambiguous cases — 'nuovo' (new) ends in -ovo but is adj
        if tok not in {"nuovo", "uovo", "ovvio"}:
            return ("I",)

    # 3sg imperfetto -ava/-eva/-iva (parlava, leggeva, dormiva)
    if tok.endswith(("ava", "eva", "iva")) and len(tok) >= 4:
        if tok not in {"nuova", "uova", "ovvia"}:
            return ("He", "She", "It", "He/She")

    # 1pl imperfetto -avamo/-evamo/-ivamo
    if tok.endswith(("avamo", "evamo", "ivamo")) and len(tok) >= 6:
        return ("We",)

    # 2pl imperfetto -avate/-evate/-ivate
    if tok.endswith(("avate", "evate", "ivate")) and len(tok) >= 6:
        return ("You",)

    # 3pl imperfetto -avano/-evano/-ivano
    if tok.endswith(("avano", "evano", "ivano")) and len(tok) >= 6:
        return ("They",)

    # Future tense -ò (1sg), -ai (2sg, but ambiguous with hai/dai), -à (3sg)
    # 1sg future: -erò/-irò (parlerò, dormirò)
    if tok.endswith(("erò", "irò", "arò")) and len(tok) >= 4:
        return ("I",)
    # 3sg future: -erà/-irà/-arà
    if tok.endswith(("erà", "irà", "arà")) and len(tok) >= 4:
        return ("He", "She", "It", "He/She")
    # 1pl future: -emo/-imo (parleremo, dormiremo)
    if tok.endswith(("eremo", "iremo", "aremo")) and len(tok) >= 5:
        return ("We",)
    # 3pl future: -anno (parleranno, dormiranno) — careful, 'anno' (year)
    # is a noun. Require length >= 7 to exclude 'anno'.
    if tok.endswith(("eranno", "iranno", "aranno")) and len(tok) >= 7:
        return ("They",)

    # Condizionale 1sg: -ei (parlerei, dormirei) — too short and ambiguous, skip
    # Condizionale 3sg: -ebbe — distinctive
    if tok.endswith("ebbe") and len(tok) >= 6:
        return ("He", "She", "It", "He/She")
    # Condizionale 3pl: -ebbero
    if tok.endswith("ebbero") and len(tok) >= 7:
        return ("They",)
    # Condizionale 1pl: -emmo
    if tok.endswith("emmo") and len(tok) >= 6:
        return ("We",)

    # Passato remoto 3sg -ò (parlò, dormì) — single-char ending, too risky to detect alone

    return None


def _find_main_verb_and_subjects(italian_text: str):
    """
    Scan the Italian sentence for the first finite verb that gives us
    unambiguous subject info. Skips negation, adverbs, conjunctions, and
    proclitic pronouns. Returns (verb_token, allowed_subjects) or
    (None, None) if no detectable finite verb is found in the first
    several tokens.
    """
    if not italian_text:
        return (None, None)
    import re as _re
    tokens_raw = _re.split(r"\s+", italian_text.strip())
    tokens = [t.lower().strip(".,;:!?\"'’()[]") for t in tokens_raw if t.strip()]

    # Look at the first 6 tokens — beyond that, the sentence is likely
    # complex enough that our heuristics aren't trustworthy.
    for tok in tokens[:6]:
        if not tok or tok in _PRE_VERB_SKIP:
            continue
        # Verb-noun ambiguous forms (vivo, canto, gioco, lavoro, etc.):
        # could be either a 1sg verb or a noun/adjective. Treat as
        # NOT a verb so we don't generate false positives.
        if tok in _VERB_NOUN_AMBIGUOUS:
            continue
        subjects = _looks_like_finite_verb(tok)
        if subjects is not None:
            return (tok, subjects)
        # If we hit a non-skip non-verb word, the sentence structure is
        # something like "Il bambino mangia" — already covered because
        # "il" is in _PRE_VERB_SKIP but "bambino" isn't a verb. We
        # continue scanning a couple more tokens to handle this.

    return (None, None)


def _english_starts_with_subject(english_text: str, allowed_subjects):
    """
    Check if the English string starts (after optional leading adverbs)
    with one of the allowed subject pronouns.

    Handles:
      - Contractions: "I'm", "You're", "He's", "They've"
      - Leading adverbs: "Tomorrow I will...", "Yesterday he ate..."
      - Capital and lowercase variants
      - Explicit noun subjects when the verb is 3rd person ("The dog
        barks", "My friend speaks", "Maria sings") — these are accepted
        for 3sg/3pl verbs because pronoun-replacement isn't required.
    """
    if not english_text:
        return False
    import re as _re
    eng = english_text.strip()

    # Strip leading adverbial phrases like "Tomorrow,", "Yesterday,",
    # "Now", "Often", "Sometimes", "Today" etc. These can prefix the
    # English without affecting subject agreement.
    leading_skip = (
        "tomorrow", "yesterday", "today", "now", "often", "sometimes",
        "always", "never", "soon", "later", "early", "perhaps", "maybe",
        "actually", "really", "quickly", "slowly", "here", "there",
        "in fact", "of course", "at home", "at school", "at work",
    )
    eng_lower = eng.lower()
    for prefix in leading_skip:
        if eng_lower.startswith(prefix + " ") or eng_lower.startswith(prefix + ","):
            eng = eng[len(prefix):].lstrip(" ,")
            break

    # Direct pronoun match
    for subj in allowed_subjects:
        # Match "I ", "I'm", "I've", "You're", "He's", "They've",
        # including straight and curly apostrophes.
        pattern = rf"^{_re.escape(subj)}\b['’]?\w*"
        if _re.match(pattern, eng, _re.IGNORECASE):
            return True

    # Fallback: if the allowed subjects are 3rd-person (He/She/It/They),
    # also accept any noun-phrase subject ("The dog barks", "My friend
    # speaks", "Maria sings"). We do this by checking that the first
    # token is NOT any English personal pronoun — if it's a real noun,
    # determiner, or proper name, pronoun-replacement isn't required.
    if any(s in ("He", "She", "It", "He/She", "They") for s in allowed_subjects):
        first_token = _re.split(r"\W+", eng, maxsplit=1)[0].lower()
        ALL_ENGLISH_PRONOUNS = {
            "i", "we", "you", "he", "she", "it", "they",
            "me", "us", "him", "her", "them",
            "my", "our", "your", "his", "their",  # possessive determiners — these DO precede nouns
        }
        # If the first token is a pronoun, the direct match above would
        # have caught it if it were valid. Anything else (noun, proper
        # name, "The", "A", "Maria", etc.) is treated as a valid 3rd-
        # person noun subject.
        # NOTE: possessive determiners like "My friend" CAN validly
        # introduce a 3rd-person subject. So we accept them too.
        POSSESSIVE_DETS = {"my", "our", "your", "his", "her", "their", "its"}
        if first_token in POSSESSIVE_DETS:
            return True
        if first_token and first_token not in ALL_ENGLISH_PRONOUNS:
            return True

    return False


def _verify_subject_agreement(italian_text: str, english_text: str):
    """
    Verify that the English translation's subject pronoun is consistent
    with the conjugation of the main Italian verb.

    Returns:
        (is_consistent, allowed_subjects_or_None, detected_verb_or_None)

    is_consistent is True iff:
      - We detected a clear finite verb in the Italian AND the English's
        leading subject matches what that verb allows, OR
      - We could NOT detect a finite verb (bail out — don't flag), OR
      - The sentence uses a piacere-family verb (inverted IT↔EN subject,
        so the check doesn't apply).
    """
    # Skip verification entirely for piacere-family verbs. These flip the
    # subject between Italian and English: 'Mi piace X' = 'I like X', where
    # X is the Italian subject but the English object. Our normal check
    # would wrongly flag "I like coffee" against piace (3sg).
    if _has_piacere_family_verb(italian_text):
        return (True, None, None)

    verb, allowed = _find_main_verb_and_subjects(italian_text)
    if not verb or not allowed:
        # No detectable verb — don't flag. The LLM might be right or
        # wrong; we just can't tell, so we let it through.
        return (True, None, None)
    if _english_starts_with_subject(english_text, allowed):
        return (True, allowed, verb)
    return (False, allowed, verb)


# Piacere-family verbs invert subject/object between Italian and English.
# When any of these forms appear, our normal subject-agreement check
# doesn't apply because the Italian subject becomes the English object.
_PIACERE_FAMILY = {
    # piacere
    "piace", "piacciono", "piaceva", "piacevano", "piacerà", "piaceranno",
    "piacque", "piacquero", "piaciuto", "piaciuta", "piaciuti", "piaciute",
    # mancare ("to be missing/lacking" — "mi manca casa" = "I miss home")
    "manca", "mancano", "mancava", "mancavano", "mancherà", "mancheranno",
    # servire ("to be needed" — "mi serve un libro" = "I need a book")
    "serve", "servono", "serviva", "servivano", "servirà", "serviranno",
    # bastare ("to be enough" — "mi basta così" = "that's enough for me")
    "basta", "bastano", "bastava", "bastavano", "basterà", "basteranno",
    # restare/rimanere ("to remain/be left" — used like piacere sometimes)
    # importare ("to matter" — "non mi importa" = "I don't care")
    "importa", "importano", "importava", "importeranno",
    # interessare ("to interest" — "mi interessa" = "I'm interested")
    "interessa", "interessano", "interessava", "interessavano",
    # dispiacere ("to be sorry" — "mi dispiace" = "I'm sorry")
    "dispiace", "dispiacciono", "dispiaceva",
    # occorrere ("to be needed")
    "occorre", "occorrono",
}


def _has_piacere_family_verb(italian_text: str):
    """True if the sentence contains a piacere-family verb form."""
    if not italian_text:
        return False
    import re as _re
    tokens = [t.lower().strip(".,;:!?\"'’()[]")
              for t in _re.split(r"\s+", italian_text.strip()) if t.strip()]
    return any(tok in _PIACERE_FAMILY for tok in tokens)


# Words that have an identical-spelling noun/adjective AND a 1sg verb form.
# When these appear, we can't tell from spelling alone which sense is meant,
# so the safe move is to NOT treat them as verbs at all.
#
# Common offenders:
#   vivo    = "I live" OR "alive/live"     (musica dal vivo = live music)
#   canto   = "I sing" OR "song/corner"
#   gioco   = "I play" OR "game"
#   lavoro  = "I work" OR "job/work"
#   aiuto   = "I help" OR "help" (noun)
#   bacio   = "I kiss" OR "a kiss"
#   sogno   = "I dream" OR "a dream"
#   pago    = "I pay" OR "pay" (noun, rare)
#   passo   = "I pass" OR "step"
#   tocco   = "I touch" OR "touch" (noun)
#   pranzo  = "I lunch" OR "lunch"
#   cena    = "I dine" (3sg actually) OR "dinner" — tricky
#   regalo  = "I gift" OR "a gift"
#   abito   = "I live/dwell" OR "outfit/dress"
#   conto   = "I count" OR "bill/account"
#   fumo    = "I smoke" OR "smoke"
#   gusto   = "I taste" OR "taste"
#   inizio  = "I begin" OR "beginning"
#   incontro = "I meet" OR "a meeting"
#   ritorno = "I return" OR "return" (noun)
#   saluto  = "I greet" OR "a greeting"
#   sguardo = (not a verb)
#   suono   = "I play (instrument)/sound" OR "sound"
#   tiro    = "I throw/pull" OR "throw/draw"
#   uso     = "I use" OR "use" (noun)
_VERB_NOUN_AMBIGUOUS = {
    "vivo", "canto", "gioco", "lavoro", "aiuto", "bacio", "sogno",
    "passo", "tocco", "pranzo", "regalo", "abito", "conto", "fumo",
    "gusto", "inizio", "incontro", "ritorno", "saluto", "suono",
    "tiro", "uso", "vino",  # vino isn't a verb but listed for safety
    "viaggio", "studio", "amo",  # amo could be hook OR "I love"
    "porto", "porto",  # I carry OR port
    "credo", "credo",  # I believe OR creed
    "dubbio",  # (only noun, listed for safety)
}


# ----------------------------------------------------------------------
# Per-category distractor playbooks — INJECTED into the prompt based on
# what kind of target word is being tested.
# ----------------------------------------------------------------------
DISTRACTOR_PLAYBOOKS = {
    "article": """
    TARGET CATEGORY: ARTICLE — this card trains ear for gender, number,
    and definiteness.

    Distractors MUST vary article-driven features while keeping the noun
    and verb identical:
      - Definite vs indefinite ('the book' vs 'a book')
      - Gender mismatch reflected in the English where natural
        (e.g. distractor implies wrong-gender article was used)
      - Singular vs plural ('the book' vs 'the books')
      - For articulated prepositions: vary which preposition contracted
        ('in the book' / 'of the book' / 'on the book')

    Example for target 'gli' (m. plural definite):
      Correct:    "The boys are eating pizza."           (i ragazzi... wait — gli before vowel/z/s+cons)
      Distractor: "A boy is eating pizza."               (un ragazzo)
      Distractor: "The boy is eating pizza."             (il ragazzo)
      Distractor: "Some boys are eating pizza."          (dei ragazzi)

    Do NOT swap the noun itself — the article is what's being tested.
    """,

    "preposition": """
    TARGET CATEGORY: PREPOSITION — train preposition selection, which is
    notoriously non-transferable from English.

    Distractors swap the preposition only:
      - 'I go TO Rome' / 'I go FROM Rome' / 'I'm IN Rome' / 'I'm AT Rome'
      - 'a book OF John' vs 'a book FOR John' vs 'a book FROM John'

    For articulated prepositions, vary the BASE preposition (nel vs sul
    vs dal vs col), not the embedded article. Do not change other content.
    """,

    "clitic": """
    TARGET CATEGORY: CLITIC PRONOUN — train ear for the tiny pronouns
    that swap object / reflexive / partitive meaning, and which Italians
    fire off at machine-gun speed.

    Distractors vary the clitic's referent or function:
      - Direct-object person: 'I see HIM' / 'I see HER' / 'I see THEM' (lo/la/li/le)
      - Partitive vs DO: 'I have SOME' (ne) vs 'I have IT' (lo)
      - Reflexive vs transitive: 'he washes HIMSELF' (si) vs 'he washes HIM' (lo)
      - Combined clitics: 'I give IT to HIM' (glielo) vs 'I give IT to HER' (gliela)
        vs 'I give SOME to HIM' (gliene)
      - Indirect vs direct: 'I speak TO HIM' (gli) vs 'I see HIM' (lo)

    Keep verbs, tense, and surrounding nouns identical. The clitic is
    the WHOLE point of the test.
    """,

    "verb_form": """
    TARGET CATEGORY: VERB FORM — train ear for person, tense, and mood,
    Italian's main grammatical battlefield.

    Pick the axis the target form actually exercises and vary ONE feature
    per distractor:

      - PERSON axis (if target distinguishes person):
          'I eat' / 'you (s.) eat' / 'he eats' / 'they eat'
          IMPORTANT: in Italian, pronouns are usually dropped, so person
          is encoded in the ending — this is exactly what we're testing.

      - TENSE axis:
          'I eat' (presente) / 'I ate' (passato prossimo) /
          'I was eating' (imperfetto) / 'I will eat' (futuro semplice) /
          'I had eaten' (trapassato prossimo)

      - MOOD axis:
          'I think he IS tired'    (indicativo, learner mistake)
          'I think he BE tired'    (congiuntivo, correct)
          'I think he WOULD BE'    (condizionale)
          'BE tired!'              (imperativo)
          If target is CONGIUNTIVO, at LEAST ONE distractor must be the
          INDICATIVO version of the same sentence — this is the single
          most common learner error.

      - AUXILIARY axis (passato prossimo):
          Hint at avere-vs-essere confusion through participle agreement
          mismatches in English when natural.

    Do not change nouns, adverbs, or sentence structure. Just the verb form.
    """,

    "time": """
    TARGET CATEGORY: TIME WORD — vary the time reference only.

    Distractors substitute the time word: yesterday / today / tomorrow,
    this morning / last night / next week, always / never / sometimes /
    rarely. Keep the verb tense IDENTICAL across all four options — this
    isolates the time-word as the learning target, which is the whole
    point. Do NOT swap nouns or verbs.

    Example for target 'domani':
      Correct:    "I'll see you tomorrow."
      Distractor: "I'll see you today."
      Distractor: "I'll see you yesterday."   (allow if grammatical in EN)
      Distractor: "I'll see you next week."
    """,

    "negation": """
    TARGET CATEGORY: NEGATION — vary negation scope and emphasis.

    Distractors:
      - The AFFIRMATIVE version ('I eat' vs 'I don't eat')
      - Strong negation (mica, neanche, nemmeno) vs simple non:
        'I don't eat AT ALL' vs 'I don't eat'
      - Negative pronoun swap: 'NOBODY came' vs 'NOTHING came' vs 'NEVER came'
      - Double-negation scope: 'I don't see anyone' vs 'I see nobody'

    Keep verb and object stable.
    """,

    "quantifier": """
    TARGET CATEGORY: QUANTIFIER — vary the quantity only.

    Numbers (three / thirteen / thirty), partitives (some / all / none /
    a few / many / too many), indefinites (every / each / any), and
    gender/number-agreed forms (molto vs molti vs molta vs molte where
    the English can reflect this). Keep everything else identical.
    """,

    "expression": """
    TARGET CATEGORY: SET EXPRESSION / IDIOM / FILLER — train recognition
    of Italian idioms and discourse markers whose literal meaning is
    misleading.

    Distractors:
      - The LITERAL (wrong) translation of the idiom
      - A semantically-adjacent idiom that means something different
      - A neutral paraphrase that loses the idiomatic register

    Example for target 'in bocca al lupo':
      Correct:    "Good luck!"
      Distractor: "In the wolf's mouth."     (literal trap)
      Distractor: "Watch out for the wolf!"  (literal-plausible)
      Distractor: "Break a leg!"             (right meaning, wrong idiom — sneaky)

    Example for target 'magari':
      Correct:    "I wish!" / "If only!"
      Distractor: "Maybe."                    (the other valid sense — context-dependent)
      Distractor: "Of course!"
      Distractor: "Never!"
    """,

    "content_word": """
    TARGET CATEGORY: CONTENT WORD (noun / verb / adjective).

    Build distractors on three axes (one each):
      1. NEAR-HOMOPHONE TRAP: swap the target for a phonetically similar
         Italian word — especially geminate-vs-single-consonant pairs:
           capelli/cappelli (hair/hats), pena/penna (pain/pen),
           sete/sette (thirst/seven), nono/nonno (ninth/grandfather),
           caro/carro (dear/cart), sono/sonno (I am/sleep).
      2. SEMANTIC NEIGHBOUR: a meaning-adjacent word (buy/sell, hot/cold,
         enter/leave, big/small).
      3. SECONDARY FEATURE: vary tense, negation, or quantifier of the
         surrounding sentence while keeping the target word.

    All four options must be grammatical English of similar length.
    """,

    "general": """
    TARGET CATEGORY: general — vary ONE specific feature per distractor:
    agreement (gender or number), tense, a single key noun/verb, or a
    quantifier. Do not stack changes.
    """,
}

# ======================================================================
# Universal rules appended to every prompt
# ======================================================================
UNIVERSAL_RULES = """
    UNIVERSAL DISTRACTOR RULES:
    1. Every distractor varies the TARGET word's learning axis (clitic→clitic,
       verb form→verb form, noun→that noun).
    2. BANNED: plain polarity flips (unless target IS negation), absurd
       sentences, pure synonym swaps, word-order-only changes.
    3. All four options: grammatical English, within 3 words of each other
       in length, plausible for someone who heard ~80% of the audio.
    4. If the sentence contains near-homophones (anno/hanno, a/ha, o/ho,
       e/è, la/là, capelli/cappelli, sete/sette, sono/sonno etc.),
       distractors must NOT differ from the correct answer ONLY on such a
       pair — audio can't disambiguate them.

    SUBJECT-VERB AGREEMENT (CRITICAL): Italian drops subject pronouns; the
    verb ending determines the English subject. sono=I/They, sei=You,
    è/ha=He/She/It, siamo=We, siete=You-all, hanno=They; endings -o=I,
    -i=You, -a/-e=He/She, -iamo=We, -ate/-ete/-ite=You-all, -ano/-ono=They.
    HARD RULES: "sono" is NEVER "He/She is". "è" is NEVER "I am" or "They
    are". "ha" is NEVER "They have". "hanno" is NEVER "He has".
    Before returning, re-check english_correct's subject against the main
    verb's conjugation; rewrite if mismatched. Hedge as "He/She" only for
    3sg verbs with no explicit subject.
"""


def generate_dictation_exercise(target_word_dict):
    """
    Generate one listening MCQ exercise for the given target Italian word/phrase.

    Returns a dict with:
      italian, pinyin (legacy alias for italian, kept for UI compat), english_correct,
      english_distractors, word_breakdown, grammar_point, expression_note, target_category
    """
    italian_text = (
        target_word_dict.get('italian')
        or target_word_dict.get('chinese')  # legacy fallback if mixed schema
        or target_word_dict.get('hanzi', '')
    )
    english = target_word_dict.get('english', '')
    hint = target_word_dict.get('hint', '') or ''

    # Word count, not character count — Italian is a word-language
    is_locked_phrase = len(italian_text.split()) > 1

    # Identify the learning target & which playbook applies
    target_category = _classify_target(italian_text, english, hint)
    playbook = DISTRACTOR_PLAYBOOKS.get(target_category, DISTRACTOR_PLAYBOOKS["general"])

    if is_locked_phrase:
        behavior_prompt = f"""
        LOCKED SENTENCE / PHRASE: '{italian_text}'
        Meaning: '{english}'
        {f"Hint: {hint}" if hint else ""}

        CRITICAL RULES FOR THIS ENTRY:
        1. DO NOT alter the Italian text. Analyse EXACTLY '{italian_text}' and nothing else.
        2. Generate the breakdown and translation for that exact phrase.
        3. Treat the entry as a fixed expression — do not "improve" it.
        """
        colloquialisms_in_play = _find_relevant_colloquialisms(italian_text)
    else:
        behavior_prompt = f"""
        Create ONE Italian sentence (5 to 12 words) that naturally uses the
        target word '{italian_text}' ({english}){f", hint: {hint}" if hint else ""}.

        - The sentence should be a real, plausible thing an Italian speaker
          would say.
        - Use natural Italian rhythm. Drop subject pronouns by default
          (Italian is pro-drop).
        - If the target is a verb form (e.g. congiuntivo or imperfetto),
          pick a context that REQUIRES that exact form — that's the point.
        """
        colloquialisms_in_play = _find_relevant_colloquialisms(italian_text)

    # Build colloquialism context if any are immediately relevant
    colloquialism_section = ""
    if colloquialisms_in_play:
        lines = "\n".join(
            f"   - {expr}: {meaning[1]}" for expr, meaning in colloquialisms_in_play.items()
        )
        colloquialism_section = f"""
    COLLOQUIAL EXPRESSION CONTEXT (translate idiomatically, NOT literally):
    The following expressions appear here and have non-literal meaning:
{lines}
    """
    # NOTE: the full ~20-entry colloquialism glossary used to be shipped in
    # EVERY prompt (~800+ tokens/call). Removed — the free tier is limited
    # by tokens/minute, and the glossary only matters when a colloquialism
    # is actually in play (handled above).

    prompt = f"""
    You are an expert Italian tutor designing a LISTENING comprehension
    multiple-choice question. The learner already speaks some Italian; this
    app trains their ear for native-pace Italian, with particular focus on
    gender/number agreement, verb conjugation (especially congiuntivo and
    passato prossimo vs imperfetto), clitic pronouns, articulated
    prepositions, and idiomatic discourse markers.

    {behavior_prompt}

    {colloquialism_section}

    GENERAL: use proper accents (never 'e' for 'è') and elision (l'amico,
    un'amica); no invented proper names (use generic subjects); write
    numerals as Italian words; drop subject pronouns where natural.

    ═══════════════════════════════════════════════════════════════════
    TARGET-AWARE DISTRACTOR DESIGN (THE MOST IMPORTANT SECTION)
    ═══════════════════════════════════════════════════════════════════

    This card was scheduled to teach the target: '{italian_text}' ("{english}")
    Target category detected: {target_category.upper()}

    {playbook}

    {UNIVERSAL_RULES}

    NOTES: 'grammar_point' names the construction (e.g. "congiuntivo
    presente after credo che"). 'expression_note' explains any
    colloquialism/idiom present (magari, dai, in bocca al lupo...); null
    if none.

    WORD BREAKDOWN: one entry per word: italian, contextual english, short
    note (gender/number or verb person+tense). MANDATORY: congiuntivo verb
    notes must name the person AND preempt indicativo confusion (e.g.
    'parli' after "penso che" → "3rd sg. congiuntivo — NOT indicativo 'tu
    parli'; io/tu/lui/lei all take -i"). Self-check every person/tense
    label against the sentence's subject. Optional 'stress' field ONLY for
    non-penultimate stress, e.g. "TE-le-fo-no".

    Output a raw JSON object EXACTLY in this format (no prose, no markdown):
    {{
        "italian": "<the sentence with proper accents>",
        "english_correct": "<accurate translation, hedging on dropped pronouns where needed>",
        "english_distractors": ["<dist1>", "<dist2>", "<dist3>"],
        "word_breakdown": [
            {{"italian": "casa", "english": "house", "note": "f. sing."}},
            {{"italian": "telefono", "english": "telephone", "note": "m. sing.", "stress": "TE-le-fo-no"}}
        ],
        "grammar_point": {{
            "structure": "<syntax pattern name>",
            "explanation": "<short explanation>"
        }},
        "expression_note": {{
            "expression": "<the colloquial expression if present, else null>",
            "explanation": "<emotional/pragmatic force, register>"
        }}
    }}
    """

    # Call the LLM with up to 3 attempts. If Groq's server-side json
    # validation rejects the output (400 json_validate_failed), the next
    # attempt runs WITHOUT response_format and parses client-side — that
    # path can never 400 on validation. Rate limits and auth fail fast.
    raw_data = None
    last_error_category = None
    last_error_desc = None
    use_json_mode = True
    for attempt in range(3):
        try:
            raw_data = _call_llm_json(prompt, use_json_mode=use_json_mode)
            break  # success
        except Exception as e:
            category, desc = _classify_llm_error(e)
            last_error_category, last_error_desc = category, desc
            logging.warning(
                f"Dictation generation attempt {attempt + 1}/3 failed "
                f"[{category}] (json_mode={use_json_mode}): {desc}"
            )
            if category == "auth":
                logging.error("Auth failure — not retrying. Check GROQ_API_KEY in Streamlit secrets.")
                break
            if category == "model_gone":
                logging.error(
                    "Model unavailable/retired — not retrying. "
                    "Update GENERATION_MODEL in config.py."
                )
                break
            if category == "rate_limit":
                # Don't waste retries — we'd just hit the same limit again.
                logging.warning("Rate-limited — not retrying. Caller should surface 'wait' message.")
                break
            if category == "json_parse":
                # Server rejected the JSON — retry in fallback mode where
                # we parse client-side instead.
                use_json_mode = False
            if attempt < 2:
                _wait_before_retry(category, attempt)

    if raw_data is None:
        # Before giving up, try a MINIMAL exercise that asks for very
        # little JSON — this can't truncate and is far more robust. Only
        # skip this fallback for rate limits / auth, where any further
        # call is pointless.
        if last_error_category not in ("rate_limit", "auth", "model_gone"):
            minimal = _generate_minimal_dictation(italian_text, english, hint)
            if minimal is not None:
                logging.info("Recovered via minimal-dictation fallback.")
                _record_last_error(None)
                return minimal
        logging.error(
            f"Dictation generation FAILED. Last error [{last_error_category}]: {last_error_desc}"
        )
        _record_last_error(last_error_category)
        return None

    # Clear the last-error slot on success
    _record_last_error(None)

    try:
        # --- SURGICAL LOCK for multi-word entries ---
        if is_locked_phrase:
            final_italian = italian_text
            final_english = english if english else raw_data.get("english_correct", "")
        else:
            final_italian = raw_data.get("italian", raw_data.get("hanzi", ""))
            final_english = raw_data.get("english_correct", "")

        # Normalise word breakdown
        word_breakdown = []
        for item in raw_data.get("word_breakdown", []):
            wb_entry = {
                "italian": item.get("italian", item.get("hanzi", item.get("chinese", ""))),
                "english": item.get("english", ""),
                "note":    item.get("note", ""),
            }
            if item.get("stress"):
                wb_entry["stress"] = item["stress"]
            word_breakdown.append(wb_entry)

        # Extract the translation fields up front so all the downstream
        # safety nets can reference them.
        english_correct = raw_data.get("english_correct", final_english)
        english_distractors = raw_data.get("english_distractors", [])

        # --- Verb/pronoun agreement safety net (bulletproof verifier) ---
        # Verifies that the English translation's subject pronoun matches the
        # person of the main Italian verb. Catches the LLM's most common
        # generation bug: e.g. labelling "Sono a casa" as "He/She is at home".
        #
        # SKIP this entirely for locked phrases (multi-word CSV entries like
        # "musica dal vivo", "in bocca al lupo") — those are fixed
        # expressions, not sentences with a subject.
        if is_locked_phrase:
            agreement_ok, allowed_subjects, detected_verb = (True, None, None)
        else:
            agreement_ok, allowed_subjects, detected_verb = _verify_subject_agreement(
                final_italian, english_correct
            )
        if not agreement_ok and allowed_subjects:
            logging.warning(
                f"Verb/subject mismatch detected: Italian verb '{detected_verb}' "
                f"requires English subject in {allowed_subjects}, but got: "
                f"'{english_correct}'."
            )
            # Filter distractors with the same mismatch so we don't show
            # multiple wrong-person options.
            english_distractors = [
                d for d in english_distractors
                if _english_starts_with_subject(d, allowed_subjects)
            ]

        # --- Spelling-trap post-processing safety net ---
        present_traps = _detect_spelling_traps(final_italian)

        # Hedge English pronouns on 3rd-person verbs without explicit subject.
        # Heuristic: starts with "He " or "She " and Italian has no Lui/Lei/Egli/Ella
        # at the start.
        italian_lower = final_italian.lower().strip()
        starts_with_pronoun_it = any(
            italian_lower.startswith(p + " ") for p in ("lui", "lei", "egli", "ella", "loro")
        )
        if not starts_with_pronoun_it:
            if english_correct.startswith("He ") and "She" not in english_correct.split()[0:3]:
                english_correct = "He/She " + english_correct[3:]
            elif english_correct.startswith("She ") and "He" not in english_correct.split()[0:3]:
                english_correct = "He/She " + english_correct[4:]

        # Drop any distractor that differs from the correct answer ONLY by a
        # known spelling-trap pair — that's unfair audio-wise.
        if present_traps:
            def _normalise_for_trap(s):
                s_low = s.lower()
                for a, b in present_traps:
                    s_low = s_low.replace(a, "X").replace(b, "X")
                return s_low.strip()
            correct_norm = _normalise_for_trap(english_correct)
            english_distractors = [
                d for d in english_distractors
                if _normalise_for_trap(d) != correct_norm
            ]

        exercise_data = {
            # NOTE: 'chinese' and 'pinyin' kept as aliases for any UI code
            # still expecting them — but 'italian' is the canonical key now.
            "italian": final_italian,
            "chinese": final_italian,   # alias for legacy UI references
            "pinyin":  final_italian,   # alias — "what to type" is the italian text
            "english_correct": english_correct,
            "english_distractors": english_distractors,
            "word_breakdown": word_breakdown,
            "grammar_point": raw_data.get("grammar_point", {}),
            "expression_note": raw_data.get("expression_note", {}),
            "target_category": target_category,  # exposed for debugging in UI
        }

        if len(exercise_data["english_distractors"]) < 3:
            logging.warning(
                f"Only {len(exercise_data['english_distractors'])} distractors after filtering "
                f"for sentence: {final_italian}"
            )

        return exercise_data

    except Exception as e:
        # We already succeeded at the LLM call; this is a post-processing
        # bug (likely in the verifier / breakdown normalisation / etc).
        # Log with the full type so we can see what to fix.
        logging.error(
            f"Post-processing error in generate_dictation_exercise: "
            f"{type(e).__name__}: {e}"
        )
        import traceback
        logging.error("Traceback:\n" + traceback.format_exc())
        return None


# ======================================================================
# CONJUGATION DRILL — for verb infinitives in the recall flow.
# ======================================================================
# Weighted (tense, person) combos. Weights bias toward forms English
# speakers struggle with: congiuntivo, condizionale, passato prossimo
# (with avere/essere choice), imperfetto. Presente gets baseline weight
# as periodic refresh.
CONJUGATION_FORMS = [
    # (tense, person, weight)
    ("presente",                 "io",       1),
    ("presente",                 "tu",       1),
    ("presente",                 "lui/lei",  1),
    ("presente",                 "noi",      1),
    ("presente",                 "voi",      1),
    ("presente",                 "loro",     1),

    ("passato prossimo",         "io",       3),
    ("passato prossimo",         "tu",       2),
    ("passato prossimo",         "lui/lei",  3),
    ("passato prossimo",         "noi",      2),
    ("passato prossimo",         "voi",      2),
    ("passato prossimo",         "loro",     3),

    ("imperfetto",               "io",       3),
    ("imperfetto",               "tu",       2),
    ("imperfetto",               "lui/lei",  3),
    ("imperfetto",               "noi",      2),
    ("imperfetto",               "voi",      2),
    ("imperfetto",               "loro",     3),

    ("trapassato prossimo",      "io",       1),
    ("trapassato prossimo",      "lui/lei",  1),

    ("futuro semplice",          "io",       2),
    ("futuro semplice",          "tu",       1),
    ("futuro semplice",          "lui/lei",  2),
    ("futuro semplice",          "noi",      1),
    ("futuro semplice",          "loro",     2),

    ("condizionale presente",    "io",       3),
    ("condizionale presente",    "tu",       2),
    ("condizionale presente",    "lui/lei",  3),
    ("condizionale presente",    "noi",      2),
    ("condizionale presente",    "loro",     2),

    ("condizionale passato",     "io",       2),
    ("condizionale passato",     "lui/lei",  2),
    ("condizionale passato",     "loro",     1),

    ("congiuntivo presente",     "io",       3),
    ("congiuntivo presente",     "tu",       3),
    ("congiuntivo presente",     "lui/lei",  4),  # most common (credo che + 3sg)
    ("congiuntivo presente",     "noi",      2),
    ("congiuntivo presente",     "loro",     3),

    ("congiuntivo imperfetto",   "io",       3),
    ("congiuntivo imperfetto",   "tu",       2),
    ("congiuntivo imperfetto",   "lui/lei",  3),
    ("congiuntivo imperfetto",   "loro",     2),

    ("imperativo (informale)",   "tu",       2),
    ("imperativo (formale Lei)", "lui/lei",  1),
    ("imperativo",               "voi",      1),
    ("imperativo negativo",      "tu",       2),  # non + infinitive
]


def _pick_conjugation_target():
    """Weighted random selection of (tense, person)."""
    import random as _random
    population = [(t, p) for t, p, _w in CONJUGATION_FORMS]
    weights    = [w for _t, _p, w in CONJUGATION_FORMS]
    return _random.choices(population, weights=weights, k=1)[0]


def _is_bare_infinitive(italian_text: str, hint: str = "") -> bool:
    """
    True iff the entry is a bare verb infinitive (-are/-ere/-ire), suitable
    for conjugation drilling. Excludes already-conjugated forms like
    'ho mangiato', 'mangiavo', 'sia', etc.
    """
    if not italian_text:
        return False
    word = italian_text.strip().lower()
    if " " in word:
        return False  # multi-word entries are phrases/idioms, not bare infinitives
    if not word.endswith(("are", "ere", "ire")):
        return False
    if len(word) < 4:
        return False
    # If the hint explicitly names a specific tense/form, this entry IS already
    # locked to that form (e.g. an entry called 'sia' with hint 'congiuntivo')
    # — but we don't reach here in that case because the word wouldn't end
    # in -are/-ere/-ire. Belt-and-braces: re-check anyway.
    hint_lc = (hint or "").lower()
    locked_form_signals = ("passato prossimo", "imperfetto", "congiuntivo",
                           "condizionale", "trapass", "futuro anteriore",
                           "imperativo")
    if any(sig in hint_lc for sig in locked_form_signals):
        return False
    return True


def _is_cloze_target_category(category: str) -> bool:
    """Categories where blanking the target word makes a useful cloze test."""
    return category in ("clitic", "article", "preposition")


def choose_recall_strategy(target_word_dict):
    """
    Decide which kind of recall exercise to generate for this card.
    Returns one of: 'conjugation', 'cloze', 'free'.

    Routing rules (probabilities live in config.py):
      - Bare verb infinitive               → conjugation drill (CONJUGATION_DRILL_PROB)
      - Clitic / article / articulated prep → cloze (CLOZE_DRILL_PROB)
      - Otherwise                          → free recall
    """
    import random as _random
    try:
        from config import CONJUGATION_DRILL_PROB, CLOZE_DRILL_PROB
    except ImportError:
        CONJUGATION_DRILL_PROB = 0.70
        CLOZE_DRILL_PROB       = 0.70

    italian_text = (target_word_dict.get('italian') or '').strip()
    english      = target_word_dict.get('english', '')
    hint         = target_word_dict.get('hint', '') or ''

    if _is_bare_infinitive(italian_text, hint):
        if _random.random() < CONJUGATION_DRILL_PROB:
            return 'conjugation'
        return 'free'

    category = _classify_target(italian_text, english, hint)
    if _is_cloze_target_category(category):
        if _random.random() < CLOZE_DRILL_PROB:
            return 'cloze'
        return 'free'

    return 'free'


def generate_conjugation_drill(target_word_dict, tense=None, person=None):
    """
    Generate a conjugation drill: a short English prompt that NATURALLY
    requires a specific (tense, person) form of the target verb infinitive.

    Returns a dict shaped to plug into the existing recall UI:
      {
        "exercise_type":     "conjugation",
        "infinitive":        "mangiare",
        "tense":             "condizionale passato",
        "person":            "io",
        "english_prompt":    "I would have eaten the whole pizza.",
        "italian":           "Avrei mangiato tutta la pizza.",   # full expected sentence
        "english_correct":   "I would have eaten the whole pizza.",
        "expected_form":     "avrei mangiato",                    # just the verb form
        "form_explanation":  "...",
        "word_breakdown":    [...],
        "grammar_point":     {...},
        "target_category":   "verb_form",
      }

    The generator includes a post-generation verifier that checks the
    Italian translation actually uses the target person, with correct
    agreement on adjectives and participles. If a mismatch is detected,
    it retries up to 2 more times with a sharper prompt before returning
    the best result available.

    Caller can pass explicit tense/person to deterministically pick one
    (useful for tests); otherwise weighted random selection is used.
    """
    infinitive = (target_word_dict.get('italian') or '').strip()
    english    = target_word_dict.get('english', '')

    if tense is None or person is None:
        tense, person = _pick_conjugation_target()

    # Try up to 2 times — first attempt normal, retry gets a sharper prompt
    # explaining what went wrong. Reduced from 3 to keep quota friendly.
    last_result = None
    for attempt in range(2):
        result = _generate_conjugation_drill_once(
            infinitive, english, tense, person, attempt=attempt,
            prior_error=(_describe_drill_error(last_result) if last_result else None),
        )
        if result is None:
            # If the once-call hit a rate limit, don't bother retrying
            # — we'd just hit the same wall.
            if get_last_error_category() == "rate_limit":
                return None
            continue
        last_result = result
        # Verify the generated Italian matches the target person
        is_consistent, issue = _verify_conjugation_drill(result, tense, person)
        if is_consistent:
            return result
        logging.warning(
            f"Conjugation drill attempt {attempt + 1} failed verification "
            f"(target {tense}, {person}, infinitive={infinitive!r}): {issue}. "
            f"Italian was: {result.get('italian')!r}."
        )

    # Verifier failed but we got SOME result — return it anyway so the user
    # gets a card. The grader has a "trust the English prompt" instruction
    # to handle the case where the expected Italian is buggy.
    if last_result is not None:
        logging.warning(
            f"Conjugation drill returning unverified result after 2 attempts. "
            f"Italian: {last_result.get('italian')!r}"
        )
    return last_result


def _generate_conjugation_drill_once(infinitive, english, tense, person, attempt=0, prior_error=None):
    """
    A single attempt at generating a conjugation drill via the LLM.
    Retries get an extra "you got it wrong because X — fix it" section
    at the top of the prompt.
    """
    # Person-specific examples to anchor the AI's output. These are the
    # single biggest lever we have: showing canonical examples for every
    # person × major-tense combination dramatically reduces the rate of
    # person/agreement mistakes.
    PERSON_AGREEMENT_NOTES = {
        "io":      "1st person singular. Verb ends in -o (or auxiliary 'ho/sono'). Adjectives/participles agree with the speaker's gender if known, else default to masculine singular (-o ending): 'io sono stanco' or 'io sono stanca'.",
        "tu":      "2nd person singular. Verb ends in -i (or auxiliary 'hai/sei'). Adjective: m.sg. -o or f.sg. -a depending on addressee.",
        "lui/lei": "3rd person singular. Verb ends in -a (-are) / -e (-ere, -ire) (or auxiliary 'ha/è'). Adjective: m.sg. -o (lui) or f.sg. -a (lei).",
        "noi":     "1st person plural. Verb ends in -iamo (or auxiliary 'abbiamo/siamo'). Adjective: m.pl. -i (mixed/male group) or f.pl. -e (all-female group). Example: 'noi siamo stanchi' (m.pl.) or 'noi siamo stanche' (f.pl.). NEVER -o or -a (singular).",
        "voi":     "2nd person plural (you all). Verb ends in -ate/-ete/-ite (or auxiliary 'avete/siete'). Adjective: m.pl. -i or f.pl. -e. Example: 'voi siete stanchi' or 'voi siete stanche'. NEVER -o or -a.",
        "loro":    "3rd person plural (they). Verb ends in -ano/-ono (or auxiliary 'hanno/sono'). Adjective: m.pl. -i or f.pl. -e. Example: 'loro sono stanchi' (m./mixed) or 'loro sono stanche' (all-female). NEVER -o or -a singular!",
    }
    agreement_note = PERSON_AGREEMENT_NOTES.get(person, "")

    # Build a sharper prompt on retry attempts
    retry_correction = ""
    if attempt > 0 and prior_error:
        retry_correction = f"""
    ⚠️ PREVIOUS ATTEMPT WAS WRONG:
    {prior_error}

    FIX IT THIS TIME. Re-read the agreement rules below carefully.
    """

    prompt = f"""
    You are an expert Italian tutor designing a CONJUGATION DRILL.
    {retry_correction}
    The learner is given an English sentence to translate. The translation
    MUST require a specific Italian verb form. Your job is to write a
    natural, short English sentence (5–10 words) that, when translated to
    Italian, REQUIRES the exact tense and person below.

    INFINITIVE:    {infinitive} ("{english}")
    TARGET TENSE:  {tense}
    TARGET PERSON: {person}

    PERSON-SPECIFIC AGREEMENT (CRITICAL):
    {agreement_note}

    HARD CONSTRAINTS:
    1. English must FORCE the target tense+person (congiuntivo: after
       "I think/doubt that…"; condizionale: "would…"; imperfetto:
       "was …-ing / used to"; passato prossimo: completed action;
       imperativo: a command).
    2. English subject matches person: io=I, tu=you(sg), lui/lei=he/she
       or singular noun, noi=we, voi=you all, loro=they or plural noun.
    3. Italian: verb in exact target tense+person; ALL adjectives &
       participles agree in number/gender — for noi/voi/loro NEVER end
       an adjective/participle in singular -o/-a (use -i or -e).
    4. Compound tenses: correct auxiliary (avere vs essere) + participle
       agreement. -ire verbs: mind -isco class (finire) vs plain (dormire).
    SELF-CHECK before returning: verb ending matches person; every
    adjective/participle agrees; plural subject ⇒ no singular -o/-a.

    Output ONLY valid JSON, no prose, no markdown:
    {{
        "english_prompt":   "<short English sentence to translate>",
        "italian":          "<full Italian translation with proper accents>",
        "expected_form":    "<just the conjugated form of {infinitive}, e.g. 'avrei mangiato'>",
        "form_explanation": "<one sentence naming the construction (e.g. 'condizionale passato, 1st pers. sing., avere as auxiliary')>",
        "word_breakdown": [
            {{"italian": "Avrei", "english": "I would have", "note": "1st sg. condizionale of avere (auxiliary)"}},
            {{"italian": "mangiato", "english": "eaten", "note": "past participle of mangiare"}}
        ],
        "grammar_point": {{
            "structure":   "<construction name>",
            "explanation": "<one sentence on why this English forces this Italian form>"
        }}
    }}
    """

    # Up to 3 sub-attempts for THIS call (separate from the outer retry
    # loop that retries on verifier failures). After a server-side json
    # validation failure, the next sub-attempt runs without response_format
    # and parses client-side (can never 400 on validation).
    raw_data = None
    last_error_category = None
    last_error_desc = None
    use_json_mode = True
    for sub_attempt in range(3):
        try:
            raw_data = _call_llm_json(prompt, use_json_mode=use_json_mode)
            break
        except Exception as e:
            category, desc = _classify_llm_error(e)
            last_error_category, last_error_desc = category, desc
            logging.warning(
                f"Conjugation drill API attempt {sub_attempt + 1}/3 failed "
                f"[{category}] (json_mode={use_json_mode}): {desc}"
            )
            if category == "auth":
                logging.error("Auth failure — not retrying. Check GROQ_API_KEY.")
                break
            if category == "model_gone":
                logging.error("Model unavailable/retired — not retrying.")
                break
            if category == "rate_limit":
                logging.warning("Rate-limited — not retrying.")
                break
            if category == "json_parse":
                use_json_mode = False
            if sub_attempt < 2:
                _wait_before_retry(category, sub_attempt)

    if raw_data is None:
        logging.error(
            f"Conjugation drill API call FAILED (outer attempt {attempt + 1}). "
            f"Last error [{last_error_category}]: {last_error_desc}"
        )
        _record_last_error(last_error_category)
        return None

    try:
        # Normalise word breakdown (same shape as free-recall exercises)
        word_breakdown = []
        for item in raw_data.get("word_breakdown", []):
            wb = {
                "italian": item.get("italian", ""),
                "english": item.get("english", ""),
                "note":    item.get("note", ""),
            }
            if item.get("stress"):
                wb["stress"] = item["stress"]
            word_breakdown.append(wb)

        # Clear last-error slot on success
        _record_last_error(None)
        return {
            "exercise_type":    "conjugation",
            "infinitive":       infinitive,
            "tense":             tense,
            "person":           person,
            "english_prompt":   raw_data.get("english_prompt", english),
            "italian":          raw_data.get("italian", ""),
            "english_correct":  raw_data.get("english_prompt", english),
            "expected_form":    raw_data.get("expected_form", ""),
            "form_explanation": raw_data.get("form_explanation", ""),
            "word_breakdown":   word_breakdown,
            "grammar_point":    raw_data.get("grammar_point", {}),
            "expression_note":  {},
            "english_distractors": [],   # unused in drills but kept for schema parity
            "target_category":  "verb_form",
        }
    except Exception as e:
        logging.error(
            f"Conjugation drill post-processing error (attempt {attempt + 1}): "
            f"{type(e).__name__}: {e}"
        )
        import traceback
        logging.error("Traceback:\n" + traceback.format_exc())
        return None


# ----------------------------------------------------------------------
# Agreement verifier for conjugation drills.
# ----------------------------------------------------------------------

# Map our internal person labels to the English subject pronouns that
# should appear (or be implied) in the English prompt.
_PERSON_TO_ENGLISH_SUBJECTS = {
    "io":      ("I",),
    "tu":      ("You",),
    "lui/lei": ("He", "She", "It", "He/She"),
    "noi":     ("We",),
    "voi":     ("You",),  # "you all" / "you guys" — still starts with "You"
    "loro":    ("They",),
}

# Map our internal person labels to the verb-ending sets expected in
# present-tense regular conjugations. (For irregulars we lean on the
# existing _IRREGULAR_VERBS table.)
_PERSON_TO_REGULAR_ENDINGS = {
    "io":      ("o",),                                # parlo, mangio, dormo, vivo
    "tu":      ("i",),                                # parli, mangi, dormi
    "lui/lei": ("a", "e"),                            # parla, legge, dorme
    "noi":     ("iamo",),                             # parliamo, leggiamo, dormiamo
    "voi":     ("ate", "ete", "ite"),                 # parlate, leggete, dormite
    "loro":    ("ano", "ono"),                        # parlano, leggono, dormono
}

# A "subject is plural" check — used to enforce that adjectives/participles
# can't be singular -o or -a alone.
_PLURAL_PERSONS = {"noi", "voi", "loro"}


def _verify_conjugation_drill(result, tense, person):
    """
    Verify that a generated conjugation drill is internally consistent.

    Returns (is_consistent, issue_description).
    issue_description is None if consistent, otherwise a short string
    explaining what's wrong (used to feed back into the retry prompt).

    Three categories of check:
      1. English subject matches target person
      2. Italian sentence contains a finite verb conjugated for target person
      3. Adjectives / participles in predicative position agree with target
         person's number (singular vs plural)
    """
    italian = (result.get("italian") or "").strip()
    english_prompt = (result.get("english_prompt") or "").strip()

    if not italian:
        return (False, "No Italian translation was provided.")

    # ---- Check 1: English subject ----
    expected_english_subjects = _PERSON_TO_ENGLISH_SUBJECTS.get(person)
    if expected_english_subjects:
        if not _english_starts_with_subject(english_prompt, expected_english_subjects):
            return (
                False,
                f"English prompt {english_prompt!r} doesn't start with a subject "
                f"appropriate for target person {person!r} "
                f"(expected one of: {expected_english_subjects}).",
            )

    # ---- Tokenise the Italian ----
    italian_tokens = [t.lower().strip(".,;:!?\"'’()[]")
                      for t in italian.split() if t.strip()]
    if not italian_tokens:
        return (False, "Italian translation has no tokens.")

    # ---- Check 2: Verb conjugation ----
    # Look for a verb that UNAMBIGUOUSLY matches the target person.
    # We trust two things:
    #   (a) Forms in _IRREGULAR_VERBS (essere, avere, andare, etc.)
    #   (b) Distinctive plural endings that can't be confused with nouns
    #       or adjectives: -iamo (1pl), -ate/-ete/-ite (2pl), -ano/-ono (3pl)
    #       PROVIDED the word isn't in the verb-noun-ambiguous list.
    #
    # For 2sg, 3sg, 1sg we lean on the irregular table + the expected_form
    # field, since their endings (-i, -a, -e, -o) overlap with nouns.
    target_english_subjects = _PERSON_TO_ENGLISH_SUBJECTS.get(person, ())

    DISTINCTIVE_PERSON_ENDINGS = {
        "noi":  ("iamo",),
        "voi":  ("ate", "ete", "ite"),
        "loro": ("ano", "ono"),
    }
    distinctive_endings = DISTINCTIVE_PERSON_ENDINGS.get(person)

    verb_person_ok = False
    wrong_person_verb = None  # for better error messages

    for tok in italian_tokens:
        # Skip past participles (-ato, -uto, -ito and feminine/plural variants):
        # they're not finite verbs and shouldn't be used to infer person.
        if _is_past_participle(tok):
            continue

        if tok in _IRREGULAR_VERBS:
            allowed = _IRREGULAR_VERBS[tok]
            if any(s in allowed for s in target_english_subjects):
                verb_person_ok = True
                break
            # Track the mismatched verb for diagnostic
            if not wrong_person_verb:
                wrong_person_verb = (tok, allowed)
            continue

        # Distinctive plural endings (only when target is plural)
        if distinctive_endings:
            if any(tok.endswith(e) for e in distinctive_endings) and len(tok) >= 4:
                if tok not in _VERB_NOUN_AMBIGUOUS:
                    verb_person_ok = True
                    break

    # If we have a wrong-person irregular verb and no right-person verb,
    # that's a hard failure.
    if not verb_person_ok and wrong_person_verb:
        verb, allowed = wrong_person_verb
        return (
            False,
            f"Italian uses verb {verb!r} (subject must be one of {allowed}), "
            f"but target person is {person!r} (needs subject "
            f"{target_english_subjects}). Wrong auxiliary or main verb.",
        )

    # If we couldn't find any matching verb, fall back to checking
    # expected_form (a more lenient check).
    expected_form = (result.get("expected_form") or "").strip().lower()
    if not verb_person_ok and expected_form:
        first_word = expected_form.split()[0] if expected_form else ""
        if first_word in _IRREGULAR_VERBS:
            allowed = _IRREGULAR_VERBS[first_word]
            if any(s in allowed for s in target_english_subjects):
                # The expected_form's first word is a valid auxiliary for
                # the target person. AND the form should appear in the Italian.
                if expected_form in italian.lower():
                    verb_person_ok = True

    if not verb_person_ok:
        # Couldn't confirm OR refute the verb. Don't fail just because we
        # don't recognise the conjugation pattern (could be a tense
        # we don't have rules for, like passato remoto).
        # Instead, mark as inconclusive and proceed to the adjective check.
        pass

    # ---- Check 3: Predicate adjective / participle agreement ----
    # Look specifically at the word AFTER a copula (essere / sembrare /
    # diventare / rimanere) — that's the predicate position where adjectives
    # must agree with the subject. This avoids false positives on object
    # nouns (Parlano italiano — italiano is OBJECT, not predicate).
    COPULAS_AND_AUX = {
        "sono", "sei", "è", "siamo", "siete",
        "era", "eri", "ero", "eravamo", "eravate", "erano",
        "sarò", "sarai", "sarà", "saremo", "sarete", "saranno",
        "sia", "siano", "fossi", "fosse", "fossero",
        "sembra", "sembrano", "sembravo", "sembrava", "sembravano",
        "diventa", "diventano", "diventato", "diventati",
        "rimane", "rimangono", "resta", "restano",
    }

    # Find a copula in the sentence
    copula_index = None
    for i, tok in enumerate(italian_tokens):
        if tok in COPULAS_AND_AUX:
            copula_index = i
            break

    if copula_index is not None and copula_index + 1 < len(italian_tokens):
        # Look at the predicate (word right after the copula, optionally
        # past a short adverb like 'molto' or 'così')
        SKIPPABLE_ADV = {"molto", "poco", "tanto", "troppo", "abbastanza",
                         "così", "davvero", "veramente", "proprio", "ancora",
                         "già", "sempre", "spesso", "non"}
        predicate_idx = copula_index + 1
        while (predicate_idx < len(italian_tokens) and
               italian_tokens[predicate_idx] in SKIPPABLE_ADV):
            predicate_idx += 1

        if predicate_idx < len(italian_tokens):
            predicate = italian_tokens[predicate_idx]
            # Strip articles in case structure is "Maria è la professoressa"
            # (where "la" precedes a noun, not adjective)
            if predicate in {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una"}:
                # Followed by a noun — not a predicate adjective. Skip check.
                pass
            elif len(predicate) >= 3:
                # Check agreement
                singular_ending = predicate.endswith(("o", "a"))
                plural_ending = predicate.endswith(("i", "e"))

                if person in _PLURAL_PERSONS and singular_ending:
                    if predicate not in _PREDICATE_NON_ADJECTIVE_WHITELIST:
                        return (
                            False,
                            f"Target person {person!r} is plural, but predicate "
                            f"{predicate!r} after copula {italian_tokens[copula_index]!r} "
                            f"ends in singular -o/-a. For plural subjects, use "
                            f"-i (m.pl.) or -e (f.pl.). Example: 'stanco' → 'stanchi'.",
                        )

                if person in _SINGULAR_PERSONS and plural_ending:
                    # Catch the inverse: "io sono stanchi" (1sg, plural adj)
                    # Be careful — many feminine singular adjectives end in -e
                    # (felice, grande, intelligente). Only flag if it ends in -i
                    # OR is clearly a plural-only form.
                    if predicate.endswith("i") and predicate not in _SINGULAR_ENDING_I_WHITELIST:
                        return (
                            False,
                            f"Target person {person!r} is singular, but predicate "
                            f"{predicate!r} after copula {italian_tokens[copula_index]!r} "
                            f"ends in -i (looks plural). For singular subjects, "
                            f"use -o (m.sg.), -a (f.sg.), or -e (m./f. sg. for words "
                            f"like 'felice', 'grande').",
                        )

    return (True, None)


# Helper sets used by the verifier
_SINGULAR_PERSONS = {"io", "tu", "lui/lei"}

# Words ending in -i that are ACTUALLY singular (not adjective plurals).
# Mostly prepositions, conjunctions, common nouns. Used to avoid false
# positives in the singular-target plural-adjective check.
_SINGULAR_ENDING_I_WHITELIST = {
    "i", "lì", "li", "qui", "sì", "tre",  # common short -i words
    "così", "perché", "poiché", "finché", "purché",
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato",
    "oggi", "ieri",
    # Many singular nouns end in -i:
    "caffè",  # well, ends in è, but listed for safety
    "tassi", "brindisi", "analisi", "crisi", "tesi", "ipotesi", "sintesi",
    "thai", "lui",
}

# Words ending in -o/-a that are NOT predicate adjectives (nouns, adverbs,
# prepositions). Used to avoid false positives in the plural-target
# singular-adjective check.
_PREDICATE_NON_ADJECTIVE_WHITELIST = {
    # adverbs and discourse particles
    "molto", "poco", "tanto", "troppo", "abbastanza", "anche", "ancora",
    "allora", "domani", "ieri", "oggi", "adesso", "ora", "subito",
    "presto", "tardi", "sempre", "mai", "spesso", "raramente",
    "fa", "via", "ecco",
    # very common nouns where the singular ending is fine even after copula
    # (e.g. "Maria è la professoressa" — but we already skip after articles)
    # Conservative: leave empty here. The verifier only fires after a copula,
    # before which we already filter out article+noun patterns.
}


# Detect past participles by ending. -ato/-ito/-uto (m.sg.), -ata/-ita/-uta
# (f.sg.), -ati/-iti/-uti (m.pl.), -ate/-ite/-ute (f.pl.).
# Plus some common irregular past participles.
_IRREGULAR_PAST_PARTICIPLES = {
    "fatto", "fatti", "fatta", "fatte",
    "detto", "detti", "detta", "dette",
    "letto", "letti", "letta", "lette",
    "scritto", "scritti", "scritta", "scritte",
    "visto", "visti", "vista", "viste",
    "preso", "presi", "presa", "prese",
    "messo", "messi", "messa", "messe",
    "rotto", "rotti", "rotta", "rotte",
    "aperto", "aperti", "aperta", "aperte",
    "chiuso", "chiusi", "chiusa", "chiuse",
    "venuto", "venuti", "venuta", "venute",
    "andato", "andati", "andata", "andate",
    "stato", "stati", "stata", "state",
    "morto", "morti", "morta", "morte",
    "nato", "nati", "nata", "nate",
    "rimasto", "rimasti", "rimasta", "rimaste",
    "vissuto", "vissuti", "vissuta", "vissute",
    "successo", "successi",
    "perso", "persi", "persa", "perse",
    "scelto", "scelti", "scelta", "scelte",
    "offerto", "offerti", "offerta", "offerte",
    "risposto", "risposti", "risposta", "risposte",
}


def _is_past_participle(tok: str) -> bool:
    """Return True if tok looks like a past participle (not a finite verb)."""
    if not tok or len(tok) < 4:
        return False
    if tok in _IRREGULAR_PAST_PARTICIPLES:
        return True
    # Regular endings: -ato, -uto, -ito (m.sg.) and their plurals/feminines.
    # Be strict about minimum length to avoid false positives.
    if tok.endswith(("ato", "uto", "ito")) and len(tok) >= 4:
        return True
    if tok.endswith(("ati", "uti", "iti", "ate", "ute", "ite", "ata", "uta", "ita")) and len(tok) >= 5:
        # But -ate/-ete/-ite is also the 2pl present ending! So we need to
        # distinguish. A past participle in -ate/-ite would only appear
        # after a form of essere (sono partite, sono arrivate, etc.).
        # Without context, treat anything in -ate/-ite/-ute as ambiguous
        # and call it a participle.
        # This means we won't catch a 2pl verb-only check, but we'll also
        # not misread participles as 2pl present-tense verbs.
        return True
    return False


def _describe_drill_error(result):
    """Generate a brief description of what was wrong with a previous
    drill attempt, for inclusion in a retry prompt."""
    if not result:
        return None
    return (
        f"Your previous attempt: Italian = {result.get('italian')!r}, "
        f"English prompt = {result.get('english_prompt')!r}. "
        f"This failed agreement verification — the verb's person OR the "
        f"adjective/participle agreement was wrong."
    )


def generate_cloze_exercise(target_word_dict):
    """
    Generate a cloze exercise: a normal dictation exercise, plus metadata
    about which word(s) to blank out. The learner speaks the missing
    word(s); Whisper transcribes; we grade just that word.

    Wraps generate_dictation_exercise and adds:
      "exercise_type":  "cloze"
      "cloze_blank":    "<the word being blanked>"
      "cloze_display":  "<Italian sentence with the target replaced by ___>"
    """
    base = generate_dictation_exercise(target_word_dict)
    if base is None:
        return None

    target_word = (target_word_dict.get('italian') or '').strip()
    sentence    = base.get("italian", "")

    # Locate the target word in the sentence (case-insensitive, whole-word match)
    import re as _re
    pattern = _re.compile(rf"(?<!\w){_re.escape(target_word)}(?!\w)", _re.IGNORECASE)
    match = pattern.search(sentence)

    if not match:
        # Fallback: the AI generated a sentence that doesn't contain the target
        # verbatim (Italian articles/preps can elide/contract — un'amica, dell',
        # etc.). Just return as free recall.
        logging.info(f"Cloze target '{target_word}' not found in sentence — falling back to free recall.")
        base["exercise_type"] = "free"
        return base

    # Build the blanked display, preserving the original casing context.
    matched_text = match.group(0)
    cloze_display = sentence[:match.start()] + "____" + sentence[match.end():]

    base.update({
        "exercise_type": "cloze",
        "cloze_blank":   matched_text,        # what the user must say
        "cloze_display": cloze_display,        # what to show with blank
    })
    return base
