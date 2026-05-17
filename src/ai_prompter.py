# src/ai_prompter.py

import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

from config import GENERATION_MODEL

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    UNIVERSAL DISTRACTOR RULES (apply on top of the category playbook above):

    1. ALL distractors must attack the TARGET word's learning axis, not random
       other words in the sentence. If the target is a clitic, vary the clitic.
       If it's a verb form, vary verb form. If it's a noun, vary that noun.

    2. NO TRIVIAL DISTRACTORS — BANNED:
       - Sentences with directly opposite polarity (e.g. "He came" vs "He didn't come")
         UNLESS the target IS a negation.
       - Sentences that are grammatically impossible or absurd
       - Pure synonym swaps ("purchase" vs "buy")
       - Word-order-only differences ("gave him the book" vs "gave the book to him")

    3. ALL four options must be grammatical English of roughly equal length
       (within 3 words of each other).

    4. PLAUSIBILITY TEST: Could a learner who heard ~80% of the audio correctly
       genuinely pick this distractor? If not, rewrite it.

    SPELLING-TRAP / NEAR-HOMOPHONE HANDLING:

    If the Italian sentence contains any of:
      - anno/hanno, a/ha, o/ho, ai/hai  (h-drop in avere)
      - e/è, se/sé, ne/né, la/là, li/lì, si/sì, da/dà  (accent-only)
      - capelli/cappelli, pena/penna, sete/sette, nono/nonno, caro/carro,
        sono/sonno, casa/cassa, note/notte, pala/palla, copia/coppia
        (geminate consonants)
    ...then DISTRACTORS MUST NOT differ from the correct answer ONLY on
    one of these pairs. That would be unfair, since Whisper-quality audio
    cannot reliably disambiguate them.

    PRONOUN HEDGING / SUBJECT-VERB AGREEMENT (CRITICAL — read carefully):

    Italian drops subject pronouns, so the verb ending tells you the person.
    The English translation MUST match the person of the Italian verb.

    Reference table of avere / essere endings — ALL OTHER VERBS FOLLOW THE
    SAME PERSON PATTERN:
      io      sono / ho           → "I am"  / "I have"
      tu      sei  / hai          → "You are" / "You have"
      lui/lei è    / ha           → "He/She is" / "He/She has"
      noi     siamo / abbiamo     → "We are" / "We have"
      voi     siete / avete       → "You all are" / "You all have"
      loro    sono / hanno        → "They are" / "They have"

    Generic verb endings:
      -o   (parlo, mangio, dormo)        → "I"
      -i   (parli, mangi, dormi)         → "You (singular)"
      -a / -e  (parla, mangia, dorme)    → "He/She"
      -iamo  (parliamo, mangiamo)        → "We"
      -ate / -ete / -ite                 → "You all"
      -ano / -ono  (parlano, mangiano,
                    dormono, SONO)       → "They"

    HARD RULES:
    1. If the Italian verb is "sono" — the English MUST be "I am" OR
       "They are", NEVER "He is" or "She is" or "He/She is". (The same
       form serves both 1sg and 3pl; pick whichever the sentence context
       makes natural.)
    2. If the Italian verb is "è" — the English MUST be "He is" / "She is"
       / "He/She is" or "It is" depending on subject. Never "I am" or
       "They are".
    3. If the Italian verb is "ha" — "He/She has" or "It has", never
       "I have" or "They have".
    4. If the Italian verb is "hanno" — "They have", never "He has".
    5. Before returning, RE-READ your "english_correct" field and verify
       its subject pronoun matches the conjugation of the main Italian
       verb. If they don't match, REWRITE the english_correct.

    Only when the Italian sentence has a 3rd-person singular verb (è, ha,
    parla, mangia, dorme, etc.) with NO explicit subject should you hedge
    as "He/She". For 1st and 2nd person verbs there's nothing to hedge —
    the verb ending makes it unambiguous.
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

    # Always include the full glossary so the model has reference
    full_colloquialism_reference = "ITALIAN COLLOQUIAL GLOSSARY (reference for natural phrasing):\n"
    for expr, (literal, real) in ITALIAN_COLLOQUIALISMS.items():
        full_colloquialism_reference += f"   {expr}: literal = {literal}; actual usage = {real}\n"

    prompt = f"""
    You are an expert Italian tutor designing a LISTENING comprehension
    multiple-choice question. The learner already speaks some Italian; this
    app trains their ear for native-pace Italian, with particular focus on
    gender/number agreement, verb conjugation (especially congiuntivo and
    passato prossimo vs imperfetto), clitic pronouns, articulated
    prepositions, and idiomatic discourse markers.

    {behavior_prompt}

    {colloquialism_section}

    {full_colloquialism_reference}

    GENERAL INSTRUCTIONS:
    1. ACCENT MARKS MATTER: use proper Italian accents (à, è, é, ì, ò, ù).
       Never write 'e' when you mean 'è'.
    2. APOSTROPHES: use proper elision (l'amico, un'amica, dell'acqua, etc.)
       when the next word begins with a vowel.
    3. NO HALLUCINATED CONTEXT: don't invent random proper names. Use
       generic subjects ("un amico", "la mia famiglia") if needed.
    4. NUMERAL CONVERSION: if the target contains Arabic numerals, write
       them as Italian words.
    5. PRO-DROP: drop subject pronouns where natural. This is what makes
       Italian audio hard for learners — embrace it.

    ═══════════════════════════════════════════════════════════════════
    TARGET-AWARE DISTRACTOR DESIGN (THE MOST IMPORTANT SECTION)
    ═══════════════════════════════════════════════════════════════════

    This card was scheduled to teach the target: '{italian_text}' ("{english}")
    Target category detected: {target_category.upper()}

    {playbook}

    {UNIVERSAL_RULES}

    GRAMMAR AND EXPRESSION NOTES:
    Provide TWO teaching notes in the output:
      1. 'grammar_point': pure SYNTAX — name the construction (e.g.
         "passato prossimo with avere", "congiuntivo presente after credo che",
         "articulated preposition in + il = nel", "ne as partitive pronoun").
      2. 'expression_note': if the sentence contains a colloquialism, filler,
         idiom, or discourse marker (magari, dai, allora, cioè, in bocca al lupo,
         non vedo l'ora, etc.), explain its pragmatic/emotional force.
         Return null ONLY if no such expression is present.

    WORD BREAKDOWN:
    Break the sentence into word units. For each word, give:
      - the Italian word
      - its English meaning IN THIS CONTEXT
      - a short note: gender (m./f.), number (sing./pl.), or verb info
        (e.g. "1st pers. sing. passato prossimo") if useful.
      - OPTIONAL 'stress' field for words with NON-DEFAULT (non-penultimate) stress.
        Default stress in Italian is penultimate, so OMIT this field for normal words.
        Include ONLY for sdrucciole (antepenultimate stress) and bisdrucciole.
        Format: capitalise the stressed syllable, e.g. "TE-le-fo-no", "À-bi-to",
        "PRÀ-ti-co", "VEN-do-no", "MAN-gia-no", "DI-co-no". Words written with
        a final written accent (città, perché, papà) are unambiguous — no need.

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

    try:
        response = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=GENERATION_MODEL,
            response_format={"type": "json_object"}
        )

        raw_json_str = response.choices[0].message.content
        raw_data = json.loads(raw_json_str)

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
        logging.error(f"Generation Error via Groq: {e}")
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

    Caller can pass explicit tense/person to deterministically pick one
    (useful for tests); otherwise weighted random selection is used.
    """
    infinitive = (target_word_dict.get('italian') or '').strip()
    english    = target_word_dict.get('english', '')

    if tense is None or person is None:
        tense, person = _pick_conjugation_target()

    prompt = f"""
    You are an expert Italian tutor designing a CONJUGATION DRILL.

    The learner is given an English sentence to translate. The translation
    MUST require a specific Italian verb form. Your job is to write a
    natural, short English sentence (5–10 words) that, when translated to
    Italian, REQUIRES the exact tense and person below.

    INFINITIVE:    {infinitive} ("{english}")
    TARGET TENSE:  {tense}
    TARGET PERSON: {person}

    HARD CONSTRAINTS:
    1. The English sentence MUST naturally force exactly the target tense
       and person. For congiuntivo, frame it after a trigger like
       "I think that...", "I doubt that...", "Although...", "It's important
       that...", "If only...".
       For condizionale, use "would..." / "I would have...".
       For imperfetto, use "I was eating", "I used to eat", "while I was...",
       background-description framing.
       For passato prossimo, use a completed action with a specific moment.
       For imperativo, use a command ("Eat your vegetables!").

    2. The English sentence should be natural and useful — something a
       learner would actually want to say.

    3. The Italian translation must use {infinitive} (or its correct
       conjugated form) as its main verb.

    4. For compound tenses (passato prossimo, trapassato, condizionale
       passato), pick the correct auxiliary (avere vs essere) AND apply
       past-participle agreement where required.

    5. For -ire verbs, distinguish -isco verbs (finire, capire, preferire)
       from non-isco verbs (dormire, partire, sentire).

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

    try:
        response = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=GENERATION_MODEL,
            response_format={"type": "json_object"}
        )
        raw_data = json.loads(response.choices[0].message.content)

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

        return {
            "exercise_type":    "conjugation",
            "infinitive":       infinitive,
            "tense":            tense,
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
        logging.error(f"Conjugation drill generation failed: {e}")
        return None


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
