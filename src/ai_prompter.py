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

    PRONOUN HEDGING:
    Italian drops subject pronouns. If the sentence has a 3rd-person singular
    verb with no explicit subject (parla, mangia, è, ha, ecc.), the English
    correct answer should hedge: "He/She speaks", not "He speaks", unless
    context makes it unambiguous.
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

        # --- Spelling-trap post-processing safety net ---
        present_traps = _detect_spelling_traps(final_italian)
        english_correct = raw_data.get("english_correct", final_english)
        english_distractors = raw_data.get("english_distractors", [])

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
