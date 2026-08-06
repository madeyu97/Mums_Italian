# src/speech_engine.py
"""
Handles the recall side of the app: transcribe with Whisper, grade with LLM.
Italian-aware: knows about gender/number agreement, conjugation traps,
near-homophone confusables (anno/hanno, e/è, a/ha), and clitic pronouns.
"""

import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

from config import WHISPER_MODEL, GRADING_MODEL, LANGUAGE_CODE

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


def transcribe_audio(audio_bytes, filename="speech.webm"):
    """
    Send recorded audio to Groq's Whisper.
    Returns: {"text": str, "language": str, "duration": float} or None on failure.
    """
    if not audio_bytes:
        logging.error("transcribe_audio received empty bytes.")
        return None
    try:
        resp = _get_client().audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=WHISPER_MODEL,
            language=LANGUAGE_CODE,           # "it" — pinned by config
            response_format="verbose_json",
            temperature=0.0,
        )
        return {
            "text": (resp.text or "").strip(),
            "language": getattr(resp, "language", LANGUAGE_CODE),
            "duration": getattr(resp, "duration", 0.0),
        }
    except Exception as e:
        logging.error(f"Whisper transcription failed: {e}")
        return None


def grade_speech(
    expected_italian,
    expected_english,
    transcribed_text,
    hint="",
    drill_context=None,
):
    """
    Ask the LLM to grade vocab, grammar, and (proxy) pronunciation
    for an Italian utterance.

    drill_context (optional dict) lets the caller add focused context:
      For conjugation drills:
        {"type": "conjugation", "tense": "...", "person": "...",
         "infinitive": "...", "expected_form": "..."}
      For cloze drills:
        {"type": "cloze", "target_word": "...",
         "full_sentence": "...", "blanked_display": "..."}
    """
    hint_line = f"  Hint:     {hint}\n" if hint else ""

    # Build drill-specific addendum (or empty string for free recall)
    drill_section = ""
    if drill_context and drill_context.get("type") == "conjugation":
        drill_section = f"""
DRILL CONTEXT — THIS IS A CONJUGATION DRILL:
  Infinitive:       {drill_context.get('infinitive', '')}
  Target tense:     {drill_context.get('tense', '')}
  Target person:    {drill_context.get('person', '')}
  Expected form:    {drill_context.get('expected_form', '')}
  ENGLISH PROMPT THE LEARNER WAS GIVEN: {drill_context.get('english_prompt', expected_english)!r}

For conjugation drills, GRAMMAR is the PRIMARY axis. Be STRICT about:
  - Using the exact target tense (don't accept presente when imperfetto
    was required, don't accept indicativo when congiuntivo was required)
  - Correct PERSON ending (-o, -i, -a, -iamo, -ate, -ano, etc.)
  - For compound tenses: correct AUXILIARY (avere vs essere) and correct
    past-participle agreement
  - For PLURAL persons (noi, voi, loro), adjectives and participles MUST
    be plural too (-i or -e), NOT singular (-o or -a).
  - Irregular forms (e.g. fatto not facuto, detto not dicato, andato/a
    with essere)

CRITICAL — GRADE AGAINST THE ENGLISH PROMPT, NOT JUST THE EXPECTED ITALIAN:
The "expected Italian" sentence above was AI-generated and may itself contain
errors. The ENGLISH PROMPT the learner was given is the ground truth.
If the learner's spoken Italian is a CORRECT translation of the English prompt
(with proper agreement for the target person), give them FULL CREDIT even if
their answer differs from the "expected Italian" shown.

For example:
  English prompt: "They are tired."
  Expected Italian (possibly wrong): "Sono stanco."
  Learner said:   "Sono stanchi."
  → Learner is CORRECT. "Sono stanchi" is the right translation; the
    expected Italian had a gender/number agreement bug.
  → Give full credit. Mention in feedback that their answer is correct.

If the learner produced the right MEANING but wrong tense/person,
GRAMMAR should be low (3-5/10). Vocab can still be high since they
picked the right verb.
"""
    elif drill_context and drill_context.get("type") == "cloze":
        drill_section = f"""
DRILL CONTEXT — THIS IS A CLOZE (FILL-IN-THE-BLANK):
  Full sentence:     {drill_context.get('full_sentence', '')}
  Blanked display:   {drill_context.get('blanked_display', '')}
  Expected word:     {drill_context.get('target_word', '')}

The learner is speaking ONLY the missing word, not the whole sentence.
Grade purely on whether they produced the target word correctly.
Be lenient on surrounding noise (Whisper may pick up breath/fillers).
GRAMMAR score for cloze ≈ VOCAB score — they're the same axis here.
"""

    prompt = f"""
You are a strict but encouraging Italian tutor grading a student's
spoken attempt. They were asked to say a specific Italian sentence; below
is what it was supposed to be, and what Whisper transcribed.

EXPECTED SENTENCE
  Italian:  {expected_italian}
  Meaning:  {expected_english}
{hint_line}
WHISPER TRANSCRIPTION OF STUDENT'S SPEECH
  {transcribed_text or "(empty - Whisper heard nothing intelligible)"}
{drill_section}
GROUND RULE — GRADE AGAINST WHAT THE LEARNER WAS ACTUALLY ASKED FOR:
The learner was shown ONLY the English meaning above and asked to say it in
Italian. The "Expected sentence / Italian" line is one AI-generated model
answer, and it is sometimes NOT a faithful translation of that English — it
may contain extra details (a time, place, or weather phrase) the English
never mentioned, or omit something the English did mention.

Therefore: if the learner's Italian is a correct translation of the ENGLISH
MEANING shown above, award full marks — even where it differs from the model
answer. NEVER penalise the learner for omitting content that does not appear
in the English they were given.
  Example: English shown: "The city is beautiful."
           Model answer:  "Stasera la città è bella."
           Learner said:  "La città è bella."
           → CORRECT. Full marks. "Stasera" was never asked for; say so
             briefly in the feedback rather than marking it as an omission.
Mark down only genuine errors in what WAS asked for: wrong gender/number
agreement, wrong tense or person, wrong vocabulary, wrong clitic choice.

Grade three criteria, each 0-10:

1. VOCAB: right words? LENIENT on Whisper near-homophones (anno/hanno,
   a/ha, o/ho, ai/hai, e/è, se/sé, la/là) — full credit if meaning
   preserved. Penalise genuinely wrong word choice.

2. GRAMMAR: gender & number agreement, verb person/tense/mood (esp.
   congiuntivo after penso/credo che; passato prossimo vs imperfetto;
   avere-vs-essere + participle agreement), articulated prepositions,
   clitic choice/placement. Grammatical-but-different-meaning scores
   low HERE, not in vocab.

3. PRONUNCIATION (proxy from Whisper fidelity): 9-10 near-exact,
   6-8 mostly right, 3-5 partly different, 0-2 garbled. This is
   INDIRECT — it cannot judge open/closed vowels or double consonants;
   say so in feedback.

Then map the overall performance to an SRS grade:
    "again" = effectively failed
    "hard"  = struggled but the gist was there
    "good"  = solid attempt with minor issues
    "easy"  = essentially perfect

Return ONLY a JSON object, no prose around it:
{{
  "vocab_score": <int 0-10>,
  "grammar_score": <int 0-10>,
  "pronunciation_score": <int 0-10>,
  "overall_grade": "again" | "hard" | "good" | "easy",
  "feedback": "<2-4 sentences, specific and actionable. Call out any agreement, conjugation, or clitic errors by name. Acknowledge the pronunciation score is indirect and cannot grade vowel quality or double consonants.>"
}}
""".strip()

    try:
        try:
            from config import REASONING_EFFORT, MAX_GRADE_TOKENS
        except ImportError:
            REASONING_EFFORT, MAX_GRADE_TOKENS = "low", 1024
        grade_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "model": GRADING_MODEL,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_completion_tokens": MAX_GRADE_TOKENS,
        }
        if REASONING_EFFORT and GRADING_MODEL.startswith("openai/gpt-oss"):
            grade_kwargs["reasoning_effort"] = REASONING_EFFORT
        resp = _get_client().chat.completions.create(**grade_kwargs)
        data = json.loads(resp.choices[0].message.content)
        for k in ("vocab_score", "grammar_score", "pronunciation_score"):
            data[k] = max(0, min(10, int(data.get(k, 0))))
        if data.get("overall_grade") not in ("again", "hard", "good", "easy"):
            avg = (data["vocab_score"] + data["grammar_score"] + data["pronunciation_score"]) / 3
            data["overall_grade"] = (
                "again" if avg < 3 else
                "hard" if avg < 6 else
                "good" if avg < 8.5 else
                "easy"
            )
        return data
    except Exception as e:
        logging.error(f"Speech grading failed: {e}")
        return None


GRADE_MAP = {"again": 0, "hard": 1, "good": 2, "easy": 3}
