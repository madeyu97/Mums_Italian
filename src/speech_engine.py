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
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        resp = client.audio.transcriptions.create(
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

For conjugation drills, GRAMMAR is the PRIMARY axis. Be STRICT about:
  - Using the exact target tense (don't accept presente when imperfetto
    was required, don't accept indicativo when congiuntivo was required)
  - Correct PERSON ending (-o, -i, -a, -iamo, -ate, -ano, etc.)
  - For compound tenses: correct AUXILIARY (avere vs essere) and correct
    past-participle agreement
  - Irregular forms (e.g. fatto not facuto, detto not dicato, andato/a
    with essere)
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
Grade three criteria on a 0-10 integer scale:

1. VOCAB - did they produce the right words?
   Be LENIENT on near-homophones Whisper routinely mishears:
     - "anno" / "hanno"  (year / they have)
     - "a" / "ha"        (to / has)
     - "o" / "ho"        (or / I have)
     - "ai" / "hai"      (to-the / you have)
     - "e" / "è"         (and / is)
     - "se" / "sé"       (if / oneself)
     - "la" / "là"       (the / there)
   If meaning is preserved, full credit. Penalise genuinely wrong word
   choice that changes meaning.

2. GRAMMAR - check ALL of the following where relevant:
     - Gender agreement (il/la, un/una, -o/-a endings, past-participle
       agreement with essere or with preceding direct object)
     - Number agreement across article + noun + adjective
     - Verb conjugation: right PERSON, TENSE, and MOOD
         * Especially congiuntivo where required (after credo che, penso che,
           voglio che, è necessario che, etc.)
         * passato prossimo vs imperfetto distinction
         * auxiliary choice (avere vs essere) and past-participle agreement
     - Articulated prepositions (nel / dal / sul / della / etc.)
     - Clitic pronoun choice and placement (mi/ti/ci/ne/lo/la, combos like
       glielo, ce ne, me lo)
   A sentence that is grammatical but means something different from the
   target should score LOWER here, not in vocab.

3. PRONUNCIATION (proxy) - inferred from Whisper transcription fidelity:
     9-10: Whisper transcribed the expected words exactly or near-exactly
     6-8 : Most words right, a few errors
     3-5 : Whisper produced a partly-different sentence
     0-2 : Whisper produced garbled / empty output
   IMPORTANT: This score is INDIRECT. It CANNOT assess:
     - Open vs closed vowels (è/é, ò/ó)
     - Geminate (double) consonants crispness (anno vs ano, pala vs palla)
     - Italian rhythm and vowel quality
   State this caveat explicitly in your feedback.

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
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GRADING_MODEL,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
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
