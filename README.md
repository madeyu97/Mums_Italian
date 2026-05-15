# Italian Immersion Study

A fork of the Pinyin Immersion app, adapted for learning Italian.

## What changed vs the Mandarin version (initial port)

### Schema
- `vocab_progress` columns are now `italian, english, hint` (was `chinese, pinyin, english`)
- `italian_vocab.csv` replaces `vocab_export.csv` and has columns: `Italian, English, Hint`
- The `Hint` column is optional and seeds the AI generator (e.g. "subj.", "passato prossimo", "f. pl.")

### Session mix
- 50% listening, 50% recall
- **Sequential 50% + random 50%** — walks the CSV from beginning to end as you progress, plus random breadth

### Listening flow
- "Type what you hear (in Italian)" — spelling comparison flags missing accents / double consonants
- Optional slow audio (-25% rate) on demand

### Recall flow — smart routing by target
- **Verb infinitives → Conjugation Drill** (70% of the time)
- **Clitics / articles / articulated prepositions → Cloze** (70% of the time)
- **Everything else → Free Recall** (the original behaviour)

### AI prompter
- `ITALIAN_COLLOQUIALISMS` glossary (magari, dai, boh, allora, in bocca al lupo, etc.)
- `ITALIAN_SPELLING_TRAPS` covers accent pairs, h-drop pairs, geminate-consonant pairs
- Target categories: `article`, `preposition`, `clitic`, `verb_form`, `time`, `negation`, `quantifier`, `expression`, `content_word`, `general`
- Category-specific distractor playbooks
- JSON output uses `italian` and `expression_note`
- **Stress markers** (`TE-le-fo-no`, `À-bi-to`) shown in breakdown for non-default stress

### Audio engine
- 11 Italian voices, accent-character filter, optional `slow=True` for -25% rate

### UI
- Dictionary links: WordReference, Reverso Context, Treccani

---

## v2 features (just added)

### 1. Conjugation Drill

When a recall card's target is a **bare verb infinitive** (`mangiare`, `bere`, `dormire`, etc.), there's a 70% chance it becomes a conjugation drill instead of free recall.

- A weighted random `(tense, person)` combo is picked. Weights bias toward learner hurdles:
  - **Imperfetto** ~17% · **Passato prossimo** ~16%
  - **Congiuntivo presente** ~14% · **Congiuntivo imperfetto** ~12%
  - **Condizionale presente** ~12%
  - Presente only ~5%; trapassato/imperativo round out the rest
- The AI writes a short English sentence that **naturally forces** that exact tense + person:
  - `mangiare` + condizionale passato + io → *"I would have eaten the whole pizza"*
  - `andare` + congiuntivo presente + lui/lei → *"I think he goes to Rome on Fridays"*
- The user speaks the full Italian translation
- Grader is **strict on tense, person, auxiliary, and past-participle agreement** (new `drill_context` on `grade_speech`)
- Expected form (e.g. `avrei mangiato`) shown in solution view, plus a construction note

Entries that are *already specific forms* (`ho mangiato`, `mangiavo`, `sia`) are correctly excluded from drilling — those go to free recall.

### 2. Cloze mode

When a recall card's target is a **clitic, article, or articulated preposition**, there's a 70% chance it becomes a cloze:

- AI generates a normal sentence around the target word
- The target is blanked: *"____ dico domani"* with translation *"I'll tell it to him tomorrow"* — answer: `glielo`
- User speaks **only the missing word**
- Grader is told it's a cloze (vocab = grammar axis, lenient on surrounding noise)
- Optional "🔊 Hear the full sentence first" expander as audio crutch

### 3. Stress-aware display

AI prompt asks for an optional `stress` field on words with non-default stress:
- `telefono` → `TE-le-fo-no`
- `abito` → `À-bi-to`
- `prendono` → `PREN-do-no`

Shown with a 🔤 prefix in the word's expander when present. Omitted for normal penultimate-stress words.

### 4. Sequential CSV selection

The 50% sequential portion uses `ORDER BY review_count ASC, id ASC`:
- Unseen words come first, **in your CSV order** (put priority stuff at the top)
- Once everything's been seen at least once, least-reviewed words fill the queue
- You progress forward through the CSV automatically; the random 50% provides breadth

---

## Config knobs (in `config.py`)

```python
LISTENING_PCT = 0.50      # listen/recall split
RECALL_PCT    = 0.50

RANDOM_BREADTH_PCT = 0.50  # within session composition
SEQUENTIAL_PCT     = 0.50

CONJUGATION_DRILL_PROB = 0.70   # P(conjugation drill | bare infinitive)
CLOZE_DRILL_PROB       = 0.70   # P(cloze | clitic/article/articulated prep)
```

Set either drill prob to `0.0` to revert to free recall.

---

## File map

```
italian_app/
├── data/
│   └── italian_vocab.csv       # 70+ starter words across every category
├── src/
│   ├── ai_prompter.py          # generate_dictation_exercise
│   │                           # + generate_conjugation_drill
│   │                           # + generate_cloze_exercise
│   │                           # + choose_recall_strategy, CONJUGATION_FORMS
│   ├── audio_engine.py         # Italian voices + accent filter + slow mode
│   ├── config.py               # paths, mix percentages, drill probabilities
│   ├── db_manager.py           # italian/english/hint schema; sequential walk
│   ├── main_app.py             # UI — routes recall to drill/cloze/free
│   ├── speech_engine.py        # Whisper "it" + drill-aware grading
│   └── srs_engine.py           # unchanged
└── requirements.txt
```

## Running

```bash
cd src
streamlit run main_app.py
```

Env vars: `GROQ_API_KEY`, `DATABASE_URL`.

## Future ideas

- **Bucket tags**: a `tags` column to filter sessions toward specific tenses/themes
- **Form-coverage tracking**: log which `(verb, tense, person)` cells have been drilled and bias toward unseen ones
- **Stress audio emphasis**: TTS can be coaxed with SSML to stress non-default syllables more clearly
