# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. DIRECTORY & FILE PATHS
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data"

VOCAB_CSV_PATH = DATA_DIR / "italian_vocab.csv"
DB_PATH = DATA_DIR / "user_progress.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. ENVIRONMENT VARIABLES (API KEYS)
# ==========================================
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LLM_API_KEY = os.getenv("LLM_API_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# 3. LANGUAGE SETTINGS
# ==========================================
LANGUAGE_CODE = "it"          # Whisper language hint
LANGUAGE_NAME = "Italian"     # Used in prompts

# ==========================================
# 4. SESSION SIZE
# ==========================================
MAX_REVIEWS_PER_DAY = 20  # Total cards in a session
NEW_WORDS_PER_DAY = 5     # Legacy — kept for backward compat

# ==========================================
# 5. SESSION COMPOSITION
# ==========================================
# How the daily batch is built. Must sum to 1.0.
#   RANDOM_BREADTH_PCT: random sample across the whole CSV (breadth)
#   SEQUENTIAL_PCT:     walk beginning-to-end through the CSV
#                       (review_count ASC, then id ASC — so unseen words
#                        come first, then least-reviewed; progresses over time)
RANDOM_BREADTH_PCT = 0.50
SEQUENTIAL_PCT = 0.50

# ==========================================
# 6. MODE MIX  (50/50 as requested)
# ==========================================
# Within a session, what proportion is each exercise type. Must sum to 1.0.
#   LISTENING_PCT: hear audio → type Italian → MCQ English
#   RECALL_PCT:    see English → speak Italian → graded by Whisper+LLM
LISTENING_PCT = 0.50
RECALL_PCT = 0.50

# ==========================================
# 7. RECALL SUB-MODE ROUTING
# ==========================================
# Within recall, smart routing by target category. These are the
# probabilities of routing to the specialised sub-mode (vs. free recall)
# when the target qualifies.
#
#   - Verb infinitives  → conjugation drill (specific tense+person)
#   - Clitics/articles/articulated preps → cloze (blank the target word)
CONJUGATION_DRILL_PROB = 0.70
CLOZE_DRILL_PROB       = 0.70

# ==========================================
# 7. AI MODELS
# ==========================================
# NOTE: llama-3.3-70b-versatile was DEPRECATED by Groq on 17 June 2026.
# qwen/qwen3.6-27b is currently the strongest model on Groq (and notably
# good multilingually — important for Italian grammar quality).
# Alternative if qwen has issues: "openai/gpt-oss-120b".
GENERATION_MODEL = "qwen/qwen3.6-27b"
GRADING_MODEL = "qwen/qwen3.6-27b"
WHISPER_MODEL = "whisper-large-v3"

# ==========================================
# 8. SRS MULTIPLIERS
# ==========================================
EASY_MULTIPLIER = 2.5
GOOD_MULTIPLIER = 1.5
HARD_MULTIPLIER = 1.2

# ==========================================
# 9. MASTERY & FLUENCY
# ==========================================
# Interval (in days) pushed for the "Already Mastered" override button.
# Must be >= 21 to register as mastered in get_progress_stats().
MASTERED_INTERVAL_DAYS = 365

# Target vocabulary size for the fluency progress indicator.
# Conventional rough threshold for functional fluency in a language.
FLUENCY_TARGET = 10000

# ==========================================
# 10. BREATH PAUSE (spaced consolidation)
# ==========================================
# Every N cards, the app shows a brief guided box-breathing pause.
# Helps with retention (short rests between learning episodes aid
# consolidation) AND keeps Groq's per-minute rate limit happy.
# Set BREATH_PAUSE_EVERY = 0 to disable.
BREATH_PAUSE_EVERY    = 8     # Show a breath every 8 graded cards
BREATH_PAUSE_SECONDS  = 30    # Duration of each pause
