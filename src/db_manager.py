# src/db_manager.py

import os
import pandas as pd
from datetime import datetime, date
import logging
import psycopg2
import psycopg2.extras
import streamlit as st

from config import (
    VOCAB_CSV_PATH,
    MAX_REVIEWS_PER_DAY,
    NEW_WORDS_PER_DAY,
    RANDOM_BREADTH_PCT,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def classify_db_error(exc):
    """
    Turn a database exception into (category, headline, remedy_steps).

    Categories:
      'paused'    - Supabase project asleep/removed (pooler doesn't know the tenant)
      'missing'   - DATABASE_URL not configured
      'auth'      - wrong password / role
      'network'   - DNS, refused, unreachable
      'timeout'   - connect timed out
      'unknown'   - anything else

    This exists so the app can show a human explanation with exact fix
    steps instead of a Python traceback.
    """
    s = str(exc)
    low = s.lower()

    if "database_url is missing" in low:
        return (
            "missing",
            "The database address isn't configured.",
            [
                "Open share.streamlit.io and find this app.",
                "Click ⋮ → Settings → Secrets.",
                'Add a line: DATABASE_URL = "postgresql://..." '
                "(copy it from Supabase → Project Settings → Database → Connection string → Session pooler).",
                "Save, then reboot the app.",
            ],
        )

    # The signature of a paused (or deleted) Supabase project: the pooler
    # no longer recognises the project reference.
    if ("tenant or user not found" in low
            or "tenant/user" in low and "not found" in low
            or "enotfound" in low):
        return (
            "paused",
            "Your Supabase database is paused (or no longer exists).",
            [
                "Go to supabase.com and sign in.",
                "Open the project for this app.",
                'If it shows as paused, click "Restore project" and wait ~1-2 minutes.',
                "Come back here and press Try again.",
                "Free-tier Supabase projects pause after about a week of inactivity — "
                "the keep-alive workflow in this repo is designed to prevent that.",
            ],
        )

    if ("password authentication failed" in low
            or "role" in low and "does not exist" in low):
        return (
            "auth",
            "The database rejected the username or password.",
            [
                "In Supabase: Project Settings → Database → reset or copy the password.",
                "In Streamlit: ⋮ → Settings → Secrets → update DATABASE_URL with the correct password.",
                "Passwords with special characters must be URL-encoded.",
                "Save, then reboot the app.",
            ],
        )

    if "timeout" in low or "timed out" in low:
        return (
            "timeout",
            "The database didn't respond in time.",
            [
                "This is usually temporary — press Try again.",
                "If it keeps happening, check status.supabase.com for an outage.",
            ],
        )

    if ("could not translate host name" in low
            or "name or service not known" in low
            or "connection refused" in low
            or "could not connect" in low):
        return (
            "network",
            "Couldn't reach the database server.",
            [
                "Check status.supabase.com for an outage.",
                "Verify the host in DATABASE_URL matches your Supabase connection string.",
                "Press Try again in a minute.",
            ],
        )

    return (
        "unknown",
        "The database connection failed.",
        [
            "Press Try again — some failures are temporary.",
            "If it persists, check that the Supabase project is running and "
            "that DATABASE_URL in Streamlit Secrets is current.",
        ],
    )


def _read_setting(name):
    """
    Read a setting from Streamlit secrets, falling back to environment
    variables.

    NOTE: `name in st.secrets` RAISES (not returns False) when no
    secrets.toml exists, which previously made the environment-variable
    fallback unreachable and turned a clear "missing setting" into an
    opaque "No secrets found" error. Hence the try/except.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def get_connection():
    """Establishes a connection to the Supabase PostgreSQL database."""
    db_url = _read_setting("DATABASE_URL")
    if not db_url:
        raise ValueError("CRITICAL ERROR: DATABASE_URL is missing! Streamlit cannot find it in the Secrets menu.")
    # connect_timeout stops the app hanging for minutes when Supabase is
    # paused (free tier pauses after ~1 week idle) or unreachable.
    # keepalives stop long-idle connections being silently dropped by the
    # pooler mid-session (a past source of mysterious mid-card failures).
    return psycopg2.connect(
        db_url,
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vocab_progress (
            id SERIAL PRIMARY KEY,
            italian TEXT NOT NULL,
            english TEXT NOT NULL,
            hint TEXT DEFAULT '',
            date_added TEXT NOT NULL,
            next_review_date TEXT NOT NULL,
            interval INTEGER DEFAULT 0,
            ease_factor REAL DEFAULT 2.5,
            review_count INTEGER DEFAULT 0,
            priority_weight INTEGER DEFAULT 1
        )
    ''')
    # Ensure a unique constraint on 'italian' so batched ON CONFLICT works.
    # Wrapped in a DO block so it's a no-op if the constraint already exists
    # (older versions of Postgres don't support IF NOT EXISTS on constraints).
    cursor.execute('''
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'vocab_progress_italian_key'
            ) THEN
                ALTER TABLE vocab_progress
                ADD CONSTRAINT vocab_progress_italian_key UNIQUE (italian);
            END IF;
        END
        $$;
    ''')
    conn.commit()
    conn.close()
    logging.info("Supabase database initialized successfully.")

def import_vocab_from_csv():
    """
    Expects italian_vocab.csv with columns:
        Italian, English, Hint (optional)

    FAST PATH: before doing any real work, compare the DB row count to
    the CSV line count. If the DB already has at least as many rows,
    skip the import entirely — no pandas read, no 10k-row INSERT shipped
    to Supabase. This turns every warm boot from seconds of dead time
    into a single COUNT query.

    Performs a single batched INSERT with ON CONFLICT DO NOTHING when an
    import IS needed, so duplicates are skipped at the database level.
    """
    if not VOCAB_CSV_PATH.exists():
        logging.warning("CSV file not found. Skipping import.")
        return

    # --- Fast path: cheap line count vs DB count ---
    try:
        with open(VOCAB_CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
            csv_line_count = sum(1 for _ in f) - 1  # minus header
    except Exception:
        csv_line_count = None

    if csv_line_count is not None and csv_line_count > 0:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vocab_progress")
            db_count = cursor.fetchone()[0]
            conn.close()
            if db_count >= csv_line_count:
                logging.info(
                    f"CSV import skipped: DB has {db_count} rows, "
                    f"CSV has {csv_line_count} lines. Nothing new."
                )
                return
        except Exception as e:
            # Table might not exist yet on first boot — fall through to import
            logging.info(f"Fast-path count check failed ({e}); running full import.")

    df = pd.read_csv(VOCAB_CSV_PATH)

    # Normalise columns — accept case variations
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c.lower() == "italian":
            rename_map[c] = "Italian"
        elif c.lower() == "english":
            rename_map[c] = "English"
        elif c.lower() == "hint":
            rename_map[c] = "Hint"
    df = df.rename(columns=rename_map)

    if "Hint" not in df.columns:
        df["Hint"] = ""

    df['Italian'] = df['Italian'].astype(str).str.strip()
    df['English'] = df['English'].astype(str).str.strip()
    df['Hint']    = df['Hint'].astype(str).str.strip().replace('nan', '')

    df = df.replace('', pd.NA).dropna(subset=['Italian', 'English'])
    df['Hint'] = df['Hint'].fillna('')

    if df.empty:
        logging.info("CSV had no valid rows after cleaning. Nothing to import.")
        return

    today_str = date.today().isoformat()
    rows = [
        (row['Italian'], row['English'], row['Hint'], today_str, today_str)
        for _, row in df.iterrows()
    ]

    # Count current rows so we can report how many were actually new.
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM vocab_progress")
    before_count = cursor.fetchone()[0]

    # Single batched insert; relies on the UNIQUE constraint on italian
    # added in init_db(). ON CONFLICT DO NOTHING skips existing rows.
    psycopg2.extras.execute_values(
        cursor,
        '''
        INSERT INTO vocab_progress
            (italian, english, hint, date_added, next_review_date)
        VALUES %s
        ON CONFLICT (italian) DO NOTHING
        ''',
        rows,
        page_size=1000,
    )
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM vocab_progress")
    after_count = cursor.fetchone()[0]
    conn.close()

    new_words_added = after_count - before_count
    skipped = len(rows) - new_words_added
    if new_words_added > 0:
        logging.info(f"Imported {new_words_added} new words to Supabase.")
    if skipped > 0:
        logging.info(f"Skipped {skipped} rows already in database.")

def flag_word_in_database(italian_word):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE vocab_progress SET priority_weight = priority_weight + 10 WHERE italian = %s
    ''', (italian_word,))
    conn.commit()
    conn.close()


def get_session_words(total=MAX_REVIEWS_PER_DAY, random_pct=RANDOM_BREADTH_PCT):
    """
    Build a session of `total` cards composed of:
      - `random_pct` random across the whole vocabulary (breadth)
      - the rest walked sequentially from the start of the CSV
        (review_count ASC, then id ASC — so unseen words come first
         in CSV order, then least-reviewed words fill in afterwards)
    Deduplicated and shuffled.
    """
    import random as _random

    random_count = int(round(total * random_pct))
    sequential_count = total - random_count

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Sequential portion: walk from the front of the CSV.
    # Unseen first (review_count = 0, id ASC), then least-reviewed.
    cursor.execute('''
        SELECT * FROM vocab_progress
        ORDER BY review_count ASC, id ASC
        LIMIT %s
    ''', (sequential_count,))
    sequential_rows = [dict(r) for r in cursor.fetchall()]
    sequential_ids = [r['id'] for r in sequential_rows]

    # Random portion: random sample, excluding what we've already picked.
    if sequential_ids:
        placeholders = ','.join(['%s'] * len(sequential_ids))
        cursor.execute(f'''
            SELECT * FROM vocab_progress
            WHERE id NOT IN ({placeholders})
            ORDER BY RANDOM() LIMIT %s
        ''', sequential_ids + [random_count])
    else:
        cursor.execute('SELECT * FROM vocab_progress ORDER BY RANDOM() LIMIT %s', (random_count,))
    random_rows = [dict(r) for r in cursor.fetchall()]

    conn.close()
    session = sequential_rows + random_rows
    _random.shuffle(session)
    return session


def get_due_words():
    """Legacy SRS-based fetcher. Kept for fallback."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    today_str = date.today().isoformat()
    cursor.execute('''
        SELECT * FROM vocab_progress
        WHERE review_count > 0 AND next_review_date <= %s
        ORDER BY priority_weight DESC, next_review_date ASC
    ''', (today_str,))
    due_reviews = [dict(row) for row in cursor.fetchall()]
    needed_new_words = MAX_REVIEWS_PER_DAY - len(due_reviews)
    if needed_new_words > 0:
        cursor.execute('''
            SELECT * FROM vocab_progress
            WHERE review_count = 0
            ORDER BY priority_weight DESC, id DESC LIMIT %s
        ''', (needed_new_words,))
        new_words = [dict(row) for row in cursor.fetchall()]
    else:
        new_words = []
    conn.close()
    return (due_reviews + new_words)[:MAX_REVIEWS_PER_DAY]


def update_word_progress(word_id, next_review_date, new_interval, new_ease):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE vocab_progress
        SET next_review_date = %s, interval = %s, ease_factor = %s,
            review_count = review_count + 1,
            priority_weight = GREATEST(1, priority_weight - 2)
        WHERE id = %s
    ''', (next_review_date, new_interval, new_ease, word_id))
    conn.commit()
    conn.close()

def get_progress_stats():
    """Single-query stats (was 4 separate COUNTs = 4 round trips)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*)                                        AS total,
            COUNT(*) FILTER (WHERE review_count = 0)        AS unseen,
            COUNT(*) FILTER (WHERE interval >= 21)          AS mastered
        FROM vocab_progress
    """)
    total_words, unseen, mastered = cursor.fetchone()
    conn.close()
    if total_words == 0:
        return {"unseen": 0, "learning": 0, "mastered": 0, "total": 0}
    learning = total_words - unseen - mastered
    return {"unseen": unseen, "learning": learning, "mastered": mastered, "total": total_words}

def undo_word_progress(word_id, old_next_review_date, old_interval, old_ease, old_review_count, old_priority):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE vocab_progress
        SET next_review_date = %s, interval = %s, ease_factor = %s,
            review_count = %s, priority_weight = %s
        WHERE id = %s
    ''', (old_next_review_date, old_interval, old_ease, old_review_count, old_priority, word_id))
    conn.commit()
    conn.close()

def get_more_words(exclude_ids, amount=5):
    """Same beginning-to-end + random composition, excluding what's already been seen this session."""
    import random as _random
    if not exclude_ids:
        exclude_ids = [-1]

    random_count = int(round(amount * RANDOM_BREADTH_PCT))
    sequential_count = amount - random_count

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    exclude_placeholders = ','.join(['%s'] * len(exclude_ids))
    cursor.execute(f'''
        SELECT * FROM vocab_progress
        WHERE id NOT IN ({exclude_placeholders})
        ORDER BY review_count ASC, id ASC LIMIT %s
    ''', exclude_ids + [sequential_count])
    sequential_rows = [dict(r) for r in cursor.fetchall()]

    all_excluded = exclude_ids + [r['id'] for r in sequential_rows]
    all_placeholders = ','.join(['%s'] * len(all_excluded))
    cursor.execute(f'''
        SELECT * FROM vocab_progress
        WHERE id NOT IN ({all_placeholders})
        ORDER BY RANDOM() LIMIT %s
    ''', all_excluded + [random_count])
    random_rows = [dict(r) for r in cursor.fetchall()]

    conn.close()
    extra = sequential_rows + random_rows
    _random.shuffle(extra)
    return extra[:amount]

def delete_word_from_db(word_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM vocab_progress WHERE id = %s", (word_id,))
    conn.commit()
    conn.close()

def update_word_in_db(word_id, new_italian, new_english, new_hint=''):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE vocab_progress SET italian = %s, english = %s, hint = %s WHERE id = %s
    ''', (new_italian, new_english, new_hint, word_id))
    conn.commit()
    conn.close()

def save_flagged_card(exercise_json, user_note=""):
    """
    Store a card the learner reported as wrong. Reviewable later in
    Supabase (Table Editor → flagged_cards) to spot error patterns and
    tune prompts. Creates the table lazily on first use.
    """
    import json as _json
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flagged_cards (
            id SERIAL PRIMARY KEY,
            flagged_at TEXT NOT NULL,
            user_note TEXT DEFAULT '',
            exercise JSONB
        )
    ''')
    cursor.execute('''
        INSERT INTO flagged_cards (flagged_at, user_note, exercise)
        VALUES (%s, %s, %s)
    ''', (datetime.now().isoformat(), user_note,
          _json.dumps(exercise_json, ensure_ascii=False, default=str)))
    conn.commit()
    conn.close()


def mark_word_mastered(word_id, interval_days):
    """
    "Already Mastered" override: push the word's interval far into the
    future so it counts as mastered and won't be reviewed for ages.
    Also increments review_count by 1 (so it's no longer 'unseen') and
    resets priority weight to 1.
    """
    from datetime import date, timedelta
    next_review = (date.today() + timedelta(days=interval_days)).isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE vocab_progress
        SET interval = %s,
            next_review_date = %s,
            review_count = GREATEST(review_count, 1) + 1,
            priority_weight = 1,
            ease_factor = GREATEST(ease_factor, 2.5)
        WHERE id = %s
    ''', (interval_days, next_review, word_id))
    conn.commit()
    conn.close()

# ----------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------
# NOTE: this used to run init_db() + import_vocab_from_csv() at MODULE
# IMPORT time. That meant any database problem (most commonly Supabase
# free-tier auto-pausing after ~a week idle) crashed the entire app with
# a raw Python traceback before a single pixel rendered.
#
# Initialisation is now lazy and guarded: main_app calls
# ensure_initialized() and renders a human explanation on failure.

_INIT_DONE = False


def ensure_initialized(force=False):
    """
    Run one-time DB setup (create tables, import CSV if needed).

    Idempotent: real work happens once per process unless force=True.

    Returns (ok, info) where info is None on success, or a dict:
        {"category": str, "headline": str, "steps": [str], "detail": str}
    Never raises — callers can render the result instead of crashing.
    """
    global _INIT_DONE
    if _INIT_DONE and not force:
        return True, None
    try:
        init_db()
        import_vocab_from_csv()
        _INIT_DONE = True
        return True, None
    except Exception as e:
        category, headline, steps = classify_db_error(e)
        logging.error(f"Database initialisation failed [{category}]: {e}")
        return False, {
            "category": category,
            "headline": headline,
            "steps": steps,
            "detail": str(e)[:400],
        }


def check_connection():
    """
    Lightweight liveness probe. Returns (ok, info) with the same shape as
    ensure_initialized. Used by the retry button so we can re-test without
    redoing the full import.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        return True, None
    except Exception as e:
        category, headline, steps = classify_db_error(e)
        return False, {
            "category": category,
            "headline": headline,
            "steps": steps,
            "detail": str(e)[:400],
        }
