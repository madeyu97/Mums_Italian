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

def get_connection():
    """Establishes a connection to the Supabase PostgreSQL database."""
    db_url = None
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
    elif "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
    if not db_url:
        raise ValueError("CRITICAL ERROR: DATABASE_URL is missing! Streamlit cannot find it in the Secrets menu.")
    return psycopg2.connect(db_url)

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

    Performs a single batched INSERT with ON CONFLICT DO NOTHING so
    duplicates are skipped at the database level. Handles 10k+ rows
    in seconds instead of minutes.
    """
    if not VOCAB_CSV_PATH.exists():
        logging.warning("CSV file not found. Skipping import.")
        return

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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM vocab_progress")
    total_words = cursor.fetchone()[0]
    if total_words == 0:
        conn.close()
        return {"unseen": 0, "learning": 0, "mastered": 0, "total": 0}
    cursor.execute("SELECT COUNT(*) FROM vocab_progress WHERE review_count = 0")
    unseen = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM vocab_progress WHERE interval >= 21")
    mastered = cursor.fetchone()[0]
    learning = total_words - unseen - mastered
    conn.close()
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

# --- Initialization Block ---
init_db()
import_vocab_from_csv()
