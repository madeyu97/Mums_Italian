# src/main_app.py

import streamlit as st
import os
import random
import json
import logging
from datetime import date
from urllib.parse import quote_plus

from srs_engine import get_todays_quiz_batch, process_review
from ai_prompter import (
    generate_dictation_exercise,
    generate_conjugation_drill,
    generate_cloze_exercise,
    choose_recall_strategy,
    get_last_error_category,
)
from audio_engine import create_audio_file
from db_manager import (
    flag_word_in_database, get_progress_stats, undo_word_progress,
    get_more_words, delete_word_from_db, update_word_in_db,
    mark_word_mastered, save_flagged_card,
    ensure_initialized, check_connection, classify_db_error,
)
from speech_engine import transcribe_audio, grade_speech, GRADE_MAP
from config import (
    LISTENING_PCT, MAX_REVIEWS_PER_DAY, MASTERED_INTERVAL_DAYS,
    FLUENCY_TARGET, BREATH_PAUSE_EVERY, BREATH_PAUSE_SECONDS, DATA_DIR,
)

# ==========================================
# 1. CACHE MANAGEMENT
# ==========================================
# Absolute path under DATA_DIR. Previously a bare relative filename, which
# resolved against the process working directory (inside the git checkout
# on Streamlit Cloud) and could be wiped by a redeploy.
CACHE_FILE = str(DATA_DIR / "session_cache.json")

def load_cached_session():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("date") == str(date.today()):
                return cache
        except Exception:
            pass
    return None

def save_cached_session():
    cache = {
        "date": str(date.today()),
        "words_due": st.session_state.words_due,
        "modes": st.session_state.modes,
        "current_index": st.session_state.current_index,
        "current_exercise": st.session_state.current_exercise,
        "audio_path": st.session_state.audio_path,
        "stage": st.session_state.stage,
        "shuffled_options": st.session_state.shuffled_options,
        "user_typed": st.session_state.user_typed,
        "mcq_correct": st.session_state.mcq_correct,
        "exercise_history": st.session_state.exercise_history,
        "audio_history": st.session_state.audio_history,
        "recall_result": st.session_state.recall_result,
        "recall_history": st.session_state.recall_history,
        "cards_since_break": st.session_state.get("cards_since_break", 0),
        "breath_pause_active": st.session_state.get("breath_pause_active", False),
        "gen_failed_this_turn": st.session_state.get("gen_failed_this_turn", False),
        "last_gen_error": st.session_state.get("last_gen_error", None),
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def clear_cached_session():
    """Delete the on-disk cache so the next run starts from setup."""
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass

# ==========================================
# 2. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Italian Immersion", page_icon="🇮🇹", layout="centered")

# ==========================================
# 2b. DATABASE HEALTH GATE
# ==========================================
# The database is set up lazily here rather than at import time, so a
# sleeping/unreachable database produces a readable explanation instead
# of a Python traceback.
def render_db_problem(info):
    """Human-readable database failure screen with concrete fix steps."""
    st.title("🇮🇹 Italian Immersion Study")
    st.error(f"**{info['headline']}**")

    if info["category"] == "paused":
        st.markdown(
            "Nothing is lost — all your words and progress are still saved. "
            "The database just needs waking up."
        )

    st.markdown("**How to fix it:**")
    for i, step in enumerate(info["steps"], 1):
        st.markdown(f"{i}. {step}")

    if st.button("🔄 Try again", type="primary"):
        ok, _ = check_connection()
        if ok:
            ensure_initialized(force=True)
        st.rerun()

    with st.expander("Technical detail"):
        st.code(info["detail"], language="text")

_db_ok, _db_info = ensure_initialized()
if not _db_ok:
    render_db_problem(_db_info)
    st.stop()

@st.cache_data(ttl=60, show_spinner=False)
def cached_progress_stats():
    """
    Sidebar stats, cached for 60s. Without this, EVERY button click
    opened a fresh Supabase connection and ran COUNT queries — a big
    contributor to per-interaction lag.
    """
    return get_progress_stats()

def assign_modes(words):
    """Assign each card 'listen' or 'recall', then shuffle."""
    n = len(words)
    listening_count = round(n * LISTENING_PCT)
    modes = ['listen'] * listening_count + ['recall'] * (n - listening_count)
    random.shuffle(modes)
    return modes

# Midnight reset
if 'session_date' in st.session_state and st.session_state.session_date != str(date.today()):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# ==========================================
# 3. SESSION INITIALISATION — restore from cache OR show setup screen
# ==========================================
if 'words_due' not in st.session_state:
    cached_state = load_cached_session()
    if cached_state:
        # Resume an in-progress session
        for key, value in cached_state.items():
            if key != "date":
                st.session_state[key] = value
        if 'modes' not in st.session_state or not st.session_state.modes:
            st.session_state.modes = assign_modes(st.session_state.words_due)
        if 'recall_result' not in st.session_state:
            st.session_state.recall_result = None
        if 'recall_history' not in st.session_state:
            st.session_state.recall_history = {}
        if 'user_typed' not in st.session_state:
            st.session_state.user_typed = ""
        if 'cards_since_break' not in st.session_state:
            st.session_state.cards_since_break = 0
        if 'breath_pause_active' not in st.session_state:
            st.session_state.breath_pause_active = False
        if 'gen_failed_this_turn' not in st.session_state:
            st.session_state.gen_failed_this_turn = False
        if 'last_gen_error' not in st.session_state:
            st.session_state.last_gen_error = None
        st.session_state.session_date = str(date.today())
    else:
        # No cache — show the setup screen
        st.title("🇮🇹 Italian Immersion Study")
        stats = cached_progress_stats()
        st.markdown(f"You have **{stats['total']}** words in your vocabulary database.")
        st.markdown("### How long should today's session be?")

        with st.form("session_setup"):
            session_size = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=max(1, stats['total']) if stats['total'] > 0 else 100,
                value=min(MAX_REVIEWS_PER_DAY, stats['total']) if stats['total'] > 0 else MAX_REVIEWS_PER_DAY,
                step=1,
                help=f"Pick anywhere from 1 to {stats['total']} (your full vocabulary).",
            )
            listen_count = round(session_size * LISTENING_PCT)
            recall_count = session_size - listen_count
            st.caption(
                f"That's roughly **🎧 {listen_count} listening + 🎤 {recall_count} recall** "
                f"based on your {int(LISTENING_PCT*100)}/{int((1-LISTENING_PCT)*100)} mix."
            )
            start = st.form_submit_button("▶️ Start Session", type="primary", width="stretch")

        if not start:
            st.stop()

        # User clicked Start — build the session
        with st.spinner("Building your session..."):
            st.session_state.words_due = get_todays_quiz_batch(session_size=int(session_size))
            st.session_state.modes = assign_modes(st.session_state.words_due)
            st.session_state.current_index = 0
            st.session_state.current_exercise = None
            st.session_state.audio_path = None
            st.session_state.stage = 1
            st.session_state.shuffled_options = []
            st.session_state.user_typed = ""
            st.session_state.mcq_correct = None
            st.session_state.exercise_history = {}
            st.session_state.audio_history = {}
            st.session_state.recall_result = None
            st.session_state.recall_history = {}
            st.session_state.cards_since_break = 0
            st.session_state.breath_pause_active = False
            st.session_state.gen_failed_this_turn = False
            st.session_state.last_gen_error = None
            st.session_state.session_date = str(date.today())
            save_cached_session()
        st.rerun()

# ==========================================
# 4. HELPERS
# ==========================================
def reset_card_state():
    st.session_state.current_exercise = None
    st.session_state.audio_path = None
    st.session_state.stage = 1
    st.session_state.shuffled_options = []
    st.session_state.user_typed = ""
    st.session_state.mcq_correct = None
    st.session_state.recall_result = None
    st.session_state.gen_failed_this_turn = False
    st.session_state.last_gen_error = None

def _maybe_trigger_breath_pause():
    """
    After a card is graded, check whether we've hit the breath-pause
    threshold. If so, activate the pause UI for the next render.
    Skipped if there are no more cards (don't pause at session end).
    """
    if BREATH_PAUSE_EVERY <= 0:
        return
    st.session_state.cards_since_break = st.session_state.get("cards_since_break", 0) + 1
    if (st.session_state.cards_since_break >= BREATH_PAUSE_EVERY and
            st.session_state.current_index < len(st.session_state.words_due)):
        st.session_state.breath_pause_active = True
        st.session_state.cards_since_break = 0

def _safe_db_write(fn, *args, **kwargs):
    """
    Run a database write, converting failures into a readable message
    instead of a traceback.

    Mid-session the database can still blip (pooler drops an idle
    connection, Supabase restarts, network hiccup). Previously that raised
    straight through a button handler and killed the session. Now the card
    stays put and the learner can retry without losing their place.

    Returns True on success, False on failure.
    """
    try:
        fn(*args, **kwargs)
        st.session_state.db_write_error = None
        return True
    except Exception as e:
        category, headline, steps = classify_db_error(e)
        logging.error(f"Database write failed [{category}]: {e}")
        st.session_state.db_write_error = {
            "headline": headline,
            "steps": steps,
            "detail": str(e)[:300],
        }
        return False

def grade_word_and_next(grade):
    current_word = st.session_state.words_due[st.session_state.current_index]
    ok = _safe_db_write(
        process_review,
        word_id=current_word['id'],
        current_interval=current_word['interval'],
        current_ease=current_word['ease_factor'],
        grade=grade
    )
    if not ok:
        # Don't advance — the grade wasn't saved. The learner keeps their
        # place and can press the grade button again.
        return
    st.session_state.current_index += 1
    reset_card_state()
    _maybe_trigger_breath_pause()
    save_cached_session()

def mark_mastered_and_next():
    """
    "Already Mastered" override — pushes the current word far into the
    future and advances. Undo works the same as a normal grade.
    """
    current_word = st.session_state.words_due[st.session_state.current_index]
    ok = _safe_db_write(mark_word_mastered, current_word['id'], MASTERED_INTERVAL_DAYS)
    if not ok:
        return
    st.session_state.current_index += 1
    reset_card_state()
    _maybe_trigger_breath_pause()
    save_cached_session()

def undo_last_grade():
    if st.session_state.current_index > 0:
        prev_index = st.session_state.current_index - 1
        original_word = st.session_state.words_due[prev_index]
        ok = _safe_db_write(
            undo_word_progress,
            word_id=original_word['id'],
            old_next_review_date=original_word['next_review_date'],
            old_interval=original_word['interval'],
            old_ease=original_word['ease_factor'],
            old_review_count=original_word['review_count'],
            old_priority=original_word.get('priority_weight', 1)
        )
        if not ok:
            return
        st.session_state.current_index = prev_index
        idx_str = str(prev_index)
        st.session_state.current_exercise = st.session_state.exercise_history.get(idx_str)
        st.session_state.audio_path = st.session_state.audio_history.get(idx_str)
        if st.session_state.modes[prev_index] == 'recall':
            st.session_state.recall_result = st.session_state.recall_history.get(idx_str)
            st.session_state.stage = 2
        else:
            st.session_state.stage = 3
        save_cached_session()

def advance_to_stage(n):
    st.session_state.stage = n
    save_cached_session()

def start_new_session():
    """Wipe everything so the user gets the setup screen again."""
    clear_cached_session()
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# ==========================================
# 5. MAIN UI — SHARED HEADER
# ==========================================
st.title("🇮🇹 Italian Immersion Study")

# Session-complete screen
if st.session_state.current_index >= len(st.session_state.words_due):
    st.success("🎉 Bravissimo! You're all caught up for today.")
    st.balloons()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Do 5 More Words", type="secondary", width="stretch"):
            with st.spinner("Fetching more words..."):
                exclude_ids = [w['id'] for w in st.session_state.words_due]
                extra_words = get_more_words(exclude_ids, amount=5)
                if extra_words:
                    st.session_state.words_due.extend(extra_words)
                    st.session_state.modes.extend(assign_modes(extra_words))
                    save_cached_session()
                    st.rerun()
                else:
                    st.warning("You've completely exhausted your database!")
    with col_b:
        if st.button("🔄 Start New Session", type="primary", width="stretch"):
            start_new_session()
            st.rerun()
    st.stop()

# ==========================================
# 5.25 BREATH PAUSE — consolidation rest between batches
# ==========================================
def render_breath_pause():
    """
    Box-breathing pause UI. Animated client-side via HTML/CSS/JS so the
    countdown ticks smoothly without blocking the server.

    Cycle: inhale 4s → hold 4s → exhale 4s → hold 4s (16s box breath).
    Default duration is BREATH_PAUSE_SECONDS, which fits roughly two
    full cycles at 30s.
    """
    seconds = BREATH_PAUSE_SECONDS
    progress_done = st.session_state.current_index
    progress_total = len(st.session_state.words_due)

    st.markdown("### 🌬️ Breath pause")
    st.caption(
        "A short box-breathing rest to let what you've practised settle. "
        "Brief rests between batches measurably improve memory consolidation — "
        "this is part of the practice, not an interruption."
    )

    # Inline HTML + CSS for the breathing animation and countdown.
    html_block = f"""
<style>
.breath-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 8px 32px 8px;
    color: #d9d9e3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.breath-circle {{
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%,
                #6ea8ff 0%, #3a6fd1 55%, #14306b 100%);
    box-shadow: 0 0 40px rgba(110, 168, 255, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 24px;
    font-weight: 500;
    letter-spacing: 0.5px;
    animation: breath-cycle 16s ease-in-out infinite;
    transition: transform 0.5s ease;
}}
@keyframes breath-cycle {{
    /* 0-4s: inhale (expand). 4-8s: hold. 8-12s: exhale (shrink). 12-16s: hold. */
    0%   {{ transform: scale(0.65); }}
    25%  {{ transform: scale(1.0); }}
    50%  {{ transform: scale(1.0); }}
    75%  {{ transform: scale(0.65); }}
    100% {{ transform: scale(0.65); }}
}}
.breath-phase {{
    margin-top: 18px;
    font-size: 18px;
    color: #b3b3c0;
    min-height: 26px;
    text-align: center;
}}
.breath-countdown {{
    margin-top: 6px;
    font-size: 38px;
    font-weight: 300;
    color: #f0f0f5;
    font-variant-numeric: tabular-nums;
}}
.breath-progress-row {{
    margin-top: 14px;
    font-size: 13px;
    color: #8a8a99;
    letter-spacing: 0.3px;
}}
</style>

<div class="breath-wrap">
  <div class="breath-circle" id="breath-circle">
    <span id="breath-phase-text">Breathe</span>
  </div>
  <div class="breath-phase" id="breath-phase">Inhale slowly through your nose…</div>
  <div class="breath-countdown" id="breath-countdown">{seconds}</div>
  <div class="breath-progress-row">
    {progress_done} of {progress_total} cards complete · {seconds}-second rest
  </div>
</div>

<script>
(function() {{
    const total = {seconds};
    const phaseTexts = [
        "Inhale slowly through your nose…",
        "Hold gently…",
        "Exhale through your mouth…",
        "Hold and rest…",
    ];
    const circleLabel = ["Inhale", "Hold", "Exhale", "Hold"];
    let elapsed = 0;
    const countdownEl = document.getElementById("breath-countdown");
    const phaseEl = document.getElementById("breath-phase");
    const labelEl = document.getElementById("breath-phase-text");
    function tick() {{
        const remaining = Math.max(0, total - elapsed);
        if (countdownEl) countdownEl.textContent = remaining;
        const cycle_t = elapsed % 16;
        const phase_idx = Math.floor(cycle_t / 4);
        if (phaseEl) phaseEl.textContent = phaseTexts[phase_idx];
        if (labelEl) labelEl.textContent = circleLabel[phase_idx];
        elapsed += 1;
    }}
    tick();
    const iv = setInterval(tick, 1000);
    setTimeout(function() {{ clearInterval(iv); }}, (total + 1) * 1000);
}})();
</script>
"""
    st.html(html_block)

    # Server-side controls beneath the animation
    skip_col, _dummy, done_col = st.columns([1, 1, 1])
    with skip_col:
        if st.button("⏭️ Skip pause", width="stretch", key="breath_skip"):
            st.session_state.breath_pause_active = False
            save_cached_session()
            st.rerun()
    with done_col:
        if st.button("✓ I'm ready", type="primary", width="stretch", key="breath_done"):
            st.session_state.breath_pause_active = False
            save_cached_session()
            st.rerun()

    st.caption(
        f"Configurable in `config.py` — currently every "
        f"{BREATH_PAUSE_EVERY} cards, {BREATH_PAUSE_SECONDS} seconds. "
        f"Set `BREATH_PAUSE_EVERY = 0` to disable."
    )

if st.session_state.get("breath_pause_active", False):
    render_breath_pause()
    st.stop()

current_word = st.session_state.words_due[st.session_state.current_index]
current_mode = st.session_state.modes[st.session_state.current_index]

# If a grade/undo failed to save, say so here rather than crashing. The
# card stays put so nothing is lost — pressing the button again retries.
_write_err = st.session_state.get("db_write_error")
if _write_err:
    st.error(f"**Couldn't save that to the database.** {_write_err['headline']}")
    st.caption("Your place is kept — press the button again to retry once it's back.")
    with st.expander("What to do"):
        for i, step in enumerate(_write_err["steps"], 1):
            st.markdown(f"{i}. {step}")
        st.code(_write_err["detail"], language="text")

# Header row: progress / mode / undo
total_words = len(st.session_state.words_due)
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.progress((st.session_state.current_index) / total_words)
    st.caption(f"Reviewing word {st.session_state.current_index + 1} of {total_words}")
with col2:
    badge = "🎧 Listen" if current_mode == 'listen' else "🎤 Recall"
    st.caption(f"**{badge}**")
with col3:
    if st.session_state.current_index > 0:
        if st.button("↩️ Undo", width="stretch"):
            undo_last_grade()
            st.rerun()

st.markdown("---")

# ==========================================
# 5.5 SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📊 Global Progress")
    stats = cached_progress_stats()

    # ---- Fluency-to-10k indicator ----
    mastered_count = stats.get('mastered', 0)
    fluency_pct = (mastered_count / FLUENCY_TARGET) * 100 if FLUENCY_TARGET > 0 else 0
    # Cap the bar at 1.0 in case someone really does hit 10k
    bar_value = min(mastered_count / FLUENCY_TARGET, 1.0) if FLUENCY_TARGET > 0 else 0

    st.markdown("#### 🇮🇹 Fluency Progress")
    # Pick a friendly precision: %, .1f%, or .2f% depending on size
    if fluency_pct >= 10:
        pct_str = f"{fluency_pct:.1f}%"
    elif fluency_pct >= 1:
        pct_str = f"{fluency_pct:.2f}%"
    else:
        pct_str = f"{fluency_pct:.3f}%"
    st.metric(
        label=f"Toward {FLUENCY_TARGET:,}-word fluency",
        value=pct_str,
        delta=f"{mastered_count:,} / {FLUENCY_TARGET:,} mastered",
        delta_color="off",
    )
    st.progress(bar_value)
    st.caption(
        "Rough rule-of-thumb: ~10,000 mastered words ≈ functional fluency. "
        "Multi-word entries count as one unit."
    )
    st.markdown("---")
    if stats['total'] > 0:
        unseen_pct = int((stats['unseen'] / stats['total']) * 100)
        learning_pct = int((stats['learning'] / stats['total']) * 100)
        mastered_pct = int((stats['mastered'] / stats['total']) * 100)
        st.metric("Total CSV Vocabulary", stats['total'])
        st.markdown("---")
        st.write(f"**👀 Unseen:** {stats['unseen']} words ({unseen_pct}%)")
        st.progress(stats['unseen'] / stats['total'])
        st.write(f"**🧠 Learning:** {stats['learning']} words ({learning_pct}%)")
        st.progress(stats['learning'] / stats['total'])
        st.write(f"**🏆 Mastered:** {stats['mastered']} words ({mastered_pct}%)")
        st.progress(stats['mastered'] / stats['total'])
        st.markdown("---")
        st.caption("Mastered = pushed 21+ days into the future.")

        st.markdown("---")
        listen_left = sum(1 for i, m in enumerate(st.session_state.modes)
                          if i >= st.session_state.current_index and m == 'listen')
        recall_left = sum(1 for i, m in enumerate(st.session_state.modes)
                          if i >= st.session_state.current_index and m == 'recall')
        st.caption(f"This session: 🎧 {listen_left} listening · 🎤 {recall_left} recall")
        st.caption("Recall auto-routes verbs → conjugation drills, "
                   "clitics/articles/preps → cloze, rest → free recall.")

        st.markdown("---")
        if st.button("🔄 End & Start New Session", width="stretch"):
            start_new_session()
            st.rerun()
    else:
        st.write("No vocabulary found. Please check your CSV.")

# ==========================================
# 6. EXERCISE GENERATION (routes by mode + recall sub-mode)
# ==========================================
def _attempt_generate_exercise():
    """
    Try to generate an exercise for the current card. Returns the
    exercise dict on success, None on failure.

    Routes to the right generator based on listen vs recall mode and,
    for recall, on the chosen strategy.
    """
    if current_mode == 'listen':
        ex = generate_dictation_exercise(current_word)
        if ex is not None:
            ex.setdefault("exercise_type", "listen")
        return ex

    # Recall: pick strategy
    strategy = choose_recall_strategy(current_word)
    if strategy == 'conjugation':
        ex = generate_conjugation_drill(current_word)
        if ex is None:
            # Don't burn another LLM call on free recall if we got rate-limited
            if get_last_error_category() == "rate_limit":
                return None
            ex = generate_dictation_exercise(current_word)
            if ex is not None:
                ex["exercise_type"] = "free"
        return ex
    elif strategy == 'cloze':
        return generate_cloze_exercise(current_word)
    else:
        ex = generate_dictation_exercise(current_word)
        if ex is not None:
            ex["exercise_type"] = "free"
        return ex


if st.session_state.current_exercise is None:
    # Only run generation if we haven't already failed this turn.
    # st.session_state.gen_failed_this_turn prevents the silent re-attempt
    # on every rerun when the user is on the failure screen.
    if not st.session_state.get("gen_failed_this_turn", False):
        with st.spinner("Generating Italian scenario..."):
            exercise_data = _attempt_generate_exercise()

        if exercise_data:
            st.session_state.current_exercise = exercise_data
            audio_script = exercise_data['italian']
            st.session_state.audio_path = create_audio_file(audio_script)

            idx_str = str(st.session_state.current_index)
            st.session_state.exercise_history[idx_str] = exercise_data
            st.session_state.audio_history[idx_str] = st.session_state.audio_path

            if current_mode == 'listen' and exercise_data.get("english_distractors"):
                options = exercise_data['english_distractors'] + [exercise_data['english_correct']]
                random.shuffle(options)
                st.session_state.shuffled_options = options
            else:
                st.session_state.shuffled_options = []
            st.session_state.gen_failed_this_turn = False
            save_cached_session()
        else:
            # Mark so the next rerun won't silently retry generation —
            # user must click a button.
            st.session_state.gen_failed_this_turn = True
            st.session_state.last_gen_error = get_last_error_category() or "unknown"
            save_cached_session()

    # Failure UI (shown on the same rerun if generation just failed, or
    # on subsequent reruns if user is sitting on the failure screen).
    if st.session_state.current_exercise is None:
        err_cat = st.session_state.get("last_gen_error", "unknown")
        if err_cat == "rate_limit":
            st.error(
                "🚦 **You've hit Groq's rate limit.** "
                "This happens on the free tier after about 30 requests per minute. "
                "It will reset automatically — please wait around **60 seconds** before continuing."
            )
            st.caption(
                "Tip: the breath pause every 8 cards helps keep this from happening. "
                "If you're seeing it often, consider upgrading to Groq's Developer tier "
                "(about $5/mo of usage, 10× the throughput)."
            )
        elif err_cat == "auth":
            st.error(
                "🔑 **API key error.** Your Groq API key seems to be missing or invalid. "
                "Go to Streamlit Cloud → your app → ⋮ → Settings → Secrets, "
                "and check that GROQ_API_KEY is correct."
            )
        elif err_cat == "model_gone":
            st.error(
                "🧠 **The AI model this app uses is no longer available.** "
                "Groq retires older models periodically."
            )
            st.caption(
                "Fix: open `src/config.py`, and set `GENERATION_MODEL` and "
                "`GRADING_MODEL` to a current model from console.groq.com/docs/models. "
                "Then commit and reboot the app."
            )
        elif err_cat == "json_parse":
            st.error(
                "🤖 **The AI returned a malformed response.** This usually clears on the next try. "
                "Click 'Try again' below."
            )
        else:
            st.error(
                "⚠️ **Couldn't generate this card right now.** "
                "Usually a transient glitch — try again or skip."
            )
            st.caption("Check Manage app → Logs for details if this keeps happening.")

        retry_col, skip_col = st.columns(2)
        with retry_col:
            if st.button("🔄 Try again", type="primary", width="stretch", key="retry_gen"):
                st.session_state.gen_failed_this_turn = False
                st.session_state.last_gen_error = None
                save_cached_session()
                st.rerun()
        with skip_col:
            if st.button("⏭️ Skip this card", width="stretch", key="skip_gen"):
                # Move to the next card; clear the failure flag so the
                # next card gets a fresh attempt.
                st.session_state.current_index += 1
                st.session_state.gen_failed_this_turn = False
                st.session_state.last_gen_error = None
                reset_card_state()
                save_cached_session()
                st.rerun()
        st.stop()

# ==========================================
# Reusable renderers
# ==========================================
def render_breakdown():
    ex = st.session_state.current_exercise

    gp = ex.get('grammar_point')
    if gp and gp.get('structure'):
        st.markdown("#### 🧠 Grammar Point")
        st.info(f"**{gp['structure']}**: {gp['explanation']}")

    en = ex.get('expression_note')
    if en and en.get('expression'):
        st.markdown("#### 🗣️ Expression / Idiom")
        st.warning(f"**{en['expression']}**: {en['explanation']}")

    st.markdown("#### 📖 Word Breakdown")
    words = ex.get('word_breakdown', [])
    cols_per_row = 3
    for i in range(0, len(words), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(words):
                word = words[i + j]
                italian_word = word.get('italian', word.get('chinese', '?'))
                english_meaning = word.get('english', '')
                note = word.get('note', '')

                # Italian dictionary links
                encoded = quote_plus(italian_word)
                wordref_url = f"https://www.wordreference.com/iten/{encoded}"
                reverso_url = f"https://context.reverso.net/translation/italian-english/{encoded}"
                treccani_url = f"https://www.treccani.it/vocabolario/ricerca/{encoded}/"

                with col:
                    label = f"{italian_word}"
                    with st.expander(label):
                        st.write(f"**{english_meaning}**")
                        if note:
                            st.caption(note)
                        stress = word.get('stress', '')
                        if stress:
                            st.caption(f"🔤 Stress: **{stress}**")
                        st.caption(f"Word: {italian_word}")
                        button_key = f"flag_btn_{st.session_state.current_index}_{i}_{j}_{italian_word}"
                        if st.button("🚩 Needs Practice", key=button_key):
                            flag_word_in_database(italian_word)
                            st.toast(f"Flagged '{italian_word}' for more practice!")
                        st.markdown("---")
                        st.markdown(f"📚 [WordReference]({wordref_url})")
                        st.markdown(f"💬 [Reverso Context]({reverso_url})")
                        st.markdown(f"🇮🇹 [Treccani]({treccani_url})")

    # ---- Report a mistake (one-tap error flagging) ----
    with st.expander("🚩 Something wrong with this card?"):
        st.caption(
            "If the Italian, the translation, or a grammar note looks wrong, "
            "report it here. Reports are saved for review so recurring "
            "mistakes can be fixed at the source. "
            "(Heads-up: after 'penso che', forms like 'parli' really are "
            "3rd person — that's the congiuntivo, not a bug!)"
        )
        report_note = st.text_input(
            "What's wrong? (optional)",
            key=f"report_note_{st.session_state.current_index}",
            placeholder="e.g. wrong tense label on 'parli'",
        )
        if st.button("🚩 Report this card", key=f"report_btn_{st.session_state.current_index}"):
            try:
                save_flagged_card(ex, user_note=report_note or "")
                st.toast("Reported — thank you! Saved for review.")
            except Exception as e:
                st.warning(f"Couldn't save the report right now ({e}).")

def render_card_settings():
    st.markdown("---")
    with st.expander("⚙️ Card Settings (Edit or Delete)"):
        st.caption("Tweak this word's context to guide the AI, or remove it entirely.")
        edit_col1, edit_col2, edit_col3 = st.columns(3)
        with edit_col1:
            edit_italian = st.text_input("Italian", current_word.get('italian', ''))
        with edit_col2:
            edit_english = st.text_input("English", current_word.get('english', ''))
        with edit_col3:
            edit_hint = st.text_input("Hint (AI Prompt Hint)", current_word.get('hint', '') or '')

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("💾 Save & Regenerate Card", width="stretch"):
                update_word_in_db(current_word['id'], edit_italian, edit_english, edit_hint)
                st.session_state.words_due[st.session_state.current_index]['italian'] = edit_italian
                st.session_state.words_due[st.session_state.current_index]['english'] = edit_english
                st.session_state.words_due[st.session_state.current_index]['hint'] = edit_hint
                reset_card_state()
                save_cached_session()
                st.rerun()
        with btn_col2:
            if st.button("🗑️ Delete Word Permanently", type="secondary", width="stretch"):
                delete_word_from_db(current_word['id'])
                st.session_state.words_due.pop(st.session_state.current_index)
                st.session_state.modes.pop(st.session_state.current_index)
                reset_card_state()
                save_cached_session()
                st.rerun()

def render_grade_buttons(suggested_grade=None):
    st.markdown("---")
    st.markdown("#### Grade yourself (Be honest!):")
    labels = ["Again (0)\nFailed", "Hard (1)\nStruggled", "Good (2)\nSolid", "Easy (3)\nInstant"]
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, labels)):
        with col:
            btn_type = "primary" if i == suggested_grade else "secondary"
            if st.button(label, width="stretch", key=f"grade_{i}", type=btn_type):
                grade_word_and_next(i)
                st.rerun()

    # ---- Mastered override ----
    st.markdown("")  # small visual gap
    mc1, mc2, mc3 = st.columns([1, 2, 1])
    with mc2:
        if st.button(
            "🏆 Already Mastered — skip future reviews",
            width="stretch",
            key="mastered_override",
            help=(
                f"Marks this word as mastered and won't review it for "
                f"{MASTERED_INTERVAL_DAYS} days. Use this for words you've "
                f"known cold for ages and don't want cluttering sessions. "
                f"You can still undo via the ↩️ Undo button."
            ),
        ):
            mark_mastered_and_next()
            st.rerun()


# ==========================================
# 7A. LISTENING FLOW
# ==========================================
if current_mode == 'listen':
    st.subheader("Listen & Transcribe:")
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        st.audio(st.session_state.audio_path, format="audio/mp3")
    else:
        st.warning("⚠️ The audio engine failed to generate the voice file.")
        if st.button("🔄 Retry Audio", type="primary"):
            with st.spinner("Retrying audio..."):
                audio_script = st.session_state.current_exercise['italian']
                st.session_state.audio_path = create_audio_file(audio_script)
                st.session_state.audio_history[str(st.session_state.current_index)] = st.session_state.audio_path
                save_cached_session()
                st.rerun()

    # Optional slow replay
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        with st.expander("🐢 Need it slower?"):
            if st.button("Generate slow version (-25%)", key=f"slow_{st.session_state.current_index}"):
                with st.spinner("Generating slow audio..."):
                    slow_path = create_audio_file(
                        st.session_state.current_exercise['italian'], slow=True
                    )
                    if slow_path and os.path.exists(slow_path):
                        st.audio(slow_path, format="audio/mp3")

    if st.session_state.stage == 1:
        st.text_input(
            "Type what you hear (in Italian):",
            key="typed_input",
            placeholder="e.g. Ho mangiato la pizza",
        )
        if st.button("Submit", type="primary", width="stretch"):
            st.session_state.user_typed = st.session_state.typed_input
            advance_to_stage(2)
            st.rerun()

    if st.session_state.stage >= 2:
        st.success(f"**You typed:** {st.session_state.user_typed}")

    if st.session_state.stage == 2:
        st.markdown("### What does the sentence mean?")
        st.info("Select the most accurate, nuanced translation:")
        if not st.session_state.shuffled_options or len(st.session_state.shuffled_options) < 2:
            st.error("⚠️ The AI failed to generate the multiple-choice options properly.")
            if st.button("🔄 Regenerate This Word", type="primary"):
                reset_card_state()
                save_cached_session()
                st.rerun()
        else:
            selected_meaning = st.radio(
                "Choose translation:",
                st.session_state.shuffled_options,
                index=None,
                label_visibility="collapsed",
            )
            if st.button(
                "Submit Meaning",
                type="primary",
                width="stretch",
                disabled=(selected_meaning is None),
            ):
                st.session_state.mcq_correct = (
                    selected_meaning == st.session_state.current_exercise['english_correct']
                )
                advance_to_stage(3)
                st.rerun()

    if st.session_state.stage == 3:
        st.markdown("---")
        st.markdown("### The Solution")
        if st.session_state.mcq_correct is not None:
            if st.session_state.mcq_correct:
                st.success("✅ **Translation:** Correct!")
            else:
                st.error("❌ **Translation:** Incorrect.")
        st.info(f"**Correct Italian:** {st.session_state.current_exercise['italian']}")
        st.info(f"**Correct English:** {st.session_state.current_exercise['english_correct']}")

        # Spelling comparison: what they typed vs the answer
        if st.session_state.user_typed:
            typed_norm = st.session_state.user_typed.strip().lower()
            answer_norm = st.session_state.current_exercise['italian'].strip().lower()
            if typed_norm == answer_norm:
                st.caption("📝 Spelling: **perfect match** ✅")
            else:
                st.caption(f"📝 Spelling: not exact — check accents and double consonants")

        render_breakdown()
        render_card_settings()
        render_grade_buttons()


# ==========================================
# 7B. RECALL FLOW
# ==========================================
elif current_mode == 'recall':
    ex = st.session_state.current_exercise
    exercise_type = ex.get('exercise_type', 'free')

    # ------- Prompt header (varies by sub-mode) -------
    if exercise_type == 'conjugation':
        st.subheader("🎤 Conjugation Drill")
        tense  = ex.get('tense', '')
        person = ex.get('person', '')
        st.caption(f"Target form: **{tense}**, **{person}**")
        st.markdown(f"**Translate into Italian:**")
        st.info(f"_{ex['english_prompt']}_")
        st.caption(
            f"Infinitive: **{ex.get('infinitive', current_word.get('italian', ''))}** "
            f"({current_word.get('english', '')})"
        )

    elif exercise_type == 'cloze':
        st.subheader("🎤 Fill the Blank (Cloze)")
        st.markdown("**Read the sentence below and SPEAK the missing word:**")
        st.info(ex.get('cloze_display', ex.get('italian', '')))
        st.caption(f"Meaning: _{ex['english_correct']}_")
        # Optional: let them hear the full sentence first
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            with st.expander("🔊 Hear the full sentence first"):
                st.audio(st.session_state.audio_path, format="audio/mp3")
                st.caption("Tip: try to say the blank without listening, then check.")

    else:
        # Free recall (original behaviour)
        st.subheader("🎤 Speak the Italian:")
        st.markdown(f"**Say this in Italian:** _{ex['english_correct']}_")
        target_meaning = current_word.get('english', '')
        if target_meaning:
            st.caption(f"Target word meaning: _{target_meaning}_")

    # ------- Stage 1: record + submit (shared across sub-modes) -------
    if st.session_state.stage == 1:
        st.info("Tap to record, speak, then submit.")
        audio_value = st.audio_input("🎙️ Your attempt", key=f"mic_{st.session_state.current_index}")

        if audio_value is not None:
            if st.button("✅ Submit Recording", type="primary", width="stretch"):
                audio_bytes = audio_value.getvalue()
                with st.spinner("Transcribing with Whisper..."):
                    transcription = transcribe_audio(audio_bytes)
                if transcription is None:
                    st.error("Transcription failed — try again.")
                    st.stop()

                # Build the right grading call based on sub-mode
                drill_context = None
                if exercise_type == 'conjugation':
                    grade_expected_italian = ex.get('italian', '')
                    grade_expected_english = ex.get('english_prompt', ex.get('english_correct', ''))
                    drill_context = {
                        "type": "conjugation",
                        "infinitive": ex.get('infinitive', ''),
                        "tense":      ex.get('tense', ''),
                        "person":     ex.get('person', ''),
                        "expected_form": ex.get('expected_form', ''),
                        "english_prompt": grade_expected_english,
                    }
                elif exercise_type == 'cloze':
                    grade_expected_italian = ex.get('cloze_blank', '')
                    grade_expected_english = ex.get('english_correct', '')
                    drill_context = {
                        "type": "cloze",
                        "target_word":     ex.get('cloze_blank', ''),
                        "full_sentence":   ex.get('italian', ''),
                        "blanked_display": ex.get('cloze_display', ''),
                    }
                else:
                    grade_expected_italian = ex.get('italian', '')
                    grade_expected_english = ex.get('english_correct', '')

                with st.spinner("Grading your attempt..."):
                    grading = grade_speech(
                        expected_italian=grade_expected_italian,
                        expected_english=grade_expected_english,
                        transcribed_text=transcription['text'],
                        hint=current_word.get('hint', '') or '',
                        drill_context=drill_context,
                    )
                if grading is None:
                    st.error("Grading failed — try again, or skip to grade yourself manually.")
                    st.stop()
                st.session_state.recall_result = {
                    "transcription": transcription,
                    "grading": grading,
                }
                st.session_state.recall_history[str(st.session_state.current_index)] = st.session_state.recall_result
                advance_to_stage(2)
                st.rerun()

        with st.expander("Can't record right now?"):
            if st.button("⏭️ Skip recording and self-grade"):
                st.session_state.recall_result = None
                advance_to_stage(2)
                st.rerun()

    # ------- Stage 2: solution + grading + breakdown (shared) -------
    if st.session_state.stage == 2:
        result = st.session_state.recall_result
        st.markdown("---")
        st.markdown("### The Solution")

        if exercise_type == 'conjugation':
            st.success(f"**Expected form:** `{ex.get('expected_form', '')}`")
            st.info(f"**Full sentence:** {ex.get('italian', '')}")
            st.caption(f"*(Meaning: {ex.get('english_prompt', ex.get('english_correct', ''))})*")
            form_expl = ex.get('form_explanation', '')
            if form_expl:
                st.caption(f"📖 {form_expl}")

        elif exercise_type == 'cloze':
            st.success(f"**Missing word:** `{ex.get('cloze_blank', '')}`")
            st.info(f"**Full sentence:** {ex.get('italian', '')}")
            st.caption(f"*(Meaning: {ex['english_correct']})*")

        else:
            st.info(f"**Correct Italian:** {ex.get('italian', '')}")
            st.caption(f"*(Meaning: {ex['english_correct']})*")

        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            st.caption("How it should sound:")
            st.audio(st.session_state.audio_path, format="audio/mp3")

        if result is not None:
            grading = result['grading']
            transcription = result['transcription']

            st.markdown("#### 📝 Whisper heard:")
            st.code(transcription['text'] or "(nothing audible)", language=None)

            s1, s2, s3 = st.columns(3)
            with s1: st.metric("Vocab", f"{grading['vocab_score']}/10")
            with s2: st.metric("Grammar", f"{grading['grammar_score']}/10")
            with s3: st.metric("Pronunciation*", f"{grading['pronunciation_score']}/10")
            st.caption(
                "*Pronunciation is inferred from Whisper transcription fidelity — "
                "it can't grade open/closed vowels or double-consonant crispness directly."
            )

            st.markdown("#### 💬 Feedback")
            st.write(grading['feedback'])

            suggested = GRADE_MAP.get(grading['overall_grade'], 2)
            st.caption(f"Suggested SRS grade: **{grading['overall_grade']}** (you can override below).")
        else:
            suggested = None
            st.caption("No recording submitted — grade yourself below.")

        render_breakdown()
        render_card_settings()
        render_grade_buttons(suggested_grade=suggested)
