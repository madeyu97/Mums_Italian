# src/audio_engine.py

import asyncio
import edge_tts
import logging
import random
import re
import time
import uuid
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Every clip gets its OWN file.
#
# Previously all audio was written to a single shared "current_audio.mp3",
# which caused three real bugs:
#   1. Generating a slow (-25%) replay OVERWROTE the normal-speed clip, so
#      every later replay of that card played the slow version.
#   2. audio_history (used by Undo) stored the same path for every card, so
#      undoing replayed the WRONG sentence's audio.
#   3. Two people using the app at once overwrote each other's audio
#      mid-playback.
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# How long generated clips are kept before cleanup. Comfortably longer than
# any single study session, so Undo can still replay earlier cards.
AUDIO_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours


def _new_audio_path() -> str:
    """A fresh, collision-free path for one clip."""
    return str(AUDIO_DIR / f"tts_{uuid.uuid4().hex}.mp3")


def _cleanup_old_audio():
    """
    Delete clips older than AUDIO_MAX_AGE_SECONDS so unique filenames can't
    fill the disk over time. Best-effort: never raises, never blocks a card.
    """
    try:
        cutoff = time.time() - AUDIO_MAX_AGE_SECONDS
        for path in AUDIO_DIR.glob("tts_*.mp3"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass
        # Remove the legacy shared file from older deployments if present.
        legacy = DATA_DIR / "current_audio.mp3"
        if legacy.exists() and legacy.stat().st_mtime < cutoff:
            legacy.unlink()
    except Exception:
        pass

# ==========================================
# THE ITALIAN VOICE CAST
# ==========================================
# All standard Italian. Mixing male/female + neural variants gives real
# ear-training value (different timbres, speaking rates, expressiveness).
VOICE_CAST = [
    # Only voices confirmed reliable on edge-tts. The expressive voices
    # (Benigno, Fabiola, Fiamma, Palmira, Lisandro, Calimero) and Gianni
    # have been returning empty audio responses — kept out of the cast.
    "it-IT-ElsaNeural",        # F, standard
    "it-IT-IsabellaNeural",    # F, standard
    "it-IT-GiuseppeNeural",    # M, standard
    "it-IT-DiegoNeural",       # M, standard
]

# Fallback if a chosen voice fails for any reason
FALLBACK_VOICE = "it-IT-ElsaNeural"

async def _generate_audio_async(text: str, voice: str, output_path: str, rate: str = "+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)

# Hard cap per TTS attempt. Without this, a hung Edge TTS request froze
# the whole card until edge-tts's own (long) internal timeout expired —
# twice, if the fallback also hung.
TTS_TIMEOUT_SECONDS = 12

def _run_tts_with_timeout(text: str, voice: str, output_path: str, rate: str):
    asyncio.run(
        asyncio.wait_for(
            _generate_audio_async(text, voice, output_path, rate=rate),
            timeout=TTS_TIMEOUT_SECONDS,
        )
    )

def create_audio_file(italian_text: str, voice: str = None, slow: bool = False):
    """
    Generate Italian TTS audio.
    Args:
        italian_text: the sentence to speak
        voice: optional specific voice override
        slow: if True, generates at -25% rate (useful for learner playback)
    """
    if not italian_text or len(italian_text.strip()) == 0:
        logging.error("Audio Engine received empty text.")
        return None

    # 1. Clean the text — KEEP Italian accented characters, spaces, apostrophes
    #    Italian uses: à á è é ì í î ò ó ù ú and standard punctuation.
    clean_text = re.sub(
        r"[^A-Za-zÀÁÈÉÌÍÎÒÓÙÚàáèéìíîòóùú,\.\!\?;:\s'’]",
        "",
        italian_text,
    )

    if len(clean_text.strip()) == 0:
        logging.warning("Filter stripped all text! Falling back to raw text.")
        clean_text = italian_text

    # NOTE: No "Malaysian spoofer" needed for Italian — TTS handles spelling correctly.

    # 2. Select Voice
    selected_voice = voice if voice else random.choice(VOICE_CAST)
    rate = "-25%" if slow else "+0%"
    logging.info(f"Attempting audio for: '{clean_text}' using {selected_voice} (rate={rate})")

    # 3. Generate Audio into a fresh file (see AUDIO_DIR note above)
    _cleanup_old_audio()
    out_path = _new_audio_path()
    try:
        _run_tts_with_timeout(clean_text, selected_voice, out_path, rate)
        return out_path
    except Exception as e:
        logging.warning(f"Voice {selected_voice} failed ({e}). Trying fallback...")
        try:
            _run_tts_with_timeout(clean_text, FALLBACK_VOICE, out_path, rate)
            return out_path
        except Exception as e_final:
            logging.error(f"Total Audio Failure: {e_final}")
            # Don't leave a truncated/empty file behind for st.audio to choke on
            try:
                import os as _os
                if _os.path.exists(out_path):
                    _os.remove(out_path)
            except OSError:
                pass
            return None
