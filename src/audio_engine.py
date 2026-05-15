# src/audio_engine.py

import asyncio
import edge_tts
import logging
import random
import re
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
AUDIO_PATH = DATA_DIR / "current_audio.mp3"

# ==========================================
# THE ITALIAN VOICE CAST
# ==========================================
# All standard Italian. Mixing male/female + neural variants gives real
# ear-training value (different timbres, speaking rates, expressiveness).
VOICE_CAST = [
    "it-IT-ElsaNeural",        # F, standard
    "it-IT-IsabellaNeural",    # F, standard
    "it-IT-GiuseppeNeural",    # M, standard
    "it-IT-DiegoNeural",       # M, standard
    "it-IT-BenignoNeural",     # M, expressive
    "it-IT-FabiolaNeural",     # F
    "it-IT-FiammaNeural",      # F
    "it-IT-GianniNeural",      # M
    "it-IT-CalimeroNeural",    # M
    "it-IT-PalmiraNeural",     # F
    "it-IT-LisandroNeural",    # M
]

# Fallback if a chosen voice fails for any reason
FALLBACK_VOICE = "it-IT-ElsaNeural"

async def _generate_audio_async(text: str, voice: str, output_path: str, rate: str = "+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)

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

    # 3. Generate Audio
    try:
        asyncio.run(_generate_audio_async(clean_text, selected_voice, str(AUDIO_PATH), rate=rate))
        return str(AUDIO_PATH)
    except Exception as e:
        logging.warning(f"Voice {selected_voice} failed ({e}). Trying fallback...")
        try:
            asyncio.run(_generate_audio_async(clean_text, FALLBACK_VOICE, str(AUDIO_PATH), rate=rate))
            return str(AUDIO_PATH)
        except Exception as e_final:
            logging.error(f"Total Audio Failure: {e_final}")
            return None
