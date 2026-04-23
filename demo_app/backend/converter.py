"""
converter.py
============
MIDIConverter — converts REMIGEN2 token lists to MIDI files.

Copied from notebooks/xLSTM-3/generate/converter.py and refactored:
  - midiprocessor_path constructor param removed (midiprocessor is pip-installed)
  - sys.path manipulation removed
  - token_analysis import fixed for same-directory layout

Encoding note:
  We use MidiDecoder('REMIGEN2') because the LMD training data was encoded
  with enc_remigen2_utils (s-X t-X emitted at the START OF EVERY BAR).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from token_analysis import clean_tokens

logger = logging.getLogger(__name__)


class MIDIConverter:
    """
    Converts REMIGEN2 token lists to MIDI files via midiprocessor.

    midiprocessor must be pip-installed (midiprocessor==0.1.5).
    """

    def __init__(self) -> None:
        import midiprocessor as mp
        self._decoder = mp.MidiDecoder("REMIGEN2")
        logger.debug("MIDIConverter initialised with REMIGEN2 decoder.")

    def tokens_to_midi(
        self,
        tokens: List[str],
        output_path: str,
        *,
        use_clean_fallback: bool = True,
    ) -> bool:
        """
        Convert a REMIGEN2 token list to a MIDI file and write it to disk.

        Tries direct decode first. If that fails and use_clean_fallback=True,
        applies clean_tokens() and retries.

        Args:
            tokens:              List of raw REMIGEN2 token strings.
            output_path:         Destination .mid file path (directories created).
            use_clean_fallback:  Whether to retry with cleaned tokens on failure.

        Returns:
            True on success, False if decoding failed even after cleaning.
        """
        # First attempt: raw tokens
        success = self._try_decode(tokens, output_path)
        if success:
            return True

        if not use_clean_fallback:
            logger.error("Decode failed for %s (no fallback).", output_path)
            return False

        # Second attempt: cleaned tokens
        logger.warning("Direct decode failed for %s — retrying with cleaned tokens.", output_path)
        cleaned = clean_tokens(tokens)
        if not cleaned:
            logger.error("Token list empty after cleaning — giving up.")
            return False

        success = self._try_decode(cleaned, output_path)
        if not success:
            logger.error("Decode still failed after cleaning for %s.", output_path)
        return success

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_decode(self, tokens: List[str], output_path: str) -> bool:
        """Attempt to decode tokens and write MIDI to output_path."""
        try:
            midi_obj = self._decoder.decode_from_token_str_list(tokens)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            midi_obj.dump(output_path)
            return True
        except Exception as exc:
            logger.debug("Decode error (%s): %s", type(exc).__name__, exc)
            return False
