"""
token_analysis.py
=================
Pure functions for REMIGEN2 token stream analysis.
No side effects — all functions take a token list and return data.

Copied as-is from notebooks/xLSTM-3/generate/token_analysis.py.
No modifications needed — zero external dependencies.

Token format reference (from repos/MidiProcessor/midiprocessor/const.py):
    b-X  BAR_ABBR     — bar boundary (marks END of a bar), value always 1
    o-X  POS_ABBR     — onset/position within bar
    s-X  TS_ABBR      — time signature (emitted at START of every bar)
    t-X  TEMPO_ABBR   — tempo class (emitted at START of every bar)
    i-X  INST_ABBR    — instrument (MIDI program)
    p-X  PITCH_ABBR   — pitch
    d-X  DURATION_ABBR — duration class (follows p-X)
    v-X  VELOCITY_ABBR — velocity class (follows d-X)

Grammar rule: each note == p-X d-X v-X (triplet). Any deviation is an error.
A complete bar == s-X t-X ... b-1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import statistics


# ---------------------------------------------------------------------------
# Token prefix constants (mirroring const.py — no import needed)
# ---------------------------------------------------------------------------
BAR_PREFIX   = "b"
POS_PREFIX   = "o"
TS_PREFIX    = "s"
TEMPO_PREFIX = "t"
INST_PREFIX  = "i"
PITCH_PREFIX = "p"
DUR_PREFIX   = "d"
VEL_PREFIX   = "v"


def _prefix(tok: str) -> Optional[str]:
    """Return the single-letter prefix of a token, or None if malformed."""
    idx = tok.find("-")
    if idx < 1:
        return None
    return tok[:idx]


# ---------------------------------------------------------------------------
# Grammar checking
# ---------------------------------------------------------------------------

@dataclass
class GrammarError:
    position: int
    token: str
    message: str

    def __str__(self) -> str:
        return f"[{self.position}] {self.token!r}: {self.message}"


def check_grammar(tokens: List[str]) -> List[GrammarError]:
    """
    Scan a REMIGEN2 token list for grammar errors.

    Rules checked:
    1. Every token must have a valid 'prefix-value' format.
    2. Every p-X must be immediately followed by d-X then v-X.
    3. Every d-X or v-X must be part of a valid p-d-v triplet
       (i.e., preceded by the correct predecessor).
    4. No unknown prefixes (anything not in the known set).

    Returns a list of GrammarError objects (empty = no errors).
    """
    errors: List[GrammarError] = []
    known_prefixes = {BAR_PREFIX, POS_PREFIX, TS_PREFIX, TEMPO_PREFIX,
                      INST_PREFIX, PITCH_PREFIX, DUR_PREFIX, VEL_PREFIX}

    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        pfx = _prefix(tok)

        # Rule 1 — valid format
        if pfx is None:
            errors.append(GrammarError(i, tok, "malformed token (no '-' separator)"))
            i += 1
            continue

        # Rule 4 — known prefix
        if pfx not in known_prefixes:
            errors.append(GrammarError(i, tok, f"unknown prefix '{pfx}'"))
            i += 1
            continue

        # Rule 2 — p-X must be the start of a complete p-d-v triplet
        if pfx == PITCH_PREFIX:
            if i + 1 >= n or _prefix(tokens[i + 1]) != DUR_PREFIX:
                errors.append(GrammarError(i, tok, "p-X not followed by d-X"))
            elif i + 2 >= n or _prefix(tokens[i + 2]) != VEL_PREFIX:
                errors.append(GrammarError(i, tok, "p-X d-X not followed by v-X"))

        # Rule 3 — d-X must be preceded by p-X
        elif pfx == DUR_PREFIX:
            if i == 0 or _prefix(tokens[i - 1]) != PITCH_PREFIX:
                errors.append(GrammarError(i, tok, "orphan d-X (not preceded by p-X)"))

        # Rule 3 — v-X must be preceded by d-X
        elif pfx == VEL_PREFIX:
            if i == 0 or _prefix(tokens[i - 1]) != DUR_PREFIX:
                errors.append(GrammarError(i, tok, "orphan v-X (not preceded by d-X)"))

        i += 1

    return errors


# ---------------------------------------------------------------------------
# Bar segmentation
# ---------------------------------------------------------------------------

def segment_bars(tokens: List[str]) -> List[List[str]]:
    """
    Split a token list into bars.
    Each bar is the slice between two consecutive b-1 tokens (inclusive of b-1 at the end).
    Tokens before the first b-1 form bar 0 (may be the prompt header).

    Returns:
        List of bars, each bar is a list of tokens ending with 'b-1'.
    """
    bars: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        current.append(tok)
        if _prefix(tok) == BAR_PREFIX:
            bars.append(current)
            current = []
    # Any trailing tokens that didn't close with b-1 are a partial bar — discard.
    return bars


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class TokenAnalysis:
    total_tokens: int
    num_bars: int
    num_notes: int                         # count of p-X tokens
    num_instruments: int                   # unique i-X values seen
    instruments: List[str]                 # sorted list of i-X strings
    tokens_per_bar_mean: float
    tokens_per_bar_std: float
    tokens_per_bar_min: int
    tokens_per_bar_max: int
    ends_with_bar: bool
    grammar_errors: List[GrammarError]
    grammar_error_rate: float              # len(errors) / total_tokens
    repair_needed: bool                    # True if any errors found


def analyse_tokens(tokens: List[str]) -> TokenAnalysis:
    """
    Full structural analysis of a REMIGEN2 token list.

    Args:
        tokens: List of token strings (already split, not a single string).

    Returns:
        TokenAnalysis dataclass.
    """
    errors = check_grammar(tokens)
    bars = segment_bars(tokens)
    bar_lengths = [len(b) for b in bars]

    instruments: List[str] = []
    seen_insts = set()
    for tok in tokens:
        if _prefix(tok) == INST_PREFIX and tok not in seen_insts:
            seen_insts.add(tok)
            instruments.append(tok)

    num_notes = sum(1 for t in tokens if _prefix(t) == PITCH_PREFIX)
    ends_with_bar = bool(tokens) and _prefix(tokens[-1]) == BAR_PREFIX

    tpb_mean = statistics.mean(bar_lengths) if bar_lengths else 0.0
    tpb_std  = statistics.stdev(bar_lengths) if len(bar_lengths) > 1 else 0.0
    tpb_min  = min(bar_lengths) if bar_lengths else 0
    tpb_max  = max(bar_lengths) if bar_lengths else 0

    total = len(tokens)
    error_rate = len(errors) / total if total > 0 else 0.0

    return TokenAnalysis(
        total_tokens=total,
        num_bars=len(bars),
        num_notes=num_notes,
        num_instruments=len(seen_insts),
        instruments=sorted(instruments),
        tokens_per_bar_mean=tpb_mean,
        tokens_per_bar_std=tpb_std,
        tokens_per_bar_min=tpb_min,
        tokens_per_bar_max=tpb_max,
        ends_with_bar=ends_with_bar,
        grammar_errors=errors,
        grammar_error_rate=error_rate,
        repair_needed=len(errors) > 0,
    )


# ---------------------------------------------------------------------------
# Token stream cleaning (for MIDI decode fallback)
# ---------------------------------------------------------------------------

def clean_tokens(tokens: List[str]) -> List[str]:
    """
    Remove invalid tokens and repair the p-d-v triplet structure.

    Strategy:
    - Skip tokens with no '-' separator.
    - For p-X: only keep if followed immediately by d-X then v-X, else skip all three.
    - Skip orphan d-X or v-X tokens.
    - All other valid tokens are kept as-is.

    This is applied ONLY as a last resort before MIDI decoding.
    """
    cleaned: List[str] = []
    known_prefixes = {BAR_PREFIX, POS_PREFIX, TS_PREFIX, TEMPO_PREFIX,
                      INST_PREFIX, PITCH_PREFIX, DUR_PREFIX, VEL_PREFIX}
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        pfx = _prefix(tok)

        if pfx is None or pfx not in known_prefixes:
            i += 1
            continue

        if pfx == PITCH_PREFIX:
            if (i + 2 < n
                    and _prefix(tokens[i + 1]) == DUR_PREFIX
                    and _prefix(tokens[i + 2]) == VEL_PREFIX):
                cleaned.extend(tokens[i:i + 3])
                i += 3
            else:
                i += 1
            continue

        if pfx in (DUR_PREFIX, VEL_PREFIX):
            i += 1
            continue

        cleaned.append(tok)
        i += 1

    return cleaned
