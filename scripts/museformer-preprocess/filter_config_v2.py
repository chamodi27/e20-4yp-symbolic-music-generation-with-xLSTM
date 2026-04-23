"""
Improved Configuration for MuseFormer MIDI Filtering with Robust Track Detection

This version handles inconsistent track naming from MuseScore normalization
and provides multiple methods for detecting the required melody track.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Input: Directory containing MuseScore-normalized MIDI files
INPUT_NORMALIZED_DIR = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/29k/04_musescore_norm")

# Output: Directory for filtered files (before pitch normalization)
OUTPUT_FILTERED_DIR = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/29k/05_filtered")

# Output: Directory for pitch-normalized files (final output)
OUTPUT_PITCH_NORM_DIR = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/29k/06_pitch_normalize")

# Manifest CSV file to update with processing results
MANIFEST_PATH = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/29k/logs/29k_manifest.csv")

# Stage names for manifest tracking
STAGE_FILTER = "05_filtered"
STAGE_PITCH_NORM = "06_pitch_normalize"


# =============================================================================
# TRACK DETECTION (IMPROVED)
# =============================================================================

# Method for detecting the required melody track:
#   "name_exact": Exact match on track name (original, strict)
#   "name_flexible": Contains match on track name (handles "Piano, piano")
#   "program": Match by MIDI program number (most robust)
#   "program_or_name": Try program first, fall back to name (recommended)
TRACK_DETECTION_METHOD = "program_or_name"

# Required track name for "name_exact" and "name_flexible" methods
# MuseFormer uses "square_synth" for melody track
REQUIRED_TRACK_NAME = "square_synth"

# Alternative track names to check (for "name_flexible" method)
# These will be checked in order if exact match fails
ALTERNATIVE_TRACK_NAMES = [
    "square_synth",      # Exact
    "square_synthesizer", # Variation
    "lead",              # Generic alternative
    "melody",            # Generic alternative
]

# MIDI Program number for square wave/lead synth
# Square wave is typically program 80 (Lead 1 - square)
# This is the most reliable method for MuseScore-normalized files
REQUIRED_PROGRAM_NUMBER = 80

# Whether to check for program number in range (more flexible)
# Useful if MuseScore uses slightly different programs
REQUIRED_PROGRAM_RANGE = [80, 81]  # Lead 1 (square) and Lead 2 (sawtooth)

# Whether a file needs exactly one melody track or at least one
REQUIRE_SINGLE_MELODY_TRACK = False  # False = at least one is OK


# =============================================================================
# TIME SIGNATURE FILTERING
# =============================================================================

# Allowed time signatures (MuseFormer paper: only 4/4)
ALLOWED_TIME_SIGNATURES = ["4/4"]

# If True, files with no time signature will be treated as valid (assumed 4/4)
# If False, files must have an explicit time signature
ALLOW_MISSING_TIME_SIGNATURE = True


# =============================================================================
# TRACK REQUIREMENTS
# =============================================================================

# Minimum number of non-empty tracks required
# MuseFormer paper: at least 2 instruments
MIN_NONEMPTY_TRACKS = 2


# =============================================================================
# TEMPO FILTERING
# =============================================================================

# Minimum tempo in BPM (MuseFormer paper: 24 BPM)
TEMPO_MIN = 24

# Maximum tempo in BPM (MuseFormer paper: 200 BPM)
TEMPO_MAX = 200


# =============================================================================
# PITCH RANGE FILTERING
# =============================================================================

# Minimum MIDI pitch allowed (MuseFormer paper: 21 = A0)
PITCH_MIN = 21

# Maximum MIDI pitch allowed (MuseFormer paper: 108 = C8)
PITCH_MAX = 108


# =============================================================================
# NOTE DURATION FILTERING
# =============================================================================

# Maximum note duration in beats (MuseFormer paper: 16 beats = 4 bars in 4/4)
MAX_NOTE_DURATION_BEATS = 16.0


# =============================================================================
# EMPTY BARS FILTERING
# =============================================================================

# Maximum number of consecutive empty bars allowed
# MuseFormer paper: filter files with 4 or more consecutive empty bars
MAX_EMPTY_BARS_ALLOWED = 3

# Method for counting empty bars:
#   "onset" - A bar is empty if no notes START in that bar (original)
#   "sounding" - A bar is empty if no notes are SOUNDING during that bar (stricter)
EMPTY_BAR_METHOD = "sounding"

# Whether to count drum notes when determining if a bar is empty
# True = drums count as notes (bar with only drums is not empty)
# False = ignore drums (bar with only drums is considered empty)
COUNT_DRUMS_FOR_EMPTY_BARS = True


# =============================================================================
# DEGENERATE CONTENT FILTERING
# =============================================================================

# If True, filter out files where all notes have the same pitch
# or all notes have the same duration (degenerate/repetitive content)
DROP_DEGENERATE_CONTENT = True


# =============================================================================
# PITCH NORMALIZATION
# =============================================================================

# Target pitch class for major keys (MuseFormer: C major = 0)
MAJOR_TARGET_PC = 0  # C

# Target pitch class for minor keys (MuseFormer: A minor = 9)
MINOR_TARGET_PC = 9  # A

# Krumhansl-Kessler key profiles for major and minor keys
# These are used for automatic key detection
# Values represent the perceived stability of each pitch class in the key

# Major profile (C major: C=1st position has highest weight)
KK_MAJOR_PROFILE = [
    6.35,  # C  (tonic)
    2.23,  # C#
    3.48,  # D  (supertonic)
    2.33,  # D#
    4.38,  # E  (mediant)
    4.09,  # F  (subdominant)
    2.52,  # F#
    5.19,  # G  (dominant)
    2.39,  # G#
    3.66,  # A  (submediant)
    2.29,  # A#
    2.88,  # B  (leading tone)
]

# Minor profile (A minor: A=1st position has highest weight)
KK_MINOR_PROFILE = [
    6.33,  # A  (tonic)
    2.68,  # A#
    3.52,  # B  (supertonic)
    5.38,  # C  (mediant)
    2.60,  # C#
    3.53,  # D  (subdominant)
    2.54,  # D#
    4.75,  # E  (dominant)
    3.98,  # F  (submediant)
    2.69,  # F#
    3.34,  # G  (subtonic)
    3.17,  # G#
]


# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

# Logging verbosity
# 0 = Errors only
# 1 = Progress updates every 50 files
# 2 = Detailed per-file information
VERBOSITY = 1


# =============================================================================
# TRACK NAME NORMALIZATION (NEW)
# =============================================================================

def normalize_track_name(track_name: str) -> str:
    """
    Normalize MuseScore track names to handle inconsistent formatting.
    
    MuseScore sometimes creates names like:
    - "Square Synthesizer, square_synth" instead of "square_synth"
    - "Piano, piano" instead of "piano"
    
    This function extracts the canonical name.
    
    Args:
        track_name: Raw track name from MIDI file
        
    Returns:
        Normalized track name (lowercase)
    """
    if not track_name:
        return ""
    
    # Convert to lowercase
    name = track_name.lower().strip()
    
    # If name contains comma, extract the part after comma
    # Example: "Piano, piano" -> "piano"
    if "," in name:
        parts = name.split(",")
        # Take the last part (usually the canonical name)
        name = parts[-1].strip()
    
    # Remove extra whitespace
    name = " ".join(name.split())
    
    return name


def track_matches_requirement(inst, method: str = None) -> bool:
    """
    Check if a track matches the melody track requirement.
    
    Supports multiple detection methods to handle MuseScore inconsistencies.
    
    Args:
        inst: miditoolkit Instrument object
        method: Detection method override (uses TRACK_DETECTION_METHOD if None)
        
    Returns:
        True if track matches requirement, False otherwise
    """
    if method is None:
        method = TRACK_DETECTION_METHOD
    
    # Don't consider drum tracks
    if inst.is_drum:
        return False
    
    # Method 1: Exact name match (original, strict)
    if method == "name_exact":
        return inst.name == REQUIRED_TRACK_NAME
    
    # Method 2: Flexible name match (handles "Piano, piano" format)
    elif method == "name_flexible":
        normalized_name = normalize_track_name(inst.name)
        
        # Check exact match first
        if normalized_name == REQUIRED_TRACK_NAME.lower():
            return True
        
        # Check alternatives
        for alt_name in ALTERNATIVE_TRACK_NAMES:
            if normalized_name == alt_name.lower():
                return True
            # Also check if normalized name contains the alternative
            if alt_name.lower() in normalized_name:
                return True
        
        return False
    
    # Method 3: Program number match (most robust)
    elif method == "program":
        # Check exact program number
        if inst.program == REQUIRED_PROGRAM_NUMBER:
            return True
        # Check program range
        if inst.program in REQUIRED_PROGRAM_RANGE:
            return True
        return False
    
    # Method 4: Program OR name (recommended)
    elif method == "program_or_name":
        # Try program first (most reliable)
        if track_matches_requirement(inst, "program"):
            return True
        # Fall back to flexible name matching
        if track_matches_requirement(inst, "name_flexible"):
            return True
        return False
    
    else:
        raise ValueError(f"Unknown track detection method: {method}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration parameters.
    Raises ValueError if any parameters are invalid.
    """
    errors = []
    
    # Path validation
    if not INPUT_NORMALIZED_DIR.exists():
        errors.append(f"Input directory does not exist: {INPUT_NORMALIZED_DIR}")
    
    if MANIFEST_PATH.exists() and not MANIFEST_PATH.is_file():
        errors.append(f"Manifest path exists but is not a file: {MANIFEST_PATH}")
    
    # Numeric range validation
    if TEMPO_MIN <= 0:
        errors.append(f"TEMPO_MIN must be positive, got {TEMPO_MIN}")
    
    if TEMPO_MAX <= TEMPO_MIN:
        errors.append(f"TEMPO_MAX ({TEMPO_MAX}) must be greater than TEMPO_MIN ({TEMPO_MIN})")
    
    if PITCH_MIN < 0 or PITCH_MIN > 127:
        errors.append(f"PITCH_MIN must be in range [0, 127], got {PITCH_MIN}")
    
    if PITCH_MAX < 0 or PITCH_MAX > 127:
        errors.append(f"PITCH_MAX must be in range [0, 127], got {PITCH_MAX}")
    
    if PITCH_MAX <= PITCH_MIN:
        errors.append(f"PITCH_MAX ({PITCH_MAX}) must be greater than PITCH_MIN ({PITCH_MIN})")
    
    if MAX_NOTE_DURATION_BEATS <= 0:
        errors.append(f"MAX_NOTE_DURATION_BEATS must be positive, got {MAX_NOTE_DURATION_BEATS}")
    
    if MAX_EMPTY_BARS_ALLOWED < 0:
        errors.append(f"MAX_EMPTY_BARS_ALLOWED must be non-negative, got {MAX_EMPTY_BARS_ALLOWED}")
    
    if MIN_NONEMPTY_TRACKS < 1:
        errors.append(f"MIN_NONEMPTY_TRACKS must be at least 1, got {MIN_NONEMPTY_TRACKS}")
    
    # String validation
    if EMPTY_BAR_METHOD not in ["onset", "sounding"]:
        errors.append(f"EMPTY_BAR_METHOD must be 'onset' or 'sounding', got '{EMPTY_BAR_METHOD}'")
    
    if TRACK_DETECTION_METHOD not in ["name_exact", "name_flexible", "program", "program_or_name"]:
        errors.append(f"TRACK_DETECTION_METHOD must be one of: name_exact, name_flexible, program, program_or_name")
    
    if not ALLOWED_TIME_SIGNATURES:
        errors.append("ALLOWED_TIME_SIGNATURES cannot be empty")
    
    if not REQUIRED_TRACK_NAME:
        errors.append("REQUIRED_TRACK_NAME cannot be empty")
    
    # Program number validation
    if REQUIRED_PROGRAM_NUMBER < 0 or REQUIRED_PROGRAM_NUMBER > 127:
        errors.append(f"REQUIRED_PROGRAM_NUMBER must be in range [0, 127], got {REQUIRED_PROGRAM_NUMBER}")
    
    # Pitch class validation
    if MAJOR_TARGET_PC < 0 or MAJOR_TARGET_PC > 11:
        errors.append(f"MAJOR_TARGET_PC must be in range [0, 11], got {MAJOR_TARGET_PC}")
    
    if MINOR_TARGET_PC < 0 or MINOR_TARGET_PC > 11:
        errors.append(f"MINOR_TARGET_PC must be in range [0, 11], got {MINOR_TARGET_PC}")
    
    # Key profile validation
    if len(KK_MAJOR_PROFILE) != 12:
        errors.append(f"KK_MAJOR_PROFILE must have 12 elements, got {len(KK_MAJOR_PROFILE)}")
    
    if len(KK_MINOR_PROFILE) != 12:
        errors.append(f"KK_MINOR_PROFILE must have 12 elements, got {len(KK_MINOR_PROFILE)}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))


# Run validation when config is imported
validate_config()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def load_museformer_strict():
    """
    Load strict MuseFormer paper configuration.
    Uses program-based detection for robustness.
    """
    global TEMPO_MIN, TEMPO_MAX, PITCH_MIN, PITCH_MAX
    global MAX_NOTE_DURATION_BEATS, MAX_EMPTY_BARS_ALLOWED
    global ALLOWED_TIME_SIGNATURES, MIN_NONEMPTY_TRACKS
    global DROP_DEGENERATE_CONTENT, EMPTY_BAR_METHOD
    global ALLOW_MISSING_TIME_SIGNATURE, TRACK_DETECTION_METHOD
    
    TEMPO_MIN = 24
    TEMPO_MAX = 200
    PITCH_MIN = 21
    PITCH_MAX = 108
    MAX_NOTE_DURATION_BEATS = 16.0
    MAX_EMPTY_BARS_ALLOWED = 3
    ALLOWED_TIME_SIGNATURES = ["4/4"]
    MIN_NONEMPTY_TRACKS = 2
    DROP_DEGENERATE_CONTENT = True
    EMPTY_BAR_METHOD = "onset"
    ALLOW_MISSING_TIME_SIGNATURE = False
    TRACK_DETECTION_METHOD = "program_or_name"  # More robust than name_exact
    
    validate_config()
    print("Loaded: MuseFormer Strict configuration (with robust track detection)")


def load_permissive():
    """
    Load more permissive configuration.
    Allows wider ranges and more flexibility.
    """
    global TEMPO_MIN, TEMPO_MAX, PITCH_MIN, PITCH_MAX
    global MAX_NOTE_DURATION_BEATS, MAX_EMPTY_BARS_ALLOWED
    global ALLOWED_TIME_SIGNATURES, MIN_NONEMPTY_TRACKS
    global DROP_DEGENERATE_CONTENT, EMPTY_BAR_METHOD
    global ALLOW_MISSING_TIME_SIGNATURE, TRACK_DETECTION_METHOD
    
    TEMPO_MIN = 20
    TEMPO_MAX = 240
    PITCH_MIN = 12
    PITCH_MAX = 120
    MAX_NOTE_DURATION_BEATS = 32.0
    MAX_EMPTY_BARS_ALLOWED = 7
    ALLOWED_TIME_SIGNATURES = ["4/4", "3/4", "2/4", "6/8", "5/4"]
    MIN_NONEMPTY_TRACKS = 1
    DROP_DEGENERATE_CONTENT = False
    EMPTY_BAR_METHOD = "onset"
    ALLOW_MISSING_TIME_SIGNATURE = True
    TRACK_DETECTION_METHOD = "program_or_name"
    
    validate_config()
    print("Loaded: Permissive configuration")