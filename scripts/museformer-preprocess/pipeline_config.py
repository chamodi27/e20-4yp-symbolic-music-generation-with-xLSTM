"""
Centralized Configuration for MuseFormer Preprocessing Pipeline

This module contains all configuration settings for the preprocessing pipeline,
including paths, stage settings, and processing parameters.
"""

from pathlib import Path
from typing import Literal, List

# =============================================================================
# BASE PATHS
# =============================================================================

# Root directory for the project
FYP_MUSICGEN_ROOT = Path("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen")

# Base directory for dataset (can be overridden via CLI)
DEFAULT_BASE_DIR = FYP_MUSICGEN_ROOT / "data" / "museformer_baseline" / "29k"

# =============================================================================
# PROCESSING MODE
# =============================================================================

# Processing mode: 'dev' or 'prod'
# - dev: Keep all intermediate files (for testing/debugging)
# - prod: Minimal storage, streaming pipeline (for large datasets)
ProcessingMode = Literal['dev', 'prod']
DEFAULT_MODE: ProcessingMode = 'dev'

# Auto-select mode based on dataset size
AUTO_MODE_THRESHOLD = 5000  # Files: use 'prod' if >= this many files

# =============================================================================
# STAGE DIRECTORIES (relative to base_dir)
# =============================================================================

STAGE_DIRS = {
    'raw': '00_raw',
    'parsed': '01_parsed_results',
    'midiminer_input': '02_a_midiminer',
    'midiminer_output': '02_b_midiminer_results',
    'compressed': '03_compressed6',
    'musescore_norm': '04_musescore_norm',
    'filtered': '05_filtered',
    'pitch_norm': '06_pitch_normalize',
    'logs': 'logs',
}

# =============================================================================
# MANIFEST
# =============================================================================

MANIFEST_FILENAME = 'manifest.csv'

# Stage names for manifest tracking
STAGE_NAMES = {
    'initial': 'initial',
    'parsing': '01_parsed_meta',
    'midiminer': '02_b_midiminer',
    'compress': '03_compressed6',
    'musescore': '04_musescore_norm',
    'filter': '05_filtered',
    'pitch_norm': '06_pitch_normalize',
}

# =============================================================================
# STAGE 1: PARSING
# =============================================================================

# MIDI file extensions to search for
MIDI_EXTENSIONS = ['*.mid', '*.midi', '*.MID', '*.MIDI']

# Whether to copy parsed files in dev mode
PARSING_COPY_FILES_DEV = True

# =============================================================================
# STAGE 2: MIDIMINER
# =============================================================================

# Midiminer repository path
MIDIMINER_REPO = FYP_MUSICGEN_ROOT / "repos" / "midi-miner"

# Midiminer script path
MIDIMINER_SCRIPT = MIDIMINER_REPO / "track_separate.py"

# Midiminer conda environment name
MIDIMINER_CONDA_ENV = "midiminer"

# Temp directories for midiminer (required for joblib)
MIDIMINER_TEMP_DIR = FYP_MUSICGEN_ROOT / "temp"
MIDIMINER_JOBLIB_TEMP = MIDIMINER_TEMP_DIR / "joblib"

# Midiminer execution parameters
MIDIMINER_TARGET_TRACK = "melody"  # Track type to extract
MIDIMINER_NUM_CORES = 4  # Number of CPU cores to use

# Midiminer output JSON filename
MIDIMINER_JSON_FILENAME = 'program_result.json'
MIDIMINER_JSON_PATH = FYP_MUSICGEN_ROOT / "data" / "museformer_baseline" / "29k" / "02_b_midiminer_results" / MIDIMINER_JSON_FILENAME

# Whether to delete midiminer input directory after processing (prod mode)
MIDIMINER_CLEANUP_INPUT = True

# =============================================================================
# STAGE 2: MIDIMINER BATCH PROCESSING
# =============================================================================

# Batch size for midiminer processing (to prevent OOM crashes)
# Process files in batches to reduce memory usage
MIDIMINER_BATCH_SIZE = 5000  # Process 15k files per batch

# Whether to enable batch processing (auto-detect if None)
# - None: Auto-enable if total files > batch size
# - True: Always use batch processing
# - False: Never use batch processing (process all at once)
MIDIMINER_ENABLE_BATCHING = True

# =============================================================================
# STAGE 3: COMPRESSION TO 6 TRACKS
# =============================================================================

# Canonical MIDI programs for 6-track output
CANONICAL_PROGRAMS = {
    'square_synth': 80,  # Lead 1 (square)
    'piano': 0,          # Acoustic Grand Piano
    'guitar': 24,        # Acoustic Guitar (nylon)
    'string': 48,        # String Ensemble 1
    'bass': 32,          # Acoustic Bass
}

# GM family ranges for instrument classification
FAMILY_RANGES = {
    'piano': [(0, 7)],
    'guitar': [(24, 31)],
    'string': [(40, 51), (88, 95)],  # Strings + synth strings/pads
}

# Polyphony caps for non-melody tracks
POLYPHONY_CAP = {
    'piano': 10,
    'guitar': 6,
    'string': 6,
}

# Number of top instruments to keep per family
KEEP_TOP_K_PER_FAMILY = 2

# Whether to save compressed files in dev mode
COMPRESS_SAVE_FILES_DEV = True

# =============================================================================
# STAGE 4: MUSESCORE NORMALIZATION
# =============================================================================

# MuseScore AppImage path
MUSESCORE_APPIMAGE = FYP_MUSICGEN_ROOT / "musescore" / "MuseScore-Studio-4.6.5.253511702-x86_64.AppImage"

# MIDI import options XML
MUSESCORE_IMPORT_OPTIONS = FYP_MUSICGEN_ROOT / "musescore" / "midi_import_options.xml"

# MuseScore environment variables
MUSESCORE_ENV = {
    'SKIP_LIBJACK': '1',
    'QT_QPA_PLATFORM': 'offscreen',
    'XDG_CONFIG_HOME': str(FYP_MUSICGEN_ROOT / "musescore" / ".config"),
    'XDG_DATA_HOME': str(FYP_MUSICGEN_ROOT / "musescore" / ".local" / "share"),
    'XDG_CACHE_HOME': str(FYP_MUSICGEN_ROOT / "musescore" / ".cache"),
}

# Timeout per file (seconds)
MUSESCORE_TIMEOUT = 300

# Whether to fix track names after MuseScore processing
MUSESCORE_FIX_TRACK_NAMES = True

# Whether to save normalized files in dev mode
MUSESCORE_SAVE_FILES_DEV = True

# =============================================================================
# STAGE 5: FILTERING AND PITCH NORMALIZATION
# =============================================================================

# Import filter config from existing module
# (filter_config_v2.py contains all filtering parameters)

# Filter preset: None, 'strict', or 'permissive'
# - None: Use default configuration from filter_config_v2.py
# - 'strict': Use strict MuseFormer paper configuration
# - 'permissive': Use more permissive filtering
FILTER_PRESET = 'strict'  # Options: None, 'strict', 'permissive'

# Whether to save filtered files (before pitch norm) in dev mode
FILTER_SAVE_INTERMEDIATE_DEV = True

# Batch size for stage 5 processing (0 = process all at once)
# Recommended: 5000 for ~99k files to avoid timeout
STAGE5_BATCH_SIZE = 5000

# Subprocess timeout for stage 5 in seconds (None = no limit)
# Set to None when using batch mode since batches self-manage
STAGE5_TIMEOUT = None

# =============================================================================
# LOGGING
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = 'INFO'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Whether to log to file
LOG_TO_FILE = True

# Log filename (relative to logs directory)
LOG_FILENAME = 'pipeline.log'

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

# Number of parallel workers (None = auto-detect)
NUM_WORKERS = None

# Batch size for processing
BATCH_SIZE = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

class PipelineConfig:
    """Configuration object for pipeline execution."""
    
    def __init__(
        self,
        base_dir: Path = DEFAULT_BASE_DIR,
        mode: ProcessingMode = DEFAULT_MODE,
        auto_mode: bool = True,
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            base_dir: Base directory for dataset
            mode: Processing mode ('dev' or 'prod')
            auto_mode: Whether to auto-select mode based on dataset size
        """
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.auto_mode = auto_mode

        # Resume flag: if True, stage 5 skips already-processed files
        # Set by pipeline_main.py --resume flag
        self.resume = False
        
        # Create directory paths
        self.dirs = {
            name: self.base_dir / path
            for name, path in STAGE_DIRS.items()
        }
        
        # Manifest path
        self.manifest_path = self.dirs['logs'] / MANIFEST_FILENAME
        
        # Midiminer JSON path
        self.midiminer_json = self.dirs['midiminer_input'] / MIDIMINER_JSON_FILENAME
        
    def setup_directories(self, mode: ProcessingMode = None) -> None:
        """
        Create necessary directories based on mode.
        
        Args:
            mode: Processing mode (uses self.mode if None)
        """
        mode = mode or self.mode
        
        # Always create these
        always_create = ['raw', 'logs', 'pitch_norm']
        
        if mode == 'dev':
            # Create all directories in dev mode
            dirs_to_create = list(STAGE_DIRS.keys())
        else:
            # Only create essential directories in prod mode
            dirs_to_create = always_create + ['midiminer_input']
        
        for name in dirs_to_create:
            self.dirs[name].mkdir(parents=True, exist_ok=True)
    
    def determine_mode(self) -> ProcessingMode:
        """
        Determine processing mode based on dataset size.
        
        Returns:
            Processing mode ('dev' or 'prod')
        """
        if not self.auto_mode:
            return self.mode
        
        # Count files in raw directory
        if not self.dirs['raw'].exists():
            return self.mode
        
        num_files = sum(
            1 for ext in MIDI_EXTENSIONS
            for _ in self.dirs['raw'].glob(ext)
        )
        
        if num_files >= AUTO_MODE_THRESHOLD:
            return 'prod'
        else:
            return 'dev'
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check base directory exists
        if not self.base_dir.exists():
            errors.append(f"Base directory does not exist: {self.base_dir}")
        
        # Check raw directory exists
        if not self.dirs['raw'].exists():
            errors.append(f"Raw directory does not exist: {self.dirs['raw']}")
        
        # Check MuseScore AppImage exists
        if not MUSESCORE_APPIMAGE.exists():
            errors.append(f"MuseScore AppImage not found: {MUSESCORE_APPIMAGE}")
        
        # Check MIDI import options exist
        if not MUSESCORE_IMPORT_OPTIONS.exists():
            errors.append(f"MIDI import options not found: {MUSESCORE_IMPORT_OPTIONS}")
        
        return errors
    
    def __repr__(self) -> str:
        return f"PipelineConfig(base_dir={self.base_dir}, mode={self.mode})"


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()
