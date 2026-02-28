"""
Stage 5: Filter and Pitch Normalization

Temporarily modifies filter_config_v2.py and runs filter_and_normalize_v2.py.
Supports batch mode and resume via pipeline_config.STAGE5_BATCH_SIZE and
pipeline_config.STAGE5_TIMEOUT, and config.resume flag.
"""

from pathlib import Path
import subprocess
import logging
import shutil
from typing import Dict, Any
import sys

from manifest_utils import ManifestManager
from pipeline_config import PipelineConfig, STAGE_NAMES, STAGE5_BATCH_SIZE, STAGE5_TIMEOUT

logger = logging.getLogger(__name__)


def run_stage_5(config: PipelineConfig, manifest: ManifestManager) -> dict:
    """
    Run Stage 5: Filtering and pitch normalization.
    
    Temporarily modifies filter_config_v2.py paths and runs filter_and_normalize_v2.py.
    
    Args:
        config: Pipeline configuration
        manifest: Manifest manager
    
    Returns:
        Dict with stage statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 5: Filtering and Pitch Normalization")
    logger.info("=" * 80)
    logger.info("")
    
    # Create output directories
    if config.mode == 'dev':
        config.dirs['filtered'].mkdir(parents=True, exist_ok=True)
    config.dirs['pitch_norm'].mkdir(parents=True, exist_ok=True)
    
    # Get path to filter config
    script_dir = Path(__file__).parent
    filter_config_path = script_dir / "filter_config_v2.py"
    filter_config_backup = script_dir / "filter_config_v2.py.backup"
    
    # Backup original config
    shutil.copy2(filter_config_path, filter_config_backup)
    logger.info(f"Backed up filter config to: {filter_config_backup}")
    
    try:
        # Read original config
        with open(filter_config_path, 'r') as f:
            lines = f.readlines()
        
        # Modify paths in the config file
        modified_lines = []
        for line in lines:
            if 'INPUT_NORMALIZED_DIR = Path(' in line and not line.strip().startswith('#'):
                modified_lines.append(f'INPUT_NORMALIZED_DIR = Path("{config.dirs["musescore_norm"]}")\n')
            elif 'OUTPUT_FILTERED_DIR = Path(' in line and not line.strip().startswith('#'):
                modified_lines.append(f'OUTPUT_FILTERED_DIR = Path("{config.dirs["filtered"]}")\n')
            elif 'OUTPUT_PITCH_NORM_DIR = Path(' in line and not line.strip().startswith('#'):
                modified_lines.append(f'OUTPUT_PITCH_NORM_DIR = Path("{config.dirs["pitch_norm"]}")\n')
            elif 'MANIFEST_PATH = Path(' in line and not line.strip().startswith('#'):
                modified_lines.append(f'MANIFEST_PATH = Path("{config.manifest_path}")\n')
            else:
                modified_lines.append(line)
        
        # Write modified config
        with open(filter_config_path, 'w') as f:
            f.writelines(modified_lines)
        
        logger.info("Updated filter_config_v2.py with pipeline paths")
        
        # Get filter preset from pipeline config (reload to get fresh value)
        import importlib
        import pipeline_config
        importlib.reload(pipeline_config)
        FILTER_PRESET = pipeline_config.FILTER_PRESET
        
        # Log configuration
        logger.info("Filter Configuration:")
        logger.info(f"  Input dir: {config.dirs['musescore_norm']}")
        logger.info(f"  Output filtered dir: {config.dirs['filtered']}")
        logger.info(f"  Output pitch norm dir: {config.dirs['pitch_norm']}")
        logger.info(f"  Manifest: {config.manifest_path}")
        
        if FILTER_PRESET:
            logger.info(f"  Filter preset: {FILTER_PRESET}")
        
        # Import to log other settings
        import filter_config_v2 as filter_config
        logger.info(f"  Track detection: {filter_config.TRACK_DETECTION_METHOD}")
        logger.info(f"  Required program: {filter_config.REQUIRED_PROGRAM_NUMBER}")
        logger.info(f"  Tempo range: {filter_config.TEMPO_MIN}-{filter_config.TEMPO_MAX} BPM")
        logger.info(f"  Pitch range: {filter_config.PITCH_MIN}-{filter_config.PITCH_MAX}")
        logger.info(f"  Max note duration: {filter_config.MAX_NOTE_DURATION_BEATS} beats")
        logger.info(f"  Max empty bars: {filter_config.MAX_EMPTY_BARS_ALLOWED}")
        logger.info(f"  Min tracks: {filter_config.MIN_NONEMPTY_TRACKS}")
        logger.info(f"  Allowed time signatures: {filter_config.ALLOWED_TIME_SIGNATURES}")
        logger.info("")
        
        # Get path to filter script
        filter_script = script_dir / "filter_and_normalize_v2.py"
        
        if not filter_script.exists():
            logger.error(f"Filter script not found: {filter_script}")
            return {'error': 'filter_script_not_found'}
        
        # Build command with preset argument if configured
        cmd = [sys.executable, str(filter_script)]
        if FILTER_PRESET:
            cmd.extend(['--preset', FILTER_PRESET])

        # Add batch-size argument
        batch_size = STAGE5_BATCH_SIZE
        if batch_size and batch_size > 0:
            cmd.extend(['--batch-size', str(batch_size)])
            logger.info(f"  Batch size: {batch_size}")
        else:
            logger.info("  Batch size: unlimited (single pass)")

        # Add resume flag if requested
        if getattr(config, 'resume', False):
            cmd.append('--resume')
            logger.info("  Resume mode: ON (skipping already-processed files)")

        logger.info(f"Running command: {' '.join(str(c) for c in cmd)}")
        logger.info("")

        # Determine timeout
        timeout = STAGE5_TIMEOUT  # None = no limit
        if timeout:
            logger.info(f"  Subprocess timeout: {timeout}s")
        else:
            logger.info("  Subprocess timeout: none (batch mode manages its own progress)")

        # Run the filter script
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Log the output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"  STDERR: {line}")
        
        if result.returncode != 0:
            logger.error(f"Filter script failed with exit code {result.returncode}")
            return {'error': f'filter_script_failed_code_{result.returncode}'}
        
        logger.info("")
        logger.info("Filter script completed successfully")
        logger.info("")
        
        # Clean up temporary directories in prod mode
        if config.mode == 'prod':
            logger.info("Production mode: Cleaning up temporary directories...")
            
            # Remove filtered directory if it exists
            if config.dirs['filtered'].exists():
                shutil.rmtree(config.dirs['filtered'])
                logger.info(f"  Removed: {config.dirs['filtered']}")
            
            logger.info("")
        
        # Read summary statistics if available
        summary_path = config.dirs['pitch_norm'] / "pipeline_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            
            return {
                'total': summary.get('input_files', 0),
                'parse_failures': summary.get('parse_failures', 0),
                'duplicates': summary.get('duplicates_removed', 0),
                'filtered_kept': summary.get('filtered_kept', 0),
                'filtered_dropped': summary.get('filtered_dropped', 0),
                'pitch_normalized': summary.get('pitch_normalized', 0),
                'pitch_norm_failures': summary.get('pitch_norm_failures', 0),
            }
        else:
            logger.warning("Summary file not found, returning basic stats")
            return {'status': 'completed'}
    
    except subprocess.TimeoutExpired:
        logger.error("Filter script timed out after 1 hour")
        return {'error': 'timeout'}
    except Exception as e:
        logger.error(f"Error running filter script: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': repr(e)}
    finally:
        # Always restore original config
        if filter_config_backup.exists():
            shutil.move(filter_config_backup, filter_config_path)
            logger.info(f"Restored original filter_config_v2.py")


if __name__ == '__main__':
    # For testing
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from pipeline_config import get_default_config
    
    config = get_default_config()
    config.setup_directories()
    
    manifest = ManifestManager(config.manifest_path)
    manifest.load()
    
    stats = run_stage_5(config, manifest)
    print(f"\nStage 5 Statistics: {stats}")
