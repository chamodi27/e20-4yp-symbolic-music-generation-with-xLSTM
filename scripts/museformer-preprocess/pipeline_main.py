#!/usr/bin/env python3
"""
MuseFormer Preprocessing Pipeline - Main Orchestrator

This script orchestrates the complete preprocessing pipeline for MuseFormer data.
It supports running individual stages or the full pipeline with dev/prod modes.

Usage:
    python pipeline_main.py --stage all --mode dev
    python pipeline_main.py --stage 1
    python pipeline_main.py --stage 3-5 --mode prod
    python pipeline_main.py --stage 2-pre
    python pipeline_main.py --stage 2-post
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Import configuration and utilities
from pipeline_config import PipelineConfig, DEFAULT_BASE_DIR, DEFAULT_MODE
from manifest_utils import ManifestManager

# Import stage modules
import stage_01_parsing as stage1
import stage_02_midiminer_helper as stage2
import stage_03_compress6 as stage3
import stage_04_musescore_norm as stage4
import stage_05_filter_wrapper as stage5


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: PipelineConfig, verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_file = config.dirs['logs'] / 'pipeline.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


# =============================================================================
# STAGE RUNNERS
# =============================================================================

STAGE_RUNNERS = {
    '1': ('Stage 1: Parsing', stage1.run_stage_1),
    '2-pre': ('Stage 2 Pre: Prepare for Midiminer', stage2.run_stage_2_pre),
    '2': ('Stage 2: Automated Midiminer', None),  # Special handling in run_stage
    '2-post': ('Stage 2 Post: Process Midiminer Results', stage2.run_stage_2_post),
    '3': ('Stage 3: Compress to 6 Tracks', stage3.run_stage_3),
    '4': ('Stage 4: MuseScore Normalization', stage4.run_stage_4),
    '5': ('Stage 5: Filter and Pitch Normalization', stage5.run_stage_5),
}


def run_stage(
    stage_id: str,
    config: PipelineConfig,
    manifest: ManifestManager,
    logger: logging.Logger
) -> dict:
    """
    Run a single stage.
    
    Args:
        stage_id: Stage identifier (e.g., '1', '2-pre', '3')
        config: Pipeline configuration
        manifest: Manifest manager
        logger: Logger instance
    
    Returns:
        Dict with stage statistics
    """
    if stage_id not in STAGE_RUNNERS:
        logger.error(f"Unknown stage: {stage_id}")
        return {'error': f'unknown_stage_{stage_id}'}
    
    stage_name, stage_func = STAGE_RUNNERS[stage_id]
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"RUNNING: {stage_name}")
    logger.info("=" * 80)
    logger.info("")
    
    # Special handling for automated Stage 2
    if stage_id == '2':
        try:
            # Step 1: Prepare files (Stage 2 Pre)
            logger.info("Step 1/3: Preparing files for midiminer...")
            stats_pre = stage2.run_stage_2_pre(config, manifest)
            logger.info(f"Prepared {stats_pre.get('copied', 0)} files")
            
            # Step 2: Run midiminer automatically
            logger.info("")
            logger.info("Step 2/3: Running midiminer track separation...")
            stats_midiminer = stage2.run_stage_2_midiminer(config)
            
            if stats_midiminer.get('status') != 'success':
                logger.error("Midiminer execution failed")
                return {
                    'error': 'midiminer_failed',
                    'pre': stats_pre,
                    'midiminer': stats_midiminer
                }
            
            logger.info(f"Processed {stats_midiminer.get('num_files', 0)} files")
            
            # Step 3: Process results (Stage 2 Post)
            logger.info("")
            logger.info("Step 3/3: Processing midiminer results...")
            stats_post = stage2.run_stage_2_post(config, manifest)
            
            logger.info(f"Found {stats_post.get('has_melody', 0)} files with melody")
            
            # Combine statistics
            return {
                'pre': stats_pre,
                'midiminer': stats_midiminer,
                'post': stats_post,
                'total_files': stats_midiminer.get('num_files', 0),
                'files_with_melody': stats_post.get('has_melody', 0),
                'files_without_melody': stats_post.get('no_melody', 0),
            }
        
        except Exception as e:
            logger.error(f"Stage 2 automated failed with exception: {e}", exc_info=True)
            return {'error': repr(e)}
    
    # Standard stage execution
    try:
        stats = stage_func(config, manifest)
        return stats
    except Exception as e:
        logger.error(f"Stage {stage_id} failed with exception: {e}", exc_info=True)
        return {'error': repr(e)}

def parse_stage_spec(stage_spec: str) -> List[str]:
    """
    Parse stage specification into list of stage IDs.
    
    Examples:
        'all' -> ['1', '2-pre', '3', '4', '5']
        '1' -> ['1']
        '3-5' -> ['3', '4', '5']
        '2-pre' -> ['2-pre']
    
    Args:
        stage_spec: Stage specification string
    
    Returns:
        List of stage IDs
    """
    if stage_spec == 'all':
        return ['1', '2', '3', '4', '5']
    
    if stage_spec in STAGE_RUNNERS:
        return [stage_spec]
    
    # Handle range (e.g., '3-5')
    if '-' in stage_spec and stage_spec not in STAGE_RUNNERS:
        try:
            start, end = stage_spec.split('-')
            start_num = int(start)
            end_num = int(end)
            
            stages = []
            for i in range(start_num, end_num + 1):
                stages.append(str(i))
            return stages
        except ValueError:
            pass
    
    return [stage_spec]


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MuseFormer Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages in dev mode
  python pipeline_main.py --stage all --mode dev
  
  # Run only parsing stage
  python pipeline_main.py --stage 1
  
  # Run stages 3-5 in production mode
  python pipeline_main.py --stage 3-5 --mode prod
  
  # Prepare for midiminer
  python pipeline_main.py --stage 2-pre
  
  # Process midiminer results
  python pipeline_main.py --stage 2-post
  
  # Run with strict MuseFormer filtering
  python pipeline_main.py --stage all --mode prod --filter-preset strict

  # Resume stage 5 after a reboot/crash
  python pipeline_main.py --stage 5 --mode prod --resume

Stage IDs:
  1       - Parse MIDI files
  2-pre   - Prepare files for midiminer (copy to input directory)
  2       - Run midiminer automatically (activates conda, runs script)
  2-post  - Process midiminer results
  3       - Compress to 6 tracks
  4       - MuseScore normalization
  5       - Filter and pitch normalization
  all     - Run all automated stages (1, 2, 3, 4, 5)
  3-5     - Run stages 3 through 5
        """
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        help='Stage(s) to run: stage ID, range (e.g., 3-5), or "all"'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dev', 'prod', 'auto'],
        default='auto',
        help='Processing mode: dev (keep intermediates), prod (minimal storage), auto (auto-select)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f'Base directory for dataset (default: {DEFAULT_BASE_DIR})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually running'
    )
    
    parser.add_argument(
        '--filter-preset',
        type=str,
        choices=['strict', 'permissive'],
        default=None,
        help='Filter preset for Stage 5: strict (MuseFormer paper) or permissive (more lenient)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume stage 5: skip files already in 06_pitch_normalize/ and continue from where it left off'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        base_dir=args.base_dir,
        mode=args.mode if args.mode != 'auto' else DEFAULT_MODE,
        auto_mode=(args.mode == 'auto')
    )
    
    # Determine mode if auto
    if args.mode == 'auto':
        config.mode = config.determine_mode()
        print(f"Auto-selected mode: {config.mode}")
    
    # Override filter preset if specified
    if args.filter_preset:
        import pipeline_config
        pipeline_config.FILTER_PRESET = args.filter_preset
        print(f"Filter preset: {args.filter_preset}")

    # Set resume flag on config (consumed by stage_05_filter_wrapper)
    config.resume = args.resume
    if args.resume:
        print("Resume mode: stage 5 will skip already-processed files")
    
    # Setup directories
    config.setup_directories(mode=config.mode)
    
    # Setup logging
    logger = setup_logging(config, verbose=args.verbose)
    
    logger.info("=" * 80)
    logger.info("MuseFormer Preprocessing Pipeline")
    logger.info("=" * 80)
    logger.info(f"Base directory: {config.base_dir}")
    logger.info(f"Processing mode: {config.mode}")
    logger.info(f"Stage(s): {args.stage}")
    logger.info("")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Parse stage specification
    stages_to_run = parse_stage_spec(args.stage)
    logger.info(f"Stages to run: {', '.join(stages_to_run)}")
    logger.info("")
    
    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        logger.info(f"Would run stages: {', '.join(stages_to_run)}")
        sys.exit(0)
    
    # Load manifest
    manifest = ManifestManager(config.manifest_path)
    manifest.load()
    
    logger.info(f"Loaded manifest: {len(manifest.df)} files")
    logger.info("")
    
    # Run stages
    all_stats = {}
    
    for stage_id in stages_to_run:
        stats = run_stage(stage_id, config, manifest, logger)
        all_stats[stage_id] = stats
        
        # Check for errors
        if 'error' in stats:
            logger.error(f"Stage {stage_id} failed: {stats['error']}")
            
            # Stop on critical errors
            if stage_id in ['1', '2-post', '3']:
                logger.error("Critical stage failed, stopping pipeline")
                break
    
    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    for stage_id, stats in all_stats.items():
        stage_name = STAGE_RUNNERS.get(stage_id, (stage_id, None))[0]
        logger.info(f"{stage_name}:")
        
        if 'error' in stats:
            logger.info(f"  Status: FAILED - {stats['error']}")
        else:
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
    
    logger.info("")
    logger.info(f"Manifest: {config.manifest_path}")
    logger.info(f"Final output: {config.dirs['pitch_norm']}")
    logger.info("")
    
    # Get final statistics from manifest
    manifest_stats = manifest.get_statistics()
    logger.info("Manifest Statistics:")
    logger.info(f"  Total files: {manifest_stats['total_files']}")
    logger.info(f"  By status: {manifest_stats['by_status']}")
    
    logger.info("")
    logger.info("Pipeline complete!")


if __name__ == '__main__':
    main()
