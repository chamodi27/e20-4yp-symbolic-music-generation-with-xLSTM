# Stage 5: Filter & Pitch Normalization — Background Execution Guide

Stage 5 processes **99,573 files** through filtering and pitch normalization. It runs in
batches of 5,000 files (configurable in `pipeline_config.py: STAGE5_BATCH_SIZE`) and
saves progress after each batch so it can safely resume after a reboot or crash.

---

## What Was Modified (Batch Mode Implementation)

The original stage 5 ran `filter_and_normalize_v2.py` as a single subprocess with a
**hard 1-hour timeout** and processed all ~99k files in one pass — causing a timeout
failure. The following changes were made:

### `pipeline_config.py`
- Added `STAGE5_BATCH_SIZE = 5000` — number of files per batch
- Added `STAGE5_TIMEOUT = None` — subprocess timeout (None = no limit)
- Added `self.resume = False` to `PipelineConfig` — set by `pipeline_main.py --resume`

### `pipeline_main.py`
- Added `--resume` CLI flag
- Sets `config.resume = args.resume` after argument parsing, which flows through to stage 5

### `stage_05_filter_wrapper.py`
- Reads `STAGE5_BATCH_SIZE` and `STAGE5_TIMEOUT` from `pipeline_config`
- Passes `--batch-size <N>` and `--resume` to the subprocess command
- Replaced hard-coded `timeout=3600` with configurable `STAGE5_TIMEOUT`

### `filter_and_normalize_v2.py` (core changes)
- Added `--batch-size N` CLI arg (0 = all at once, for backward compatibility)
- Added `--resume` CLI arg
- Extracted batch logic into `_process_batch()` helper function
- Refactored `main()` to:
  - Split input file list into batches of size N
  - Skip already-processed files on `--resume` (checks `06_pitch_normalize/*.mid`)
  - Maintain a **global duplicate-signature set** across all batches (persisted to `progress.json`)
  - Save manifest CSV **after every batch** (not just at the end)
  - Write `progress.json` checkpoint after each batch for safe resume

---

## Run (First Time)

```bash
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/scripts/museformer-preprocess

nohup /home/e20363/miniconda3/envs/museformer/bin/python pipeline_main.py \
  --stage 5 --mode prod \
  >> /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/nohup_stage5.log 2>&1 &

echo $! > /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid
echo "Started with PID: $!"
```

---

## Resume After Reboot / Crash

```bash
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/scripts/museformer-preprocess

nohup /home/e20363/miniconda3/envs/museformer/bin/python pipeline_main.py \
  --stage 5 --mode prod --resume \
  >> /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/nohup_stage5.log 2>&1 &

echo $! > /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid
echo "Resumed with PID: $!"
```

> **`--resume`** skips files already present in `06_pitch_normalize/` and restores the
> global duplicate-signature set from `06_pitch_normalize/progress.json`. Use this every
> time you restart after an interruption.

---

## Monitoring

```bash
# Live pipeline log (structured, with timestamps)
tail -f /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.log

# nohup log (stdout/stderr, useful for tracebacks)
tail -f /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/nohup_stage5.log

# Check if process is alive
ps -p $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid)

# One-liner status
kill -0 $(cat .../logs/pipeline.pid) && echo "RUNNING" || echo "DEAD"

# Check progress (files produced so far)
ls /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/06_pitch_normalize/*.mid 2>/dev/null | wc -l
```

---

## Kill

```bash
# Graceful
kill $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid)

# Force kill
kill -9 $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid)
```

---

## Output Files

| Path | Description |
|------|-------------|
| `06_pitch_normalize/*.mid` | Final output MIDI files |
| `06_pitch_normalize/progress.json` | Batch checkpoint (used by `--resume`) |
| `06_pitch_normalize/pipeline_summary.json` | Final stats (written on completion) |
| `logs/pipeline.log` | Structured pipeline log |
| `logs/nohup_stage5.log` | nohup stdout/stderr |
| `logs/pipeline.pid` | PID of the running process |

---

## Configuration

Edit `pipeline_config.py` to adjust:

```python
STAGE5_BATCH_SIZE = 5000   # files per batch (lower = less memory, more frequent checkpoints)
STAGE5_TIMEOUT   = None    # subprocess timeout in seconds (None = no limit)
```
