# Running the Pipeline with nohup

Use `nohup` to run the pipeline as a background process that **survives terminal disconnects and tmux crashes**.

## Why nohup?

- Process is owned by `init` (PID 1), not your shell or tmux session
- Survives SSH disconnects, terminal closures, and tmux server crashes
- No dependency on any session manager

---

## Running the Pipeline

```bash
cd /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/scripts/museformer-preprocess

nohup /home/e20363/miniconda3/envs/museformer/bin/python pipeline_main.py --stage 4-5 --mode prod \
  > /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/nohup_stage45.log 2>&1 &

echo $! > /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid
echo "Pipeline started with PID: $!"
```

> **Note:** Uses the full path to the `museformer` conda environment's Python binary.
> This avoids the need for `conda activate`, which doesn't work in non-interactive shells.

### Common Stage Variants

```bash
# Stage 4 only (MuseScore normalization)
nohup /home/e20363/miniconda3/envs/museformer/bin/python pipeline_main.py --stage 4 --mode prod \
  > .../logs/nohup_stage4.log 2>&1 &

# Stage 5 only (Filter & pitch normalization)
nohup /home/e20363/miniconda3/envs/museformer/bin/python pipeline_main.py --stage 5 --mode prod \
  > .../logs/nohup_stage5.log 2>&1 &

# Full pipeline
nohup /home/e20363/miniconda3/envs/museformer/bin/python pipeline_main.py --stage all --mode prod \
  > .../logs/nohup_all.log 2>&1 &
```

---

## Observability

### Check if the process is running

```bash
ps -p $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid)
```

One-liner status check:
```bash
kill -0 $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid) && echo "RUNNING" || echo "DEAD"
```

### Watch live pipeline progress

```bash
# Pipeline's own structured log (timestamps + log levels)
tail -f /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.log

# nohup log (stdout/stderr, useful for catching crashes/tracebacks)
tail -f /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/nohup_stage45.log
```

### Check for errors

```bash
grep ERROR .../logs/pipeline.log | tail -20
```

### View recent progress

```bash
grep "Normalized" .../logs/pipeline.log | tail -5
```

---

## Killing the Process

### Graceful kill (SIGTERM)

```bash
kill $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid)
```

### Force kill (SIGKILL) — if graceful kill doesn't work

```bash
kill -9 $(cat /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/full/logs/pipeline.pid)
```

### Clean up orphaned MuseScore child processes

After a force kill, MuseScore subprocesses may linger:
```bash
pkill -f "MuseScore"
```

---

## Log Files Reference

| File | Description |
|------|-------------|
| `logs/pipeline.log` | Structured pipeline log (written by the pipeline itself) |
| `logs/nohup_stage45.log` | Raw stdout/stderr from nohup (crashes, tracebacks) |
| `logs/pipeline.pid` | PID of the last nohup-launched pipeline process |
