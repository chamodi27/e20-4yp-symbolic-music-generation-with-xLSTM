# Backend API Plan — xLSTM Music Generation

> **Goal**: Create a self-contained Flask API that accepts a prompt and sequence length, generates music using the recurrent `xLSTMGenerator`, converts to MIDI, and returns the file. Deployable via Docker on any server with an NVIDIA GPU.

---

## 1. Architecture Overview

```
Client (browser / curl / frontend)
       │
       │  POST /generate  { prompt, length, temperature }
       ▼
┌─────────────────────────┐
│     Flask API (app.py)  │
│                         │
│  1. Validate input      │
│  2. generator.generate()│  ← xLSTMGenerator (recurrent, O(N))
│  3. Convert tokens→MIDI │  ← MIDIConverter (midiprocessor)
│  4. Return .mid file    │
└─────────────────────────┘
       │
       │  Model loaded ONCE at startup
       │  GPU memory held permanently
       ▼
   checkpoint-46000-last/
```

The API is **synchronous** — the client sends a request and waits for the response (up to ~77s for 12,288 tokens). For a demo with 1–3 concurrent users, this is sufficient.

---

## 2. File Structure

```
backend/
├── backend-plan.md          ← This file
│
├── app.py                   ← Flask server (endpoints + CORS + validation)
├── generator.py             ← xLSTMGenerator (copied from inference.py, helibrunna removed)
├── converter.py             ← MIDIConverter (copied from xLSTM-3, simplified)
├── token_analysis.py        ← Grammar analysis (copied from xLSTM-3, as-is)
│
├── environment.yml          ← Conda env for manual setup (no Docker)
├── requirements.txt         ← Pip packages (used by Dockerfile)
├── Dockerfile               ← Docker image definition
├── .dockerignore             ← Exclude unnecessary files from image
│
└── README.md                ← Setup & deployment instructions
```

### Model Checkpoint — NOT in the backend folder

The checkpoint folder (~1 GB) is **not copied** into `backend/`. Instead, the app reads its path from an environment variable `MODEL_PATH` at startup.

```
# Checkpoint lives here (on visionsrv, not inside backend/):
/scratch1/.../helibrunna/output/
  xlstm_lmd_preprocessed_512d_4096ctx_12b/
    run_20260301-1906/
      checkpoint-46000-last/
        ├── config.yaml
        ├── model.safetensors    (~1 GB — stays here, not copied)
        └── tokenizer.json
```

**Why not copy it:**
1. **Size** — 1 GB in a code folder is impractical for version control or packaging.
2. **Docker volumes** — For Docker deployment, the checkpoint is mounted at runtime (not baked into the image), keeping the image small and allowing checkpoint swaps without rebuilds.
3. **Flexibility** — You can point to any checkpoint by changing one environment variable.

**How the app finds it:**
```bash
# Local (on visionsrv):
export MODEL_PATH=/path/to/checkpoint-46000-last
python app.py

# Docker:
docker run --gpus all -p 5000:5000 \
  -e MODEL_PATH=/app/model \
  -v /path/to/checkpoint-46000-last:/app/model \
  xlstm-music-api
```

---

## 3. Zero External Repo Dependencies

The backend is **fully self-contained** — no helibrunna or MidiProcessor repos needed at deploy time.

| Original dependency | How we handle it |
|---|---|
| `helibrunna` (only used `model_from_config()`) | **Inlined** into `generator.py` (~7 lines of logic for xLSTM) |
| `helibrunna.LanguageModel` | **Not used** — we bypass it entirely with `xLSTMGenerator` |
| `MidiProcessor` repo | **Already pip-installable** as `midiprocessor==0.1.5` — no cloning needed |

---

## 4. Changes to Copied Files

### `generator.py` (from `notebooks/xLSTM-4-recurrent-state/inference.py`)

| Change | Reason |
|---|---|
| Remove `from helibrunna.source.utilities import model_from_config` | Eliminate helibrunna dependency |
| Remove `from helibrunna.source.languagemodel import LanguageModel` | Not needed |
| Remove `sys.path.append(...)` for repos | Not needed |
| Inline `model_from_config()` as a private method | Self-contained (~7 lines: uses `dacite.from_dict` + `xLSTMLMModelConfig`) |

### `converter.py` (from `notebooks/xLSTM-3/generate/converter.py`)

| Change | Reason |
|---|---|
| Remove `midiprocessor_path` constructor param | `midiprocessor` is pip-installed |
| Remove `sys.path.insert(0, ...)` | Not needed |
| Fix `from token_analysis import clean_tokens` import path | Same directory |

### `token_analysis.py` (from `notebooks/xLSTM-3/generate/token_analysis.py`)

| Change | Reason |
|---|---|
| None | Pure Python with zero external deps — copies as-is |

---

## 5. API Specification

### `POST /generate`

**Request** (JSON):
```json
{
  "prompt": "s-9 o-0 t-38",
  "length": 4096,
  "temperature": 0.8,
  "seed": 42
}
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `prompt` | string | `"s-9 o-0 t-38"` | Non-empty REMIGEN2 tokens |
| `length` | int | `2048` | `100 ≤ length ≤ 12288` |
| `temperature` | float | `0.8` | `0.1 ≤ temperature ≤ 2.0` |
| `seed` | int or null | `null` (random) | Optional |

**Response**: Binary `.mid` file (`Content-Type: audio/midi`).

**Custom response headers** (metadata):
- `X-Generation-Time`, `X-Tokens-Per-Second`, `X-Actual-Tokens`, `X-Grammar-Error-Rate`

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`.

---

## 6. Environment & Dependencies

### Dependency Sources (Pinned Versions)

We use **exact tested versions** from two verified environments to avoid compatibility issues:

**From the xlstm official environment (`environment_pt240cu124.yaml`):**
- Python 3.11, PyTorch 2.4.0, CUDA 12.4, ninja, cmake, gxx compiler
- `dacite==1.9.2`, `omegaconf==2.3.0`, `numpy<2.0`

**From the tested midi-decode environment (`midi-decode-test-env`):**
- `midiprocessor==0.1.5`, `miditoolkit==0.1.16`, `mido==1.3.3`
- `pretty-midi==0.2.11`, `numpy==1.23.5`, `scipy==1.10.1`

> **⚠️ Important:** The `numpy` and `miditoolkit` versions are critical. Different versions cause silent failures in MidiProcessor. Do not change them without testing.

**Additional packages:**
- `xlstm==2.0.5`, `safetensors==0.7.0`, `transformers==4.44.0`
- `flask>=3.0`, `flask-cors>=4.0`

### Two Setup Paths

We provide **both** files so the backend can be deployed with or without Docker:

#### Path A: Conda (manual setup, no Docker)

`environment.yml` — combines the xlstm base env + midi packages + flask:
```bash
conda env create -n xlstm-api -f environment.yml
conda activate xlstm-api
pip install midiprocessor==0.1.5   # from PyPI or cloned repo
python app.py
```

#### Path B: Docker (recommended for deployment)

`Dockerfile` — uses `nvidia/cuda:12.4.0-devel-ubuntu22.04` as base, installs Python 3.11 and all pip packages. No conda needed inside Docker.

`requirements.txt` — used internally by the Dockerfile, contains all pip packages with pinned versions.

```bash
docker build -t xlstm-music-api .
docker run --gpus all -p 5000:5000 -v /path/to/checkpoint:/app/model xlstm-music-api
```

The sLSTM CUDA kernels compile automatically inside the container on first startup (~30s), then are cached.

---

## 7. Model Checkpoint

The checkpoint folder must be accessible at runtime:

```
checkpoint-46000-last/
├── config.yaml
├── model.safetensors    (~1 GB)
└── tokenizer.json
```

**Location**: `/scratch1/.../helibrunna/output/xlstm_lmd_preprocessed_512d_4096ctx_12b/run_20260301-1906/checkpoint-46000-last`

For Docker, this is mounted as a volume (`-v`) rather than baked into the image (keeps image small, allows swapping checkpoints).

---

## 8. Running the Server

### Local development (on visionsrv):
```bash
conda activate xlstm
pip install flask flask-cors    # one-time
python backend/app.py
```

### Docker deployment (on target server):
```bash
docker build -t xlstm-music-api .
docker run --gpus all -p 5000:5000 \
  -v /path/to/checkpoint-46000-last:/app/model \
  xlstm-music-api
```

### Test:
```bash
# Health check
curl http://localhost:5000/health

# Generate a MIDI file
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "s-9 o-0 t-38", "length": 2048}' \
  --output generated.mid
```

---

## 9. Implementation Checklist

- [ ] Copy + refactor `generator.py` (inline model_from_config, remove helibrunna)
- [ ] Copy + refactor `converter.py` (remove sys.path hacks)
- [ ] Copy `token_analysis.py` (as-is)
- [ ] Create `app.py` (Flask API with `/generate` and `/health`)
- [ ] Create `requirements.txt` (pinned versions)
- [ ] Create `environment.yml` (conda alternative)
- [ ] Create `Dockerfile`
- [ ] Create `.dockerignore`
- [ ] Create `README.md` (setup & deployment guide)
- [ ] Test locally: `GET /health`
- [ ] Test locally: `POST /generate` → valid MIDI file
