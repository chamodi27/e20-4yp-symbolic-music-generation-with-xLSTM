# Data Preprocessing — Slide Content
# Final Year Project: Symbolic Music Generation with xLSTM
#
# This file is structured as one slide per section (---).
# Each section has a SLIDE TITLE, SPEAKER NOTES, and VISUAL SUGGESTIONS.
# Adapt the layout (e.g. two-column, numbered steps) to your slide tool.

---

## [SLIDE 1] Data Preprocessing Pipeline — Overview

### Key Points (bullet points on slide)

- Adopted the **MuseFormer** preprocessing methodology (Lv et al., 2023)
- 5-stage pipeline: Parsing → Melody Detection → Instrument Compression → MIDI Normalization → Filtering & Pitch Normalization
- Input: raw multi-track MIDI files from the **Lakh MIDI Dataset (LMD)**
- Output: clean, normalised, pitch-corrected MIDI files ready for tokenisation
- Dataset split: **~29,940 files** (train / validation / test)

### Visual Suggestion
Horizontal pipeline diagram with 5 boxes connected by arrows:
```
[Raw MIDI] → [Stage 1: Parse] → [Stage 2: Melody Detect] → [Stage 3: Compress 6 Tracks] → [Stage 4: MuseScore Norm] → [Stage 5: Filter + Pitch Norm] → [Clean MIDI]
```

### Speaker Notes
"Our data preprocessing follows a 5-stage pipeline closely based on the MuseFormer paper's methodology. We start with raw MIDI files from the Lakh MIDI Dataset and progressively clean, structure, and normalise them into a consistent format that the model can learn from efficiently. Each stage removes or transforms data in a principled way."

---

## [SLIDE 2] Dataset Source — Lakh MIDI Dataset

### Key Points

- Source: **Lakh MIDI Dataset (LMD)** — ~170,000 unique multi-track MIDI files
- Rich in genre diversity: pop, rock, jazz, classical, electronic
- Files vary widely in: number of instruments, tempo, time signature, pitch range, length
- Challenge: raw MIDI is **noisy** — duplicates, inconsistent instrument labels, extreme pitch ranges, degenerate content
- We worked with a **29k subset** (after progressive filtering from the full dataset)

### Visual Suggestion
Two-column layout:
- Left: Stats table — Raw files: ~170k → Subset used: ~29k → Final after filtering: ~29.9k (train+val+test)
- Right: A brief "why LMD?" bullet list (genre diversity, established benchmark, used by MuseFormer baseline)

### Speaker Notes
"We used the Lakh MIDI Dataset, one of the largest publicly available MIDI corpora with around 170,000 unique files. Raw MIDI data is inherently messy — files have inconsistent instrument names, duplicates, extreme tempos, and other quality issues. Our pipeline progressively cleans this data down to a high-quality 29k-file subset."

---

## [SLIDE 3] Stage 1 — MIDI Parsing

### Key Points

- Tool: `miditoolkit` Python library
- Parse every raw MIDI file and extract metadata into a **manifest CSV**
- Metadata extracted per file:
  - Time signatures present
  - Tempo range (BPM)
  - Pitch range (min/max MIDI note)
  - Note count, number of tracks, duration (beats)
  - Number of empty bars
- Failed-to-parse files are **flagged and skipped** (corrupt / format errors)
- Manifest CSV is the **single source of truth** for all downstream stages

### Visual Suggestion
Simple flow box:
```
Raw MIDI → miditoolkit parser → manifest.csv (metadata per file)
```
Plus a small table showing example manifest columns:
`file_id | raw_path | time_sig | tempo_min | tempo_max | pitch_min | pitch_max | num_tracks | num_notes`

### Speaker Notes
"Stage 1 is purely about information extraction. We parse every file and record its musical properties into a central manifest CSV. This manifest acts as a database — all subsequent stages read from it to know which files to process and update it with their results. Nothing is deleted at this stage."

---

## [SLIDE 4] Stage 2 — Melody Track Detection (Midiminer)

### Key Points

- Tool: **Midiminer** (`midi-miner` library) — ML-based track role classifier
- Runs as a **separate step** (requires its own conda environment)
- For each file, Midiminer assigns roles to tracks:
  - `melody` — lead melodic line
  - `bass` — bass line
  - `drum` — percussion
  - (other tracks are unlabelled)
- Output: `program_result.json` — maps each file to its detected track programs
- Files **without a detectable melody track are dropped** at this point
- Melody track identification is critical: it becomes the `square_synth` (lead) channel

### Visual Suggestion
Diagram showing a multi-track MIDI file being annotated:
```
Track 1 (Piano)   → [Midiminer] → melody
Track 2 (Bass)    →             → bass
Track 3 (Drums)   →             → drum
Track 4 (Strings) →             → (other)
```

### Speaker Notes
"Stage 2 uses a machine-learning-based tool called Midiminer to automatically identify which instrument track in each file is the melody. This is crucial because MuseFormer and our xLSTM model are trained on pieces that always have an explicit melody channel. Files without a detectable melody are removed from the pipeline at this stage."

---

## [SLIDE 5] Stage 3 — Instrument Compression to 6 Tracks

### Key Points

- Reduces each MIDI file to exactly **6 canonical instrument tracks**:
  | Channel | MIDI Program | Role |
  |---|---|---|
  | `square_synth` | 80 (Lead 1 — Square) | Melody |
  | `piano` | 0 (Acoustic Grand) | Harmony |
  | `guitar` | 24 (Acoustic Guitar) | Accompaniment |
  | `string` | 48 (String Ensemble) | Accompaniment |
  | `bass` | 32 (Acoustic Bass) | Bass |
  | `drum` | — (percussion channel) | Rhythm |

- Instruments are mapped by **GM family ranges** (piano: 0–7, guitar: 24–31, strings: 40–51, 88–95)
- If multiple instruments fall in the same family, keep top-K by note count (K=2)
- Per-track cleanup applied:
  - **Melody & Bass**: enforce monophony (one note per onset)
  - **Piano / Guitar / String**: polyphony capped (piano ≤10, guitar/string ≤6)
  - All tracks: deduplicate exact notes, trim same-pitch overlaps

### Visual Suggestion
Left: "Many original tracks" → Right: "6 fixed tracks" icon diagram.
Or a before/after table showing a hypothetical file: 12 tracks → 6 named tracks.

### Speaker Notes
"Stage 3 is where we impose structure on the data. MuseFormer was designed for a fixed 6-instrument format, so we compress every file into exactly those 6 categories. The melody track is always monophonic — a single note at a time — which is a deliberate musical constraint. The polyphony caps prevent any single instrument from dominating the token count."

---

## [SLIDE 6] Stage 4 — MuseScore MIDI Normalization

### Key Points

- Tool: **MuseScore Studio 4.6.5** (AppImage, headless via `xvfb-run`)
- Exports each compressed MIDI back through MuseScore's MIDI import/export engine
- Purpose:
  - Quantises note timings to a clean grid
  - Enforces consistent tempo and time signature representation
  - Standardises track names (e.g. `"Square Synthesizer, square_synth"` → `"square_synth"`)
  - Resolves edge cases in MIDI encoding (e.g. overlapping notes, pedal events)
- Custom **MIDI import options XML** controls quantisation level
- Post-processing: a Python script fixes MuseScore's inconsistent track naming (`"Piano, piano"` → `"piano"`)

### Visual Suggestion
Simple before/after: show messy MIDI timing grid → clean quantised grid.
Or a note about why MIDI normalization matters for the tokeniser downstream.

### Speaker Notes
"MuseScore is a professional notation software. By routing our MIDI through it, we leverage its robust import/export engine to produce clean, consistently formatted MIDI. This is especially important for the tokeniser in Stage downstream, which expects well-formed timing grids. We had to add a post-processing step to fix inconsistent track naming introduced by MuseScore itself."

---

## [SLIDE 7] Stage 5 — Quality Filtering (MuseFormer Rules)

### Key Points

Implements all filtering rules from **MuseFormer Table 4**:

| Filter | Rule |
|---|---|
| Time Signature | Only 4/4 allowed |
| Minimum Tracks | ≥ 2 non-empty tracks |
| Melody Track | Must have `square_synth` / program 80 |
| Tempo Range | 24 – 200 BPM |
| Pitch Range | MIDI 21 (A0) – 108 (C8) |
| Max Note Duration | ≤ 16 beats (4 bars in 4/4) |
| Empty Bars | ≤ 3 consecutive empty bars |
| Degenerate Content | Drop if all notes share the same pitch OR same duration |
| Duplicates | Removed based on musical signature (duration, bar count, note count, onsets, instruments) |

### Visual Suggestion
A clean table (as above) with each row showing the rule. Use red/green highlights to show "drop if fails".

### Speaker Notes
"Stage 5 applies strict quality filters directly from the MuseFormer paper. Each rule removes a specific type of low-quality or musically degenerate content. For example, files where all notes have the same pitch are trivially uninteresting for a music model. Duplicate detection uses a content fingerprint — not just filename — to catch re-encoded versions of the same piece."

---

## [SLIDE 8] Stage 5 — Pitch Normalization

### Key Points

- After filtering, all surviving files are **transposed to a single key**
- Algorithm: **Krumhansl-Kessler (KK) key detection**
  - Computes a duration-weighted pitch-class histogram over all notes
  - Correlates histogram with 24 pre-defined KK major/minor key profiles
  - Selects the best-matching key
- Transposition rules:
  - Major keys → **C major** (pitch class 0)
  - Minor keys → **A minor** (pitch class 9)
  - Shift is always **minimal** (within ±6 semitones)
- Octave adjustment: if transposition moves notes outside MIDI 21–108, octave-shift to fit
- Files that cannot fit in range after adjustment are dropped

### Visual Suggestion
Diagram:
```
Input: G major piece → KK detects G major → Shift -7 semitones → Output: C major piece
```
Or a pitch-class histogram graphic showing the KK correlation.

### Speaker Notes
"Pitch normalization is important for training efficiency. By transposing every file to C major or A minor, we ensure the model doesn't need to learn the same musical patterns in all 12 keys — it can focus on structure and harmony rather than absolute pitch. The Krumhansl-Kessler algorithm is a well-established psychoacoustic model for key detection."

---

## [SLIDE 9] Pipeline Engineering — Batch Mode & Scalability

### Key Points

- The full Lakh dataset has ~170k files — **naive processing causes timeouts**
- We implemented a **batch mode** for Stage 5:
  - Processes files in configurable batches (default: **5,000 files/batch**)
  - Saves a checkpoint (`progress.json`) after each batch
  - `--resume` flag skips already-processed files on restart
- Two pipeline modes:
  - **Dev mode**: keeps all intermediate files (~6× raw dataset size) — good for debugging
  - **Prod mode**: streaming pipeline, minimal intermediate storage (~1.5× raw) — used for 170k files
- Stage orchestrated via `pipeline_main.py` with CLI flags (`--stage`, `--mode`, `--resume`)
- All stages tracked in a unified **manifest CSV** (single source of truth)

### Visual Suggestion
Side-by-side comparison table:
| | Dev Mode | Prod Mode |
|---|---|---|
| Intermediate files | All kept | Streamed / deleted |
| Storage (170k files) | ~60 GB | ~20 GB |
| Use case | Debugging | Production |

### Speaker Notes
"When we scaled from our 29k subset to the full 170k dataset, a single-run Stage 5 would time out. We refactored the pipeline to batch processing with checkpointing, so a server reboot or crash doesn't mean starting over. The manifest CSV tracks every file's status across all stages, allowing seamless resume."

---

## [SLIDE 10] Dataset Statistics — Final Numbers

### Key Points

| Stage | Files |
|---|---|
| Raw input (00_raw, 29k subset) | ~7,725 |
| After melody detection (Stage 2) | ~7,725 (subset already filtered) |
| After quality filtering (Stage 5) | ~6,110 |
| After pitch normalization (final) | ~6,112 |
| **Train split** | **23,952** |
| **Validation split** | **2,994** |
| **Test split** | **2,994** |
| **Total (train+val+test)** | **29,940** |

> Note: The 29k final split incorporates files from multiple processing runs / dataset versions.

- Each file contains exactly **6 instrument tracks** in a standardised format
- All files in **4/4 time**, tempo 24–200 BPM, pitch within MIDI 21–108
- All files transposed to **C major or A minor**

### Visual Suggestion
Funnel diagram:
```
170k (Full LMD) → 29k (Working Subset) → 6.1k (After Strict Filter) → Final 29,940 (Aggregated Clean Set)
```
Plus a data split donut chart (80% train / 10% val / 10% test).

### Speaker Notes
"These are our final dataset statistics. We worked with a 29k subset for development and validation of our pipeline. The final clean dataset of ~29,940 files is split 80/10/10 into train, validation, and test. Every file in the dataset conforms to the same strict structural format: 6 tracks, 4/4 time, C major or A minor, within standard pitch and tempo bounds."

---

## [SLIDE 11] Tokenisation — Bridge to Model Input

### Key Points

- Preprocessed MIDI files are tokenised using **MidiTok** Python library
- Tokenisation scheme: **REMI+** (Revamped MIDI-derived events, extended)
  - Encodes: Bar, Position (sub-beat grid), Pitch, Velocity, Duration, Program (instrument)
  - One vocabulary shared across all instruments
- Vocabulary trained on the full training split (`train_tokenizer.py`)
- Tokeniser config: `remi_plus_xlstm_large_h100.yaml`
- Output: integer token sequences fed directly to the xLSTM model
- Alternative tokeniser tested: **Octuple** (encodes multiple attributes per token position)

### Visual Suggestion
Example of a MIDI event → token sequence mapping:
```
Bar_1  →  Position_0  →  Program_80  →  Pitch_60  →  Velocity_80  →  Duration_4  →  Bar_2  → ...
```
Or a diagram of the REMI+ vocabulary structure.

### Speaker Notes
"After preprocessing, MIDI files are converted into discrete token sequences using the REMI+ scheme. REMI+ encodes musical events as a flat sequence of tokens — bar markers, beat positions, pitch, velocity, duration, and instrument program — which the xLSTM model then processes as a language-modelling task. The tokeniser vocabulary is learned from the training data itself."

---

## [APPENDIX] Implementation Files Reference

| File | Purpose |
|---|---|
| `scripts/museformer-preprocess/pipeline_main.py` | Main pipeline orchestrator (CLI entry point) |
| `scripts/museformer-preprocess/pipeline_config.py` | All configuration: paths, thresholds, modes |
| `scripts/museformer-preprocess/stage_01_parsing.py` | Stage 1: MIDI metadata extraction |
| `scripts/museformer-preprocess/stage_02_midiminer_helper.py` | Stage 2: melody detection helper |
| `scripts/museformer-preprocess/stage_03_compress6.py` | Stage 3: compress to 6 tracks |
| `scripts/museformer-preprocess/stage_04_musescore_norm.py` | Stage 4: MuseScore normalization |
| `scripts/museformer-preprocess/stage_05_filter_wrapper.py` | Stage 5 wrapper: batch mode + resume |
| `scripts/museformer-preprocess/filter_and_normalize_v2.py` | Stage 5 core: filtering + pitch normalization |
| `scripts/museformer-preprocess/filter_config_v2.py` | Stage 5 configuration (all filter parameters) |
| `scripts/museformer-preprocess/manifest_utils.py` | Manifest CSV management utilities |
| `xlstm-fairseq/music-xlstm/tokenize/train_tokenizer.py` | REMI+ tokeniser training |
| `data/museformer_baseline/29k/` | 29k dataset with all stage directories |

---
*Document generated: 2026-03-14 | FYP: Symbolic Music Generation with xLSTM*
