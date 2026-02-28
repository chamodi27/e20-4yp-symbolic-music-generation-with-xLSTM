# Paper Plan: Implicit Hierarchical Modeling of Long-Term Structure in Symbolic Music Generation Using xLSTM

**Paper Title:** Implicit Hierarchical Modeling of Long-Term Structure in Symbolic Music Generation Using xLSTM  
**Target Venue:** IEEE Conference (2-column format)  
**Status:** Draft in progress  
**Last Updated:** 2026-02-25

---

## Progress Overview

| Section | Written | Experiments Done | Final |
|---|---|---|---|
| Abstract | [ ] | — | [ ] |
| I. Introduction | [ ] | — | [ ] |
| II. Background & Related Work | [ ] | — | [ ] |
| III. Methodology | [ ] | — | [ ] |
| IV. Experiments & Setup | [ ] | [/] Perplexity only | [ ] |
| V. Results & Discussion | [ ] | [/] Partial | [ ] |
| VI. Conclusion | [ ] | — | [ ] |
| References | [ ] | — | [ ] |

---

## Paper Structure

### Abstract
- [ ] Write abstract (~200 words)

**Content to cover:**
- The problem: long-term structural coherence in symbolic music generation
- The gap: existing models either use rigid explicit hierarchies OR are fully implicit and lose structure
- Our approach: xLSTM with implicit multi-timescale memory
- Dataset: Lakh MIDI Dataset (LMD), REMIGEN2 tokenization
- What we measure: perplexity, structural similarity (SSM), motif recurrence, and human evaluation
- Key finding: placeholder — fill in after experiments complete

**Draft outline:**
> Deep learning models for symbolic music generation have made significant progress, yet sustaining long-term structural coherence—including thematic repetition, phrase symmetry, and sectional contrast—remains a core challenge. Existing approaches face a persistent trade-off: explicit hierarchical models achieve structural consistency but rely on rigid, predefined segmentation, while standard autoregressive models are flexible but struggle to maintain coherence over long sequences. This work investigates the Extended Long Short-Term Memory (xLSTM) architecture as an approach for *implicit* multi-timescale modeling in symbolic music generation. We train xLSTM models on the Lakh MIDI Dataset using the REMIGEN2 token representation and evaluate them on perplexity, structural self-similarity, motif recurrence, and qualitative human listening tests. We compare against Museformer and LookBack RNN as baselines. [Results summary — fill in after experiments.] Our findings suggest that xLSTM's exponential gating and matrix memory enable it to capture hierarchical musical dependencies organically, without explicitly defined structural boundaries.

---

### Section I: Introduction
- [ ] Write Introduction (~500–600 words)

**Flow and content:**
1. **Opening hook** — Music generation is a rich and active area of AI research. Generated music can be judged not just on note-by-note quality but also on its large-scale structural coherence (does it sound like a complete, well-formed piece or like random notes?).
2. **The structural challenge** — Long-term structure in music (motifs, phrases, sections like verse-chorus) is what makes music feel intentional. Most models fail at this: they capture local patterns well but "forget" what happened many bars ago.
3. **Existing approaches and their limits** — Two camps:
   - *Explicit hierarchical models* (MusicVAE, Museformer, Bar Transformer): manually impose structure at bar/phrase/section level. Work well but require predefined segmentation and don't generalize across styles.
   - *Implicit autoregressive models* (Music Transformer, standard LSTMs, LLMs like MuPT): flexible and scalable, but struggle with long-term coherence — their attention or memory is dominated by local context.
4. **The gap** — No architecture simultaneously achieves both the structural modeling of explicit hierarchical designs AND the flexibility/generality of implicit sequence models.
5. **Our proposal** — xLSTM (Beck et al., 2024) introduces exponential gating and matrix memory (mLSTM) that enable implicit multi-timescale dynamics within a single recurrent layer. Unlike standard LSTMs, xLSTM can retain long-range dependencies without quadratic attention costs. We hypothesize that this makes it naturally suited for capturing hierarchical musical structure without manual segmentation.
6. **Contributions** — List 3–4 bullet points:
   - First study of xLSTM applied to symbolic music generation
   - Training xLSTM on LMD with REMIGEN2 tokenization via Helibrunna framework
   - Evaluation across context lengths with perplexity analysis
   - Structural evaluation with SSM-based metrics and human listening tests (planned)
7. **Paper structure** — One sentence per section.

---

### Section II: Background and Related Work
- [ ] Write Background (~700–900 words, split into clear subsections)

> **Note:** Our literature review (E2030.pdf) is very thorough. This section in the paper should be a condensed version of the key points. Do NOT copy the literature review — summarize and cite.

**Subsections:**

#### II-A. Musical Structure and Representation
- Briefly explain the hierarchical nature of music: note → motif → phrase → section
- REMIGEN2 / REMI-style tokenization for LMD: what tokens are used, why event-based compound representations work better than elementary tokens for long-form generation
- Reference: REMI (Huang & Yang 2020), OctupleMIDI (MusicBERT), Museformer's use of MIDI tokens

#### II-B. Models for Symbolic Music Generation
Keep concise — just enough to motivate our work. Organize as a short narrative:
- **RNN-based**: LookBack RNN and Attention RNN — captured short-term repetition but struggled with global form
- **Transformer-based**: Music Transformer (relative attention), Pop Music Transformer (REMI) — strong local modeling, but attention window limits long-range coherence
- **Explicit hierarchical**: MusicVAE (conductor RNN), Museformer (fine+coarse attention), Bar Transformer — achieved better long-term structures but needed predefined segmentation
- **xLSTM**: Introduced by Beck et al. (2024); exponential gating + mLSTM matrix memory; outperforms LSTMs and rivals Transformers on long-range tasks; applied to audio (AxLSTM) but **not yet studied for symbolic music generation**

Include a concise **summary table** of related models:

| Model | Architecture | Strength | Weakness |
|---|---|---|---|
| LookBack RNN | RNN | Short-term repetition | No global form |
| Music Transformer | Transformer | Rich style | Limited structural coherence |
| Museformer | Transformer (hierarchical attn) | Multi-scale structure | Relies on explicit bar segmentation |
| MusicVAE | Hierarchical VAE | Structural interpolation | Fixed-length bar divisions |
| xLSTM (ours) | Recurrent (mLSTM + sLSTM) | Implicit multi-timescale | Not yet evaluated for music |

#### II-C. Evaluation of Structural Coherence
- Traditional metrics: perplexity, negative log-likelihood — measure predictive quality but not structure
- Statistical features: pitch histogram, note density, rhythmic entropy — measure style but not global form
- **Structure-aware metrics**: Self-Similarity Matrix (SSM), segmentation similarity (MSCOM by de Berardinis et al.; reference [5] in E2030) — directly measure repetition and sectional structure
- Human evaluation: MOS Likert-scale, pairwise A/B preference — gold standard but expensive and subjective

---

### Section III: Methodology
- [ ] Write Methodology (~600–800 words)

**Subsections:**

#### III-A. Dataset
- **Lakh MIDI Dataset (LMD)**: 176,581 MIDI files, large-scale general-purpose corpus
- **Data subset**: Following Museformer's procedure — use only the MIDI files specified by Museformer (their train/valid/test splits). This ensures comparability with Museformer baseline.
  - Train: `train.txt`, Valid: `valid.txt`, Test: `test.txt`
  - Exact counts: `[PLACEHOLDER — check dataset sizes]`
- **Tokenization**: REMIGEN2 format using MidiProcessor (same as Museformer baseline)
  - REMIGEN2 uses elementary event-based tokens: speed (`s-`), onset (`o-`), time (`t-`), instrument (`i-`), pitch (`p-`), duration (`d-`), velocity (`v-`)
  - Vocabulary size: **675 tokens**
  - Example token sequence: `s-9 o-42 t-38 i-21 p-67 d-1 v-31 ...`
- **Note on preprocessing**: We are working with unpreprocessed MIDI for the initial experiments. Museformer's full preprocessing pipeline (as described in their Appendix) is currently being implemented and will be the subject of a separate training run.

#### III-B. Model Architecture — xLSTM
- Briefly explain xLSTM's two cell types:
  - **sLSTM** (scalar LSTM): Enhanced gating with exponential activation; memory mixing via head-wise recurrence; more "memory-efficient" for common patterns
  - **mLSTM** (matrix LSTM): Memory stored as a matrix (key-value associative memory); covariance update rule; parallelizable like Transformers; captures long-range associations precisely
- **Our configuration** (trained model):
  - Embedding dimension: **512**
  - Number of blocks: **12** (sLSTM blocks at positions 3, 6, 9; mLSTM blocks at remaining positions)
  - Context length: **2048 tokens** (training)
  - Vocabulary size: 675
  - Parameters: `[PLACEHOLDER — calculate or report]`
  - mLSTM: 4 heads, conv1d kernel 4, QKV proj blocksize 4
  - sLSTM: cuda backend, 4 heads, conv1d kernel 4, powerlaw_blockdependent bias init
  - Feedforward: proj_factor 1.3, GELU activation

#### III-C. Training Setup
- Framework: **Helibrunna** (open-source xLSTM training framework)
- Hardware: `[PLACEHOLDER — GPU type and count]`
- Optimizer: AdamW, weight_decay=0.01
- Learning rate: 0.001, with warmup (15,874 steps) and cosine decay to 0.001×LR over 158,746 steps
- Precision: float16 (AMP), weights in float32
- Batch size: 1
- Epochs: 20
- Checkpoint saved every 2000 steps
- Best checkpoint selected by validation perplexity: **checkpoint-66000** (step 66,000)
- wandb project: `lmd_remigen_xlstm`

#### III-D. Evaluation Strategy

Our evaluation follows a multi-level strategy combining automatic metrics and human judgment:

1. **Perplexity** (language model quality)
   - Evaluated at multiple context lengths: 1024, 2048, 3072, 4096, 5120, 10240
   - Measures how well the model predicts unseen token sequences
   - Used to select the best checkpoint and compare against baselines

2. **Structural Self-Similarity (SSM-based)** *(planned)*
   - Compute self-similarity matrix at bar level for generated music
   - Measure: proportion of bars that repeat (above a similarity threshold), average block size of repetitive segments
   - Motivation: SSM directly measures the kind of sectional repetition (A-A-B-A patterns etc.) that defines musical structure

3. **Motif Recurrence / Similarity Metric** *(planned)*
   - Detect short recurring motifs in generated pieces
   - Compare motif recurrence rate between xLSTM and baselines

4. **Qualitative Human Evaluation** *(planned)*
   - **Pairwise A/B test**: Compare xLSTM vs. Museformer, xLSTM vs. LookBack RNN
     - Metric: Preference rate (%)
     - Evaluate on: overall quality, structural coherence
   - **Likert Scale Rating (MOS 1–5)**:
     - Structural coherence (does the piece have clear sections/repetition?)
     - Phrase quality (are individual phrases musical?)
     - Global flow (does the piece feel complete and well-formed?)

---

### Section IV: Experiments and Results
- [ ] Write Results section (mostly placeholders for now, fill in as experiments complete)

**Subsections:**

#### IV-A. Perplexity Analysis
**Status: DONE ✅**

Summary of results for `xlstm_lmd_512d_2048ctx_12b`, best checkpoint (`checkpoint-66000`), evaluated on test set:

| Context Length | Perplexity |
|---|---|
| 1024 | 1.918 |
| 2048 | 1.776 |
| 3072 | **1.775** ← best |
| 4096 | 1.882 |
| 5120 | 2.085 |
| 10240 | 3.550 |

**Key observations to discuss in paper:**
- Perplexity decreases as context length increases from 1024 → 3072, then increases beyond that
- The model was trained with 2048-token context but achieves best PPL at 3072 — showing some generalization beyond training context
- Performance degrades significantly beyond 5120 tokens (no training signal for those lengths)
- Compare with Museformer's reported perplexity: `[PLACEHOLDER — extract from Museformer paper]`
- Compare with LookBack RNN: `[PLACEHOLDER — use published values]`

#### IV-B. Structural Self-Similarity Analysis
**Status: IN PROGRESS 🔄**

- [ ] Generate a set of MIDI samples from the trained model
- [ ] Compute bar-level SSM for generated samples
- [ ] Compare SSM patterns vs. training data (LMD) and vs. Museformer-generated music
- [ ] Report: average structural similarity score, proportion of pieces with clear repetitive structure

**What to write (placeholder):**
> We generated [N] MIDI samples of [length] bars each and computed bar-level self-similarity matrices (SSMs) to quantify structural repetition. [Results and comparison with baselines here.]

#### IV-C. Motif Recurrence
**Status: PLANNED 📋**

- [ ] Define motif detection algorithm (short n-gram matching at the note level)
- [ ] Measure recurrence rate in generated samples
- [ ] Compare with baseline models

#### IV-D. Human Evaluation
**Status: PLANNED 📋**

- [ ] Design listening experiment (A/B pairwise + MOS Likert)
- [ ] Prepare audio samples (MIDI → audio using FluidSynth or MuseScore)
- [ ] Recruit listeners (minimum 10 participants recommended)
- [ ] Conduct evaluation with Surveysparrow / Google Forms
- [ ] Report: preference rates, MOS scores with standard deviation
- **Fallback**: If time is short, use published human evaluation numbers from the Museformer paper and LookBack RNN paper for comparison instead of running new tests.

---

### Section V: Discussion
- [ ] Write Discussion (~300–400 words)

**Points to cover:**
- What does the perplexity curve across context lengths tell us? (Sweet spot at 3072 suggests the model benefits from longer context than it was trained on, up to a point)
- Does the model show evidence of structural awareness? (Link to SSM results)
- Comparison with explicit hierarchical models (does xLSTM approach their structural quality without manual segmentation?)
- Limitations: unpreprocessed MIDI data, no explicit harmonic/structural conditioning, perplexity doesn't directly measure structure
- Future direction: train on preprocessed data (Museformer pipeline), experiment with larger context lengths (4096 training context), explore conditioning strategies

---

### Section VI: Conclusion
- [ ] Write Conclusion (~200–250 words)

**Structure:**
1. Restate the problem and our approach (1–2 sentences)
2. Summarize key findings (2–3 sentences)
3. Reflect on the contribution: first study of xLSTM for symbolic music generation
4. Point to future work: preprocessing (Museformer pipeline), larger models, more thorough human evaluation

---

### References
- [ ] Compile full reference list in IEEE format
- [ ] Verify all citations used in the paper are listed

**Key references to include:**
- Beck et al. (2024) — xLSTM
- Yu et al. (2022) — Museformer
- Waite et al. (2016) — LookBack RNN (Google Magenta)
- Huang et al. (2018) — Music Transformer
- Roberts et al. (2018) — MusicVAE
- Raffel (2016) — Lakh MIDI Dataset
- de Berardinis et al. (2022) — SSM structural evaluation (MSCOM)
- Ji et al. (2023) — Survey on deep learning for symbolic music generation
- Le et al. (2025) — NLP methods for symbolic music (survey)
- Bhandari & Colton (2024) — Motifs, Phrases, and Beyond (survey)
- Alharthi & Mahmood (2024) — xLSTMTime
- Zhang et al. (2023) — Symbolic music representations for classification

---

## Pending Experiments / Action Items

### High Priority (needed for paper)
- [ ] **Train 4096-context model** — second variation with ctx=4096 for comparison with ctx=2048 model
- [ ] **Evaluate 4096 model with perplexity** at multiple context lengths
- [ ] **Generate MIDI samples** from best checkpoint for structural evaluation
- [ ] **Implement bar-level SSM** evaluation pipeline
- [ ] **Implement motif recurrence** evaluation

### Medium Priority
- [ ] **Preprocess dataset** using Museformer's full pipeline (MidiProcessor with proper steps)
- [ ] **Retrain on preprocessed data** for fair comparison with Museformer
- [ ] **Design human listening study**

### Low Priority / Fallback
- [ ] **Baseline inference**: run LookBack RNN and Museformer inference for direct comparison
- [ ] **Alternatively**: use published numbers from Museformer paper

---

## Writing Guidelines

### Style
- IEEE 2-column format, `\documentclass[conference]{IEEEtran}`
- Academic language — clear and precise, but not overly complex
- English is not the authors' first language — prefer clear, direct sentences over long or complex ones
- Aim for 6–8 pages in IEEE format (typical conference paper)
- Use citations in the format `[N]` matching references

### What to include as placeholders
Use `[PLACEHOLDER: description]` for any result or number not yet available. Examples:
- `[PLACEHOLDER — add GPU specs]`
- `[PLACEHOLDER — Museformer reported perplexity]`
- `[PLACEHOLDER — number of generated samples]`

### Tables and Figures to create
- [ ] **Table I**: Model comparison (architecture, training data, evaluation metrics)
- [ ] **Table II**: Perplexity results across context lengths (both models)
- [ ] **Figure 1**: Perplexity vs. context length curve (line chart, both models)
- [ ] **Figure 2**: Example SSM of a generated piece vs. real piece
- [ ] **Figure 3** *(optional)*: Example generated MIDI score snippet

---

## File Structure in `paper/`

```
paper/
├── main.tex           ← Main LaTeX paper (to be created)
├── paper-plan.md      ← This file
├── refs_text/         ← Extracted PDF text (gitignored, for reference only)
│   ├── E2030_literature_review.txt
│   ├── Museformer.txt
│   ├── Museformer_appendix.txt
│   ├── xLSTM_paper.txt
│   └── Motifs_Phrases_Beyond.txt
├── E2030.pdf
├── Museformer.pdf
├── Museformer-appendix.pdf
├── xLSTM Extended Long Short-Term Memory.pdf
└── Motifs, Phrases, and Beyond-The Modelling of...pdf
```

---

## Quick Reference: What We Have Done

| Item | Status | Notes |
|---|---|---|
| xLSTM model training (2048 ctx) | ✅ Done | `checkpoint-66000`, step=66k |
| Perplexity evaluation (2048 ctx model) | ✅ Done | Best PPL=1.775 at ctx=3072 |
| xLSTM model training (4096 ctx) | ⏳ Planned | Second variation |
| Perplexity eval (4096 ctx model) | ⏳ Planned | — |
| SSM bar-level analysis | 🔄 In progress | — |
| Motif recurrence | ⏳ Planned | — |
| Museformer baseline | ⏳ Planned | May use published numbers |
| LookBack RNN baseline | ⏳ Planned | May use published numbers |
| Human listening test | ⏳ Planned | Fallback: use published scores |
| Dataset preprocessing (Museformer pipeline) | 🔄 In progress (teammate) | — |
