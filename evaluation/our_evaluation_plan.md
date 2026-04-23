# Evaluation Plan

## MOS (Mean Opinion Score)

Each listener rates **9 clips total**:

- 3 clips from **xLSTM**
- 3 clips from **Museformer**
- 3 clips from **Lookback RNN**

With **20 listeners**:

- Each model gets **60 MOS ratings total**
- Each of the **20 selected clips per model** gets about **3 ratings on average**

---

# MOS Questions

## All listeners

1. **Structural coherence**  
   *“The piece feels organized and coherent over time.”*

2. **Musical flow**  
   *“The music progresses naturally.”*

3. **Overall musical quality**  
   *“Overall, this is a good musical composition.”*

## Experts only

1. **Motivic consistency**  
   *“The piece repeats and develops musical ideas in a consistent way.”*

2. **Harmonic coherence**  
   *“The chord progression sounds natural and makes musical sense.”*

---

# Pairwise A/B Test

Each listener completes **9 pairwise comparisons total**:

- 3 × **xLSTM vs Museformer**
- 3 × **xLSTM vs Lookback RNN**
- 3 × **Museformer vs Lookback RNN**

With **20 listeners**:

- Each comparison family gets **60 votes**
- Enough to estimate preference direction

---

# Pairwise A/B Questions

For each **A/B comparison**:

1. **Which piece has better long-term structural coherence?**
   - A
   - B

2. **Which piece has better overall musical quality?**
   - A
   - B

---

# Turing Test

**Question:**  
*“Is this human-composed or AI-generated?”*

**Composition of clips:**

- 4 generated clips (random from models)
- 6 human clips