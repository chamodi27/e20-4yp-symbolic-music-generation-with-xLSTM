# xLSTM Music Generation - Complete Guide

## Table of Contents
1. [What Was Wrong](#what-was-wrong)
2. [The Solution](#the-solution)
3. [Understanding Context Length](#understanding-context-length)
4. [Memory Management](#memory-management)
5. [API Reference](#api-reference)
6. [Best Practices](#best-practices)

---

## What Was Wrong

### The Memory Issue

Your original code had this critical problem:

```python
# ❌ WRONG - Causes quadratic memory growth
for iteration in range(max_iterations):
    chunk, info = generate_remigen_tokens(
        model,
        prompt=output,  # Growing context
        max_length=len(output.split()) + chunk_size,  # ← THE PROBLEM
        ...
    )
```

**Why This Failed:**

1. **Iteration 1**: 
   - Context: 40 tokens
   - `max_length = 40 + 1500 = 1540`
   - Model allocates `40 × 1540` matrix → Works fine

2. **Iteration 2**: 
   - Context: 1540 tokens (includes previous generation)
   - `max_length = 1540 + 1500 = 3040`
   - Model tries to allocate `1540 × 3040` matrix → **OOM!**

The mLSTM layer creates an N×N covariance matrix where N = `max_length`. This grows quadratically.

### Misunderstanding of max_length

```python
# Common misunderstanding:
max_length = current_length + tokens_to_generate  # ❌ WRONG

# Correct understanding:
max_length = total_output_length  # ✓ RIGHT
```

`max_length` is the **absolute maximum output length**, not a delta to add.

---

## The Solution

### Approach 1: Fixed max_length (Short Sequences)

```python
# ✓ CORRECT - For sequences under context_length
result = model.generate(
    prompt="s-9 o-0 t-35",
    max_length=2048,  # Fixed total length
    temperature=0.8
)
```

This works perfectly for short pieces (≤ 2048 tokens).

### Approach 2: Sliding Window (Long Sequences)

```python
# ✓ CORRECT - For long sequences
all_tokens = prompt.split()

for iteration in range(max_iterations):
    # Use only recent context (sliding window)
    window_size = min(context_length - chunk_size, len(all_tokens))
    context = " ".join(all_tokens[-window_size:])
    
    # Generate next chunk
    result = model.generate(
        prompt=context,
        max_length=len(context.split()) + chunk_size,  # Now safe!
        temperature=0.8
    )
    
    # Extract ONLY new tokens
    new_tokens = result.split()[len(context.split()):]
    all_tokens.extend(new_tokens)
```

**Key insight**: By using a sliding window, `len(context.split())` stays constant, so the matrix size doesn't explode.

---

## Understanding Context Length

### Training vs Inference

**During Training:**
```python
# Model was trained with:
context_length = 2048
```

**During Inference:**
```python
# You CAN use larger context:
context_length = 4096  # ✓ Allowed
context_length = 8192  # ✓ Allowed (if you have memory)
```

The model can handle sequences longer than what it was trained on, but memory requirements grow quadratically.

### Memory Requirements

| Context Length | Approximate VRAM | Feasible? |
|---------------|------------------|-----------|
| 1024 | ~2.5 GB | ✓ Yes |
| 2048 | ~10 GB | ✓ Yes |
| 4096 | ~40 GB | ⚠️ Maybe (48GB GPU) |
| 8192 | ~160 GB | ✗ No (OOM) |
| 16384 | ~640 GB | ✗ No (impossible) |

**Formula**: Memory ≈ N² × model_size

This is why the Helibrunna example with `context_length=16_384` only worked for **single-shot generation**, not iterative.

---

## Memory Management

### GPU Memory Best Practices

1. **Clear cache between generations:**
```python
import torch
torch.cuda.empty_cache()
```

2. **Monitor GPU usage:**
```python
# Check available memory
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

3. **Use gradient checkpointing (if training):**
```python
# In training config
gradient_checkpointing = True
```

4. **Batch size = 1 during generation:**
```python
# Generation always uses batch_size=1
# No need to change anything
```

### When You Get OOM

If you still get OOM errors:

1. **Reduce context_length:**
   ```python
   context_length = 2048  # Instead of 4096
   ```

2. **Reduce chunk_tokens:**
   ```python
   chunk_tokens = 512  # Instead of 1024
   ```

3. **Check other processes:**
   ```bash
   nvidia-smi
   # Kill other GPU processes if possible
   ```

4. **Use smaller temperature:**
   ```python
   temperature = 0.5  # More deterministic, uses less memory
   ```

---

## API Reference

### MusicGenerator

```python
class MusicGenerator:
    def __init__(self, model_path, context_length=2048, device="cuda")
    def generate(self, prompt, temperature=0.8, max_tokens=2048, verbose=True)
    def generate_long(self, prompt, temperature=0.8, target_bars=32, 
                     chunk_tokens=1024, max_iterations=20, verbose=True)
```

### MIDIConverter

```python
class MIDIConverter:
    def __init__(self)
    def tokens_to_midi(self, tokens_str, output_path) -> bool
```

### Simple API

```python
generate_music(
    model_path,
    num_songs=1,
    max_tokens=2048,
    temperature=0.8,
    output_dir="./generated",
    long_mode=False,
    target_bars=None
) -> List[Path]
```

---

## Best Practices

### 1. Start Simple

```python
# Begin with short generations
result = generator.generate(
    prompt="s-9 o-0 t-35",
    max_tokens=1500,  # Conservative
    temperature=0.8
)
```

### 2. Test Memory Limits

```python
# Gradually increase to find your GPU's limit
for max_tokens in [1500, 2000, 2500, 3000]:
    try:
        result = generator.generate(max_tokens=max_tokens)
        print(f"✓ {max_tokens} tokens OK")
    except RuntimeError as e:
        print(f"✗ {max_tokens} tokens: OOM")
        break
```

### 3. Use Appropriate Mode

```python
# Short pieces (< 2048 tokens): Use simple generate()
if target_length <= 2048:
    result = generator.generate(max_tokens=target_length)

# Long pieces (> 2048 tokens): Use generate_long()
else:
    result = generator.generate_long(target_bars=64)
```

### 4. Temperature Guidelines

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.3-0.5 | Conservative, repetitive | Background music |
| 0.6-0.8 | Balanced | General use |
| 0.9-1.1 | Creative | Experimental |
| 1.2-1.5 | Wild, chaotic | Maximum variety |

### 5. Prompt Engineering

Good prompts for REMIGEN:

```python
# Minimal (let model decide everything)
"s-9 o-0 t-35"

# With tempo control
"s-9 o-0 t-120"  # Fast
"s-9 o-0 t-60"   # Medium
"s-9 o-0 t-35"   # Slow

# With instrument
"s-9 o-0 t-35 i-0"    # Piano
"s-9 o-0 t-35 i-128"  # Drums

# Complete phrase (copied from training data)
"s-9 o-0 t-35 i-128 p-170 d-3 v-31 o-12 t-35 i-128 p-170 d-3 v-25"
```

### 6. Batch Processing

```python
# For experiments, generate multiple samples
for i in range(100):
    result = generator.generate(
        prompt="s-9 o-0 t-35",
        temperature=0.8,
        max_tokens=2048,
        verbose=False  # Disable logging
    )
    # Save result...
```

### 7. Quality vs Speed

```python
# Fast generation (lower quality)
result = generator.generate(
    temperature=0.5,  # More deterministic
    max_tokens=1024   # Shorter
)

# Quality generation (slower)
result = generator.generate(
    temperature=0.8,  # More creative
    max_tokens=2048   # Longer
)
```

---

## Comparison with Helibrunna Example

### Their Approach (Garland format):

```python
# They use fixed max_length
for iteration in range(20):
    output_dict = model.generate(
        prompt=output,
        max_length=16_384,  # FIXED (doesn't grow)
        end_tokens=["NEXT"],
        ...
    )
```

**Why it works**: They stop at `NEXT` token, which typically comes before hitting 16,384 tokens.

### Your Approach (REMIGEN format):

```python
# You need sliding window for long generation
result = generator.generate_long(
    prompt="s-9 o-0 t-35",
    target_bars=64,      # Stop condition
    chunk_tokens=1024,   # Safe chunk size
    ...
)
```

**Key difference**: REMIGEN doesn't have natural stopping tokens like Garland's `NEXT`, so you need explicit bar counting.

---

## Troubleshooting

### Problem: Still getting OOM

**Solution checklist:**
1. Reduce `context_length` to 2048
2. Reduce `chunk_tokens` to 512
3. Use `max_tokens=1500` for short generation
4. Clear CUDA cache: `torch.cuda.empty_cache()`
5. Check other GPU processes: `nvidia-smi`
6. Restart Python kernel

### Problem: Generation is too repetitive

**Solutions:**
1. Increase temperature: `temperature=1.0`
2. Use longer context: `context_length=4096`
3. Try different prompts
4. Check if model is properly trained

### Problem: Invalid tokens in output

**Solutions:**
1. The code already filters invalid tokens
2. Check if `[PAD]` and `[EOS]` are in `forbidden_tokens`
3. Verify REMIGEN format is correct

### Problem: MIDI conversion fails

**Solutions:**
1. Check tokens are valid REMIGEN format
2. Ensure `midiprocessor` is installed
3. Verify decoder: `mp.MidiDecoder('REMIGEN')`
4. Check output directory exists

---

## For Your Research

### Generating Comparison Dataset

```python
# Generate matched pairs for xLSTM vs Museformer comparison

prompts = ["s-9 o-0 t-35", "s-9 o-0 t-60", "s-9 o-0 t-120"]
temperatures = [0.5, 0.8, 1.0]

for prompt_id, prompt in enumerate(prompts):
    for temp_id, temp in enumerate(temperatures):
        for sample in range(10):  # 10 samples per condition
            
            result = generator.generate(
                prompt=prompt,
                temperature=temp,
                max_tokens=2048,
                verbose=False
            )
            
            filename = f"xlstm_p{prompt_id}_t{temp_id}_s{sample:03d}.mid"
            converter.tokens_to_midi(result['tokens'], f"./dataset/{filename}")

# Total: 3 prompts × 3 temps × 10 samples = 90 pieces
# Generate same from Museformer for comparison
```

### Metrics to Compare

1. **Objective metrics:**
   - Note density
   - Pitch range
   - Rhythmic complexity
   - Harmonic coherence

2. **Subjective evaluation:**
   - Listening test
   - Musicality rating
   - Coherence over time

3. **Computational:**
   - Generation speed (tokens/sec)
   - Memory usage
   - Context window capabilities

---

## Summary

**Key takeaways:**

1. ✓ Use fixed `max_length` for short sequences
2. ✓ Use sliding window for long sequences
3. ✓ Context length can exceed training length
4. ✓ Memory grows quadratically with context
5. ✓ Clear CUDA cache between generations
6. ✓ Filter invalid tokens from output
7. ✓ Use appropriate temperature for your use case

**Your new workflow:**

```python
from xlstm_music_generation import MusicGenerator, MIDIConverter

# 1. Initialize
generator = MusicGenerator(model_path, context_length=2048)
converter = MIDIConverter()

# 2. Generate
result = generator.generate(prompt="s-9 o-0 t-35", max_tokens=2048)

# 3. Convert
converter.tokens_to_midi(result['tokens'], "output.mid")
```

Done! 🎵
