"""
xLSTM Music Generation Pipeline
Clean, modular code for generating music with REMIGEN representation
"""

import sys
import os
from pathlib import Path
import torch

# Add helibrunna to path
sys.path.append("/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna")
from source.languagemodel import LanguageModel
import midiprocessor as mp


class MusicGenerator:
    """Handles music generation with xLSTM model"""
    
    def __init__(self, model_path, context_length=2048, device="cuda"):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to trained model
            context_length: Context window for generation (can exceed training length)
            device: 'cuda' or 'cpu'
        """
        print(f"Loading model from: {model_path}")
        self.model = LanguageModel(
            model_path,
            config_overrides={"context_length": context_length},
            device=device
        )
        self.device = device
        self.context_length = context_length
        print(f"✓ Model loaded (context: {context_length} tokens)")
        
    def generate(self, 
                 prompt="s-9 o-0 t-38",
                 temperature=0.8,
                 max_tokens=2048,
                 verbose=True):
        """
        Generate a single music sequence.
        
        Args:
            prompt: Starting REMIGEN tokens
            temperature: Sampling temperature (0.5-1.5)
            max_tokens: Total tokens to generate (including prompt)
            verbose: Print progress
            
        Returns:
            Dictionary with tokens and metadata
        """
        if verbose:
            print(f"🎵 Generating...")
            print(f"   Prompt: {prompt[:60]}...")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Temperature: {temperature}")
        
        # Generate
        output_dict = self.model.generate(
            prompt=prompt,
            temperature=temperature,
            max_length=max_tokens,  # FIXED: this is total length, not delta
            end_tokens=[],
            forbidden_tokens=["[PAD]", "[EOS]"],
            return_structured_output=True
        )
        
        # Extract and filter tokens
        tokens_raw = output_dict["output"]
        tokens_list = tokens_raw.split()
        
        # Filter: keep only valid REMIGEN tokens (format: prefix-value)
        valid_tokens = [t for t in tokens_list if '-' in t and not t.startswith('[')]
        
        if verbose:
            invalid_count = len(tokens_list) - len(valid_tokens)
            bars = sum(1 for t in valid_tokens if t == "b-1")
            print(f"✓ Generated {len(valid_tokens)} tokens ({bars} bars)")
            if invalid_count > 0:
                print(f"   Filtered {invalid_count} invalid tokens")
        
        return {
            "tokens": " ".join(valid_tokens),
            "num_tokens": len(valid_tokens),
            "bars": sum(1 for t in valid_tokens if t == "b-1")
        }
    
    def generate_long(self,
                      prompt="s-9 o-0 t-38", 
                      temperature=0.8,
                      target_bars=32,
                      chunk_tokens=1024,
                      max_iterations=20,
                      verbose=True):
        """
        Generate long sequences by chunking (avoids OOM).
        
        Key insight: Generate in fixed-size chunks, using only recent context
        as prompt for next chunk (sliding window approach).
        
        Args:
            prompt: Starting tokens
            temperature: Sampling temperature
            target_bars: Stop after N bars (or None for max_iterations)
            chunk_tokens: Tokens per chunk
            max_iterations: Maximum chunks to generate
            verbose: Print progress
            
        Returns:
            Dictionary with tokens and metadata
        """
        if verbose:
            print(f"🎵 Long generation (chunked)...")
            print(f"   Target: {target_bars} bars" if target_bars else f"   Max iterations: {max_iterations}")
            print(f"   Chunk size: {chunk_tokens} tokens")
        
        all_tokens = prompt.split()
        total_bars = sum(1 for t in all_tokens if t == "b-1")
        
        for iteration in range(max_iterations):
            # Use sliding window: last N tokens as context
            # This prevents context from growing unbounded
            window_size = min(self.context_length - chunk_tokens - 100, len(all_tokens))
            context = " ".join(all_tokens[-window_size:])
            
            if verbose:
                print(f"\n📝 Iteration {iteration + 1}/{max_iterations}")
                print(f"   Context: {len(context.split())} tokens")
            
            # Generate next chunk
            chunk_result = self.generate(
                prompt=context,
                temperature=temperature,
                max_tokens=len(context.split()) + chunk_tokens,
                verbose=False
            )
            
            # Extract only NEW tokens (after the context)
            chunk_tokens_list = chunk_result["tokens"].split()
            context_tokens_list = context.split()
            
            # Find where new content starts
            if len(chunk_tokens_list) > len(context_tokens_list):
                new_tokens = chunk_tokens_list[len(context_tokens_list):]
                all_tokens.extend(new_tokens)
                
                # Count new bars
                new_bars = sum(1 for t in new_tokens if t == "b-1")
                total_bars += new_bars
                
                if verbose:
                    print(f"   Added: {len(new_tokens)} tokens ({new_bars} bars)")
                    print(f"   Total: {len(all_tokens)} tokens ({total_bars} bars)")
                
                # Check stopping condition
                if target_bars and total_bars >= target_bars:
                    if verbose:
                        print(f"\n✓ Reached target: {total_bars} bars")
                    break
            else:
                if verbose:
                    print(f"⚠️  No new tokens generated, stopping")
                break
            
            # Clear CUDA cache after each iteration
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        final_tokens = " ".join(all_tokens)
        
        if verbose:
            print(f"\n✓ Generation complete!")
            print(f"   Final: {len(all_tokens)} tokens, {total_bars} bars")
        
        return {
            "tokens": final_tokens,
            "num_tokens": len(all_tokens),
            "bars": total_bars
        }


class MIDIConverter:
    """Handles REMIGEN → MIDI conversion"""
    
    def __init__(self):
        self.decoder = mp.MidiDecoder('REMIGEN')
    
    def tokens_to_midi(self, tokens_str, output_path, clean=True):
        """
        Convert REMIGEN tokens to MIDI file.
        
        Args:
            tokens_str: Space-separated REMIGEN tokens
            output_path: Path to save .mid file
            clean: Whether to clean tokens (default True)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if clean:
                cleaned_tokens = self.clean_tokens(tokens_str)
                tokens = cleaned_tokens.split()
            else:
                tokens = tokens_str.strip().split()
            
            midi_obj = self.decoder.decode_from_token_str_list(tokens)
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            midi_obj.dump(output_path)
            
            return True
        except Exception as e:
            print(f"✗ Decoding error: {type(e).__name__}: {e}")
            return False
    
    def clean_tokens(self, tokens_str):
        """Remove incomplete and invalid tokens"""
        tokens = tokens_str.strip().split()
        cleaned = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Skip invalid tokens (must have format 'X-Y')
            if '-' not in token:
                i += 1
                continue
            
            # If it's a pitch token, validate full p-d-v triplet
            if token.startswith('p-'):
                # Need: p-X d-Y v-Z
                if (i + 2 < len(tokens) and 
                    tokens[i + 1].startswith('d-') and 
                    tokens[i + 2].startswith('v-')):
                    cleaned.extend([tokens[i], tokens[i + 1], tokens[i + 2]])
                    i += 3
                else:
                    i += 1  # Skip incomplete triplet
            # Skip orphan durations or velocities
            elif token.startswith('d-') or token.startswith('v-'):
                i += 1
            else:
                cleaned.append(token)
                i += 1
        
        return " ".join(cleaned)




# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_short_generation():
    """Generate short pieces (< 2048 tokens)"""
    
    # Initialize
    generator = MusicGenerator(
        model_path="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/lmd_remigen_xlstm/run_20260115-1028",
        context_length=2048,
        device="cuda"
    )
    
    converter = MIDIConverter()
    output_dir = Path("./generated_music")
    output_dir.mkdir(exist_ok=True)
    
    # Generate 3 short pieces
    for i in range(3):
        print(f"\n{'='*60}")
        print(f"Generating song {i+1}/3")
        print('='*60)
        
        result = generator.generate(
            prompt="s-9 o-0 t-35 i-128 p-170 d-3 v-31",
            temperature=0.8,
            max_tokens=2048,  # Stay within context limit
            verbose=True
        )
        
        # Save to MIDI
        midi_path = output_dir / f"song_{i:03d}.mid"
        success = converter.tokens_to_midi(result["tokens"], str(midi_path))
        
        if success:
            print(f"✓ Saved: {midi_path}")
        
        # Also save tokens
        token_path = output_dir / f"song_{i:03d}_tokens.txt"
        with open(token_path, 'w') as f:
            f.write(result["tokens"])
    
    print(f"\n✓ All songs generated in: {output_dir}")


def example_long_generation():
    """Generate long pieces using chunking"""
    
    # Initialize with LARGER context for better quality
    generator = MusicGenerator(
        model_path="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/lmd_remigen_xlstm/run_20260115-1028",
        context_length=4096,  # Increase context for inference
        device="cuda"
    )
    
    converter = MIDIConverter()
    output_dir = Path("./generated_music_long")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating long piece")
    print('='*60)
    
    result = generator.generate_long(
        prompt="s-9 o-0 t-35 i-128 p-170 d-3 v-31",
        temperature=0.8,
        target_bars=64,      # Generate 64 bars
        chunk_tokens=1024,   # 1024 tokens per chunk
        max_iterations=20,
        verbose=True
    )
    
    # Save
    midi_path = output_dir / "long_song.mid"
    success = converter.tokens_to_midi(result["tokens"], str(midi_path))
    
    if success:
        print(f"\n✓ Saved: {midi_path}")
    
    # Save tokens
    token_path = output_dir / "long_song_tokens.txt"
    with open(token_path, 'w') as f:
        f.write(result["tokens"])
    
    print(f"✓ Tokens saved: {token_path}")


def example_batch_with_variety():
    """Generate batch with different temperatures"""
    
    generator = MusicGenerator(
        model_path="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/lmd_remigen_xlstm/run_20260115-1028",
        context_length=2048,
        device="cuda"
    )
    
    converter = MIDIConverter()
    output_dir = Path("./generated_variety")
    output_dir.mkdir(exist_ok=True)
    
    temperatures = [0.5, 0.8, 1.0, 1.2]  # Low to high creativity
    
    for i, temp in enumerate(temperatures):
        print(f"\n{'='*60}")
        print(f"Song {i+1}/4 - Temperature: {temp}")
        print('='*60)
        
        result = generator.generate(
            prompt="s-9 o-0 t-35 i-128 p-170 d-3 v-31",
            temperature=temp,
            max_tokens=2048,
            verbose=True
        )
        
        # Save
        midi_path = output_dir / f"song_temp{temp:.1f}.mid"
        converter.tokens_to_midi(result["tokens"], str(midi_path))
        print(f"✓ Saved: {midi_path}")
    
    print(f"\n✓ Variety batch complete: {output_dir}")


# =============================================================================
# SIMPLE API FOR YOUR EXPERIMENTS
# =============================================================================

def generate_music(model_path,
                   num_songs=1,
                   max_tokens=2048,
                   temperature=0.8,
                   output_dir="./generated",
                   long_mode=False,
                   target_bars=None):
    """
    Simple one-function API for music generation.
    
    Args:
        model_path: Path to your trained model
        num_songs: Number of songs to generate
        max_tokens: Max tokens per song (for short mode)
        temperature: Creativity (0.5=safe, 1.2=wild)
        output_dir: Where to save files
        long_mode: Use chunking for long sequences
        target_bars: For long mode, target number of bars
        
    Returns:
        List of output paths
    """
    # Setup
    context_length = 4096 if long_mode else 2048
    generator = MusicGenerator(model_path, context_length=context_length)
    converter = MIDIConverter()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    outputs = []
    
    for i in range(num_songs):
        print(f"\n{'='*60}")
        print(f"Song {i+1}/{num_songs}")
        print('='*60)
        
        # Generate
        if long_mode:
            result = generator.generate_long(
                prompt="s-9 o-0 t-35",
                temperature=temperature,
                target_bars=target_bars or 64,
                chunk_tokens=1024,
                verbose=True
            )
        else:
            result = generator.generate(
                prompt="s-9 o-0 t-35",
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=True
            )
        
        # Save
        midi_path = output_dir / f"song_{i:03d}.mid"
        converter.tokens_to_midi(result["tokens"], str(midi_path))
        outputs.append(midi_path)
        
        print(f"✓ Saved: {midi_path}")
    
    return outputs


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_short_generation()
    # example_long_generation()
    # example_batch_with_variety()
    
    # Or use the simple API:
    generate_music(
        model_path="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/repos/helibrunna/output/lmd_remigen_xlstm/run_20260115-1028",
        num_songs=2,
        max_tokens=2048,
        temperature=0.8,
        output_dir="./my_music"
    )
