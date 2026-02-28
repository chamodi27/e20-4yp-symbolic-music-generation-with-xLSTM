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
            max_length=max_tokens,
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

    def _find_last_bar_index(self, tokens_list):
        """
        Find the index of the last 'b-1' token.

        Args:
            tokens_list: List of token strings

        Returns:
            Index of last 'b-1', or None if not found
        """
        for i in range(len(tokens_list) - 1, -1, -1):
            if tokens_list[i] == "b-1":
                return i
        return None

    def generate_long(self,
                      prompt="s-9 o-0 t-38",
                      temperature=0.8,
                      target_bars=32,
                      max_iterations=50,
                      verbose=True):
        """
        Generate long sequences by bar-aware chunking.

        Strategy:
        1. Generate chunks of ~400 new tokens (2-3 bars)
        2. Always cut at last complete bar (b-1)
        3. Use last 1600 tokens as context for continuity
        4. This ensures NO incomplete triplets or orphan tokens

        Args:
            prompt: Starting tokens
            temperature: Sampling temperature
            target_bars: Stop after N bars
            max_iterations: Maximum chunks to generate
            verbose: Print progress

        Returns:
            Dictionary with tokens and metadata
        """
        if verbose:
            print(f"🎵 Long generation (bar-aware chunking)...")
            print(f"   Target: {target_bars} bars")
            print(f"   Strategy: Generate 2-3 bars per iteration, cut at b-1")

        all_tokens = prompt.split()
        total_bars = sum(1 for t in all_tokens if t == "b-1")

        # Parameters tuned for REMIGEN (avg 158 tokens/bar from Museformer)
        CONTEXT_SIZE = 1500  # ~10 bars of context
        NEW_TOKENS = 400     # ~2-3 bars per iteration

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n📝 Iteration {iteration + 1}/{max_iterations}")

            # Step 1: Prepare context (last CONTEXT_SIZE tokens)
            if len(all_tokens) > CONTEXT_SIZE:
                context_tokens = all_tokens[-CONTEXT_SIZE:]
            else:
                context_tokens = all_tokens

            context = " ".join(context_tokens)

            if verbose:
                print(f"   Context: {len(context_tokens)} tokens ({sum(1 for t in context_tokens if t == 'b-1')} bars)")

            # Step 2: Generate chunk (context + ~400 new tokens)
            target_length = len(context_tokens) + NEW_TOKENS

            chunk_result = self.generate(
                prompt=context,
                temperature=temperature,
                max_tokens=target_length,
                verbose=False
            )

            # Step 3: Extract only NEW tokens
            chunk_tokens = chunk_result["tokens"].split()

            if len(chunk_tokens) <= len(context_tokens):
                if verbose:
                    print(f"⚠️  No new tokens generated, stopping")
                break

            new_tokens = chunk_tokens[len(context_tokens):]

            # Step 4: Find last complete bar in new tokens
            last_bar_idx = self._find_last_bar_index(new_tokens)

            if last_bar_idx is None:
                if verbose:
                    print(f"⚠️  No complete bar (b-1) found in new tokens, stopping")
                break

            # Step 5: Take only complete bars (up to and including last b-1)
            complete_new_tokens = new_tokens[:last_bar_idx + 1]
            all_tokens.extend(complete_new_tokens)

            # Count bars
            new_bars = sum(1 for t in complete_new_tokens if t == "b-1")
            total_bars += new_bars

            if verbose:
                print(f"   Generated: {len(new_tokens)} tokens")
                print(f"   Kept (complete bars): {len(complete_new_tokens)} tokens ({new_bars} bars)")
                print(f"   Total: {len(all_tokens)} tokens ({total_bars} bars)")

            # Step 6: Check stopping condition
            if total_bars >= target_bars:
                if verbose:
                    print(f"\n✓ Reached target: {total_bars} bars")
                break

            # Clear CUDA cache
            if self.device == "cuda":
                import torch
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

    def generate_batch(self,
                   temperatures=[0.6, 0.7, 0.8, 0.9],
                   target_bars_list=[32, 48, 64],
                   pieces_per_combination=5,
                   prompt="s-9 o-0 t-35",
                   output_base_dir="./generated_batch",
                   verbose=True):
        """
        Generate a batch of music pieces with different parameter combinations.
        
        Args:
            temperatures: List of temperature values to try
            target_bars_list: List of target bar counts
            pieces_per_combination: Number of pieces per (temp, bars) combination
            prompt: Starting prompt for all generations
            output_base_dir: Base directory for outputs
            verbose: Print progress
            
        Returns:
            Dictionary with batch results
        """
        import time
        from datetime import datetime
        import os
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            print("="*70)
            print("BATCH GENERATION")
            print("="*70)
            print(f"Temperatures: {temperatures}")
            print(f"Target bars: {target_bars_list}")
            print(f"Pieces per combination: {pieces_per_combination}")
            print(f"Total pieces: {len(temperatures) * len(target_bars_list) * pieces_per_combination}")
            print(f"Output directory: {output_dir}")
            print("="*70)
        
        # Results storage
        results = []
        piece_number = 0
        total_pieces = len(temperatures) * len(target_bars_list) * pieces_per_combination
        
        # Open analysis file
        analysis_path = os.path.join(output_dir, "analysis_report.txt")
        
        with open(analysis_path, 'w') as f:
            # Write header
            f.write("="*70 + "\n")
            f.write("BATCH GENERATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total pieces: {total_pieces}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("="*70 + "\n\n")
            
            # Write table header
            f.write("SUMMARY TABLE:\n")
            f.write("-"*140 + "\n")
            f.write(f"{'#':<4} | {'Filename':<30} | {'Temp':<5} | {'Target':<6} | {'Actual':<6} | {'Tokens':<6} | {'Insts':<5} | {'Errors':<6} | {'Clean':<5} | {'Time(s)':<7} | {'Status':<6}\n")
            f.write("-"*140 + "\n")
        
        # Generate all combinations
        for temp in temperatures:
            for target_bars in target_bars_list:
                for piece_idx in range(pieces_per_combination):
                    piece_number += 1
                    
                    # Generate filename
                    filename = f"temp_{temp:.1f}_bars_{target_bars}_{piece_idx+1:03d}.mid"
                    filepath = os.path.join(output_dir, filename)
                    
                    if verbose:
                        print(f"\n[{piece_number}/{total_pieces}] Generating {filename}...")
                    
                    # Record start time
                    start_time = time.time()
                    
                    try:
                        # Generate music
                        result = self.generate_long(
                            prompt=prompt,
                            temperature=temp,
                            target_bars=target_bars,
                            verbose=False
                        )
                        
                        generation_time = time.time() - start_time
                        
                        # Analyze tokens
                        analysis = self.analyze_tokens(result['tokens'], verbose=False)
                        
                        # Try converting without cleaning first
                        converter = MIDIConverter()
                        success_no_clean = converter.tokens_to_midi(result['tokens'], filepath, clean=False)
                        
                        used_cleaning = False
                        status = "✓"
                        
                        if not success_no_clean:
                            # Try with cleaning
                            if verbose:
                                print(f"   First attempt failed, trying with cleaning...")
                            success_with_clean = converter.tokens_to_midi(result['tokens'], filepath, clean=True)
                            
                            if success_with_clean:
                                used_cleaning = True
                                status = "✓"
                            else:
                                status = "✗"
                                if verbose:
                                    print(f"   Both attempts failed!")
                        
                        # Store result
                        result_data = {
                            "piece_number": piece_number,
                            "filename": filename,
                            "temperature": temp,
                            "target_bars": target_bars,
                            "actual_bars": analysis['bars'],
                            "tokens": analysis['total_tokens'],
                            "instruments": analysis['num_instruments'],
                            "errors": len(analysis['grammar_errors']),
                            "used_cleaning": used_cleaning,
                            "generation_time": generation_time,
                            "status": status,
                            "analysis": analysis
                        }
                        results.append(result_data)
                        
                        if verbose:
                            print(f"   ✓ Success - {result_data['actual_bars']} bars, {result_data['tokens']} tokens")
                            print(f"   Time: {generation_time:.1f}s, Cleaning: {'Yes' if used_cleaning else 'No'}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"   ✗ Failed: {e}")
                        
                        generation_time = time.time() - start_time
                        
                        result_data = {
                            "piece_number": piece_number,
                            "filename": filename,
                            "temperature": temp,
                            "target_bars": target_bars,
                            "actual_bars": 0,
                            "tokens": 0,
                            "instruments": 0,
                            "errors": -1,
                            "used_cleaning": False,
                            "generation_time": generation_time,
                            "status": "✗",
                            "error": str(e),
                            "analysis": None
                        }
                        results.append(result_data)
                    
                    # Write to table immediately (incremental logging)
                    with open(analysis_path, 'a') as f:
                        clean_str = "Yes" if result_data['used_cleaning'] else "No"
                        f.write(f"{result_data['piece_number']:<4} | {result_data['filename']:<30} | {result_data['temperature']:<5.1f} | {result_data['target_bars']:<6} | {result_data['actual_bars']:<6} | {result_data['tokens']:<6} | {result_data['instruments']:<5} | {result_data['errors']:<6} | {clean_str:<5} | {result_data['generation_time']:<7.1f} | {result_data['status']:<6}\n")
        
        # Write detailed analysis section
        with open(analysis_path, 'a') as f:
            f.write("-"*140 + "\n\n")
            f.write("="*70 + "\n")
            f.write("DETAILED ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            for result_data in results:
                f.write(f"[{result_data['piece_number']}/{total_pieces}] {result_data['filename']}\n")
                f.write("-"*70 + "\n")
                
                if result_data['status'] == "✗":
                    f.write(f"Status: ✗ FAILED\n")
                    if 'error' in result_data:
                        f.write(f"Error: {result_data['error']}\n")
                else:
                    clean_status = "with cleaning" if result_data['used_cleaning'] else "no cleaning needed"
                    f.write(f"Status: ✓ Success ({clean_status})\n")
                
                f.write(f"Temperature: {result_data['temperature']}\n")
                f.write(f"Target bars: {result_data['target_bars']}, Actual bars: {result_data['actual_bars']}\n")
                f.write(f"Generation time: {result_data['generation_time']:.1f}s\n\n")
                
                # Write full analysis if available
                if result_data['analysis']:
                    analysis = result_data['analysis']
                    
                    f.write(f"📊 OVERALL STATISTICS:\n")
                    f.write(f"   Total tokens: {analysis['total_tokens']}\n")
                    f.write(f"   Total bars: {analysis['bars']}\n")
                    f.write(f"   Total notes: {analysis['notes']}\n")
                    f.write(f"   Unique instruments: {analysis['num_instruments']}\n\n")
                    
                    f.write(f"🎵 BAR STATISTICS:\n")
                    f.write(f"   Average bar length: {analysis['avg_bar_length']:.1f} tokens\n")
                    f.write(f"   Min bar length: {analysis['min_bar_length']} tokens\n")
                    f.write(f"   Max bar length: {analysis['max_bar_length']} tokens\n\n")
                    
                    f.write(f"🔤 TOKEN TYPES:\n")
                    for token_type, count in sorted(analysis['token_types'].items()):
                        f.write(f"   {token_type}-: {count}\n")
                    f.write("\n")
                    
                    f.write(f"🎹 INSTRUMENTS:\n")
                    for inst in sorted(analysis['instruments']):
                        f.write(f"   {inst}\n")
                    f.write("\n")
                    
                    f.write(f"✅ SEQUENCE HEALTH:\n")
                    f.write(f"   Ends with b-1: {'✓' if analysis['ends_with_bar'] else '✗'}\n")
                    f.write(f"   Grammar errors: {len(analysis['grammar_errors'])}\n")
                    
                    if analysis['grammar_errors']:
                        f.write(f"\n⚠️  GRAMMAR ERRORS (showing first 5):\n")
                        for error in analysis['grammar_errors'][:5]:
                            f.write(f"   - {error}\n")
                        if len(analysis['grammar_errors']) > 5:
                            f.write(f"   ... and {len(analysis['grammar_errors']) - 5} more\n")
                    
                    f.write(f"\n🔍 SEQUENCE EDGES:\n")
                    f.write(f"   First 10: {' '.join(analysis['first_10'])}\n")
                    f.write(f"   Last 10: {' '.join(analysis['last_10'])}\n")
                
                f.write("\n" + "-"*70 + "\n\n")
            
            # Write summary statistics
            f.write("="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*70 + "\n")
            
            successful = [r for r in results if r['status'] == "✓"]
            failed = [r for r in results if r['status'] == "✗"]
            with_cleaning = [r for r in successful if r['used_cleaning']]
            
            f.write(f"Total pieces: {len(results)}\n")
            f.write(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)\n")
            f.write(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)\n")
            f.write(f"Required cleaning: {len(with_cleaning)} ({len(with_cleaning)/len(successful)*100:.1f}% of successful)\n\n")
            
            if successful:
                avg_bars = sum(r['actual_bars'] for r in successful) / len(successful)
                avg_tokens = sum(r['tokens'] for r in successful) / len(successful)
                avg_time = sum(r['generation_time'] for r in successful) / len(successful)
                avg_instruments = sum(r['instruments'] for r in successful) / len(successful)
                avg_errors = sum(r['errors'] for r in successful) / len(successful)
                
                f.write(f"Average bars generated: {avg_bars:.1f}\n")
                f.write(f"Average tokens: {avg_tokens:.1f}\n")
                f.write(f"Average generation time: {avg_time:.1f}s\n")
                f.write(f"Average instruments: {avg_instruments:.1f}\n")
                f.write(f"Average grammar errors: {avg_errors:.1f}\n")
            
            f.write("="*70 + "\n")
        
        if verbose:
            print("\n" + "="*70)
            print("BATCH GENERATION COMPLETE")
            print("="*70)
            print(f"Output directory: {output_dir}")
            print(f"Analysis report: {analysis_path}")
            print(f"Success rate: {len(successful)}/{len(results)}")
        
        return {
            "output_dir": output_dir,
            "analysis_path": analysis_path,
            "results": results,
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "with_cleaning": len(with_cleaning)
            }
        }

    def analyze_tokens(self, tokens_str, verbose=True):
        """
        Analyze REMIGEN token sequence structure and statistics.
        
        Args:
            tokens_str: Space-separated REMIGEN tokens
            verbose: Print detailed report
            
        Returns:
            Dictionary with analysis results
        """
        tokens = tokens_str.strip().split()
        
        # Initialize counters
        analysis = {
            "total_tokens": len(tokens),
            "token_types": {},
            "bars": 0,
            "instruments": set(),
            "notes": 0,
            "grammar_errors": [],
            "bar_lengths": [],
            "first_10": tokens[:10] if len(tokens) >= 10 else tokens,
            "last_10": tokens[-10:] if len(tokens) >= 10 else tokens,
            "ends_with_bar": tokens[-1] == "b-1" if tokens else False
        }
        
        # Count token types
        for token in tokens:
            if '-' not in token:
                analysis["grammar_errors"].append(f"Invalid token format: {token}")
                continue
            
            prefix = token.split('-')[0]
            analysis["token_types"][prefix] = analysis["token_types"].get(prefix, 0) + 1
            
            # Track specific types
            if prefix == "b":
                analysis["bars"] += 1
            elif prefix == "i":
                analysis["instruments"].add(token)
            elif prefix == "p":
                analysis["notes"] += 1
        
        # Check grammar (p-d-v triplets)
        i = 0
        current_bar_length = 0
        bar_start = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == "b-1":
                # Record bar length
                analysis["bar_lengths"].append(current_bar_length)
                current_bar_length = 0
                bar_start = i + 1
                i += 1
                continue
            
            if token.startswith("p-"):
                # Expect d- and v- to follow
                if i + 1 >= len(tokens) or not tokens[i + 1].startswith("d-"):
                    analysis["grammar_errors"].append(f"Incomplete triplet at token {i}: {token} (missing duration)")
                elif i + 2 >= len(tokens) or not tokens[i + 2].startswith("v-"):
                    analysis["grammar_errors"].append(f"Incomplete triplet at token {i}: {token} {tokens[i+1]} (missing velocity)")
            
            elif token.startswith("d-") or token.startswith("v-"):
                # Check if it's an orphan (not part of p-d-v)
                if i == 0 or not tokens[i-1].startswith("p-") and not tokens[i-1].startswith("d-"):
                    analysis["grammar_errors"].append(f"Orphan token at {i}: {token}")
            
            current_bar_length += 1
            i += 1
        
        # Calculate statistics
        analysis["num_instruments"] = len(analysis["instruments"])
        analysis["avg_bar_length"] = sum(analysis["bar_lengths"]) / len(analysis["bar_lengths"]) if analysis["bar_lengths"] else 0
        analysis["min_bar_length"] = min(analysis["bar_lengths"]) if analysis["bar_lengths"] else 0
        analysis["max_bar_length"] = max(analysis["bar_lengths"]) if analysis["bar_lengths"] else 0
        analysis["has_errors"] = len(analysis["grammar_errors"]) > 0
        
        # Print report if verbose
        if verbose:
            print("="*60)
            print("TOKEN ANALYSIS REPORT")
            print("="*60)
            print(f"\n📊 OVERALL STATISTICS:")
            print(f"   Total tokens: {analysis['total_tokens']}")
            print(f"   Total bars: {analysis['bars']}")
            print(f"   Total notes: {analysis['notes']}")
            print(f"   Unique instruments: {analysis['num_instruments']}")
            
            print(f"\n🎵 BAR STATISTICS:")
            print(f"   Average bar length: {analysis['avg_bar_length']:.1f} tokens")
            print(f"   Min bar length: {analysis['min_bar_length']} tokens")
            print(f"   Max bar length: {analysis['max_bar_length']} tokens")
            
            print(f"\n🔤 TOKEN TYPES:")
            for token_type, count in sorted(analysis['token_types'].items()):
                print(f"   {token_type}-: {count}")
            
            print(f"\n🎹 INSTRUMENTS:")
            for inst in sorted(analysis['instruments']):
                print(f"   {inst}")
            
            print(f"\n✅ SEQUENCE HEALTH:")
            print(f"   Ends with b-1: {'✓' if analysis['ends_with_bar'] else '✗'}")
            print(f"   Grammar errors: {len(analysis['grammar_errors'])}")
            
            if analysis['grammar_errors']:
                print(f"\n⚠️  GRAMMAR ERRORS (showing first 5):")
                for error in analysis['grammar_errors'][:5]:
                    print(f"   - {error}")
                if len(analysis['grammar_errors']) > 5:
                    print(f"   ... and {len(analysis['grammar_errors']) - 5} more")
            
            print(f"\n🔍 SEQUENCE EDGES:")
            print(f"   First 10 tokens: {' '.join(analysis['first_10'])}")
            print(f"   Last 10 tokens: {' '.join(analysis['last_10'])}")
            print("="*60)
        
        return analysis

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
