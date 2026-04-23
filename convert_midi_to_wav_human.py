import os
import random
import glob
import subprocess

midi_dir = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/data/museformer_baseline/29k_backup/final_output"
wav_dir = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/evaluation/chamodi_eval/human"
soundfont = "/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/assets/soundfonts/TimGM6mb.sf2"

os.makedirs(wav_dir, exist_ok=True)

all_midis = glob.glob(os.path.join(midi_dir, "*.mid"))
random.seed(43) # Different seed for different directory
# Ensure we don't try to sample more than what's available
num_to_sample = min(20, len(all_midis))
sampled_midis = random.sample(all_midis, num_to_sample)

for i, midi_file in enumerate(sampled_midis):
    base_name = os.path.basename(midi_file)
    wav_file = os.path.join(wav_dir, base_name.replace(".mid", ".wav"))
    print(f"Converting {i+1}/{num_to_sample}: {base_name} -> {wav_file}")
    cmd = ["fluidsynth", "-ni", "-F", wav_file, "-r", "44100", soundfont, midi_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Conversion complete!")
