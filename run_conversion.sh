#!/bin/bash
# Find conda path
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Create and activate environment
conda create -y -n midi2wav_env python=3.9
conda activate midi2wav_env

# Install fluidsynth and midi2audio
conda install -y -c conda-forge fluidsynth
pip install midi2audio

# Run conversion script
python /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/convert_midi_to_wav.py

echo "CONVERSION_DONE" > /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/conversion_status.txt
