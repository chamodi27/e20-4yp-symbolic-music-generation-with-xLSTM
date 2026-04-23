#!/bin/bash
# Find conda path
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate environment
conda activate midi2wav_env

# Run conversion script
python /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/convert_midi_to_wav_human.py

echo "CONVERSION_DONE" > /scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen/conversion_status_human.txt
