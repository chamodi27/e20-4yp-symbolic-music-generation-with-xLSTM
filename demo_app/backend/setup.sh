#!/bin/bash
# setup.sh — Complete environment setup for the xLSTM Music Generation Backend
#
# Run this ONCE after cloning the backend folder onto a new server.
# This script:
#   1. Creates the conda environment from environment.yml
#   2. Installs midiprocessor AFTER miditoolkit (required build order)
#
# Usage:
#   bash setup.sh

set -e  # exit on any error

ENV_NAME="xlstm-api"

echo "=== Step 1: Creating conda environment '$ENV_NAME' ==="
conda env create -n "$ENV_NAME" -f environment.yml

echo ""
echo "=== Step 2: Installing midiprocessor (requires miditoolkit to be present first) ==="
# midiprocessor's setup.py imports miditoolkit at build time.
# --no-build-isolation lets pip see the already-installed miditoolkit instead
# of running in a clean isolated build environment where it can't be found.
conda run -n "$ENV_NAME" pip install --no-build-isolation git+https://github.com/btyu/MidiProcessor.git

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the API server:"
echo "  bash start.sh"
