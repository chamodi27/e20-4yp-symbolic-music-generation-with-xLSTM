#!/bin/bash
source /home/e20365/miniconda3/etc/profile.d/conda.sh
conda activate museformer

mkdir -p output_log
mkdir -p output/generation

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

echo "Generating tokens..."
fairseq-generate data-bin/lmd6remi \
  --path checkpoints/backup/checkpoint_best.pt \
  --user-dir museformer \
  --task museformer_language_modeling \
  --sampling --sampling-topk 8 --beam 1 --nbest 1 \
  --min-len 1024 \
  --max-len-b 20480 \
  --num-workers 4 \
  --seed 1 \
  --batch-size 1 \
  --results-path output_log/generation_out

echo "Extracting tokens..."
python tools/batch_extract_log.py output_log/generation_out/generate-test.txt output/generation --start_idx 1

echo "Converting to MIDI..."
python tools/batch_generate_midis.py --encoding-method REMIGEN2 --input-dir output/generation --output-dir output/generation

echo "Done."
